

from __future__ import annotations

from collections.abc import Callable
import inspect

import numpy as np
import scipy.sparse.linalg as spla

import config
from quantum_hhl import (
    MQ_AVAILABLE,
    _HHL_IMPORT_ERROR,
    _dense_from_matvec,
    _run_quantum_hhl,
)


def _relative_residual(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x: np.ndarray,
) -> float:
    rhs = np.asarray(b)
    residual = np.asarray(matvec(np.asarray(x))) - rhs
    denom = float(np.linalg.norm(rhs) + 1e-12)
    return float(np.linalg.norm(residual) / denom)


def _dense_direct_solve(matrix: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    
    try:
        x = np.linalg.solve(matrix, b)
        return np.asarray(x), {"actual_backend": "dense_direct"}
    except np.linalg.LinAlgError:
        x, _, _, _ = np.linalg.lstsq(matrix, b, rcond=None)
        return np.asarray(x), {"actual_backend": "dense_lstsq"}


def _call_krylov_solver(
    name: str,
    lin_op: spla.LinearOperator,
    b: np.ndarray,
    precision: float,
    *,
    restart: int = 50,
    maxiter: int = 200,
) -> tuple[np.ndarray, int]:
    solver = getattr(spla, name)
    params = inspect.signature(solver).parameters
    kwargs: dict[str, object] = {"maxiter": maxiter}

    if "rtol" in params:
        kwargs["rtol"] = precision
        kwargs["atol"] = 0.0
    else:
        kwargs["tol"] = precision

    if "restart" in params:
        kwargs["restart"] = restart

    result = solver(lin_op, b, **kwargs)
    x, info = result[:2]
    return np.asarray(x), int(info)


def _iterative_solve(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    precision: float,
    *,
    preferred: str = "lgmres",
    restart: int = 50,
    maxiter: int = 200,
) -> tuple[np.ndarray, dict[str, object]]:
    
    rhs = np.asarray(b)
    dim = int(rhs.shape[0])

    def safe_matvec(x: np.ndarray) -> np.ndarray:
        return np.array(matvec(x), dtype=rhs.dtype, copy=True)

    lin_op = spla.LinearOperator((dim, dim), matvec=safe_matvec, dtype=rhs.dtype)

    methods = [preferred]
    if preferred != "gmres":
        methods.append("gmres")

    errors: list[str] = []
    for method in methods:
        try:
            x, info_code = _call_krylov_solver(
                method,
                lin_op,
                rhs,
                precision,
                restart=restart,
                maxiter=maxiter,
            )
            rel_res = _relative_residual(safe_matvec, rhs, x)
            if info_code == 0 and np.isfinite(rel_res) and rel_res <= max(10.0 * precision, 1e-8):
                return x, {
                    "actual_backend": method,
                    "iterative_info": info_code,
                    "iterative_method": method,
                    "iterative_accepted": True,
                    "iterative_rel_residual": rel_res,
                }

            errors.append(
                f"{method} returned info={info_code} with rel_residual={rel_res:.3e}"
            )
        except Exception as exc:  # pragma: no cover - exercised when SciPy backend raises
            errors.append(f"{method} raised {type(exc).__name__}: {exc}")

    x, info_code = _call_krylov_solver(
        methods[-1],
        lin_op,
        rhs,
        precision,
        restart=restart,
        maxiter=maxiter,
    )
    return x, {
        "actual_backend": methods[-1],
        "iterative_info": info_code,
        "iterative_method": methods[-1],
        "iterative_accepted": False,
        "iterative_rel_residual": _relative_residual(safe_matvec, rhs, x),
        "iterative_warnings": errors,
    }


def _solve_classical_with_info(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    precision: float,
    *,
    backend: str,
    dense_matrix: np.ndarray | None,
    max_dense_dim: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rhs = np.asarray(b)
    dim = int(rhs.shape[0])

    if backend == "dense":
        matrix = dense_matrix if dense_matrix is not None else _dense_from_matvec(matvec, dim)
        x, info = _dense_direct_solve(matrix, rhs)
        info["used_dense_reconstruction"] = True
        return x, info

    if backend in {"krylov", "iterative", "lgmres", "gmres"}:
        preferred = "gmres" if backend == "gmres" else "lgmres"
        x, info = _iterative_solve(matvec, rhs, precision, preferred=preferred)
        info["used_dense_reconstruction"] = False
        return x, info

    if backend != "auto":
        raise ValueError(f"Unsupported solver backend: {backend}")

    if dense_matrix is not None or dim <= max_dense_dim:
        matrix = dense_matrix if dense_matrix is not None else _dense_from_matvec(matvec, dim)
        x, info = _dense_direct_solve(matrix, rhs)
        info["used_dense_reconstruction"] = True
        return x, info

    x, info = _iterative_solve(matvec, rhs, precision, preferred="lgmres")
    info["used_dense_reconstruction"] = False
    return x, info


def solve_linear_system_with_info(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    precision: float,
    *,
    backend: str = "hhl_research",
    max_dense_dim: int = 4096,
    hhl_acceptance_rel_residual: float = 0.25,
) -> tuple[np.ndarray, dict[str, object]]:
    rhs = np.asarray(b)
    dim = int(rhs.shape[0])
    requested_backend = str(backend).lower()
    strict_hhl = bool(getattr(config, "STRICT_HHL", False))

    info: dict[str, object] = {
        "requested_backend": requested_backend,
        "dim": dim,
        "precision": float(precision),
        "max_dense_dim": int(max_dense_dim),
        "strict_hhl": strict_hhl,
        "used_mindquantum": False,
        "used_quantum_circuit": False,
        "research_mode": requested_backend == "hhl_research",
    }

    if requested_backend != "hhl_research":
        x, backend_info = _solve_classical_with_info(
            matvec,
            rhs,
            precision,
            backend=requested_backend,
            dense_matrix=None,
            max_dense_dim=max_dense_dim,
        )
        info.update(backend_info)
        info["rel_residual"] = _relative_residual(matvec, rhs, x)
        return x, info

    if dim > max_dense_dim:
        if strict_hhl:
            raise RuntimeError(
                f"strict_hhl requires dim={dim} <= max_dense_dim={max_dense_dim}"
            )
        x, backend_info = _solve_classical_with_info(
            matvec,
            rhs,
            precision,
            backend="krylov",
            dense_matrix=None,
            max_dense_dim=max_dense_dim,
        )
        info.update(backend_info)
        info["execution_mode"] = "operator_fallback"
        info["fallback_reason"] = f"dim={dim} exceeds max_dense_dim={max_dense_dim}"
        info["rel_residual"] = _relative_residual(matvec, rhs, x)
        return x, info

    dense_matrix = _dense_from_matvec(matvec, dim)
    info["used_dense_reconstruction"] = True

    if not MQ_AVAILABLE:
        if strict_hhl:
            if _HHL_IMPORT_ERROR:
                raise RuntimeError(f"strict_hhl requires mindquantum ({_HHL_IMPORT_ERROR})")
            raise RuntimeError("strict_hhl requires mindquantum")
        x, backend_info = _solve_classical_with_info(
            matvec,
            rhs,
            precision,
            backend="dense",
            dense_matrix=dense_matrix,
            max_dense_dim=max_dense_dim,
        )
        info.update(backend_info)
        info["execution_mode"] = "classical_small_scale"
        if _HHL_IMPORT_ERROR:
            info["fallback_reason"] = f"mindquantum unavailable ({_HHL_IMPORT_ERROR})"
        else:
            info["fallback_reason"] = "mindquantum unavailable"
        info["rel_residual"] = _relative_residual(matvec, rhs, x)
        return x, info

    info["used_mindquantum"] = True
    info["used_quantum_circuit"] = True
    info["quantum_attempted"] = True

    try:
        quantum_x = np.asarray(_run_quantum_hhl(dense_matrix, rhs, precision))
        quantum_rel_res = _relative_residual(matvec, rhs, quantum_x)
        info["quantum_candidate_backend"] = "mindquantum_hhl"
        info["quantum_candidate_rel_residual"] = quantum_rel_res

        if strict_hhl:
            info["actual_backend"] = "mindquantum_hhl"
            info["execution_mode"] = "mindquantum_strict"
            info["quantum_result_accepted"] = True
            info["rel_residual"] = quantum_rel_res
            return quantum_x, info

        if np.isfinite(quantum_rel_res) and quantum_rel_res <= hhl_acceptance_rel_residual:
            info["actual_backend"] = "mindquantum_hhl"
            info["execution_mode"] = "mindquantum_small_scale"
            info["quantum_result_accepted"] = True
            info["rel_residual"] = quantum_rel_res
            return quantum_x, info

        x, backend_info = _solve_classical_with_info(
            matvec,
            rhs,
            precision,
            backend="dense",
            dense_matrix=dense_matrix,
            max_dense_dim=max_dense_dim,
        )
        info.update(backend_info)
        info["execution_mode"] = "quantum_rejected_to_classical"
        info["quantum_result_accepted"] = False
        info["fallback_reason"] = (
            "quantum candidate failed residual guard: "
            f"{quantum_rel_res:.3e} > {hhl_acceptance_rel_residual:.3e}"
        )
        info["rel_residual"] = _relative_residual(matvec, rhs, x)
        return x, info
    except Exception as exc:
        if strict_hhl:
            raise RuntimeError(f"strict_hhl quantum branch failed: {type(exc).__name__}: {exc}") from exc
        x, backend_info = _solve_classical_with_info(
            matvec,
            rhs,
            precision,
            backend="dense",
            dense_matrix=dense_matrix,
            max_dense_dim=max_dense_dim,
        )
        info.update(backend_info)
        info["execution_mode"] = "quantum_exception_to_classical"
        info["quantum_result_accepted"] = False
        info["quantum_error"] = f"{type(exc).__name__}: {exc}"
        info["fallback_reason"] = "quantum branch raised an exception"
        info["rel_residual"] = _relative_residual(matvec, rhs, x)
        return x, info


def solve_linear_system(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    precision: float,
    *,
    backend: str = "hhl_research",
    max_dense_dim: int = 4096,
    hhl_acceptance_rel_residual: float = 0.25,
) -> np.ndarray:
    
    x, _ = solve_linear_system_with_info(
        matvec,
        b,
        precision,
        backend=backend,
        max_dense_dim=max_dense_dim,
        hhl_acceptance_rel_residual=hhl_acceptance_rel_residual,
    )
    return x
