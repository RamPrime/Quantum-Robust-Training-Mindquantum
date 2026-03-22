

from __future__ import annotations

import math
from typing import Callable

import importlib
import importlib.util
import traceback
import inspect

import numpy as np
import scipy.sparse.linalg as spla

_MQ_SPEC = importlib.util.find_spec("mindquantum")
_MQ_SIM_SPEC = importlib.util.find_spec("mindquantum.simulator")
MQ_AVAILABLE = _MQ_SPEC is not None and _MQ_SIM_SPEC is not None
_HHL_IMPORT_ERROR: str | None = None
if MQ_AVAILABLE:
    _mq_simulator = importlib.import_module("mindquantum.simulator")
    Simulator = getattr(_mq_simulator, "Simulator", None)
    hhl = None
    try:
        from hhl_provider import hhl as hhl  # type: ignore[assignment]
    except Exception as exc:
        _HHL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        hhl = None
    MQ_AVAILABLE = hhl is not None and Simulator is not None


def _dense_from_matvec(matvec: Callable[[np.ndarray], np.ndarray], dim: int) -> np.ndarray:
    probe = np.zeros(dim, dtype=np.float32)
    probe[0] = 1.0
    first_col = np.asarray(matvec(probe), dtype=np.complex128)
    matrix = np.empty((dim, dim), dtype=np.complex128)
    matrix[:, 0] = first_col
    for col_idx in range(1, dim):
        probe[col_idx - 1] = 0.0
        probe[col_idx] = 1.0
        matrix[:, col_idx] = np.asarray(matvec(probe), dtype=np.complex128)
    return matrix


def _call_iterative_solver(
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
) -> tuple[np.ndarray, int, str]:
    
    rhs = np.asarray(b)
    dim = int(rhs.shape[0])

    def safe_matvec(x: np.ndarray) -> np.ndarray:
        return np.array(matvec(x), dtype=rhs.dtype, copy=True)

    lin_op = spla.LinearOperator((dim, dim), matvec=safe_matvec, dtype=rhs.dtype)
    methods = [preferred]
    if preferred != "gmres":
        methods.append("gmres")

    last_x = np.zeros_like(rhs)
    last_info = maxiter
    for method in methods:
        x, info_code = _call_iterative_solver(
            method,
            lin_op,
            rhs,
            precision,
            restart=restart,
            maxiter=maxiter,
        )
        rel_res = float(np.linalg.norm(safe_matvec(x) - rhs) / (np.linalg.norm(rhs) + 1e-12))
        last_x = x
        last_info = info_code
        if info_code == 0 and np.isfinite(rel_res) and rel_res <= max(10.0 * precision, 1e-8):
            return x, info_code, method

    return last_x, last_info, methods[-1]


def _best_scalar_rescale(matrix: np.ndarray, b: np.ndarray, solution: np.ndarray) -> np.ndarray:
    
    if solution.size == 0:
        return solution
    ax = matrix @ solution
    denom = np.vdot(ax, ax)
    if np.abs(denom) <= 1e-15:
        return solution
    alpha = np.vdot(ax, b) / denom
    return solution * alpha


def _run_quantum_hhl(matrix: np.ndarray, b: np.ndarray, precision: float) -> np.ndarray:
    

    if not MQ_AVAILABLE:
        detail = f" (hhl_provider import error: {_HHL_IMPORT_ERROR})" if _HHL_IMPORT_ERROR else ""
        raise ImportError(
            "mindquantum with the local hhl_provider implementation is required for the quantum HHL branch"
            f"{detail}"
        )

    def _is_hermitian(mat: np.ndarray, atol: float = 1e-10) -> bool:
        return np.allclose(mat, mat.conj().T, atol=atol, rtol=0.0)

    def _spd_lift(mat: np.ndarray, rhs: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
        
        ah = mat.conj().T
        spd = ah @ mat
        spd = 0.5 * (spd + spd.conj().T)  # symmetrize against numerical noise
        spd = spd + ridge * np.eye(spd.shape[0], dtype=spd.dtype)
        rhs2 = ah @ rhs
        return spd, rhs2

    matrix = np.asarray(matrix, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    original_dim = int(matrix.shape[0])

    if not _is_hermitian(matrix):
        ridge = float(max(precision, 1e-12))
        matrix, b = _spd_lift(matrix, b, ridge=ridge)

    hhl_circ, state_prep, result_decoder = hhl(matrix, b)

    num_qubits = hhl_circ.n_qubits
    sim = Simulator("mqvector", num_qubits)

    state = sim.get_qs()
    state = state_prep(state)
    sim.reset()
    sim.set_qs(state)

    sim.apply_circuit(hhl_circ)
    final_state = sim.get_qs()
    solution = result_decoder(final_state, precision)
    solution = np.asarray(solution).reshape(-1)
    if solution.size >= original_dim:
        solution = solution[:original_dim]
    solution = _best_scalar_rescale(matrix, b, solution)
    return np.real_if_close(solution)


def solve_linear_system_quantum(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    precision: float,
    *,
    max_dense_dim: int = 4096,
) -> np.ndarray:
    

    dim = len(b)

    if dim > max_dense_dim:
        x, _, _ = _iterative_solve(matvec, b, precision, preferred="lgmres", restart=50, maxiter=200)
        return x

    dense_matrix = _dense_from_matvec(matvec, dim)

    try:
        x = _run_quantum_hhl(dense_matrix, b, precision)
    except Exception:
        x, _, _ = _iterative_solve(matvec, b, precision, preferred="lgmres", restart=50, maxiter=200)

    return x


def solve_linear_system_quantum_with_info(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    precision: float,
    *,
    max_dense_dim: int = 4096,
) -> tuple[np.ndarray, dict[str, object]]:
    
    dim = len(b)
    info: dict[str, object] = {"dim": dim, "precision": float(precision)}

    def _residual(x: np.ndarray) -> float:
        r = matvec(x) - b
        denom = float(np.linalg.norm(b) + 1e-12)
        return float(np.linalg.norm(r) / denom)

    if dim > max_dense_dim:
        x, iterative_info, method = _iterative_solve(matvec, b, precision, preferred="lgmres", restart=50, maxiter=200)
        info["backend"] = method
        info["iterative_info"] = int(iterative_info)
        info["rel_residual"] = _residual(x)
        return x, info

    dense_matrix = _dense_from_matvec(matvec, dim)
    try:
        x = _run_quantum_hhl(dense_matrix, b, precision)
        info["backend"] = "mindquantum_hhl"
        info["rel_residual"] = _residual(x)
        return x, info
    except Exception as exc:
        x, iterative_info, method = _iterative_solve(matvec, b, precision, preferred="lgmres", restart=50, maxiter=200)
        info["backend"] = f"{method}_fallback"
        info["iterative_info"] = int(iterative_info)
        info["hhl_error"] = f"{type(exc).__name__}: {exc}"
        info["rel_residual"] = _residual(x)
        return x, info


class MindQuantumHHL:
    

    def __init__(self, precision: float = 1e-6) -> None:
        if not MQ_AVAILABLE:
            detail = f" (hhl_provider import error: {_HHL_IMPORT_ERROR})" if _HHL_IMPORT_ERROR else ""
            raise RuntimeError(
                "mindquantum with the local hhl_provider implementation is not available; "
                "please ensure hhl_provider.py is importable."
                f"{detail}"
            )
        self.precision = precision
        self.last_backend: str | None = None
        self.last_error: str | None = None

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, float]:
        
        classical = np.linalg.solve(matrix, rhs)
        try:
            solution = _run_quantum_hhl(np.asarray(matrix), np.asarray(rhs), self.precision)
            fidelity = _solution_fidelity(np.asarray(solution), np.asarray(classical))
            self.last_backend = "quantum"
            self.last_error = None
            return solution, fidelity
        except Exception as exc:
            self.last_backend = "classical"
            self.last_error = traceback.format_exc()
            return classical, float("nan")


def _solution_fidelity(quantum_solution: np.ndarray, classical_solution: np.ndarray) -> float:
    
    q = quantum_solution.astype(np.complex128)
    c = classical_solution.astype(np.complex128)
    q_norm = np.linalg.norm(q)
    c_norm = np.linalg.norm(c)
    if q_norm == 0 or c_norm == 0:
        return 0.0
    overlap = np.vdot(q / q_norm, c / c_norm)
    return float(np.abs(overlap))
