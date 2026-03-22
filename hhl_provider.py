

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
from scipy.linalg import expm

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RY, X, UnivMathGate

DEFAULT_PHASE_QUBITS = 8


def _normalized_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Input vector b must be non-zero for HHL state preparation.")
    return vec / norm


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _qft_matrix(num_qubits: int) -> np.ndarray:
    dim = 2**num_qubits
    omega = np.exp(2j * np.pi / dim)
    indices = np.arange(dim)
    return omega ** (np.outer(indices, indices)) / math.sqrt(dim)


def _phase_register_values(num_qubits: int, t: float) -> np.ndarray:
    dim = 2**num_qubits
    k = np.arange(dim, dtype=np.float64)
    return 2 * np.pi * k / (dim * t)


def _apply_qpe(
    circ: Circuit,
    unitary: np.ndarray,
    phase_qubits: list[int],
    target_qubits: list[int],
    t: float,
) -> None:
    for q in phase_qubits:
        circ += H.on(q)

    for power, ctrl in enumerate(phase_qubits):
        u_power = expm(1j * unitary * t * (2**power))
        circ += UnivMathGate(f"U_{power}", u_power).on(target_qubits, ctrl)

    qft_dagger = UnivMathGate("QFT_dag", _qft_matrix(len(phase_qubits)).conj().T)
    circ += qft_dagger.on(phase_qubits)


def _apply_inverse_qpe(
    circ: Circuit,
    unitary: np.ndarray,
    phase_qubits: list[int],
    target_qubits: list[int],
    t: float,
) -> None:
    qft = UnivMathGate("QFT", _qft_matrix(len(phase_qubits)))
    circ += qft.on(phase_qubits)

    for power, ctrl in reversed(list(enumerate(phase_qubits))):
        u_power = expm(-1j * unitary * t * (2**power))
        circ += UnivMathGate(f"Uinv_{power}", u_power).on(target_qubits, ctrl)

    for q in phase_qubits:
        circ += H.on(q)


def _apply_controlled_rotations(
    circ: Circuit,
    ancilla: int,
    phase_qubits: list[int],
    t: float,
    c_const: float,
) -> None:
    num_phase = len(phase_qubits)
    lambdas = _phase_register_values(num_phase, t)

    for k, lam in enumerate(lambdas):
        if k == 0 or lam <= 0:
            continue
        ratio = min(c_const / lam, 1.0)
        angle = 2 * math.asin(ratio)
        if angle == 0:
            continue

        flipped: list[int] = []
        for bit_pos, qubit in enumerate(phase_qubits):
            if ((k >> bit_pos) & 1) == 0:
                circ += X.on(qubit)
                flipped.append(qubit)

        circ += RY(angle).on(ancilla, phase_qubits)

        for qubit in reversed(flipped):
            circ += X.on(qubit)


def _select_phase_qubits() -> int:
    return DEFAULT_PHASE_QUBITS


def hhl(matrix: np.ndarray, b: np.ndarray, *, phase_qubits: int | None = None) -> tuple[Circuit, Callable, Callable]:
    

    matrix = np.asarray(matrix, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square for HHL.")
    if b.ndim != 1 or b.shape[0] != matrix.shape[0]:
        raise ValueError("Vector b must be one-dimensional and match matrix size.")

    herm_tol = 1e-10
    if not np.allclose(matrix, matrix.conj().T, atol=herm_tol, rtol=0.0):
        raise ValueError("HHL requires a Hermitian (symmetric) matrix; got a non-Hermitian input.")

    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = np.real_if_close(eigvals, tol=1000)
    if np.iscomplexobj(eigvals):
        imag_max = float(np.max(np.abs(np.imag(eigvals))))
        raise ValueError(
            f"HHL expected real eigenvalues for a Hermitian matrix, but got imag max {imag_max:.2e}."
        )

    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    if min_eig <= 0.0:
        raise ValueError("HHL requires a positive-definite matrix with positive eigenvalues.")

    input_dim = matrix.shape[0]
    input_qubits = int(math.log2(_next_power_of_two(input_dim)))
    if phase_qubits is None:
        phase_qubits = _select_phase_qubits()
    if phase_qubits <= 0:
        raise ValueError("phase_qubits must be positive.")

    dim = 2**input_qubits
    if input_dim != dim:
        padded = np.zeros((dim, dim), dtype=np.complex128)
        padded[:input_dim, :input_dim] = matrix
        matrix = padded
        b_pad = np.zeros(dim, dtype=np.complex128)
        b_pad[:input_dim] = b
        b = b_pad

    t = 2 * np.pi / (2 * max_eig)
    c_const = float(min_eig) * 0.5

    ancilla = 0
    phase = list(range(1, 1 + phase_qubits))
    input_reg = list(range(1 + phase_qubits, 1 + phase_qubits + input_qubits))

    circ = Circuit()
    _apply_qpe(circ, matrix, phase, input_reg, t)
    _apply_controlled_rotations(circ, ancilla, phase, t, c_const)
    _apply_inverse_qpe(circ, matrix, phase, input_reg, t)

    b_norm = _normalized_vector(b)
    b_scale = float(np.linalg.norm(b))

    def state_prep(state_vector: np.ndarray) -> np.ndarray:
        prepared = np.zeros_like(state_vector)
        for idx, amp in enumerate(b_norm):
            basis_index = 0
            for bit_pos, qubit in enumerate(input_reg):
                if idx & (1 << bit_pos):
                    basis_index |= 1 << qubit
            prepared[basis_index] = amp
        return prepared

    def result_decoder(final_state: np.ndarray, precision: float) -> np.ndarray:
        _ = precision
        solution = np.zeros(dim, dtype=np.complex128)
        for idx in range(dim):
            basis_index = 1 << ancilla
            for bit_pos, qubit in enumerate(input_reg):
                if idx & (1 << bit_pos):
                    basis_index |= 1 << qubit
            solution[idx] = final_state[basis_index]

        return solution * (b_scale / c_const)

    return circ, state_prep, result_decoder


def _run_demo() -> None:
    

    from mindquantum.simulator import Simulator

    matrix = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.complex128)
    b = np.array([1.0, 0.0], dtype=np.complex128)

    hhl_circ, state_prep, result_decoder = hhl(matrix, b)
    sim = Simulator("mqvector", hhl_circ.n_qubits)

    state = sim.get_qs()
    state = state_prep(state)
    sim.reset()
    sim.set_qs(state)
    sim.apply_circuit(hhl_circ)
    final_state = sim.get_qs()

    hhl_solution = result_decoder(final_state, 1e-6)
    classical_solution = np.linalg.solve(matrix, b)
    print("HHL solution:", np.real_if_close(hhl_solution))
    print("Classical solution:", np.real_if_close(classical_solution))


if __name__ == "__main__":
    _run_demo()
