import importlib.util
import time
from typing import Optional

import numpy as np
import torch

import config
from classical_baseline import PolyMLP, RobustTrainer
from solver_backends import solve_linear_system, solve_linear_system_with_info

_JAX_SPEC = importlib.util.find_spec("jax")
if _JAX_SPEC is None:
    raise RuntimeError("jax is required for qrt_simulation.py (JAX-based Carleman coefficients).")

import jax
import jax.numpy as jnp

from carleman_coeffs import build_vector_field, compute_coeffs


def _resolve_w_dim(objective_mode: str, batch_size: int) -> int:
    if objective_mode == "batch_perturbation":
        w_dim_mode = str(getattr(config, "QRT_W_DIM_MODE", "shared")).lower()
        if w_dim_mode == "per_sample":
            return int(batch_size) * int(config.INPUT_DIM)
    return int(config.INPUT_DIM)

class CarlemanSystem:
    
    def __init__(self, model, data, labels, v_operating_point=None, initial_phase: str = "u_only", verbose: bool = True):
        self.model = model
        self.data = data # Fixed batch for dynamics
        self.labels = labels
        self.verbose = bool(verbose)
        self.objective_mode = str(getattr(config, "QRT_OBJECTIVE_MODE", "single_sample")).lower()
        if self.objective_mode == "batch_perturbation" and isinstance(self.data, np.ndarray):
            self.batch_size = int(self.data.shape[0])
        else:
            self.batch_size = 1
        self.w_dim = _resolve_w_dim(self.objective_mode, self.batch_size)
        self.u_params = list(model.trainable_params())
        self.u_dim = int(sum(int(p.numel()) for p in self.u_params))
        self.dim = self.w_dim + self.u_dim
        self.truncation = config.CARLEMAN_N
        
        self.param_shapes = [tuple(int(d) for d in p.shape) for p in self.u_params]
        self.param_sizes = [int(p.numel()) for p in self.u_params]
        self.proj = np.asarray(getattr(self.model, "proj").detach().cpu().numpy(), dtype=np.float32)
        
        if v_operating_point is None:
            if self.objective_mode == "batch_perturbation":
                w0 = np.zeros(self.w_dim, dtype=np.float32)
            else:
                if isinstance(self.data, np.ndarray) and self.data.size > 0:
                    w0 = np.asarray(self.data[0].flatten(), dtype=np.float32)
                else:
                    w0 = np.zeros(self.w_dim, dtype=np.float32)
            u0 = np.concatenate([np.asarray(p.detach().cpu().numpy(), dtype=np.float32).flatten() for p in self.u_params])
            self.v_op = np.concatenate([w0, u0])
        else:
            self.v_op = np.asarray(v_operating_point, dtype=np.float32)
        
        if self.verbose:
            print(
                f"Initializing Carleman System with Dimension {self.dim} "
                f"(w:{self.w_dim}, u:{self.u_dim}) and Truncation N={self.truncation}"
            )
        
        self._extract_coefficients(phase=initial_phase)
        
        if self.verbose:
            print(f"  F0 norm: {np.linalg.norm(self.F0):.6f}")
            print(f"  F1 norm: {np.linalg.norm(self.F1):.6f}")
            print(f"  F1 spectral radius: {np.max(np.abs(np.linalg.eigvals(self.F1))):.6f}")
        
    def _extract_coefficients(self, phase: str = "u_only"):
        
        if self.objective_mode == "batch_perturbation":
            label = np.asarray(self.labels, dtype=np.float32)  # (B, C)
            x_clean = np.asarray(self.data, dtype=np.float32)  # (B, INPUT_DIM)
        else:
            label = np.asarray(self.labels[0], dtype=np.float32)  # (C,)
            x_clean = None
        use_combined = bool(getattr(config, "USE_COMBINED_LOSS", False))
        alpha = float(getattr(config, "LOSS_ALPHA", 0.5))
        weight_decay = float(getattr(config, "QRT_WEIGHT_DECAY", 0.0))

        param_shapes = tuple(tuple(int(d) for d in s) for s in self.param_shapes)
        param_sizes = tuple(int(sz) for sz in self.param_sizes)
        proj = jnp.asarray(self.proj)
        y = jnp.asarray(label)
        x_batch = jnp.asarray(x_clean) if x_clean is not None else None
        input_dim = int(config.INPUT_DIM)
        batch_size = int(self.batch_size)
        w_dim_mode = str(getattr(config, "QRT_W_DIM_MODE", "shared")).lower()

        def _unpack(u_in: jnp.ndarray) -> list[jnp.ndarray]:
            mats: list[jnp.ndarray] = []
            pointer = 0
            for shape, size in zip(param_shapes, param_sizes):
                mats.append(jnp.reshape(u_in[pointer:pointer + size], shape))
                pointer += size
            return mats

        act = str(getattr(self.model, "activation", getattr(config, "ACTIVATION", "square"))).lower()

        def _apply_activation(hidden: jnp.ndarray) -> jnp.ndarray:
            if act == "relu":
                return jax.nn.relu(hidden)
            if act == "tanh":
                return jnp.tanh(hidden)
            if act == "softmax":
                return jax.nn.softmax(hidden, axis=-1)
            if act == "square":
                return jnp.square(hidden)
            raise ValueError(f"Unsupported activation: {act}")

        def loss_fn(w: jnp.ndarray, u: jnp.ndarray, _batch: object) -> jnp.ndarray:
            def forward(w_in: jnp.ndarray, u_in: jnp.ndarray) -> jnp.ndarray:
                mats = _unpack(u_in)
                fc1_w = mats[0]
                fc2_w = mats[1] if len(mats) > 1 else None

                if self.objective_mode == "batch_perturbation":
                    if w_dim_mode == "per_sample":
                        w_perturb = jnp.reshape(w_in, (batch_size, input_dim))
                        x_adv = x_batch + w_perturb
                    else:
                        x_adv = x_batch + w_in[None, :]
                    x_proj = jnp.matmul(x_adv, proj)  # (B, PROJ_DIM)
                    hidden = jnp.matmul(x_proj, fc1_w.T)  # (B, HIDDEN_DIM)
                    hidden = _apply_activation(hidden)
                    logits = jnp.matmul(hidden, fc2_w.T) if fc2_w is not None else hidden  # (B, C)
                    diff = logits - y
                    return jnp.mean(jnp.square(diff))

                w_proj = jnp.matmul(w_in, proj)  # (PROJ_DIM,)
                hidden = jnp.dot(fc1_w, w_proj)  # (HIDDEN_DIM,)
                hidden = _apply_activation(hidden)
                logits = jnp.dot(fc2_w, hidden) if fc2_w is not None else hidden
                diff = logits - y
                return jnp.mean(jnp.square(diff))

            loss_robust = forward(w, u)
            penalty = 0.5 * weight_decay * jnp.sum(jnp.square(u))

            if phase == "w_only":
                return loss_robust + penalty

            if not use_combined:
                return loss_robust + penalty

            loss_clean = forward(jnp.zeros_like(w), u)
            return alpha * loss_clean + (1.0 - alpha) * loss_robust + penalty

        v_op_jax = jnp.asarray(self.v_op, dtype=jnp.float32)
        max_order = 1 if int(self.truncation) <= 1 else 2
        coeffs = compute_coeffs(loss_fn, v_op_jax, self.w_dim, None, max_order=max_order)
        F0 = coeffs["F0"].astype(np.float32)
        F1 = coeffs["F1"].astype(np.float32)
        F2 = coeffs.get("F2")
        if F2 is None:
            F2 = np.zeros((self.dim, self.dim, self.dim), dtype=np.float32)
        else:
            F2 = F2.astype(np.float32)

        if phase == "w_only":
            F0[self.w_dim:] = 0.0
            F1[self.w_dim:, :] = 0.0
            F2[self.w_dim:, :, :] = 0.0
        elif phase == "u_only":
            F0[: self.w_dim] = 0.0
            F1[: self.w_dim, :] = 0.0
            F2[: self.w_dim, :, :] = 0.0

        lr = float(getattr(config, "QRT_LEARNING_RATE", 1.0))
        F0 = lr * F0
        F1 = lr * F1
        F2 = lr * F2

        grad_clip = float(getattr(config, "QRT_GRAD_CLIP", 0.0))
        if grad_clip > 0.0:
            f0_norm = float(np.linalg.norm(F0))
            if f0_norm > grad_clip:
                F0 = F0 * (grad_clip / (f0_norm + 1e-8))

        self.F0 = F0
        self.F1 = F1
        self.F2_dense = F2
        self.f2_mode = "full"
        self.F3_dense = None

        vf_core = build_vector_field(loss_fn, self.w_dim)
        vf_compiled = jax.jit(lambda v_in: vf_core(v_in, None))
        self._phase_mode = phase

        def _vf_np(v_np: np.ndarray) -> np.ndarray:
            out = np.asarray(vf_compiled(jnp.asarray(v_np, dtype=jnp.float32)), dtype=np.float32)
            out = lr * out
            if phase == "w_only":
                out[self.w_dim:] = 0.0
            elif phase == "u_only":
                out[: self.w_dim] = 0.0
            return out

        self.vector_field_fn = _vf_np


    def get_F2_action(self, delta_v):
        
        val_pos = self.vector_field_fn(self.v_op + delta_v)
        val_neg = self.vector_field_fn(self.v_op - delta_v)
        return 0.5 * (val_pos + val_neg - 2 * self.F0)

    def get_F1_action(self, delta_v):
        
        return self.F1 @ delta_v

    def matvec(self, y_hat):
        
        y_hat = np.asarray(y_hat, dtype=np.float32)
        D = self.dim
        N = self.truncation

        levels = []
        offset = 0
        size = D
        for _ in range(N):
            levels.append(y_hat[offset:offset + size])
            offset += size
            size *= D

        z_levels = [np.zeros_like(levels[0], dtype=np.float32)]
        if N >= 2:
            z_levels.append(np.zeros_like(levels[1], dtype=np.float32))
        if N >= 3:
            z_levels.append(np.zeros_like(levels[2], dtype=np.float32))
        if N >= 4:
            z_levels.append(np.zeros_like(levels[3], dtype=np.float32))

        y1 = levels[0]

        z1 = z_levels[0]
        z1 += self.get_F1_action(y1)

        if N >= 2:
            y2 = levels[1].reshape(D, D)
            if getattr(self, "f2_mode", "approx") == "full" and hasattr(self, "F2_dense"):
                z1 += np.einsum('ijk,jk->i', self.F2_dense, y2)
            else:
                diag_y2 = np.diag(y2)
                z1 += np.einsum('ij,j->i', self.F2_diag, diag_y2)
                if hasattr(self, "F2_cross"):
                    for (i, j, vec) in self.F2_cross:
                        z1 += vec * y2[i, j]
        
        if N >= 3 and hasattr(self, "F3_dense") and self.F3_dense is not None:
            y3 = levels[2].reshape(D, D, D)
            for j in range(D):
                z1 += self.F3_dense[:, j, j, j] * y3[j, j, j]

        if N >= 2:
            z2 = z_levels[1]
            term_21 = np.kron(self.F0, y1) + np.kron(y1, self.F0)
            z2 += term_21

            y2_mat = y2
            term_22 = self.F1 @ y2_mat + y2_mat @ self.F1.T
            z2 += term_22.flatten()

        if N >= 3:
            z3 = z_levels[2]
            y3_tensor = levels[2].reshape(D, D, D)

            term3 = np.einsum('li,ijk->ljk', self.F1, y3_tensor)
            term3 += np.einsum('lj,ijk->ilk', self.F1, y3_tensor)
            term3 += np.einsum('lk,ijk->ijl', self.F1, y3_tensor)
            z3 += term3.reshape(-1)

        if N >= 4:
            z4 = z_levels[3]
            y4_tensor = levels[3].reshape(D, D, D, D)

            term4 = np.einsum('mi,ijkl->mjkl', self.F1, y4_tensor)
            term4 += np.einsum('mj,ijkl->imkl', self.F1, y4_tensor)
            term4 += np.einsum('mk,ijkl->ijml', self.F1, y4_tensor)
            term4 += np.einsum('ml,ijkl->ijkm', self.F1, y4_tensor)
            z4 += term4.reshape(-1)

        out = []
        for lvl in z_levels:
            out.append(lvl.flatten())
        return np.concatenate(out)

    def precompute_dense_F2(self):
        
        pass

def verify_carleman_truncation(cs, test_point, num_tests=5, perturb_scale: Optional[float] = None):
    
    errors = []
    
    if perturb_scale is None:
        perturb_scale = float(getattr(config, "CARLEMAN_VERIFY_SCALE", 1e-2))

    for test_idx in range(num_tests):
        v_test = cs.v_op + np.random.randn(cs.dim).astype(np.float32) * perturb_scale
        
        v_true = cs.vector_field_fn(v_test)
        
        delta_v = v_test - cs.v_op
        v_recon = cs.F0 + cs.F1 @ delta_v
        
        if hasattr(cs, 'f2_mode'):
            if cs.f2_mode == "full" and hasattr(cs, 'F2_dense'):
                delta_v2 = np.outer(delta_v, delta_v)
                v_recon += np.einsum('ijk,jk->i', cs.F2_dense, delta_v2)
            elif hasattr(cs, 'F2_diag'):
                v_recon += np.einsum('ij,j->i', cs.F2_diag, delta_v * delta_v)
        
        if hasattr(cs, 'F3_dense') and cs.F3_dense is not None:
            for j in range(len(delta_v)):
                v_recon += cs.F3_dense[:, j, j, j] * (delta_v[j]**3)
        
        rel_error = np.linalg.norm(v_true - v_recon) / (np.linalg.norm(v_true) + 1e-8)
        errors.append(rel_error)
    
    avg_error = np.mean(errors)
    return avg_error, errors

def _estimate_cond_level1(F1, dt):
    
    A_lvl1 = np.eye(F1.shape[0]) - F1 * dt
    try:
        return float(np.linalg.cond(A_lvl1))
    except Exception:
        return float("nan")


def hhl_solve_noisy(A_matvec, b, dim, precision):
    
    backend = str(getattr(config, "LINEAR_SOLVER_BACKEND", "hhl_research")).lower()
    max_dense_dim = int(getattr(config, "LINEAR_SOLVER_MAX_DENSE_DIM", 4096))
    hhl_acceptance_rel_residual = float(getattr(config, "HHL_ACCEPTANCE_REL_RESIDUAL", 0.25))

    if bool(getattr(config, "SOLVER_DIAGNOSTICS", False)):
        x, info = solve_linear_system_with_info(
            A_matvec,
            b,
            precision,
            backend=backend,
            max_dense_dim=max_dense_dim,
            hhl_acceptance_rel_residual=hhl_acceptance_rel_residual,
        )
        requested = info.get("requested_backend", backend)
        actual = info.get("actual_backend", "unknown")
        execution_mode = info.get("execution_mode", "unknown")
        rel_res = info.get("rel_residual", None)
        iterative_info = info.get("iterative_info", None)
        fallback_reason = info.get("fallback_reason", None)
        quantum_rel_res = info.get("quantum_candidate_rel_residual", None)
        if iterative_info is not None:
            print(
                "    [solver] "
                f"requested={requested} actual={actual} mode={execution_mode} "
                f"iterative_info={iterative_info} rel_res={rel_res}"
            )
        else:
            print(
                "    [solver] "
                f"requested={requested} actual={actual} mode={execution_mode} rel_res={rel_res}"
            )
        if quantum_rel_res is not None:
            print(f"    [solver] quantum_candidate_rel_res={quantum_rel_res}")
        if fallback_reason:
            print(f"    [solver] fallback_reason={fallback_reason}")
        return x
    return solve_linear_system(
        A_matvec,
        b,
        precision,
        backend=backend,
        max_dense_dim=max_dense_dim,
        hhl_acceptance_rel_residual=hhl_acceptance_rel_residual,
    )

def _project_linf(x: np.ndarray, center: np.ndarray, epsilon: float) -> np.ndarray:
    delta = x - center
    delta = np.clip(delta, -epsilon, epsilon)
    return center + delta


def _phase_for_step(step: int, disable_w_phase: bool) -> tuple[bool, str]:
    
    if disable_w_phase:
        return False, "u_only"

    adv_first = bool(getattr(config, "QRT_ADVERSARY_FIRST", True))
    is_adversary_step = (step % 2 == 0) if adv_first else (step % 2 == 1)
    return is_adversary_step, ("w_only" if is_adversary_step else "u_only")

def train_qrt(X_train=None, y_train=None, initial_weights=None, eval_interval=5):
    
    rng = np.random.default_rng(config.RANDOM_SEED)
    objective_mode = str(getattr(config, "QRT_OBJECTIVE_MODE", "single_sample")).lower()
    if X_train is None or y_train is None:
        print("[QRT] Warning: No training data provided. Using mock data.")
        data_use = np.random.randn(1, config.INPUT_DIM).astype(np.float32)
        label_use = np.array([[1.0]]).astype(np.float32)
    else:
        BATCH_SIZE = int(getattr(config, "QRT_BATCH_SIZE", 64))
        idx = rng.choice(len(X_train), size=min(BATCH_SIZE, len(X_train)), replace=False)
        data_use = X_train[idx]
        label_use = y_train[idx]
    
    model = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    
    w_init_vec = None
    u_init_vec = None
    
    if initial_weights is not None:
        model._load_flat_weights(initial_weights)
        u_init_vec = initial_weights
        print(f"[QRT] Initialized with provided weights (norm: {np.linalg.norm(u_init_vec):.4f})")
    else:
        rng_local = np.random.default_rng(int(time.time()))
        scale = 0.05
        with torch.no_grad():
            for param in model.trainable_params():
                shape = tuple(int(d) for d in param.shape)
                p_data = rng_local.normal(0.0, scale, size=shape).astype(np.float32)
                param.copy_(torch.from_numpy(p_data))
        u_init_vec = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.trainable_params()])
        print(f"[QRT] Initialized with fresh random weights (norm: {np.linalg.norm(u_init_vec):.4f})")

    if objective_mode == "batch_perturbation":
        batch_size = int(data_use.shape[0]) if isinstance(data_use, np.ndarray) else 1
        w_dim = _resolve_w_dim(objective_mode, batch_size)
        w_init_vec = np.zeros(w_dim, dtype=np.float32)
        w_center = np.zeros_like(w_init_vec)
    else:
        w_init_vec = data_use[0].flatten()
        w_center = w_init_vec.copy()
    
    v_op = np.concatenate([w_init_vec, u_init_vec])
    v_curr = v_op.copy()

    cs_verify = CarlemanSystem(
        model,
        data_use,
        label_use,
        v_operating_point=v_op,
        initial_phase="u_only",
        verbose=True,
    )

    y_total = 0
    block_size = cs_verify.dim
    for _ in range(cs_verify.truncation):
        y_total += block_size
        block_size *= cs_verify.dim

    history = []
    snapshots = [(-1, u_init_vec.copy())]
    print(f"[QRT] Starting Time-Stepping for {config.TOTAL_STEPS} steps...")

    print("\n=== Carleman Truncation Verification ===")
    avg_error, errors = verify_carleman_truncation(cs_verify, v_op, num_tests=5)
    for idx, err in enumerate(errors):
        print(f"  Test {idx+1}: Relative Error = {err:.4%}")
    print(f"  Average Error: {avg_error:.4%}")
    if avg_error > 0.10:
        print(f"  WARNING: Error > 10%! Consider increasing CARLEMAN_N to {cs_verify.truncation + 1}")
    else:
        print(f"  ✓ Truncation order N={cs_verify.truncation} is sufficient")
    print()

    relinearize_interval = int(getattr(config, "RELINEARIZE_INTERVAL", 1))
    disable_w_phase = bool(getattr(config, "QRT_DISABLE_W_PHASE", False))
    eps_train = float(getattr(config, "EPSILON_TRAIN", 0.03))
    adv_step_size = float(getattr(config, "QRT_ADVERSARY_STEP_SIZE", getattr(config, "TIME_STEP", 0.1)))

    for step in range(config.TOTAL_STEPS):
        is_adv_step, phase_mode = _phase_for_step(step, disable_w_phase)

        if step > 0 and (step % relinearize_interval == 0) and X_train is not None and y_train is not None:
            idx = rng.choice(len(X_train), size=min(BATCH_SIZE, len(X_train)), replace=False)
            data_use = X_train[idx]
            label_use = y_train[idx]
            if objective_mode == "batch_perturbation":
                w_center = np.zeros_like(v_curr[: _resolve_w_dim(objective_mode, len(data_use))], dtype=np.float32)
            else:
                w_center = data_use[0].flatten()

        verbose_step = bool(step < 2 or step % 10 == 0)
        cs = CarlemanSystem(
            model,
            data_use,
            label_use,
            v_operating_point=v_curr,
            initial_phase=phase_mode,
            verbose=verbose_step,
        )

        if is_adv_step:
            grad_w = cs.vector_field_fn(cs.v_op)[:cs.w_dim]
            w_curr = v_curr[:cs.w_dim].copy()
            w_raw = w_curr + adv_step_size * np.sign(grad_w)
            w_next = _project_linf(w_raw, w_center, eps_train)

            v_next = v_curr.copy()
            v_next[:cs.w_dim] = w_next

            if verbose_step:
                delta_u_norm = float(np.linalg.norm(v_next[cs.w_dim:] - v_curr[cs.w_dim:]))
                delta_w_norm = float(np.linalg.norm(v_next[:cs.w_dim] - v_curr[:cs.w_dim]))
                print(
                    f"[QRT][step {step}][ADV-CLASSICAL] "
                    f"Δu={delta_u_norm:.6f} | Δdelta={delta_w_norm:.6f}"
                )
        else:
            y_curr = np.zeros(y_total, dtype=np.float32)
            b_vec = np.zeros_like(y_curr)
            b_vec[:cs.dim] = cs.F0
            rhs = y_curr + b_vec * config.TIME_STEP

            def system_operator(x):
                return x - cs.matvec(x) * config.TIME_STEP

            y_next = hhl_solve_noisy(
                system_operator,
                rhs,
                len(y_curr),
                config.HHL_PRECISION,
            )

            cond_lvl1 = _estimate_cond_level1(cs.F1, config.TIME_STEP)
            delta_v_next = y_next[:cs.dim]
            v_next = cs.v_op + delta_v_next

            v_next[:cs.w_dim] = v_curr[:cs.w_dim]

            if verbose_step:
                u_curr_norm = float(np.linalg.norm(v_next[cs.w_dim:]))
                delta_u_norm = float(np.linalg.norm(v_next[cs.w_dim:] - v_curr[cs.w_dim:]))
                delta_w_norm = float(np.linalg.norm(v_next[:cs.w_dim] - v_curr[:cs.w_dim]))
                print(
                    f"[QRT][step {step}][U-DESCENT] cond≈{cond_lvl1:.2e} "
                    f"| Δu={delta_u_norm:.6f} | Δdelta={delta_w_norm:.6f} | u_norm={u_curr_norm:.4f}"
                )

        v_curr = v_next
        history.append(np.linalg.norm(v_curr[cs.w_dim:]))

        if (step % eval_interval == 0) or (step == config.TOTAL_STEPS - 1):
            snapshots.append((step, v_curr[cs.w_dim:].copy()))
    
    u_final = v_curr[cs.w_dim:]
    
    return history, u_final, snapshots
