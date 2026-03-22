

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class PolyMLP(nn.Module):
    

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        initial_weights: np.ndarray | None = None,
        activation: str | None = None,
    ):
        super().__init__()
        proj_mode = str(getattr(config, "PROJ_MODE", "random")).lower()
        if proj_mode == "identity":
            if int(config.PROJ_DIM) != int(input_dim):
                raise ValueError("PROJ_MODE='identity' requires PROJ_DIM == input_dim.")
            proj_weights = np.eye(input_dim, dtype=np.float32)
        else:
            proj_scale = 0.1
            rng = np.random.default_rng(config.RANDOM_SEED)
            proj_weights = rng.normal(0.0, proj_scale, size=(input_dim, config.PROJ_DIM)).astype(np.float32)
        self.register_buffer("proj", torch.from_numpy(proj_weights))

        self.fc1 = nn.Linear(config.PROJ_DIM, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.05)

        self.activation = str(activation if activation is not None else getattr(config, "ACTIVATION", "square")).lower()

        if initial_weights is not None:
            self._load_flat_weights(initial_weights)

    def set_train(self, mode: bool = True) -> None:
        self.train(mode)

    def trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.fc1.weight, self.fc2.weight]

    def _load_flat_weights(self, flat_weights: np.ndarray) -> None:
        pointer = 0
        flat = np.asarray(flat_weights, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            for param in self.trainable_params():
                size = int(param.numel())
                shape = tuple(param.shape)
                p_data = flat[pointer : pointer + size].reshape(shape)
                param.copy_(torch.from_numpy(p_data))
                pointer += size
        if pointer != flat.size:
            raise ValueError(f"Flat weight length mismatch: consumed {pointer} != provided {flat.size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.proj
        x = self.fc1(x)
        act = self.activation
        if act == "relu":
            x = F.relu(x)
        elif act == "tanh":
            x = torch.tanh(x)
        elif act == "softmax":
            x = F.softmax(x, dim=-1)
        elif act == "square":
            x = x * x
        else:
            raise ValueError(f"Unsupported activation: {act}")
        x = self.fc2(x)
        return x


class RobustTrainer:
    def __init__(self, model: PolyMLP, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.trainable_params(), lr=0.01)

    def forward_loss(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits = self.model(data)
        return self.loss_fn(logits, label)

    @torch.no_grad()
    def _project_linf(self, adv: torch.Tensor, clean: torch.Tensor, epsilon: float) -> torch.Tensor:
        perturb = torch.clamp(adv - clean, min=-epsilon, max=epsilon)
        return clean + perturb

    def pgd_attack(self, data: torch.Tensor, label: torch.Tensor, epsilon: float, alpha: float, num_steps: int) -> torch.Tensor:
        
        clean = data.detach()
        adv = clean.clone().detach()

        for _ in range(int(num_steps)):
            adv.requires_grad_(True)
            with torch.enable_grad():
                loss = self.forward_loss(adv, label)
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                adv = adv + float(alpha) * torch.sign(grad)
                adv = self._project_linf(adv, clean, float(epsilon))
            adv = adv.detach()

        return adv

    def train_epoch(self, dataset: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        self.model.train(True)
        epoch_loss = 0.0
        steps = 0

        for data, label in dataset:
            data = data.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.forward_loss(data, label)
            loss.backward()
            self.optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            steps += 1

        return epoch_loss / max(steps, 1)

    def train_epoch_robust(self, dataset: list[tuple[torch.Tensor, torch.Tensor]], epoch: int = 0) -> float:
        epoch_loss = 0.0
        steps = 0
        use_combined = bool(getattr(config, "USE_COMBINED_LOSS", False))
        alpha = float(getattr(config, "LOSS_ALPHA", 0.5))

        if not use_combined:
            warmup_alpha = 0.7 if epoch <= 1 else 0.0
        else:
            warmup_alpha = alpha

        self.model.train(True)
        eps_train = float(getattr(config, "EPSILON_TRAIN", config.EPSILON))

        for data, label in dataset:
            data = data.to(self.device)
            label = label.to(self.device)

            self.model.train(False)
            adv_data = self.pgd_attack(data, label, eps_train, config.ATTACK_STEP_SIZE, config.ATTACK_STEPS)

            self.model.train(True)
            self.optimizer.zero_grad(set_to_none=True)

            if warmup_alpha > 0.0:
                loss_clean = self.forward_loss(data, label)
                loss_adv = self.forward_loss(adv_data, label)
                loss = warmup_alpha * loss_clean + (1.0 - warmup_alpha) * loss_adv
            else:
                loss = self.forward_loss(adv_data, label)

            loss.backward()
            self.optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            steps += 1

        return epoch_loss / max(steps, 1)

    @torch.no_grad()
    def evaluate(self, dataset: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[float, float, float, float]:
        self.model.train(False)
        correct_clean = 0
        correct_adv = 0
        total = 0
        total_clean_loss = 0.0
        total_robust_loss = 0.0
        diag = bool(getattr(config, "EVAL_DIAGNOSTICS", False))
        diag_linf_sum = 0.0
        diag_count = 0

        for data, label in dataset:
            data = data.to(self.device)
            label = label.to(self.device)

            logits = self.model(data)
            preds = torch.argmax(logits, dim=1)
            labels_idx = torch.argmax(label, dim=1)
            correct_clean += int((preds == labels_idx).sum().item())
            
            loss = self.loss_fn(logits, label)
            total_clean_loss += float(loss.item()) * data.shape[0]

            adv_data = self.pgd_attack(data, label, config.EPSILON, config.ATTACK_STEP_SIZE, config.ATTACK_STEPS)
            if diag:
                linf = (adv_data - data).abs().amax(dim=1)
                diag_linf_sum += float(linf.mean().item()) * data.shape[0]
                diag_count += int(data.shape[0])
            logits_adv = self.model(adv_data)
            preds_adv = torch.argmax(logits_adv, dim=1)
            correct_adv += int((preds_adv == labels_idx).sum().item())
            loss_adv = self.loss_fn(logits_adv, label)
            total_robust_loss += float(loss_adv.item()) * data.shape[0]

            total += int(label.shape[0])

        acc_clean = correct_clean / max(total, 1)
        acc_adv = correct_adv / max(total, 1)
        avg_clean_loss = total_clean_loss / max(total, 1)
        avg_robust_loss = total_robust_loss / max(total, 1)
        print(
            f"    [Eval] Clean: {acc_clean:.2%} | Robust: {acc_adv:.2%} | "
            f"Clean Loss: {avg_clean_loss:.4f} | Robust Loss: {avg_robust_loss:.4f}"
        )
        if diag and diag_count > 0:
            avg_linf = diag_linf_sum / float(diag_count)
            print(f"    [EvalDiag] PGD eps={float(config.EPSILON):.6f} avg_Linf(adv-clean)={avg_linf:.6f}")
        return acc_clean, acc_adv, avg_clean_loss, avg_robust_loss


def create_dataset_iterator(X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    
    num_samples = int(X.shape[0])
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_idx = indices[start_idx:end_idx]
        yield torch.from_numpy(np.asarray(X[batch_idx], dtype=np.float32)), torch.from_numpy(np.asarray(y[batch_idx], dtype=np.float32))

