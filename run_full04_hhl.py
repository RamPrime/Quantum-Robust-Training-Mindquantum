from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import numpy as np

import config


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--steps", type=int, default=int(config.TOTAL_STEPS))
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--disable-adv", action="store_true")
    parser.add_argument("--solver-diagnostics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def load_or_generate_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    data_dir = ROOT / "data"

    def load_images(filename: str) -> np.ndarray:
        with gzip.open(data_dir / filename, "rb") as handle:
            data = np.frombuffer(handle.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_labels(filename: str) -> np.ndarray:
        with gzip.open(data_dir / filename, "rb") as handle:
            return np.frombuffer(handle.read(), np.uint8, offset=8)

    try:
        train_images = load_images("train-images-idx3-ubyte.gz")
        train_labels = load_labels("train-labels-idx1-ubyte.gz")
        test_images = load_images("t10k-images-idx3-ubyte.gz")
        test_labels = load_labels("t10k-labels-idx1-ubyte.gz")
    except Exception:
        total = config.TRAIN_SIZE + config.TEST_SIZE
        x = (np.random.rand(total, config.INPUT_DIM).astype(np.float32) - 0.5) * 2.0
        radius_sq = np.sum(x**2, axis=1)
        threshold = np.median(radius_sq)
        y = (radius_sq > threshold).astype(np.int32) % config.NUM_CLASSES
        y_one_hot = np.eye(config.NUM_CLASSES, dtype=np.float32)[y]
        return (x[: config.TRAIN_SIZE], y_one_hot[: config.TRAIN_SIZE]), (x[config.TRAIN_SIZE :], y_one_hot[config.TRAIN_SIZE :])

    def process_all(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.ndimage import zoom

        mask = np.isin(labels, np.array(config.TARGET_DIGITS))
        images = images[mask]
        labels = labels[mask]
        if images.shape[1] != config.IMG_HEIGHT or images.shape[2] != config.IMG_WIDTH:
            scale = (config.IMG_HEIGHT / images.shape[1], config.IMG_WIDTH / images.shape[2])
            images = np.stack([zoom(img, scale, order=1) for img in images], axis=0)
        x = images.reshape(len(images), -1).astype(np.float32)
        x = (x / 255.0 - 0.5) * 2.0
        label_map = {digit: idx for idx, digit in enumerate(config.TARGET_DIGITS)}
        mapped = np.vectorize(label_map.get)(labels)
        y = np.eye(config.NUM_CLASSES, dtype=np.float32)[mapped]
        return x, y

    x_train, y_train = process_all(train_images, train_labels)
    x_test, y_test = process_all(test_images, test_labels)
    train_perm = np.random.permutation(len(x_train))
    test_perm = np.random.permutation(len(x_test))
    x_train = x_train[train_perm][: config.TRAIN_SIZE]
    y_train = y_train[train_perm][: config.TRAIN_SIZE]
    x_test = x_test[test_perm][: config.TEST_SIZE]
    y_test = y_test[test_perm][: config.TEST_SIZE]
    return (x_train, y_train), (x_test, y_test)


def evaluate_model_weights_batch(
    u_flat: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_struct,
    batch_size: int,
) -> tuple[float, float, float, float]:
    from classical_baseline import RobustTrainer, create_dataset_iterator

    model_struct._load_flat_weights(u_flat)
    trainer = RobustTrainer(model_struct)
    return trainer.evaluate(list(create_dataset_iterator(x_test, y_test, batch_size=batch_size)))


def estimate_dims(batch_size: int) -> tuple[int, int, int]:
    u_dim = int(config.PROJ_DIM) * int(config.HIDDEN_DIM) + int(config.HIDDEN_DIM) * int(config.OUTPUT_DIM)
    w_dim = int(config.INPUT_DIM) * int(batch_size)
    base_dim = w_dim + u_dim
    lifted_dim = 0
    block = base_dim
    for _ in range(int(config.CARLEMAN_N)):
        lifted_dim += block
        block *= base_dim
    return w_dim, u_dim, lifted_dim


def next_power_of_two(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def format_tib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 4):.2f} TiB"


def main() -> None:
    args = parse_args()
    config.QRT_BATCH_SIZE = int(args.batch_size)
    config.QRT_BATCH_SIZES = (int(args.batch_size),)
    config.TOTAL_STEPS = int(args.steps)
    config.USE_COMBINED_LOSS = bool(args.combined)
    config.LOSS_ALPHA = float(args.alpha)
    config.QRT_DISABLE_W_PHASE = bool(args.disable_adv)
    config.SOLVER_DIAGNOSTICS = bool(args.solver_diagnostics)
    config.LINEAR_SOLVER_BACKEND = "hhl_research"
    config.STRICT_HHL = False
    config.CARLEMAN_N = 2
    config.QRT_OBJECTIVE_MODE = "batch_perturbation"
    config.QRT_W_DIM_MODE = "per_sample"
    config.IMG_HEIGHT = 12
    config.IMG_WIDTH = 12
    config.INPUT_DIM = 144
    config.NUM_CLASSES = 5
    config.OUTPUT_DIM = 5
    config.TARGET_DIGITS = (0, 1, 2, 3, 4)
    config.PROJ_DIM = 10
    config.PROJ_MODE = "random"
    config.HIDDEN_DIM = 4
    config.ACTIVATION = "softmax"
    w_dim, u_dim, lifted_dim = estimate_dims(config.QRT_BATCH_SIZE)
    padded_dim = next_power_of_two(lifted_dim)
    dense_bytes = lifted_dim * lifted_dim * 16
    padded_bytes = padded_dim * padded_dim * 16
    print(f"batch_size={config.QRT_BATCH_SIZE}")
    print(f"steps={config.TOTAL_STEPS}")
    print(f"w_dim={w_dim}")
    print(f"u_dim={u_dim}")
    print(f"solver_dim={lifted_dim}")
    print(f"padded_dim={padded_dim}")
    print(f"dense_matrix_complex128={format_tib(dense_bytes)}")
    print(f"padded_matrix_complex128={format_tib(padded_bytes)}")
    print(f"max_dense_dim={config.LINEAR_SOLVER_MAX_DENSE_DIM}")
    print(f"strict_hhl={bool(getattr(config, 'STRICT_HHL', False))}")
    if args.dry_run:
        return
    import torch
    from classical_baseline import PolyMLP
    from qrt_simulation import train_qrt

    torch.manual_seed(int(config.RANDOM_SEED))
    np.random.seed(int(config.RANDOM_SEED))
    (x_train, y_train), (x_test, y_test) = load_or_generate_data()
    dummy_model = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    initial_u = np.concatenate([p.detach().cpu().numpy().flatten() for p in dummy_model.trainable_params()])
    start = time.time()
    history, u_final, snapshots = train_qrt(x_train, y_train, initial_weights=initial_u, eval_interval=config.QRT_EVAL_INTERVAL)
    duration = time.time() - start
    model_eval = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    clean_acc, robust_acc, clean_loss, robust_loss = evaluate_model_weights_batch(
        u_final,
        x_test,
        y_test,
        model_eval,
        batch_size=int(config.EVAL_BATCH_SIZE),
    )
    plot_dir = ROOT / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    mode = "combined" if config.USE_COMBINED_LOSS else "robust"
    adv_tag = "cleanonly" if config.QRT_DISABLE_W_PHASE else "alternating"
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = plot_dir / f"strict_hhl_full04_{mode}_{adv_tag}_bs{config.QRT_BATCH_SIZE}_steps{config.TOTAL_STEPS}{suffix}.npz"
    np.savez_compressed(
        out_path,
        batch_size=int(config.QRT_BATCH_SIZE),
        total_steps=int(config.TOTAL_STEPS),
        use_combined_loss=bool(config.USE_COMBINED_LOSS),
        loss_alpha=float(config.LOSS_ALPHA),
        disable_adv=bool(config.QRT_DISABLE_W_PHASE),
        history=np.asarray(history, dtype=np.float32),
        snapshots=np.asarray([(step, weights) for (step, weights) in snapshots], dtype=object),
        u_final=np.asarray(u_final, dtype=np.float32),
        final_metrics=np.asarray([clean_acc, robust_acc, clean_loss, robust_loss], dtype=np.float32),
        duration_seconds=np.asarray([duration], dtype=np.float32),
        w_dim=np.asarray([w_dim], dtype=np.int32),
        u_dim=np.asarray([u_dim], dtype=np.int32),
        solver_dim=np.asarray([lifted_dim], dtype=np.int32),
    )
    print(f"clean_acc={clean_acc:.6f}")
    print(f"robust_acc={robust_acc:.6f}")
    print(f"clean_loss={clean_loss:.6f}")
    print(f"robust_loss={robust_loss:.6f}")
    print(f"duration={duration:.2f}")
    print(out_path)


if __name__ == "__main__":
    main()
