import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch가 설치되어 있지 않습니다. AI 학습용 Python 환경에서 "
        "`pip install torch numpy pandas matplotlib`을 실행한 뒤 다시 "
        "시도하세요."
    ) from exc


SENSOR_NAMES = [
    "ETCHBEAMVOLTAGE",
    "IONGAUGEPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "FLOWCOOLPRESSURE",
]

NUM_SENSORS = 4
TIME_FEATURES = 4
ATTENTION_SCALE = 0.5  # 1 / sqrt(d_k), d_k=4


class SensorSelfAttentionClassifier(nn.Module):
    """
    FPGA RTL 구조와 대응되는 단일 헤드 Self-Attention 이진 분류기.

    입력 축:
        x[batch, sensor, time]

    센서가 token이고 최근 4개 시점이 각 센서의 feature이다. 따라서
    Q @ K^T 결과는 [batch, sensor, sensor]의 센서 간 4x4 행렬이다.

    하드웨어 단순화를 위해 Q/K/V와 FC 모두 bias를 사용하지 않는다.
    """

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(TIME_FEATURES, TIME_FEATURES, bias=False)
        self.k_proj = nn.Linear(TIME_FEATURES, TIME_FEATURES, bias=False)
        self.v_proj = nn.Linear(TIME_FEATURES, TIME_FEATURES, bias=False)
        self.fc = nn.Linear(NUM_SENSORS * TIME_FEATURES, 1, bias=False)

    def forward(self, x, return_attention=False):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * ATTENTION_SCALE

        # 각 query 센서가 key 센서 4개에 주는 확률의 합이 1이 된다.
        attention = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention, v)

        # sensor-major 순서로 4x4를 16개 값으로 펼친다.
        flattened = attention_output.reshape(x.shape[0], -1)
        logits = self.fc(flattened).squeeze(-1)

        if return_attention:
            # 열 합은 각 key 센서가 다른 센서들로부터 받은 총 주의량이다.
            cause_scores = attention.sum(dim=1)
            cause_index = cause_scores.argmax(dim=1)
            return logits, attention, cause_scores, cause_index

        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "sensor x time 4x4 Self-Attention 이진 이상 탐지 모델 학습"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("processed_dataset_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("attention_training_results"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--pos-weight",
        choices=["auto", "none"],
        default="auto",
        help=(
            "auto: Train의 정상/비정상 비율을 BCE positive weight로 사용, "
            "none: 가중치 없는 BCE 사용"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "코드 동작 확인용 소규모 실행. 최종 성능으로 사용하면 안 됨."
        ),
    )
    return parser.parse_args()


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested_device):
    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda가 지정됐지만 CUDA를 사용할 수 없습니다."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_split(data_dir, split_name, sample_limit=None):
    X_path = data_dir / f"X_{split_name}.npy"
    y_path = data_dir / f"y_{split_name}.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"{split_name} 데이터 파일을 찾을 수 없습니다: "
            f"{X_path}, {y_path}"
        )

    X = np.load(X_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.float32, copy=False)

    if sample_limit is not None:
        X = X[:sample_limit]
        y = y[:sample_limit]

    if X.ndim != 3 or X.shape[1:] != (NUM_SENSORS, TIME_FEATURES):
        raise ValueError(
            f"{split_name} X shape이 올바르지 않습니다: {X.shape}"
        )
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(
            f"{split_name} X/y 길이가 일치하지 않습니다: "
            f"X={X.shape}, y={y.shape}"
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{split_name} 데이터에 NaN 또는 무한대가 있습니다.")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError(f"{split_name} y에 0/1 이외의 라벨이 있습니다.")

    return X, y


def make_loader(X, y, batch_size, shuffle, device):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def calculate_binary_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))

    accuracy = safe_divide(tp + tn, len(labels))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "samples": int(len(labels)),
    }


def threshold_records(labels, probabilities):
    records = []

    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(np.int64)
        metrics = calculate_binary_metrics(labels, predictions)
        records.append(
            {
                "threshold": float(threshold),
                **metrics,
            }
        )

    return records


def choose_best_threshold(labels, probabilities):
    records = threshold_records(labels, probabilities)

    # 1순위 F1, 2순위 Recall, 3순위 0.5와 가까운 임계값
    best = max(
        records,
        key=lambda row: (
            row["f1"],
            row["recall"],
            -abs(row["threshold"] - 0.5),
        ),
    )

    return best, records


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = len(y_batch)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size

    return loss_sum / sample_count


@torch.inference_mode()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    collect_attention=False,
):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    all_labels = []
    all_probabilities = []
    all_attention = []
    all_cause_scores = []
    all_cause_indices = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if collect_attention:
            logits, attention, cause_scores, cause_indices = model(
                X_batch, return_attention=True
            )
            all_attention.append(attention.cpu().numpy())
            all_cause_scores.append(cause_scores.cpu().numpy())
            all_cause_indices.append(cause_indices.cpu().numpy())
        else:
            logits = model(X_batch)

        loss = criterion(logits, y_batch)
        probabilities = torch.sigmoid(logits)

        batch_size = len(y_batch)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        all_labels.append(y_batch.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    result = {
        "loss": loss_sum / sample_count,
        "labels": np.concatenate(all_labels).astype(np.int64),
        "probabilities": np.concatenate(all_probabilities).astype(
            np.float32
        ),
    }

    if collect_attention:
        result["attention"] = np.concatenate(all_attention).astype(
            np.float32
        )
        result["cause_scores"] = np.concatenate(all_cause_scores).astype(
            np.float32
        )
        result["cause_indices"] = np.concatenate(
            all_cause_indices
        ).astype(np.int64)

    return result


def clone_state_dict_to_cpu(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_confusion_matrix_png(metrics, output_path, title):
    matrix = np.asarray(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        dtype=np.int64,
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], labels=["Predicted Normal", "Predicted Fault"])
    ax.set_yticks([0, 1], labels=["Actual Normal", "Actual Fault"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    threshold = matrix.max() / 2.0
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=12,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_training_curve(history_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="F1")
    axes[1].plot(
        history_df["epoch"], history_df["val_recall"], label="Recall"
    )
    axes[1].plot(
        history_df["epoch"],
        history_df["val_precision"],
        label="Precision",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Validation Metrics")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_file_metrics(
    labels,
    predictions,
    file_ids,
    file_split_df,
    output_path,
):
    file_name_map = {
        int(row.file_id): str(row.base_name)
        for row in file_split_df.itertuples()
    }
    rows = []

    for file_id in sorted(np.unique(file_ids).tolist()):
        mask = file_ids == file_id
        metrics = calculate_binary_metrics(
            labels[mask], predictions[mask]
        )
        rows.append(
            {
                "file_id": int(file_id),
                "base_name": file_name_map[int(file_id)],
                **metrics,
            }
        )

    pd.DataFrame(rows).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )


def save_event_cause_summary(
    labels,
    predictions,
    cause_scores,
    event_ids,
    file_ids,
    fault_times,
    file_split_df,
    output_path,
):
    file_name_map = {
        int(row.file_id): str(row.base_name)
        for row in file_split_df.itertuples()
    }
    rows = []

    for event_id in sorted(np.unique(event_ids).tolist()):
        event_mask = event_ids == event_id
        predicted_fault_mask = event_mask & (predictions == 1)
        true_fault_mask = event_mask & (labels == 1)
        file_id = int(file_ids[event_mask][0])
        fault_time = float(fault_times[event_mask][0])

        row = {
            "event_id": int(event_id),
            "file_id": file_id,
            "base_name": file_name_map[file_id],
            "fault_time": fault_time,
            "windows": int(event_mask.sum()),
            "true_fault_windows": int(true_fault_mask.sum()),
            "predicted_fault_windows": int(predicted_fault_mask.sum()),
            "detected_true_fault_windows": int(
                np.sum(true_fault_mask & (predictions == 1))
            ),
        }

        if predicted_fault_mask.any():
            mean_scores = cause_scores[predicted_fault_mask].mean(axis=0)
            cause_index = int(np.argmax(mean_scores))
            row["predicted_cause_index"] = cause_index
            row["predicted_cause_sensor"] = SENSOR_NAMES[cause_index]

            for sensor_idx, sensor_name in enumerate(SENSOR_NAMES):
                row[f"cause_score_{sensor_idx}_{sensor_name}"] = float(
                    mean_scores[sensor_idx]
                )
        else:
            row["predicted_cause_index"] = -1
            row["predicted_cause_sensor"] = "NO_FAULT_PREDICTED"

            for sensor_idx, sensor_name in enumerate(SENSOR_NAMES):
                row[f"cause_score_{sensor_idx}_{sensor_name}"] = np.nan

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )


def main():
    args = parse_args()
    set_reproducible_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    smoke_limits = {
        "train": 20_000,
        "val": 8_000,
        "test": 8_000,
    }

    X_train, y_train = load_split(
        args.data_dir,
        "train",
        smoke_limits["train"] if args.smoke_test else None,
    )
    X_val, y_val = load_split(
        args.data_dir,
        "val",
        smoke_limits["val"] if args.smoke_test else None,
    )
    X_test, y_test = load_split(
        args.data_dir,
        "test",
        smoke_limits["test"] if args.smoke_test else None,
    )

    train_loader = make_loader(
        X_train,
        y_train,
        args.batch_size,
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        args.batch_size * 2,
        shuffle=False,
        device=device,
    )
    test_loader = make_loader(
        X_test,
        y_test,
        args.batch_size * 2,
        shuffle=False,
        device=device,
    )

    model = SensorSelfAttentionClassifier().to(device)
    train_negative = int(np.sum(y_train == 0.0))
    train_positive = int(np.sum(y_train == 1.0))

    if args.pos_weight == "auto":
        positive_weight_value = train_negative / train_positive
    else:
        positive_weight_value = 1.0

    positive_weight = torch.tensor(
        positive_weight_value,
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"Device: {device}")
    print(f"Model parameters: {total_parameters}")
    print(f"Train samples: {len(y_train):,}")
    print(f"Validation samples: {len(y_val):,}")
    print(f"Test samples: {len(y_test):,}")
    print(f"BCE positive weight: {positive_weight_value:.6f}")

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_threshold = 0.5
    best_state = None
    epochs_without_improvement = 0
    history = []
    training_start_time = time.time()

    max_epochs = min(args.epochs, 3) if args.smoke_test else args.epochs

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_result = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            collect_attention=False,
        )
        best_threshold_row, _ = choose_best_threshold(
            val_result["labels"], val_result["probabilities"]
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "val_threshold": best_threshold_row["threshold"],
            "val_accuracy": best_threshold_row["accuracy"],
            "val_precision": best_threshold_row["precision"],
            "val_recall": best_threshold_row["recall"],
            "val_f1": best_threshold_row["f1"],
            "val_specificity": best_threshold_row["specificity"],
            "seconds": time.time() - epoch_start_time,
        }
        history.append(history_row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_result['loss']:.6f} | "
            f"threshold={best_threshold_row['threshold']:.2f} | "
            f"P={best_threshold_row['precision']:.4f} | "
            f"R={best_threshold_row['recall']:.4f} | "
            f"F1={best_threshold_row['f1']:.4f}"
        )

        improved = (
            best_threshold_row["f1"] > best_val_f1 + 1e-12
            or (
                abs(best_threshold_row["f1"] - best_val_f1) <= 1e-12
                and val_result["loss"] < best_val_loss
            )
        )

        if improved:
            best_val_f1 = best_threshold_row["f1"]
            best_val_loss = val_result["loss"]
            best_epoch = epoch
            best_threshold = best_threshold_row["threshold"]
            best_state = clone_state_dict_to_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: Validation F1이 {args.patience}회 "
                "연속 개선되지 않았습니다."
            )
            break

    if best_state is None:
        raise RuntimeError("최적 모델 상태가 저장되지 않았습니다.")

    model.load_state_dict(best_state)
    model.to(device)

    # 최적 모델에서 Validation 임계값을 한 번 더 확정한다.
    final_val_result = evaluate_model(
        model,
        val_loader,
        criterion,
        device,
        collect_attention=False,
    )
    best_threshold_row, threshold_search_rows = choose_best_threshold(
        final_val_result["labels"], final_val_result["probabilities"]
    )
    best_threshold = best_threshold_row["threshold"]

    # Test는 모델과 threshold가 확정된 뒤 여기서 처음 평가한다.
    test_result = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        collect_attention=True,
    )
    test_predictions = (
        test_result["probabilities"] >= best_threshold
    ).astype(np.int64)
    test_metrics = calculate_binary_metrics(
        test_result["labels"], test_predictions
    )

    always_normal_predictions = np.zeros_like(test_result["labels"])
    baseline_metrics = calculate_binary_metrics(
        test_result["labels"], always_normal_predictions
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        args.output_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(threshold_search_rows).to_csv(
        args.output_dir / "validation_threshold_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    file_split_df = pd.read_csv(args.data_dir / "file_split.csv")
    test_metadata = np.load(args.data_dir / "metadata_test.npz")

    if args.smoke_test:
        metadata_limit = len(test_result["labels"])
        test_file_ids = test_metadata["file_ids"][:metadata_limit]
        test_event_ids = test_metadata["event_ids"][:metadata_limit]
        test_fault_times = test_metadata["fault_times"][:metadata_limit]
    else:
        test_file_ids = test_metadata["file_ids"]
        test_event_ids = test_metadata["event_ids"]
        test_fault_times = test_metadata["fault_times"]

    if len(test_file_ids) != len(test_result["labels"]):
        raise ValueError("Test metadata와 예측 결과 길이가 일치하지 않습니다.")

    save_file_metrics(
        test_result["labels"],
        test_predictions,
        test_file_ids,
        file_split_df,
        args.output_dir / "test_metrics_by_file.csv",
    )
    save_event_cause_summary(
        test_result["labels"],
        test_predictions,
        test_result["cause_scores"],
        test_event_ids,
        test_file_ids,
        test_fault_times,
        file_split_df,
        args.output_dir / "test_event_cause_summary.csv",
    )

    np.savez_compressed(
        args.output_dir / "test_predictions_and_attention.npz",
        labels=test_result["labels"],
        probabilities=test_result["probabilities"],
        predictions=test_predictions,
        attention=test_result["attention"],
        cause_scores=test_result["cause_scores"],
        cause_indices=test_result["cause_indices"],
        file_ids=test_file_ids,
        event_ids=test_event_ids,
        fault_times=test_fault_times,
        threshold=np.asarray(best_threshold, dtype=np.float32),
        sensor_names=np.asarray(SENSOR_NAMES),
    )

    checkpoint = {
        "model_state_dict": best_state,
        "model_class": "SensorSelfAttentionClassifier",
        "sensor_names": SENSOR_NAMES,
        "input_axis": "X[batch, sensor, time]",
        "attention_scale": ATTENTION_SCALE,
        "bias": False,
        "best_epoch": best_epoch,
        "validation_threshold": best_threshold,
        "positive_weight": positive_weight_value,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "best_model.pt")

    results = {
        "smoke_test": args.smoke_test,
        "device": str(device),
        "model_parameters": total_parameters,
        "model_bias": False,
        "input_axis": "X[batch, sensor, time]",
        "sensor_order": SENSOR_NAMES,
        "best_epoch": best_epoch,
        "validation_threshold": best_threshold,
        "positive_weight": positive_weight_value,
        "validation_metrics": best_threshold_row,
        "test_loss": test_result["loss"],
        "test_metrics": test_metrics,
        "always_normal_baseline": baseline_metrics,
        "training_seconds": time.time() - training_start_time,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }

    with open(
        args.output_dir / "metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    save_confusion_matrix_png(
        test_metrics,
        args.output_dir / "test_confusion_matrix.png",
        title=(
            f"Test Confusion Matrix (threshold={best_threshold:.2f})"
        ),
    )
    save_training_curve(
        history_df,
        args.output_dir / "training_curves.png",
    )

    print("\n[학습 및 Test 평가 완료]")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation threshold: {best_threshold:.2f}")
    print(f"Test accuracy:  {test_metrics['accuracy']:.6f}")
    print(f"Test precision: {test_metrics['precision']:.6f}")
    print(f"Test recall:    {test_metrics['recall']:.6f}")
    print(f"Test F1-score:  {test_metrics['f1']:.6f}")
    print(
        f"Confusion matrix: TN={test_metrics['tn']}, "
        f"FP={test_metrics['fp']}, FN={test_metrics['fn']}, "
        f"TP={test_metrics['tp']}"
    )
    print(f"결과 폴더: {args.output_dir.resolve()}")

    if args.smoke_test:
        print(
            "[주의] --smoke-test 결과는 기능 확인용이며 최종 성능으로 "
            "보고하면 안 됩니다."
        )


if __name__ == "__main__":
    main()
