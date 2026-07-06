from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\Users\SEBAS\PycharmProjects\YoloModel")
OUTPUT = Path(__file__).resolve().parents[1] / "metrics" / "model_metrics.json"
MOVINET_DIR = Path(__file__).resolve().parents[1] / "models" / "movinet_a2_cctv_direct_finetuning"
MOBILENET_BIGRU_DIR = Path(__file__).resolve().parents[1] / "models" / "runs_mobilenetv3_bigru_cctv"

RUNS = (
    {
        "id": "timesformer",
        "name": "TimeSformer",
        "type": "Video",
        "status": "Entrenado",
        "run_dir": ROOT / "runs_timesformer_cctv_clean",
        "checkpoint_dir": ROOT / "runs_timesformer_cctv_clean" / "best_timesformer_cctv",
        "model_path": ROOT / "runs_timesformer_cctv_clean" / "best_timesformer_cctv.pt",
        "repo_id": "inicial-tope/capstone-timesformer-cctv",
    },
    {
        "id": "videomae",
        "name": "VideoMAE",
        "type": "Video",
        "status": "Entrenado",
        "run_dir": ROOT / "runs_videomae_cctv_clean",
        "checkpoint_dir": ROOT / "runs_videomae_cctv_clean" / "best_videomae_cctv",
        "model_path": ROOT / "runs_videomae_cctv_clean" / "best_videomae_cctv.pt",
        "repo_id": "inicial-tope/capstone-videomae-cctv",
    },
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "epoch": int(float(row["epoch"])),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["valid_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "val_accuracy": float(row["valid_accuracy"]),
                }
            )
    return rows


def read_keras_history(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for epoch, row in enumerate(reader, start=1):
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(row["loss"]),
                    "val_loss": float(row["val_loss"]),
                    "train_accuracy": float(row["accuracy"]),
                    "val_accuracy": float(row["val_accuracy"]),
                }
            )
    return rows


def confusion_rows(matrix: list[list[int]]) -> list[dict[str, int | str]]:
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    return [
        {"actual": "No agresion", "predicted": "No agresion", "count": true_negative},
        {"actual": "No agresion", "predicted": "Agresion", "count": false_positive},
        {"actual": "Agresion", "predicted": "No agresion", "count": false_negative},
        {"actual": "Agresion", "predicted": "Agresion", "count": true_positive},
    ]


def accuracy_from_confusion(matrix: list[list[int]]) -> float:
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    total = true_negative + false_positive + false_negative + true_positive
    return (true_negative + true_positive) / total


def build_model_entry(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run["run_dir"])
    checkpoint_dir = Path(run["checkpoint_dir"])
    final_metrics = read_json(run_dir / "final_metrics.json")
    decision_config = read_json(checkpoint_dir / "cctv_decision_config.json")
    model_config = read_json(checkpoint_dir / "config.json")
    matrix = final_metrics["confusion_matrix"]
    test_metrics = final_metrics["test_at_0_5"]

    return {
        "id": run["id"],
        "name": run["name"],
        "type": run["type"],
        "status": run["status"],
        "framework": "PyTorch + Hugging Face Transformers",
        "model_path": "",
        "checkpoint_dir": "",
        "repo_id": run["repo_id"],
        "summary": {
            "accuracy": accuracy_from_confusion(matrix),
            "precision": final_metrics["test_precision_at_valid_threshold"],
            "recall": final_metrics["test_recall_at_valid_threshold"],
            "f1_score": final_metrics["test_f1_at_valid_threshold"],
            "roc_auc": test_metrics["test_roc_auc"],
            "pr_auc": test_metrics["test_pr_auc"],
        },
        "decision": {
            "threshold": final_metrics["threshold_from_valid"],
            "threshold_selected_on": "valid",
            "threshold_metric": "F2",
        },
        "preprocessing": {
            "input_size": [decision_config["image_size"], decision_config["image_size"]],
            "num_frames": decision_config["num_frames"],
            "normalize_mean": decision_config["normalization_mean"],
            "normalize_std": decision_config["normalization_std"],
            "input_layout": "B,T,C,H,W",
        },
        "labels": {
            "0": "No agresion",
            "1": "Agresion",
            "raw_id2label": model_config["id2label"],
        },
        "confusion_matrix": confusion_rows(matrix),
        "history": read_history(run_dir / "history.csv"),
        "notes": "ROC/PR AUC are real. Curve points are estimated in Streamlit because prediction scores were not exported.",
    }


def build_movinet_entry() -> dict[str, Any] | None:
    metrics_path = MOVINET_DIR / "metrics.json"
    history_path = MOVINET_DIR / "history.csv"
    model_path = MOVINET_DIR / "best_movinet_a2_cctv_direct.keras"
    if not metrics_path.exists() or not history_path.exists() or not model_path.exists():
        return None

    metrics = read_json(metrics_path)
    matrix = metrics["confusion_matrix"]
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return {
        "id": "movinet_a2",
        "name": "MoViNet-A2",
        "type": "Video",
        "status": "Entrenado",
        "framework": "Keras 3 + TensorFlow Hub",
        "model_path": "",
        "checkpoint_dir": "",
        "repo_id": "inicial-tope/capstone-movinet-a2-cctv",
        "hf_filename": "best_movinet_a2_cctv_direct.keras",
        "summary": {
            "accuracy": accuracy_from_confusion(matrix),
            "precision": precision,
            "recall": recall,
            "f1_score": metrics["test_f1"],
            "roc_auc": metrics["test_roc_auc"],
            "pr_auc": metrics["test_pr_auc"],
        },
        "decision": {
            "threshold": metrics["threshold"],
            "threshold_selected_on": "valid",
            "threshold_metric": "F1",
        },
        "preprocessing": {
            "input_size": [224, 224],
            "num_frames": 16,
            "normalize_mean": [0.0, 0.0, 0.0],
            "normalize_std": [1.0, 1.0, 1.0],
            "input_layout": "B,T,H,W,C",
        },
        "labels": {
            "0": "No agresion",
            "1": "Agresion",
        },
        "confusion_matrix": confusion_rows(matrix),
        "history": read_keras_history(history_path),
        "notes": "Metrics imported from Keras run. Inference rebuilds hub_logits_600 with the original MoViNet-A2 Kinetics-600 TensorFlow Hub backbone.",
    }


def build_mobilenet_bigru_entry() -> dict[str, Any] | None:
    metrics_path = MOBILENET_BIGRU_DIR / "final_metrics.json"
    history_path = MOBILENET_BIGRU_DIR / "history.csv"
    config_path = MOBILENET_BIGRU_DIR / "cctv_decision_config.json"
    model_path = MOBILENET_BIGRU_DIR / "best_mobilenetv3_bigru_cctv.keras"
    if not metrics_path.exists() or not history_path.exists() or not config_path.exists() or not model_path.exists():
        return None

    metrics = read_json(metrics_path)
    config = read_json(config_path)
    matrix = metrics["confusion_matrix"]
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return {
        "id": "mobilenetv3_bigru",
        "name": "MobileNetV3Small + BiGRU",
        "type": "Video",
        "status": "Entrenado",
        "framework": "Keras 3 / TensorFlow",
        "model_path": "",
        "checkpoint_dir": "",
        "repo_id": "inicial-tope/capstone-mobilenetv3-bigru-cctv",
        "hf_filename": "best_mobilenetv3_bigru_cctv.keras",
        "summary": {
            "accuracy": accuracy_from_confusion(matrix),
            "precision": precision,
            "recall": recall,
            "f1_score": metrics["test_f1_at_valid_threshold"],
            "roc_auc": metrics["test_roc_auc"],
            "pr_auc": metrics["test_pr_auc"],
        },
        "decision": {
            "threshold": metrics["threshold_from_valid"],
            "threshold_selected_on": config.get("threshold_selected_on", "valid"),
            "threshold_metric": metrics["threshold_metric"],
        },
        "preprocessing": {
            "input_size": [config["image_size"], config["image_size"]],
            "num_frames": config["num_frames"],
            "input_range": config["input_range"],
            "input_layout": "B,T,H,W,C",
        },
        "labels": {
            "0": "No agresion",
            "1": "Agresion",
            "raw_class_names": config["class_names"],
        },
        "confusion_matrix": confusion_rows(matrix),
        "history": read_keras_history(history_path),
        "notes": "Metrics imported from Keras MobileNetV3Small + BiGRU run.",
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    models = [build_model_entry(run) for run in RUNS]
    movinet_entry = build_movinet_entry()
    if movinet_entry is not None:
        models.append(movinet_entry)
    mobilenet_bigru_entry = build_mobilenet_bigru_entry()
    if mobilenet_bigru_entry is not None:
        models.append(mobilenet_bigru_entry)

    payload = {
        "source": str(ROOT),
        "models": models,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
