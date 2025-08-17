import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import librosa
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Reuse dataset utilities from training
from train_and_export import (
    PreprocessingConfig,
    MelFeatureExtractor,
    download_fsdd,
    parse_fsdd,
    split_data,
)


def load_assets(assets_dir: str) -> Tuple[ort.InferenceSession, dict]:
    model_path = os.path.join(assets_dir, "digit_cnn.onnx")
    cfg_path = os.path.join(assets_dir, "preprocess.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return session, cfg


def extract_features(wav_path: str, mean: float, std: float, cfg: dict) -> np.ndarray:
    """Extract mel features and normalize using training stats; returns (1,1,n_mels,time) float32."""
    pre = PreprocessingConfig(
        sample_rate=int(cfg["preprocessing"]["sample_rate"]),
        target_duration_seconds=float(cfg["preprocessing"]["target_duration_seconds"]),
        n_fft=int(cfg["preprocessing"]["n_fft"]),
        hop_length=int(cfg["preprocessing"]["hop_length"]),
        n_mels=int(cfg["preprocessing"]["n_mels"]),
        fmin=int(cfg["preprocessing"]["fmin"]),
        fmax=int(cfg["preprocessing"]["fmax"]),
    )
    extractor = MelFeatureExtractor(pre)
    y, _ = librosa.load(wav_path, sr=pre.sample_rate, mono=True)
    S_db = extractor.extract_from_waveform(y)
    S_db = (S_db - float(cfg["mean"])) / float(cfg["std"])  # normalize
    return S_db[None, None, :, :].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join("data", "fsdd"))
    parser.add_argument("--assets_dir", default="assets")
    parser.add_argument("--output", default=None, help="Path to write misclassifications JSON (defaults to assets)")
    args = parser.parse_args()

    # Load model and config
    session, cfg = load_assets(args.assets_dir)
    mean = float(cfg["mean"])
    std = float(cfg["std"]) if float(cfg["std"]) > 1e-12 else 1.0

    # Prepare dataset (use same split as training)
    recordings_dir = download_fsdd(args.data_dir)
    filepaths, labels = parse_fsdd(recordings_dir)
    train_files, train_labels, test_files, test_labels = split_data(filepaths, labels)

    # Evaluate on test set
    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []
    misclassified_examples = []

    input_name = session.get_inputs()[0].name

    for wav_path, true_label in tqdm(list(zip(test_files, test_labels)), desc="Evaluating test set"):
        feats = extract_features(wav_path, mean, std, cfg)
        outputs = session.run(None, {input_name: feats})
        logits = outputs[0][0]  # (10,)
        probs = softmax(logits)
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        y_true.append(int(true_label))
        y_pred.append(pred)
        confidences.append(conf)

        if pred != int(true_label):
            misclassified_examples.append({
                "file": wav_path,
                "true": int(true_label),
                "pred": pred,
                "confidence": conf,
            })

    # Metrics
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(range(10)))

    # Build top confusion pairs (by rate)
    confusion_pairs = []
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            count = int(cm[i, j])
            if count == 0:
                continue
            rate = count / int(support[i]) if int(support[i]) > 0 else 0.0
            confusion_pairs.append([i, j, count, float(rate)])
    # Sort by rate then count
    confusion_pairs.sort(key=lambda x: (x[3], x[2]), reverse=True)
    top_confusions = confusion_pairs[:10]

    results = {
        "overall_accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "confusion_matrix": cm.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "top_confusions": top_confusions,
        "num_misclassified": len(misclassified_examples),
        "misclassified_examples": misclassified_examples,
    }

    out_path = args.output or os.path.join(args.assets_dir, "misclassifications.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print a concise summary to console
    print("\nTop confusions (true -> pred, count, rate):")
    for i, (t, p, c, r) in enumerate(top_confusions, start=1):
        print(f"{i:2d}. {t} -> {p}: {c} ({r*100:.1f}%)")
    print(f"\nSaved detailed misclassification report to: {out_path}")


if __name__ == "__main__":
    main()


