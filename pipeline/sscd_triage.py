#!/usr/bin/env python3
"""Independent SSCD ensemble evidence for blinded retinal copy triage.

This stage writes evidence only. It never writes manual-review decisions and
never mutates the frozen CPU/DINO audit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def load_pairs(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(("pair_id", "left", "right")) - row.keys():
                raise RuntimeError(f"Missing pair identity at line {number}")
            if row["pair_id"] in seen:
                raise RuntimeError(f"Duplicate pair_id at line {number}")
            seen.add(row["pair_id"])
            rows.append(row)
    if not rows:
        raise RuntimeError("Private review-pair ledger is empty")
    return rows


def preprocess(image: Image.Image, size: int = 320) -> np.ndarray:
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    values = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (values - MEAN) / STD


def transformed(image: Image.Image, variant: int) -> Image.Image:
    image = image.convert("RGB")
    if variant == 0:
        return image.rotate(5, resample=Image.Resampling.BILINEAR,
                            expand=False, fillcolor=(0, 0, 0))
    if variant == 1:
        width, height = image.size
        margin_x, margin_y = max(1, width // 25), max(1, height // 25)
        return image.crop((margin_x, margin_y, width - margin_x,
                           height - margin_y)).resize(image.size,
                                                      Image.Resampling.LANCZOS)
    if variant == 2:
        return ImageEnhance.Contrast(
            ImageEnhance.Brightness(image).enhance(0.78)
        ).enhance(1.18)
    if variant == 3:
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    if variant == 4:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=55, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    raise ValueError(f"Unknown transformation variant: {variant}")


class ImageDataset:
    def __init__(self, root: Path, keys: list[str]):
        self.root = Path(root)
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        import torch
        path = self.root / self.keys[index]
        try:
            with Image.open(path) as image:
                values = preprocess(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode {path}") from exc
        return index, torch.from_numpy(np.ascontiguousarray(values))


class CalibrationDataset:
    def __init__(self, root: Path, keys: list[str]):
        self.root = Path(root)
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        import torch
        path = self.root / self.keys[index]
        try:
            with Image.open(path) as image:
                original = image.convert("RGB")
                base = preprocess(original)
                variants = np.stack([
                    preprocess(transformed(original, variant))
                    for variant in range(5)
                ])
        except Exception as exc:
            raise RuntimeError(f"Failed calibration decode: {path}") from exc
        bases = np.repeat(base[None, ...], 5, axis=0)
        return (index,
                torch.from_numpy(np.ascontiguousarray(bases)),
                torch.from_numpy(np.ascontiguousarray(variants)))


def load_model(path: Path, device: str):
    import torch
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def model_features(model, batch, device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16,
        enabled=device.startswith("cuda"),
    ):
        result = model(batch.to(device, non_blocking=True))
        if isinstance(result, (tuple, list)):
            result = result[0]
        result = functional.normalize(result.float().flatten(1), dim=1)
    return result.cpu().numpy().astype(np.float32, copy=False)


def embed(model, data_root: Path, keys: list[str], output: Path, device: str,
          workers: int, batch_size: int, model_id: str) -> np.ndarray:
    import torch
    dataset = ImageDataset(data_root, keys)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    partial = output / f"{model_id}_embeddings.partial.npy"
    final = output / f"{model_id}_embeddings.npy"
    progress = output / f"{model_id}_progress.json"
    if final.is_file():
        return np.load(final, mmap_mode="r")
    matrix = None
    completed = 0
    if partial.is_file() and progress.is_file():
        record = json.loads(progress.read_text(encoding="utf-8"))
        if record.get("keys_sha256") != identity(keys):
            raise RuntimeError(f"{model_id} resume key identity mismatch")
        completed = int(record["completed"])
        matrix = np.lib.format.open_memmap(partial, mode="r+")
    started = time.monotonic()
    for indices, batch in loader:
        indices_np = indices.numpy()
        keep = indices_np >= completed
        if not keep.any():
            continue
        selected = batch[torch.from_numpy(keep)]
        selected_indices = indices_np[keep]
        values = model_features(model, selected, device)
        if matrix is None:
            matrix = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float32,
                shape=(len(keys), values.shape[1]),
            )
        matrix[selected_indices] = values
        matrix.flush()
        completed = int(selected_indices[-1]) + 1
        elapsed = max(time.monotonic() - started, 1e-9)
        atomic_json(progress, {
            "model_id": model_id,
            "keys_sha256": identity(keys),
            "completed": completed,
            "total": len(keys),
            "images_per_second_since_start": completed / elapsed,
        })
        if completed % (batch_size * 10) == 0 or completed == len(keys):
            print(f"{model_id} embeddings: {completed}/{len(keys)}", flush=True)
    if matrix is None or completed != len(keys):
        raise RuntimeError(f"Incomplete {model_id} embeddings: {completed}/{len(keys)}")
    del matrix
    os.replace(partial, final)
    progress.unlink(missing_ok=True)
    return np.load(final, mmap_mode="r")


def calibration_features(model, data_root: Path, keys: list[str], device: str,
                         sample_size: int = 250, workers: int = 1,
                         image_batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    import torch
    chosen = sorted(keys, key=lambda key: hashlib.sha256(key.encode()).hexdigest())[
        :min(sample_size, len(keys))
    ]
    dataset = CalibrationDataset(data_root, chosen)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=image_batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    results_a, results_b = [], []
    completed = 0
    for _, bases, variants in loader:
        flat_bases = bases.flatten(0, 1)
        flat_variants = variants.flatten(0, 1)
        results_a.append(model_features(model, flat_bases, device))
        results_b.append(model_features(model, flat_variants, device))
        completed += bases.shape[0]
        if completed % 40 == 0 or completed == len(dataset):
            print(f"Calibration source images: {completed}/{len(dataset)}", flush=True)
    return np.concatenate(results_a), np.concatenate(results_b)


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability,
                             method="linear"))


def calibrate(embeddings: np.ndarray, positives_a: np.ndarray,
              positives_b: np.ndarray, seed: int = 20261104,
              negative_pairs: int = 20000) -> dict:
    positives = np.sum(positives_a * positives_b, axis=1)
    rng = random.Random(seed)
    negative_scores = []
    size = len(embeddings)
    seen = set()
    while len(negative_scores) < min(negative_pairs, size * (size - 1) // 2):
        left = rng.randrange(size)
        right = rng.randrange(size)
        if left == right:
            continue
        pair = (min(left, right), max(left, right))
        if pair in seen:
            continue
        seen.add(pair)
        negative_scores.append(float(np.dot(embeddings[left], embeddings[right])))
    negatives = np.asarray(negative_scores, dtype=np.float64)
    positive_recall_floor = quantile(positives, 0.005)
    negative_maximum = float(negatives.max())
    separation = positive_recall_floor - negative_maximum
    locked = separation > 0
    threshold = ((positive_recall_floor + negative_maximum) / 2
                 if locked else None)
    return {
        "locked": locked,
        "synthetic_positive_pairs": int(len(positives)),
        "presumed_negative_pairs": int(len(negatives)),
        "positive_similarity": {
            "minimum": float(positives.min()),
            "p005": positive_recall_floor,
            "median": quantile(positives, 0.5),
            "maximum": float(positives.max()),
        },
        "presumed_negative_similarity": {
            "minimum": float(negatives.min()),
            "median": quantile(negatives, 0.5),
            "p999": quantile(negatives, 0.999),
            "maximum": negative_maximum,
        },
        "separation_margin": separation,
        "provisional_confirmation_threshold": threshold,
        "provisional_rejection_ceiling": float(positives.min()),
        "limitations": [
            "Synthetic transformations do not span every real acquisition change",
            "Random distinct-image pairs are presumed, not adjudicated, negatives",
            "Thresholds are evidence inputs and cannot create human-review claims",
        ],
    }


def write_scores(path: Path, pairs: list[dict], keys: list[str],
                 embeddings_by_model: dict[str, np.ndarray], execution_id: str) -> None:
    index = {key: number for number, key in enumerate(keys)}
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            left, right = index[pair["left"]], index[pair["right"]]
            similarities = {
                model_id: float(np.dot(matrix[left], matrix[right]))
                for model_id, matrix in embeddings_by_model.items()
            }
            handle.write(json.dumps({
                "pair_id": pair["pair_id"],
                "execution_id": execution_id,
                "similarities": similarities,
            }, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(args) -> dict:
    data_root = args.data.resolve()
    pair_path = args.pairs.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(pair_path)
    keys = sorted({row[side] for row in pairs for side in ("left", "right")})
    models = {
        "sscd_large": args.large_weights.resolve(),
        "sscd_mixup": args.mixup_weights.resolve(),
    }
    weights = {name: digest(path) for name, path in models.items()}
    execution_id = identity({
        "pair_ledger_sha256": digest(pair_path),
        "weights": weights,
        "keys_sha256": identity(keys),
        "preprocessing": "official square-320 ImageNet normalization",
        "code_sha256": digest(Path(__file__)),
    })
    incomplete = output / "SSCD_INCOMPLETE"
    incomplete.write_text(execution_id + "\n", encoding="utf-8")
    (output / "SSCD_COMPLETE.json").unlink(missing_ok=True)
    atomic_json(output / "sscd_keys.json", keys)
    embeddings_by_model = {}
    calibrations = {}
    for model_id, weight_path in models.items():
        print(f"Loading {model_id}: {weight_path}", flush=True)
        model = load_model(weight_path, args.device)
        embeddings = embed(
            model, data_root, keys, output, args.device,
            args.workers, args.batch_size, model_id,
        )
        positive_a, positive_b = calibration_features(
            model, data_root, keys, args.device, args.calibration_images,
            workers=min(args.workers, args.calibration_workers),
            image_batch_size=args.calibration_batch_images,
        )
        calibrations[model_id] = calibrate(
            embeddings, positive_a, positive_b,
            negative_pairs=args.calibration_negatives,
        )
        embeddings_by_model[model_id] = embeddings
        del model
        import torch
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    scores = output / "sscd_scores.jsonl"
    write_scores(scores, pairs, keys, embeddings_by_model, execution_id)
    summary = {
        "execution_id": execution_id,
        "purpose": "independent machine evidence; not human adjudication",
        "images": len(keys),
        "pairs": len(pairs),
        "device": args.device,
        "weights_sha256": weights,
        "calibration": calibrations,
        "artifacts": {
            "sscd_keys.json": digest(output / "sscd_keys.json"),
            "sscd_scores.jsonl": digest(scores),
            **{
                f"{model_id}_embeddings.npy": digest(
                    output / f"{model_id}_embeddings.npy"
                ) for model_id in models
            },
        },
    }
    atomic_json(output / "SSCD_COMPLETE.json", summary)
    incomplete.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--large-weights", type=Path, required=True)
    parser.add_argument("--mixup-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration-images", type=int, default=250)
    parser.add_argument("--calibration-negatives", type=int, default=20000)
    parser.add_argument("--calibration-workers", type=int, default=64)
    parser.add_argument("--calibration-batch-images", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
