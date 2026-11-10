"""Versioned, fail-closed retinal near-duplicate audit primitives."""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.fft import dctn
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from common import digest, identity, read_json, save_json


@dataclass(frozen=True)
class AuditConfig:
    values: dict[str, Any]

    @classmethod
    def from_json(cls, path: Path) -> 'AuditConfig':
        values = read_json(path)
        required = {'schema_version', 'hash_bits'}
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError('Missing audit configuration keys: ' + ', '.join(missing))
        if values['hash_bits'] != 256:
            raise ValueError('Only the frozen 256-bit perceptual hash is supported')
        return cls(values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class NormalizedRetina:
    rgb: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    foreground_fraction: float
    edge_truncated: bool = False


@dataclass(frozen=True)
class FingerprintRecord:
    path: str
    input_sha256: str
    phash: int
    phash_flipped: int
    bbox: tuple[int, int, int, int]
    foreground_fraction: float
    edge_truncated: bool = False


@dataclass(frozen=True)
class PairEvidence:
    ssim: float
    ncc: float
    nmae: float
    mask_iou: float
    orientation: str
    rotation: float
    scale: float
    translation: tuple[float, float]
    vessel_ncc: float = -1.0
    vessel_dice: float = 0.0


def rejected_no_overlap_evidence() -> PairEvidence:
    """Return deterministic rejection evidence when alignment has no retinal overlap."""
    return PairEvidence(
        ssim=-1.0, ncc=-1.0, nmae=1.0, mask_iou=0.0,
        orientation='none', rotation=0.0, scale=1.0,
        translation=(0.0, 0.0),
    )


@dataclass(frozen=True)
class EmbeddingRecord:
    key: str
    vector: np.ndarray


class _HammingNode:
    def __init__(self, key: str, value: int):
        self.value = value
        self.keys = [key]
        self.children: dict[int, '_HammingNode'] = {}


class HammingIndex:
    """Exact BK-tree for integer Hamming distance."""
    def __init__(self):
        self.root: _HammingNode | None = None

    def add(self, key: str, value: int) -> None:
        if self.root is None:
            self.root = _HammingNode(key, value)
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            if distance == 0:
                node.keys.append(key)
                return
            if distance not in node.children:
                node.children[distance] = _HammingNode(key, value)
                return
            node = node.children[distance]

    def query(self, value: int, radius: int,
              exclude_key: str | None = None) -> list[tuple[str, int]]:
        found = []
        if self.root is None:
            return found
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= radius:
                found.extend((key, distance) for key in node.keys if key != exclude_key)
            lower, upper = distance - radius, distance + radius
            pending.extend(child for edge, child in node.children.items()
                           if lower <= edge <= upper)
        return found


def normalize_retina(image: Image.Image, config: AuditConfig,
                     context: str = '<in-memory image>') -> NormalizedRetina:
    image = ImageOps.exif_transpose(image).convert('RGB')
    rgb = np.asarray(image, dtype=np.uint8)
    scaled = rgb.astype(np.float32) / 255.0
    maximum = scaled.max(axis=2)
    minimum = scaled.min(axis=2)
    saturation = np.divide(maximum - minimum, maximum,
                           out=np.zeros_like(maximum), where=maximum > 0)
    roi = config.roi
    mask = ((maximum * 255.0 >= roi['intensity_threshold']) &
            ((saturation >= roi['saturation_threshold']) | (maximum >= 0.08)))
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError(f'Retinal foreground not found: {context}')
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(sizes.argmax())
    component = ndimage.binary_fill_holes(component)
    fraction = float(component.mean())
    if fraction < float(roi['minimum_fraction']):
        raise ValueError(f'Retinal foreground too small ({fraction:.6f}): {context}')
    # Border-filling acquisitions cannot be distinguished reliably from crops by
    # occupancy alone. Retain them for audit, but expose the condition in evidence.
    edge_truncated = fraction > float(roi['maximum_fraction'])
    signal = maximum[component]
    low, high = np.percentile(signal, [1, 99])
    if high < float(roi['minimum_signal_p99']):
        raise ValueError(f'Retinal signal is too dark ({high:.6f}): {context}')
    if high - low < float(roi['minimum_contrast_p99_p01']):
        raise ValueError(f'Retinal signal has insufficient contrast ({high-low:.6f}): {context}')
    ys, xs = np.nonzero(component)
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    side = max(right - left, bottom - top)
    side = int(np.ceil(side * (1.0 + 2.0 * float(roi['padding_fraction']))))
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_left = int(np.floor(center_x - side / 2.0))
    crop_top = int(np.floor(center_y - side / 2.0))
    box = (crop_left, crop_top, crop_left + side, crop_top + side)
    size = int(config.normalized_size)
    crop = image.crop(box).resize((size, size), Image.Resampling.LANCZOS)
    mask_image = Image.fromarray(component.astype(np.uint8) * 255).crop(box).resize(
        (size, size), Image.Resampling.NEAREST)
    normalized_rgb = np.asarray(crop, dtype=np.uint8)
    normalized_mask = np.asarray(mask_image, dtype=np.uint8) > 0
    normalized_gray = np.dot(normalized_rgb[..., :3].astype(np.float32),
                             np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
    normalized_gray *= normalized_mask
    return NormalizedRetina(normalized_rgb, normalized_gray.astype(np.float32),
                            normalized_mask, box, fraction, edge_truncated)


def perceptual_hash(gray: np.ndarray, config: AuditConfig) -> int:
    settings = config.phash
    dct_size = int(settings['dct_size'])
    low_size = int(settings['low_frequency_size'])
    if low_size * low_size != int(config.hash_bits):
        raise ValueError('pHash dimensions do not match hash_bits')
    values = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    resized = np.asarray(Image.fromarray(values).resize(
        (dct_size, dct_size), Image.Resampling.LANCZOS), dtype=np.float32)
    low = dctn(resized, type=2, norm='ortho')[:low_size, :low_size]
    median = float(np.median(low.ravel()[1:]))
    bits = (low > median).ravel()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def fingerprint_one(path: Path, config: AuditConfig) -> FingerprintRecord:
    path = Path(path)
    with Image.open(path) as image:
        normalized = normalize_retina(image, config, path.as_posix())
    return FingerprintRecord(
        path=path.as_posix(),
        input_sha256=digest(path),
        phash=perceptual_hash(normalized.gray, config),
        phash_flipped=perceptual_hash(np.fliplr(normalized.gray), config),
        bbox=normalized.bbox,
        foreground_fraction=normalized.foreground_fraction,
        edge_truncated=normalized.edge_truncated,
    )


def _center_scale(values: np.ndarray, scale: float, order: int) -> np.ndarray:
    if scale == 1.0:
        return values.copy()
    height, width = values.shape
    resized = ndimage.zoom(values, scale, order=order, prefilter=order > 1)
    result = np.zeros_like(values)
    source_y = max(0, (resized.shape[0] - height) // 2)
    source_x = max(0, (resized.shape[1] - width) // 2)
    target_y = max(0, (height - resized.shape[0]) // 2)
    target_x = max(0, (width - resized.shape[1]) // 2)
    copy_h = min(height, resized.shape[0])
    copy_w = min(width, resized.shape[1])
    result[target_y:target_y + copy_h, target_x:target_x + copy_w] = \
        resized[source_y:source_y + copy_h, source_x:source_x + copy_w]
    return result


def _transform(values: np.ndarray, rotation: float, scale: float, order: int) -> np.ndarray:
    rotated = ndimage.rotate(values, rotation, reshape=False, order=order,
                             mode='constant', cval=0.0, prefilter=order > 1)
    return _center_scale(rotated, scale, order)


def _unit_intensity(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = gray[mask]
    if values.size == 0:
        raise ValueError('No foreground overlap')
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        raise ValueError('Degenerate foreground intensity')
    result = np.clip((gray - low) / (high - low), 0, 1)
    return result.astype(np.float32)


def _ncc(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    a, b = left[mask].astype(np.float64), right[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _vessel_detail(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Enhance dark, vessel-scale structures while suppressing illumination."""
    fine = ndimage.gaussian_filter(gray, 1.0)
    background = ndimage.gaussian_filter(gray, 6.0)
    detail = np.maximum(background - fine, 0.0)
    return (detail * mask).astype(np.float32)


def _vessel_binary(detail: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = detail[mask]
    if values.size == 0:
        return np.zeros_like(mask)
    threshold = float(np.quantile(values, 0.90))
    return mask & (detail >= threshold) & (detail > 0)


def _resize_array(values: np.ndarray, size: int, nearest=False) -> np.ndarray:
    if values.shape == (size, size):
        return values
    if nearest:
        image = Image.fromarray(values.astype(np.uint8) * 255)
        return np.asarray(image.resize((size, size), Image.Resampling.NEAREST)) > 0
    image = Image.fromarray(np.clip(values * 255, 0, 255).astype(np.uint8))
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def confirm_pair(left: NormalizedRetina, right: NormalizedRetina,
                 config: AuditConfig) -> PairEvidence:
    # Select orientation/geometry on a small view, then score the chosen transform
    # at every frozen confirmation resolution.
    coarse_size = 128
    left_coarse = _resize_array(left.gray, coarse_size)
    left_mask = _resize_array(left.mask, coarse_size, nearest=True)
    best = None
    for orientation, candidate, candidate_mask in (
        ('normal', right.gray, right.mask),
        ('flipped', np.fliplr(right.gray), np.fliplr(right.mask)),
    ):
        coarse = _resize_array(candidate, coarse_size)
        coarse_mask = _resize_array(candidate_mask, coarse_size, nearest=True)
        for rotation in config.alignment['rotations_degrees']:
            for scale in config.alignment['scales']:
                transformed = _transform(coarse, float(rotation), float(scale), 1)
                transformed_mask = _transform(coarse_mask.astype(np.float32),
                                              float(rotation), float(scale), 0) > .5
                shift, _, _ = phase_cross_correlation(
                    left_coarse * left_mask, transformed * transformed_mask,
                    upsample_factor=4, normalization=None)
                maximum = (float(config.alignment['maximum_translation_pixels']) *
                           coarse_size / left.gray.shape[0])
                if np.any(np.abs(shift) > maximum + .5):
                    continue
                transformed = ndimage.shift(transformed, shift, order=1,
                                            mode='constant', cval=0.0)
                transformed_mask = ndimage.shift(transformed_mask.astype(np.float32),
                                                 shift, order=0, mode='constant', cval=0.0) > .5
                overlap = left_mask & transformed_mask
                if overlap.mean() < .1:
                    continue
                score = _ncc(_unit_intensity(left_coarse, left_mask),
                             _unit_intensity(transformed, transformed_mask), overlap)
                choice = (score, orientation, float(rotation), float(scale),
                          (float(shift[0]), float(shift[1])))
                if best is None or choice > best:
                    best = choice
    if best is None:
        # This is expected negative evidence for unrelated retinal images.  A
        # malformed input is rejected earlier by normalize_retina; reaching
        # this branch means both inputs are valid but cannot be aligned within
        # the frozen geometric limits, so the pair must be rejected rather
        # than aborting the complete calibration/audit run.
        return rejected_no_overlap_evidence()
    _, orientation, rotation, scale, coarse_shift = best
    right_gray = np.fliplr(right.gray) if orientation == 'flipped' else right.gray
    right_mask = np.fliplr(right.mask) if orientation == 'flipped' else right.mask
    transformed = _transform(right_gray, rotation, scale, 1)
    transformed_mask = _transform(right_mask.astype(np.float32), rotation, scale, 0) > .5
    factor = left.gray.shape[0] / coarse_size
    translation = (coarse_shift[0] * factor, coarse_shift[1] * factor)
    transformed = ndimage.shift(transformed, translation, order=1,
                                mode='constant', cval=0.0)
    transformed_mask = ndimage.shift(transformed_mask.astype(np.float32), translation,
                                     order=0, mode='constant', cval=0.0) > .5
    overlap = left.mask & transformed_mask
    union = left.mask | transformed_mask
    if not overlap.any():
        return rejected_no_overlap_evidence()
    sigma = float(config.confirmation['gaussian_sigma'])
    left_unit = _unit_intensity(ndimage.gaussian_filter(left.gray, sigma), left.mask)
    right_unit = _unit_intensity(ndimage.gaussian_filter(transformed, sigma), transformed_mask)
    ncc = _ncc(left_unit, right_unit, overlap)
    left_vessel = _vessel_detail(left_unit, left.mask)
    right_vessel = _vessel_detail(right_unit, transformed_mask)
    vessel_ncc = _ncc(left_vessel, right_vessel, overlap)
    left_binary = _vessel_binary(left_vessel, left.mask) & overlap
    right_binary = _vessel_binary(right_vessel, transformed_mask) & overlap
    vessel_dice = float(2 * np.logical_and(left_binary, right_binary).sum() /
                        max(1, left_binary.sum() + right_binary.sum()))
    nmae = float(np.mean(np.abs(left_unit[overlap] - right_unit[overlap])))
    ssim_values = []
    for size in config.confirmation['ssim_sizes']:
        a = _resize_array(left_unit * left.mask, int(size))
        b = _resize_array(right_unit * transformed_mask, int(size))
        ssim_values.append(float(structural_similarity(a, b, data_range=1.0)))
    return PairEvidence(
        ssim=min(ssim_values), ncc=ncc, nmae=nmae,
        mask_iou=float(overlap.sum() / max(1, union.sum())),
        orientation=orientation, rotation=rotation, scale=scale,
        translation=translation, vessel_ncc=vessel_ncc,
        vessel_dice=vessel_dice,
    )


def classify_pair(evidence: PairEvidence, locked_thresholds: dict) -> str:
    confirmed = locked_thresholds['confirmed']
    if (evidence.ssim >= confirmed['ssim'] and
            evidence.ncc >= confirmed['ncc'] and
            evidence.nmae <= confirmed['nmae'] and
            evidence.vessel_dice >= confirmed.get('vessel_dice', 0.0)):
        return 'confirmed'
    borderline = locked_thresholds['borderline']
    if (evidence.ssim >= borderline['ssim'] and
            evidence.ncc >= borderline['ncc'] and
            evidence.nmae <= borderline['nmae'] and
            evidence.vessel_dice >= borderline.get('vessel_dice', 0.0)):
        return 'borderline'
    return 'rejected'


def load_dino(config: AuditConfig, weights_path: Path, expected_sha256: str,
              device: str):
    import torch
    weights_path = Path(weights_path)
    actual = digest(weights_path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(f'DINOv2 weights SHA-256 mismatch: {actual}')
    import timm
    if timm.__version__ != config.dino['timm_version']:
        raise RuntimeError(f"timm version mismatch: {timm.__version__}")
    model = timm.create_model(config.dino['model'], pretrained=False, num_classes=0)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def embed_retinas(records, model, config: AuditConfig,
                  device: str):
    import torch
    import torch.nn.functional as torch_functional
    size = int(config.dino['input_size'])
    pretrained = getattr(model, 'pretrained_cfg', {}) or {}
    mean = torch.tensor(pretrained.get('mean', (0.485, 0.456, 0.406)),
                        dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(pretrained.get('std', (0.229, 0.224, 0.225)),
                       dtype=torch.float32, device=device).view(1, 3, 1, 1)
    model = model.to(device).eval()
    with torch.inference_mode():
        for key, rgb in records:
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().unsqueeze(0)
            tensor = tensor.to(device) / 255.0
            tensor = torch_functional.interpolate(tensor, (size, size), mode='bicubic',
                                                  align_corners=False, antialias=True)
            features = model((tensor - mean) / std)
            if isinstance(features, (tuple, list)):
                features = features[0]
            features = features.flatten(1)
            features = torch_functional.normalize(features, dim=1)
            yield EmbeddingRecord(key, features[0].detach().cpu().numpy().astype(np.float32))


def exact_cosine_neighbors(embeddings: np.ndarray, keys: list[str],
                           top_k: int, chunk_size: int) -> list[dict]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(keys) != len(embeddings):
        raise ValueError('Embedding matrix and keys do not align')
    if top_k < 1 or chunk_size < 1:
        raise ValueError('top_k and chunk_size must be positive')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Zero-norm embedding')
    embeddings = embeddings / norms
    result = []
    for start in range(0, len(embeddings), chunk_size):
        scores = embeddings[start:start + chunk_size] @ embeddings.T
        for local, row in enumerate(scores):
            index = start + local
            order = sorted((candidate for candidate in range(len(keys)) if candidate != index),
                           key=lambda candidate: (-float(row[candidate]), keys[candidate]))[:top_k]
            result.append({
                'key': keys[index],
                'neighbors': [{'key': keys[candidate], 'cosine': float(row[candidate])}
                              for candidate in order],
            })
    return result


def dino_candidate_record(left: str, right: str, similarity: float) -> dict:
    left, right = sorted((left, right))
    return {
        'pair_id': identity([left, right]),
        'left': left,
        'right': right,
        'cosine': float(similarity),
        'candidate_source': 'dinov2',
        'classification': 'unverified',
    }


def _optional_digest(path: Path | None) -> str | None:
    return digest(path) if path is not None else None


def build_audit_identity(parent_manifest: Path, source_root: Path,
                         config: AuditConfig, code_paths: list[Path],
                         dependency_lock: Path | None = None,
                         weights: Path | None = None) -> dict:
    parent_manifest = Path(parent_manifest)
    source_root = Path(source_root)
    if not parent_manifest.is_file():
        raise FileNotFoundError(parent_manifest)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    code = {Path(path).name: digest(path) for path in sorted(map(Path, code_paths))}
    components = {
        'schema_version': config.schema_version,
        'parent_manifest_sha256': digest(parent_manifest),
        'source_identity': identity(parent_manifest.read_text(encoding='utf-8')),
        'config': config.values,
        'code_sha256': code,
        'dependency_lock_sha256': _optional_digest(dependency_lock),
        'weights_sha256': _optional_digest(weights),
    }
    return {**components, 'audit_id': identity(components)}


def _line_count(path: Path) -> int:
    with path.open('r', encoding='utf-8') as handle:
        return sum(1 for line in handle if line.strip())


class AuditState:
    def __init__(self, output: Path):
        self.output = Path(output)
        self.identity_path = self.output / 'audit_identity.json'
        self.incomplete_path = self.output / 'INCOMPLETE'
        self.complete_path = self.output / 'COMPLETE.json'

    def start(self, audit_identity: dict) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        if self.identity_path.exists() and read_json(self.identity_path) != audit_identity:
            raise RuntimeError('Audit output belongs to a different identity')
        if self.complete_path.exists():
            complete = read_json(self.complete_path)
            if complete.get('audit_id') != audit_identity.get('audit_id'):
                raise RuntimeError('Completed audit identity mismatch')
            return
        save_json(self.identity_path, audit_identity)
        self.incomplete_path.write_text(audit_identity['audit_id'] + '\n', encoding='utf-8')

    def complete(self, summary: dict) -> None:
        if not self.incomplete_path.exists():
            raise RuntimeError('Audit was not started or is already complete')
        audit_identity = read_json(self.identity_path)
        for relative, expected in summary.get('expected_artifacts', {}).items():
            artifact = self.output / relative
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            actual = _line_count(artifact) if artifact.suffix in {'.jsonl', '.csv'} else 1
            if actual != expected:
                raise RuntimeError(f'Artifact count mismatch for {relative}: {actual} != {expected}')
        record = {**summary, 'audit_id': audit_identity['audit_id']}
        save_json(self.complete_path, record)
        os.replace(self.incomplete_path, self.output / 'COMPLETED')

    def require_complete(self) -> dict:
        if self.incomplete_path.exists() or not self.complete_path.exists():
            raise RuntimeError('Audit is incomplete')
        record = read_json(self.complete_path)
        audit_identity = read_json(self.identity_path)
        if record.get('audit_id') != audit_identity.get('audit_id'):
            raise RuntimeError('Completion identity mismatch')
        return record
