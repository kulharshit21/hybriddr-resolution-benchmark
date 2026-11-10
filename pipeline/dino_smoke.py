"""Single-image DINOv2 GPU preflight; not a research result."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from common import save_json
from near_duplicates import AuditConfig, embed_retinas, load_dino, normalize_retina


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--weights-sha256', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    config = AuditConfig.from_json(args.config)
    manifest = pd.read_csv(args.prepared / 'manifest.csv')
    if manifest.empty:
        raise RuntimeError('DINO smoke requires a non-empty manifest')
    relative = str(manifest.sort_values('path').iloc[0].path)
    with Image.open(args.data / relative) as image:
        rgb = normalize_retina(image, config, relative).rgb

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise RuntimeError('DINO smoke must measure a CUDA device')
    torch.cuda.set_device(device)
    free_before, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = load_dino(config, args.weights, args.weights_sha256, str(device))
    records = list(embed_retinas([(relative, rgb)], model, config, str(device)))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if len(records) != 1:
        raise RuntimeError(f'Expected one embedding, received {len(records)}')
    vector = records[0].vector
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not math.isclose(norm, 1.0, abs_tol=1e-5):
        raise RuntimeError(f'Invalid DINO embedding norm: {norm}')
    result = {
        'purpose': 'single-image DINOv2 hardware and weight compatibility check',
        'research_result': False,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device),
        'torch': torch.__version__,
        'source_path': relative,
        'embedding_dimensions': int(vector.size),
        'embedding_norm': norm,
        'elapsed_seconds_including_model_load': elapsed,
        'free_vram_before_bytes': int(free_before),
        'total_vram_bytes': int(total),
        'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
        'weights_sha256': args.weights_sha256.lower(),
        'verdict': 'PASS',
    }
    save_json(args.out, result)
    print(result, flush=True)
    print('VERDICT: PASS', flush=True)


if __name__ == '__main__':
    main()
