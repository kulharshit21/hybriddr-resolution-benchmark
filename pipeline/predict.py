"""Inference on a directory using an explicitly selected research checkpoint."""
import argparse
import csv
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torchvision.transforms.functional import to_tensor
from common import read_json
from models import HybridDR, probabilities


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']
    model = HybridDR(cfg, state['recipe'], initialize=False).to(args.device).eval()
    model.load_state_dict(state['model'], strict=True)
    temperature = read_json(args.calibration)['temperature'] if args.calibration else 1.
    paths = sorted(p for p in args.images.rglob('*') if p.suffix.lower() in ('.png', '.jpeg', '.jpg', '.tif', '.tiff'))
    if not paths:
        raise ValueError('No input images')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as stream, torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(['path', 'grade', 'p0', 'p1', 'p2', 'p3', 'p4'])
        for path in paths:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im = im.resize((cfg['cache_size'], cfg['cache_size']), Image.Resampling.LANCZOS)
                im = im.resize((cfg['image_size'], cfg['image_size']), Image.Resampling.BILINEAR)
                x = to_tensor(im).unsqueeze(0).to(args.device)
            probs = probabilities(model(x), state['recipe']['loss'] == 'ordinal', temperature)[0].cpu().tolist()
            writer.writerow([str(path), max(range(5), key=lambda k: probs[k]), *probs])


if __name__ == '__main__':
    main()
