"""Original-image inventory, decode audit, duplicate protection and frozen splits."""
import csv
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from common import digest, identity, read_json, save_json


def inventory(root):
    root = Path(root)
    rows = []
    def add(path, source, label, patient, role):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(dict(path=path.relative_to(root).as_posix(), source=source,
                         label=int(label), patient=patient, role=role))
    for row in pd.read_csv(root / 'EyePACS/trainLabels.csv').itertuples():
        add(root / 'EyePACS/train' / (row.image + '.jpeg'), 'EyePACS', row.level,
            'EyePACS:' + row.image.rsplit('_', 1)[0], 'development')
    for row in pd.read_csv(root / 'APTOS2019/train.csv').itertuples():
        add(root / 'APTOS2019/train_images' / (row.id_code + '.png'), 'APTOS', row.diagnosis,
            'APTOS:' + row.id_code, 'development')
    paths = {}
    for p in (root / 'MESSIDOR2').rglob('*'):
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            if p.stem in paths:
                raise ValueError('Ambiguous MESSIDOR identifier: ' + p.stem)
            paths[p.stem] = p
    labels = pd.read_csv(root / 'MESSIDOR2/messidor_data.csv')
    for row in labels.itertuples():
        if pd.isna(row.adjudicated_dr_grade) or row.adjudicated_gradable != 1:
            continue
        key = Path(row.image_id).stem
        if key not in paths:
            raise ValueError('Missing MESSIDOR image: ' + key)
        add(paths[key], 'MESSIDOR2', row.adjudicated_dr_grade, 'MESSIDOR2:' + key, 'external')
    # IDRiD is packaged but excluded from the primary source-only protocol.
    # EyePACS/APTOS public competition test labels are unavailable: never score them.
    return rows


def audit_one(task):
    root, out, row, size = task
    p = Path(root) / row['path']
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.load()  # fail on damaged originals; never silently replace with black images
        row['width'], row['height'] = im.size
        pixel = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        thumb = np.asarray(im.convert('L').resize((9, 8)))
        bits = np.packbits(thumb[:, 1:] > thumb[:, :-1]).tobytes().hex()
        # Deterministic resized RGB PNG cache; no random augmentation is cached.
        cache = Path(out) / 'cache' / (pixel + '.png')
        if not cache.exists():
            resized = im.resize((size, size), Image.Resampling.LANCZOS)
            tmp = cache.with_name(cache.stem + f'.{os.getpid()}.tmp')
            resized.save(tmp, format='PNG')
            os.replace(tmp, cache)
    row.update(sha256=digest(p), pixel_sha256=pixel, dhash=bits,
               cache='cache/' + pixel + '.png')
    return row


def assign_splits(frame, seed=2026):
    frame = frame.copy()
    frame['split'] = 'external'
    for source in ('EyePACS', 'APTOS'):
        indices = frame.index[(frame.source == source) & (frame.role == 'development')]
        subset = frame.loc[indices]
        if subset.empty:
            continue
        splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        for fold, (_, ix) in enumerate(splitter.split(subset, subset.label, subset.patient)):
            frame.loc[indices[ix], 'split'] = ('train' if fold < 7 else
                                             {7: 'val', 8: 'calibration', 9: 'internal'}[fold])
    return frame


def resolve_exact_duplicates(frame):
    """Select one consistent-label image per pixel hash and quarantine conflicts."""
    keep = []
    exclusions = []
    conflicts = []
    for pixel_sha256, group in frame.groupby('pixel_sha256', sort=False):
        if group.label.nunique() != 1:
            conflicts.append({
                'pixel_sha256': pixel_sha256,
                'policy': 'exclude_all_from_development_due_to_conflicting_exact_duplicate_labels',
                'members': [
                    {
                        'path': row.path,
                        'label': int(row.label),
                        'source': row.source,
                        'role': row.role,
                    }
                    for row in group.itertuples()
                ],
            })
            for row in group.itertuples():
                exclusions.append({
                    'path': row.path,
                    'reason': 'conflicting_exact_duplicate_label',
                    'pixel_sha256': pixel_sha256,
                    'observed_label': int(row.label),
                })
            continue
        # External evaluation takes priority so its pixels cannot enter development.
        chosen = group.assign(_external=(group.role == 'external').astype(int)).sort_values(
            ['_external', 'path'], ascending=[False, True]
        ).iloc[0]
        keep.append(chosen.name)
        for idx, row in group.iterrows():
            if idx != chosen.name:
                exclusions.append({
                    'path': row.path,
                    'reason': 'decoded_duplicate',
                    'retained': chosen.path,
                })
    return keep, exclusions, conflicts


def prepare(root, out, config, limit=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = inventory(root)
    for row in rows:
        info = (Path(root) / row['path']).stat()
        row['original_bytes'] = info.st_size
        row['original_mtime_ns'] = info.st_mtime_ns
    if limit:
        # Smoke fixture: stratified source/class subset; never a publication manifest.
        groups = {}
        for row in rows:
            group = (row['source'], row['label'])
            groups.setdefault(group, []).append(row)
        rows = [r for group in groups.values() for r in group[:limit]]
    signature = identity({'rows': rows, 'cache_size': config['cache_size'], 'smoke_limit': limit})
    marker = out / 'prepared.json'
    if marker.exists():
        previous = read_json(marker)
        if previous['input_signature'] != signature:
            raise RuntimeError('Manifest/config changed: use a new output directory')
        if digest(out / 'manifest.csv') != previous['manifest_sha256']:
            raise RuntimeError('Frozen manifest changed')
        return previous
    (out / 'cache').mkdir(exist_ok=True)
    tasks = ((str(root), str(out), r, config['cache_size']) for r in rows)
    workers = min(config['audit_workers'], max(1, int((os.cpu_count() or 1) * config['cpu_fraction'])))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        checked = list(pool.map(audit_one, tasks, chunksize=8))
    df = pd.DataFrame(checked)
    if not df.label.between(0, 4).all():
        raise ValueError('Labels must be five-grade integers')
    exclusions = []
    # Duplicate identity links patient groups before dropping duplicate images.
    parent = {p: p for p in df.patient}
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    for _, group in df.groupby('pixel_sha256', sort=False):
        patients = group.patient.tolist()
        for p in patients[1:]:
            parent[find(p)] = find(patients[0])
    df['original_patient'] = df.patient
    df['patient'] = df.patient.map(find)
    # Never invent a target for identical pixels carrying contradictory grades.
    # Quarantine every conflicting member and retain a machine-readable audit trail.
    keep, duplicate_exclusions, label_conflicts = resolve_exact_duplicates(df)
    exclusions.extend(duplicate_exclusions)
    df = df.loc[keep].reset_index(drop=True)
    # Remove both eyes of a development patient if any eye was excluded for external overlap.
    external_hash = set(df.loc[df.role == 'external', 'pixel_sha256'])
    contaminated = {find(r['patient']) for r in checked if r['role'] == 'development' and r['pixel_sha256'] in external_hash}
    remove = df.patient.isin(contaminated) & (df.role == 'development')
    for row in df[remove].itertuples():
        exclusions.append({'path': row.path, 'reason': 'patient_external_overlap'})
    df = df[~remove].reset_index(drop=True)
    df = assign_splits(df)
    near = []
    for dhash, group in df.groupby('dhash'):
        if len(group) > 1 and group.split.nunique() > 1:
            near.append({'dhash': dhash, 'paths': group.path.tolist(), 'splits': group.split.tolist()})
    df.to_csv(out / 'manifest.csv', index=False)
    save_json(out / 'exclusions.json', exclusions)
    save_json(out / 'label_conflicts.json', label_conflicts)
    save_json(out / 'near_duplicate_candidates.json', near)
    result = {'input_signature': signature, 'manifest_sha256': digest(out / 'manifest.csv'),
              'smoke_only': bool(limit), 'counts': df.groupby(['source', 'split']).size().to_dict(),
              'near_duplicate_groups': len(near),
              'conflicting_exact_duplicate_groups': len(label_conflicts),
              'patient_metadata_missing': ['APTOS', 'MESSIDOR2'],
              'limitations': ['dHash is candidate generation, not complete near-duplicate detection',
                             'MESSIDOR mirror/label provenance requires review', 'DDR images unavailable']}
    result['counts'] = {str(k): int(v) for k, v in result['counts'].items()}
    save_json(marker, result)
    return result


class RetinalDataset(Dataset):
    def __init__(self, frame, prepared, size, training=False):
        self.rows = frame.reset_index(drop=True)
        self.prepared = Path(prepared)
        self.size = size
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.prepared / row.cache) as image:
            image = image.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
            if self.training:
                if torch.rand(()) < .5:
                    image = TF.hflip(image)
                image = TF.rotate(image, float(torch.empty(()).uniform_(-15, 15)))
                image = TF.adjust_brightness(image, float(torch.empty(()).uniform_(.9, 1.1)))
            tensor = TF.to_tensor(image)
        # Both branches receive precisely the same geometric input.
        return tensor, int(row.label), index
