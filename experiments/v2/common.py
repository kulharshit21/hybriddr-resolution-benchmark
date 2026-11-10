"""Portable paths, atomic artifacts and bounded resource policy."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import psutil


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resources(config):
    import torch
    proc = psutil.Process()
    try:
        available = proc.cpu_affinity()
    except (AttributeError, NotImplementedError):
        available = list(range(os.cpu_count() or 1))
    budget = max(1, int(len(available) * config['cpu_fraction']))
    return {'cpu_ids': available[:budget], 'cpu_budget': budget,
            'ram_budget_bytes': int(psutil.virtual_memory().total * config['ram_fraction']),
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def bind_cpu(ids, threads):
    import torch
    try:
        psutil.Process().cpu_affinity(ids)
    except (AttributeError, NotImplementedError):
        pass
    torch.set_num_threads(max(1, threads))


def atomic_checkpoint(path, state):
    import torch
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)
