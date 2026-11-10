"""Test the real training CLI, manifest binding, completion and restart on tiny data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import pandas as pd
from PIL import Image
from common import digest, read_json, save_json


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prior = root / 'prior'
        prior.mkdir()
        (prior / 'FINAL_CAMPAIGN_COMPLETE').touch()
        prepared = root / 'prepared'
        (prepared / 'cache').mkdir(parents=True)
        rows = []
        torch.manual_seed(4)
        for split in ['train', 'val']:
            for grade in range(5):
                filename = f'{split}-{grade}.png'
                image = torch.randint(0, 256, (56, 56, 3), dtype=torch.uint8).numpy()
                Image.fromarray(image).save(prepared / 'cache' / filename)
                rows.append(dict(path=filename, cache='cache/' + filename,
                                 patient=filename, source='EyePACS', label=grade, split=split))
        pd.DataFrame(rows).to_csv(prepared / 'manifest.csv', index=False)
        save_json(prepared / 'prepared.json', dict(manifest_sha256=digest(prepared / 'manifest.csv')))
        command = [sys.executable, str(here / 'run.py'), 'train', '--config',
                   str(here / 'config_screen_ce.json'), '--out', str(root / 'smoke'),
                   '--prior-campaign', str(prior), '--prepared-source', str(prepared),
                   '--smoke', '--cpu']
        subprocess.run(command, check=True)
        markers = sorted((root / 'smoke/runs').glob('*/complete.json'))
        assert len(markers) == 2
        snapshots = {path: path.read_bytes() for path in markers}
        subprocess.run(command, check=True)
        assert all(path.read_bytes() == before for path, before in snapshots.items())
        for marker in markers:
            assert read_json(marker)['smoke_only']
            assert len(read_json(marker.parent / 'history.json')) == 2
        print('PASS: training CLI, scaled fusion/warmup, checkpoint completion, restart reuse; tiny CPU fixture only')


if __name__ == '__main__':
    main()
