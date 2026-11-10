"""Synthetic CPU integration test; never a research result."""
import csv,json,tempfile,subprocess,sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
def main():
    base=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);rows=[]
        for split in ('train','val','calibration','internal','external'):
            for source in ('APTOS','EyePACS'):
                for label in range(5):
                    name=f'{split}_{source}_{label}.png'
                    Image.fromarray(np.random.default_rng(label).integers(0,256,(64,64,3),dtype=np.uint8)).save(p/name)
                    rows.append(dict(path=name,cache=name,source=source,label=label,split=split,patient=name))
        with (p/'manifest.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        for size in (28,56):
            c=json.loads((base/'config.json').read_text())
            c.update(tiny=True,pretrained=False,image_size=size,epochs=1,workers_per_gpu=0,threads_per_gpu=2,
                auto_batch=False,batch_size=5,effective_batch_size=5,fusion_width=16,fusion_heads=2)
            job=dict(config=c,recipe=dict(name='smoke',architecture='spatial',spatial_bias=True,loss='ce'),seed=11,cpu_ids=[])
            j=p/f'job{size}.json';j.write_text(json.dumps(job));out=p/f'out{size}'
            subprocess.run([sys.executable,str(base/'baseline/train.py'),'--job',str(j),'--prepared',str(p),'--out',str(out),'--device','cpu'],check=True)
            if not json.loads((out/'complete.json').read_text())['smoke_only']:raise RuntimeError('Smoke identity missing')
            subprocess.run([sys.executable,str(base/'baseline/evaluate.py'),'--job',str(j),'--prepared',str(p),'--out',str(out),'--device','cpu'],check=True)
            if not (out/'evaluation.json').exists():raise RuntimeError('Evaluation missing')
        for resolution in (224,448):
            for seed in (11,23,42,67,101):
                shutil.copytree(out,p/'reports'/'runs'/f'aligned_ce_r{resolution}_seed{seed}')
        subprocess.run([sys.executable,str(base/'report.py'),'--out',str(p/'reports')],check=True)
        print('CPU_SMOKE_PASS: synthetic training, calibration, evaluation, paired reports and figures',flush=True)
if __name__=='__main__':main()
