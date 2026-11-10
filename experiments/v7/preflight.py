import argparse,json
from pathlib import Path
import psutil
from launch import digest
ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--package',type=Path,required=True);a=ap.parse_args()
prepared=Path((a.project/'current-quality-audit-root.txt').read_text().strip())/'prepared-quarantine-all-candidates-v1'
assert digest(prepared/'manifest.csv')=='7ecdade0df288689e6913cf0e44a23b27daed84a12b340193a4e331629e59de3','Manifest mismatch'
assert len(psutil.Process().cpu_affinity())>=60,'Need at least 60 logical CPUs'
out=a.project/'runs/v7-controls-448-01'
reference=a.project/'runs/v6-resolution-final-01'
if not (reference/'V6_COMPLETE.json').exists():raise RuntimeError('V6 must complete first')
completion=json.loads((reference/'V6_COMPLETE.json').read_text())
if digest(reference/'RESULTS.json')!=completion['results_sha256']:raise RuntimeError('V6 results hash mismatch')
record=json.loads((reference/'IDENTITY.json').read_text())
for name in ('base_models.py','models.py','train.py','data.py','metrics.py','evaluate.py','common.py'):
    if digest(a.package/'baseline'/name)!=record['baseline/'+name]:raise RuntimeError('V6 baseline source mismatch: '+name)
for seed in (11,23,42,67,101):
    job=json.loads((reference/f'aligned_ce_r448_seed{seed}.json').read_text())
    cfg=json.loads((a.package/'config.json').read_text())
    for key in ('image_size','epochs','vit','cnn','fusion_width','fusion_heads','pretrained','effective_batch_size','learning_rate','head_lr_multiplier','weight_decay','label_smoothing','gradient_checkpointing','workers_per_gpu','threads_per_gpu'):
        if cfg[key]!=job['config'][key]:raise RuntimeError('V6 training setting mismatch: '+key)
identity={str(x.relative_to(a.package)):digest(x) for x in a.package.rglob('*') if x.is_file() and x.suffix in ('.py','.json')}
if (out/'IDENTITY.json').exists():
    assert json.loads((out/'IDENTITY.json').read_text())==identity,'Existing run uses different code; preserve it'
if (out/'V7_COMPLETE.json').exists():raise RuntimeError('V7 already complete; do not stop MAMAGuard')
print('MANIFEST, CPU AND RESUME IDENTITY: PASS')
