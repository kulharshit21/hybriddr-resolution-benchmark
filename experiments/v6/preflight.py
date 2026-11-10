import argparse,json
from pathlib import Path
import psutil
from launch import digest
ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--package',type=Path,required=True);a=ap.parse_args()
prepared=Path((a.project/'current-quality-audit-root.txt').read_text().strip())/'prepared-quarantine-all-candidates-v1'
assert digest(prepared/'manifest.csv')=='7ecdade0df288689e6913cf0e44a23b27daed84a12b340193a4e331629e59de3','Manifest mismatch'
assert len(psutil.Process().cpu_affinity())>=60,'Need at least 60 logical CPUs'
out=a.project/'runs/v6-resolution-final-01'
identity={str(x.relative_to(a.package)):digest(x) for x in a.package.rglob('*') if x.is_file() and x.suffix in ('.py','.json')}
if (out/'IDENTITY.json').exists():
    assert json.loads((out/'IDENTITY.json').read_text())==identity,'Existing run uses different code; preserve it'
if (out/'V6_COMPLETE.json').exists():raise RuntimeError('V6 already complete; do not stop MAMAGuard')
print('MANIFEST, CPU AND RESUME IDENTITY: PASS')
