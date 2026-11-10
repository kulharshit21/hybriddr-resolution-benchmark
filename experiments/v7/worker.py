import subprocess,sys
from pathlib import Path
base=Path(__file__).resolve().parent/'baseline'
for stage in ('train','evaluate'):
    subprocess.run([sys.executable,'-u',str(base/(stage+'.py')),*sys.argv[1:]],check=True)
