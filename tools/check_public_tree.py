"""Reject data, weights, result artifacts and access material before publication."""
from pathlib import Path
import ast,json,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'.py','.json','.md','.txt','.yml','.yaml'}
SENSITIVE=[r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY',r'\bgh[pousr]_[A-Za-z0-9]{25,}',r'\bgithub_pat_[A-Za-z0-9_]{25,}',r'\bhf_[A-Za-z0-9]{25,}',r'\bAKIA[A-Z0-9]{16}\b',r'/home/faculty/',r'C:[/\\]Users[/\\]',r'gdrive:HybridDR']
def check():
 names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
 files=[ROOT/n for n in names if n]
 assert files,'No tracked files to inspect'
 for p in files:
  rel=p.relative_to(ROOT)
  assert p.name=='.gitignore' or p.suffix in ALLOWED,('Disallowed file type',str(rel))
  assert not any(x.lower() in {'dataset','datasets','weights','checkpoints','runs','results','outputs','cache'} for x in rel.parts),str(rel)
  assert p.stat().st_size<1024*1024,('Unexpected large file',str(rel))
  s=p.read_text(encoding='utf8')
  # This checker contains literal detection patterns, not credentials.
  if p!=Path(__file__).resolve():
   assert not any(re.search(pattern,s) for pattern in SENSITIVE),('Access material or private path',str(rel))
  if p.suffix=='.py':ast.parse(s,filename=str(rel))
  if p.suffix=='.json':json.loads(s)
 print(f'PASS: {len(files)} tracked code/config/documentation files; no dataset, result, image, weight or credential file.')
if __name__=='__main__':check()
