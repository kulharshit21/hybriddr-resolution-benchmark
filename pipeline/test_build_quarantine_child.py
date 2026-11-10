import json,tempfile,unittest
from pathlib import Path
import pandas as pd
from build_quarantine_child import build
class Tests(unittest.TestCase):
 def test_removes_every_endpoint_and_preserves_parent(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);p=d/'parent';p.mkdir();(p/'cache').mkdir()
   rows=[]
   for i in range(3):
    c=f'cache/{i}.bin';(p/c).write_bytes(bytes([i]));rows.append({'path':f'x/{i}.jpg','cache':c,'source':'X','split':'train','label':i,'patient':str(i),'role':'development'})
   pd.DataFrame(rows).to_csv(p/'manifest.csv',index=False);(p/'exclusions.json').write_text('[]')
   pairs=d/'pairs.jsonl';pairs.write_text(json.dumps({'pair_id':'a','left':'x/0.jpg','right':'x/1.jpg'})+'\n')
   s=build(p,pairs,d/'child');self.assertEqual(s['retained_images'],1);self.assertEqual(pd.read_csv(d/'child/manifest.csv').path.tolist(),['x/2.jpg']);self.assertTrue((p/'cache/0.bin').exists())
   self.assertEqual(s['manifest_sha256'],s['child_manifest_sha256']);self.assertEqual(len(s['audit_id']),64)
   self.assertEqual((d/'child/QUARANTINE_COMPLETE').read_text().strip(),s['manifest_sha256'])
if __name__=='__main__':unittest.main()
