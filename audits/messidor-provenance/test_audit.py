import csv,json,tempfile,unittest
from pathlib import Path
from audit import audit

class Tests(unittest.TestCase):
    def fixture(self,p):
        data=p/'data';data.mkdir();parent=p/'parent';parent.mkdir();child=p/'child';child.mkdir()
        (data/'messidor_data.csv').write_text('image_id,adjudicated_dr_grade,adjudicated_gradable\na.png,0,1\nb.png,1,1\nc.png,,0\n')
        for k in 'abc':(data/(k+'.png')).touch()
        header='path,source,label,split,patient\n'
        a='MESSIDOR2/a.png,MESSIDOR2,0,external,MESSIDOR2:a\n';b='MESSIDOR2/b.png,MESSIDOR2,1,external,MESSIDOR2:b\n'
        (parent/'manifest.csv').write_text(header+a+b);(child/'manifest.csv').write_text(header+a)
        (child/'exclusions.json').write_text(json.dumps([{'path':'MESSIDOR2/b.png','reason':'conservative_candidate_endpoint_quarantine'}]))
        return data,parent,child
    def test_reasons_and_no_false_provenance_claim(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=audit(*self.fixture(p),p/'out')
            self.assertEqual(r['counts'],{'retained':1,'candidate_quarantine':1,'ungradable_or_missing_grade':1})
            self.assertFalse(r['provenance_verified'])
    def test_label_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);data,parent,child=self.fixture(p)
            f=child/'manifest.csv';f.write_text(f.read_text().replace(',0,external',',2,external'))
            with self.assertRaisesRegex(ValueError,'Label mismatch'):audit(data,parent,child,p/'out')

if __name__=='__main__':unittest.main(verbosity=2)
