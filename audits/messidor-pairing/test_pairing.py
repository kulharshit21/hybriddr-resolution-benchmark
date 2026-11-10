import unittest
from pairing import validate,join
class Tests(unittest.TestCase):
    def setUp(self):
        self.pairs=[[f'a{i}',f'b{i}'] for i in range(874)]
        self.known={v for p in self.pairs for v in p}
        self.raw=('left;right\r\r\n'+'\r\r\n'.join(a+'.png;'+b+'.JPG' for a,b in self.pairs)).encode()
    def test_full_map_and_windows_line_endings(self):self.assertEqual(validate(self.raw,self.known),self.pairs)
    def test_duplicate_missing_unknown_fail(self):
        for raw in (self.raw+b'\na0.png;b0.JPG',self.raw.replace(b'a0.png',b'unknown.png'),self.raw.split(b'\r\r\n',1)[0]):
            with self.assertRaises(ValueError):validate(raw,self.known)
    def test_join_and_no_split_mutation(self):
        rows=[dict(path='MESSIDOR2/a0.png',source='MESSIDOR2',label='1',split='external'),dict(path='MESSIDOR2/b0.JPG',source='MESSIDOR2',label='2',split='external')]
        x=join(self.pairs,rows);self.assertEqual(x[0]['exam_cluster'],x[1]['exam_cluster'])
        rows[0]['split']='train'
        with self.assertRaises(ValueError):join(self.pairs,rows)
if __name__=='__main__':unittest.main(verbosity=2)
