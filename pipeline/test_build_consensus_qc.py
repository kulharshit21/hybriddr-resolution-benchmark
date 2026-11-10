import unittest
from build_consensus_qc import stratified_sample
class Tests(unittest.TestCase):
 def test_sample_is_deterministic_unique_and_exact(self):
  rows=[{'pair_id':str(i),'sources':['A' if i%2 else 'B'],'splits':['x'],'selected_classification':'borderline'} for i in range(100)]
  a=stratified_sample(rows,25,7);self.assertEqual(a,stratified_sample(rows,25,7));self.assertEqual(len(a),25);self.assertEqual(len(set(a)),25)
if __name__=='__main__':unittest.main()
