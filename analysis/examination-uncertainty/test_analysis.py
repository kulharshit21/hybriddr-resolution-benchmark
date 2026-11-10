import unittest
import numpy as np
from analyze import metrics
class Tests(unittest.TestCase):
    def test_perfect_and_grade1_absent(self):
        x=metrics(np.eye(5)*10)
        np.testing.assert_allclose(x,[1,1,1,1])
        cm=np.eye(5)*10;cm[1]=0
        self.assertTrue(np.isnan(metrics(cm)[2]))
    def test_metrics_against_sklearn(self):
        from sklearn.metrics import cohen_kappa_score,f1_score,recall_score
        rng=np.random.default_rng(11);y=rng.integers(0,5,100);pred=rng.integers(0,5,100)
        cm=np.zeros((5,5));np.add.at(cm,(y,pred),1)
        expected=[cohen_kappa_score(y,pred,weights='quadratic',labels=range(5)),f1_score(y,pred,average='macro'),recall_score(y,pred,average=None)[1],f1_score(y,pred,average=None)[1]]
        np.testing.assert_allclose(metrics(cm),expected,atol=1e-12)
    def test_cluster_weights_keep_both_eyes(self):
        cms=np.zeros((2,5,5));cms[0,0,0]=1;cms[0,1,1]=1;cms[1,2,2]=1
        sampled=(np.array([[2,0]])@cms.reshape(2,25)).reshape(5,5)
        self.assertEqual(sampled[0,0],2);self.assertEqual(sampled[1,1],2);self.assertEqual(sampled[2,2],0)
if __name__=='__main__':unittest.main(verbosity=2)
