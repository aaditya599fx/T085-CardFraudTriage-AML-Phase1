"""Behavioral checks for leakage, score orientation, ties and operational budgets."""
import unittest
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from src.data import chronological_split, FEATURES
from src.features import prepare, engineering
from src.metrics import fpr_threshold, at_threshold, at_capacity, ranking
from src.gmm_model import score

class ProtocolTests(unittest.TestCase):
    def frame(self):
        rng=np.random.default_rng(26)
        df=pd.DataFrame(rng.normal(size=(100,29)),columns=FEATURES)
        df['Amount']=np.arange(100)
        df['Time']=np.repeat(np.arange(50),2)
        df['Class']=np.tile([0,0,0,1,0],20)
        return df

    def test_time_ties_stay_together(self):
        p=chronological_split(self.frame())
        self.assertLess(p['train'].Time.max(),p['validation'].Time.min())
        self.assertLess(p['validation'].Time.max(),p['test'].Time.min())
        self.assertEqual(sum(map(len,p.values())),100)
        self.assertFalse(set(p['train'].index)&set(p['test'].index))

    def test_scaler_sees_only_legitimate_training(self):
        p=chronological_split(self.frame())
        normal=p['train'].Class==0
        p['train'].loc[~normal,'V1']=1e9
        p['test']['V1']=1e12
        train,x,scaler=prepare(p)
        np.testing.assert_allclose(scaler.mean_,engineering(p['train'])[normal].mean(axis=0))
        np.testing.assert_allclose(train.mean(axis=0),np.zeros(29),atol=1e-12)
        self.assertEqual(len(train),int(normal.sum()))

    def test_threshold_ties_cannot_exceed_budget(self):
        y=np.array([0]*10+[1,1]); s=np.array([1,2,2,2,3,3,3,4,4,5,7,8])
        for budget in [0,.1,.2,.3]:
            t=fpr_threshold(y,s,budget)
            self.assertLessEqual(at_threshold(y,s,t)['fpr'],budget)

    def test_capacity_ties_use_stable_order(self):
        y=np.array([0,1,1,0]); s=np.ones(4)
        self.assertEqual(at_capacity(y,s,.5)['precision_at_k'],.5)
        self.assertEqual(at_capacity(y,s,.5)['k'],2)

    def test_perfect_ranking_and_orientation(self):
        self.assertEqual(ranking([0,0,1,1],[0,1,2,3])['average_precision'],1.)
        rng=np.random.default_rng(26)
        model=GaussianMixture(1,random_state=26).fit(rng.normal(size=(100,2)))
        s=score(model,np.array([[0,0],[30,30]]))
        self.assertGreater(s[1],s[0])

    def test_bad_scores_rejected(self):
        with self.assertRaises(ValueError): ranking([0,1],[0,np.nan])
        with self.assertRaises(ValueError): ranking([0,0],[1,2])

if __name__=='__main__': unittest.main()
