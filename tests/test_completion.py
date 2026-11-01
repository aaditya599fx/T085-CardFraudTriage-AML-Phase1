"""Additional checks for frozen memberships, reusable preprocessing and new model families."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import FEATURES, chronological_split, save_split_indices, load_split_indices
from src.features import preprocess
from src.gmm_model import fit_candidate, covariance_diagnostics, fit_grid
from src.metrics import evaluate_scores, at_review_count, load_thresholds, fpr_threshold, at_threshold


def frame():
    rng=np.random.default_rng(26)
    df=pd.DataFrame(rng.normal(size=(120,29)),columns=FEATURES)
    df['Amount']=rng.uniform(0,200,120)
    df['Time']=np.repeat(np.arange(60),2)
    df['Class']=np.tile([0,0,0,1,0,0],20)
    return df


class CompletionTests(unittest.TestCase):
    def test_frozen_split_roundtrip_and_checksum(self):
        df=frame();parts=chronological_split(df)
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'data.csv';csv.write_text(df.to_csv(index=False))
            saved=Path(tmp)/'indices.npz'
            save_split_indices(parts,saved,csv,[.6,.2,.2])
            loaded=load_split_indices(df,saved,csv,[.6,.2,.2])
            for name in parts:np.testing.assert_array_equal(parts[name].index,loaded[name].index)
            csv.write_text(csv.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'another dataset'):
                load_split_indices(df,saved,csv,[.6,.2,.2])

    def test_frozen_split_rejects_tampering(self):
        df=frame();parts=chronological_split(df)
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'data.csv';df.to_csv(csv,index=False)
            saved=Path(tmp)/'indices.npz'
            save_split_indices(parts,saved,csv,[.6,.2,.2])
            with np.load(saved) as z: arrays={k:z[k].copy() for k in z.files}
            arrays['train_legitimate']=arrays['train_legitimate'][::-1]
            np.savez_compressed(saved,**arrays)
            with self.assertRaisesRegex(ValueError,'legitimate training'):
                load_split_indices(df,saved,csv,[.6,.2,.2])

    def test_pipeline_reuse_preserves_fit_statistics(self):
        df=frame();normal=df[df.Class==0]
        xt,pipeline=preprocess(normal,pca_dimensions=5)
        means=pipeline.named_steps['scaler'].mean_.copy()
        future=df.copy();future['V1']=1e9
        xv,same=preprocess(future,pipeline=pipeline)
        self.assertIs(same,pipeline);self.assertEqual(xv.shape,(120,5))
        np.testing.assert_array_equal(means,pipeline.named_steps['scaler'].mean_)
        self.assertEqual(xt.shape,(100,5))
        with self.assertRaises(ValueError):preprocess(df)

    def test_all_covariance_types_have_diagnostics(self):
        x=np.random.default_rng(26).normal(size=(80,3))
        with threadpool_limits(limits=1):
            for cov in ['diag','full','tied','spherical']:
                model,row=fit_candidate(x,k=2,covariance=cov,reg_covar=1e-5,n_init=1,max_iter=100,seed=26)
                self.assertIsNotNone(model);self.assertTrue(np.isfinite(row['bic']))
                self.assertGreater(row['min_cov_eigenvalue'],0)
                self.assertGreaterEqual(row['max_cov_condition'],1)
                if cov=='spherical':self.assertEqual(row['max_cov_condition'],1)

    def test_candidate_failure_is_reported(self):
        _,row=fit_candidate(np.ones((2,3)),k=4,covariance='full',reg_covar=1e-5,n_init=1,max_iter=10,seed=26)
        self.assertFalse(row['converged']);self.assertTrue(row['error'])

    def test_staged_tuning_records_and_selects_converged_fit(self):
        x=np.random.default_rng(26).normal(size=(80,3))
        config=dict(components=[1,2],covariance_types=['spherical'],reg_covar=1e-5,
            n_init=1,max_iter=100,selection='bic',tuning={'reg_covar':[1e-4],'n_init':[2],'max_iter':[1]})
        with threadpool_limits(limits=1):model,table=fit_grid(x,config,26)
        self.assertEqual(int(table.selected.sum()),1)
        self.assertTrue(table[table.selected].converged.iloc[0])
        self.assertEqual(set(table.stage),{'structure','optimizer'})

    def test_frozen_threshold_survives_csv_roundtrip_exactly(self):
        y=np.array([0,0,0,0,1])
        s=np.array([.00042315647392143774,.1,.2,.3,.4])
        threshold=fpr_threshold(y,s,.75)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'thresholds.csv'
            pd.DataFrame([dict(threshold=threshold,threshold_hex=threshold.hex())]).to_csv(path,index=False)
            loaded=load_thresholds(path).threshold.iloc[0]
            self.assertEqual(float(loaded).hex(),threshold.hex())
            self.assertEqual(at_threshold(y,s,loaded)['fp'],3)

    def test_model_independent_harness_and_count_policy(self):
        y=np.array([0,0,0,1,1]);s=np.array([0,1,2,3,4])
        result=evaluate_scores(y,s,thresholds=[2.5],review_fractions=[.4],review_counts=[2])
        self.assertEqual(result['ranking']['average_precision'],1)
        self.assertEqual(result['operating_points'][0]['fp'],0)
        self.assertEqual(result['review_counts'][0]['tp'],2)
        with self.assertRaises(ValueError):at_review_count(y,s,6)


if __name__=='__main__':unittest.main()
