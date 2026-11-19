import hashlib
import json
import unittest

import nbformat
import numpy as np
import pandas as pd

from run_phase2_hybrid import AE_WEIGHTS, fuse, select_hybrid, tail_surprise
from run_phase2_study import BUDGETS, PHASE2, ROOT


class HybridTests(unittest.TestCase):
    def test_empirical_tail_includes_ties_and_smooths_above_reference(self):
        reference = np.array([0, 1, 1, 3])
        scores = np.array([0, 1, 2, 3, 4])
        np.testing.assert_allclose(tail_surprise(reference, scores), -np.log([1, 0.8, 0.4, 0.4, 0.2]))

    def test_fusion_uses_both_signals_and_validation_selects_a_grid_weight(self):
        scores = {'GMM': np.array([0, 1, 3, 2]), 'Autoencoder': np.array([0, 2, 1, 3])}
        reference, weight, trials = select_hybrid(np.array([0, 0, 0, 1]), scores)
        self.assertEqual(tuple(trials.ae_weight), AE_WEIGHTS)
        self.assertIn(weight, AE_WEIGHTS)
        self.assertEqual(trials.loc[trials.ae_weight.eq(weight), 'validation_ap'].iloc[0], trials.validation_ap.max())
        combined = fuse(reference, scores, weight)
        for model in scores:
            null = {**scores, model: np.zeros_like(scores[model])}
            self.assertFalse(np.allclose(combined, fuse(reference, null, weight)))

    def test_committed_hybrid_evidence_preserves_frozen_baselines(self):
        hybrid = PHASE2 / 'hybrid'
        meta = json.loads((hybrid / 'metadata.json').read_text())
        self.assertTrue(meta['historical_test_previously_inspected'])
        self.assertAlmostEqual(meta['ae_weight'] + meta['gmm_weight'], 1.0)
        for file, expected in meta['inputs_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / file).read_bytes()).hexdigest(), expected)
        trials = pd.read_csv(hybrid / 'validation_weights.csv')
        winner = trials.sort_values(['validation_ap', 'ae_weight'], ascending=[False, True]).iloc[0]
        self.assertEqual(meta['ae_weight'], winner.ae_weight)
        ranking = pd.read_csv(hybrid / 'ranking.csv')
        original = pd.read_csv(PHASE2 / 'study/ranking.csv')
        pd.testing.assert_frame_equal(ranking.query("model != 'Hybrid'").reset_index(drop=True), original)
        operations = pd.read_csv(hybrid / 'operating_points.csv')
        for (_, split), group in operations.groupby(['model', 'split']):
            self.assertEqual(tuple(group.fpr_budget), BUDGETS)
            if split == 'validation':
                self.assertTrue((group.fpr <= group.fpr_budget).all())
        for _, group in operations.groupby(['model', 'fpr_budget']):
            self.assertEqual(group.threshold_hex.nunique(), 1)
        notebook = nbformat.read(ROOT / 'notebooks/phase2/02_hybrid_fusion.ipynb', as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                self.assertIsNotNone(cell.execution_count)
                self.assertFalse(any(output.output_type == 'error' for output in cell.outputs))
