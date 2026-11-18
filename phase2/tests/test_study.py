import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from run_phase2_study import (
    BUDGETS, PHASE2, ROOT, SEEDS, VARIANTS, fit_autoencoder,
    read_frozen_scores, reconstruction_error,
)


class StudyTests(unittest.TestCase):
    def test_score_reader_rejects_row_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scores.npz'
            np.savez(path, score=[0.1, 0.2], source_row=[4, 6])
            np.testing.assert_array_equal(read_frozen_scores(path, [4, 6]), [0.1, 0.2])
            with self.assertRaisesRegex(ValueError, 'split order'):
                read_frozen_scores(path, [6, 4])

    def test_checkpoint_restores_lowest_reserve_mse(self):
        data = np.random.default_rng(3).normal(size=(540, 29)).astype(np.float32)
        fit, reserve = data[:512], data[512:]
        model, history, epoch = fit_autoencoder(fit, reserve, width=8, noise_std=0.05,
                                                seed=26, max_epochs=3)
        reserve_mse = float(reconstruction_error(model, reserve).mean())
        self.assertEqual(reserve_mse, history[epoch - 1]['reserve_mse'])
        self.assertEqual(reserve_mse, min(point['reserve_mse'] for point in history))

    def test_committed_study_matches_frozen_inputs_and_operating_points(self):
        study = PHASE2 / 'study'
        metadata = json.loads((study / 'metadata.json').read_text())
        self.assertTrue(metadata['historical_test_previously_inspected'])
        self.assertEqual(metadata['ablation_mixed_split'], 'validation')
        self.assertEqual(metadata['ablation_runner_sha256'],
                         hashlib.sha256((ROOT / 'run_phase2_study.py').read_bytes()).hexdigest())
        for file, expected in metadata['inputs_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / file).read_bytes()).hexdigest(), expected)
        operations = pd.read_csv(study / 'operating_points.csv')
        historical = pd.read_csv(PHASE2 / 'metrics.csv')
        for (model, split), group in operations.groupby(['model', 'split']):
            self.assertEqual(tuple(group.fpr_budget), BUDGETS)
            if split == 'validation':
                self.assertTrue((group.fpr <= group.fpr_budget).all())
            old = historical[(historical.model == model.lower()) & (historical.split == split)].iloc[0]
            recalculated = group[group.fpr_budget.eq(0.005)].iloc[0]
            for key in ('tp', 'fp', 'fn', 'tn'):
                self.assertEqual(recalculated[key], old[f'threshold_{key}'])
        for _, group in operations.groupby(['model', 'fpr_budget']):
            self.assertEqual(group.threshold_hex.nunique(), 1)

    def test_ablation_and_executed_notebook_coverage(self):
        study = PHASE2 / 'study'
        trials = pd.read_csv(study / 'ablation_trials.csv')
        self.assertEqual(set(zip(trials.variant, trials.seed)),
                         {(variant, seed) for variant in VARIANTS for seed in SEEDS})
        self.assertEqual(len(trials), len(VARIANTS) * len(SEEDS))
        self.assertTrue((trials.best_epoch.between(1, trials.epochs)).all())
        path = ROOT / 'notebooks/phase2/01_fraud_triage.ipynb'
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                self.assertIsNotNone(cell.execution_count)
                self.assertFalse(any(out.output_type == 'error' for out in cell.outputs))
