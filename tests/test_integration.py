"""Integration-owner tests; do not require unmerged Role 4/5 modules or raw data."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('integration_runner', ROOT / 'run_all.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class IntegrationTests(unittest.TestCase):
    def config(self):
        return dict(seed=26, data_path='data/raw/creditcard.csv', split_fractions=[.6,.2,.2],
            gmm=dict(selection='bic', components=[1], covariance_types=['diag']),
            fpr_budgets=[.005], review_fractions=[.005], review_counts=[100],
            bootstrap_repetitions=2, robustness={})

    def test_valid_config_and_invalid_budgets(self):
        cfg = self.config()
        runner.validate_config(cfg)
        cfg['fpr_budgets'] = [1.0]
        with self.assertRaisesRegex(ValueError, 'FPR budgets'):
            runner.validate_config(cfg)

    def test_invalid_split_policy(self):
        cfg = self.config()
        cfg['split_fractions'] = [.7,.2,.2]
        with self.assertRaisesRegex(ValueError, 'split_fractions'):
            runner.validate_config(cfg)

    def test_preflight_is_read_only_and_names_missing_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = list(root.iterdir())
            missing = runner.preflight(root, root/'config.yaml', root/'creditcard.csv')
            self.assertEqual(before, list(root.iterdir()))
            self.assertTrue(any('Role 4' in item for item in missing))
            self.assertTrue(any('Role 5' in item for item in missing))

    def test_quick_run_skips_only_robustness_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = runner.preflight(root, root/'config.yaml', root/'creditcard.csv', True)
            self.assertFalse(any('src/robustness.py' in item for item in missing))
            self.assertTrue(any('src/evaluation.py' in item for item in missing))
            self.assertTrue(any('src/error_analysis.py' in item for item in missing))

    def test_complete_presence_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative, _ in runner.COMPONENTS.values():
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            config = root/'config.yaml'; config.touch()
            data = root/'creditcard.csv'; data.touch()
            self.assertEqual(runner.preflight(root,config,data), [])

    def test_cli_help_needs_no_pending_role_modules(self):
        for path in ['run_all.py', 'scripts/verify_outputs.py']:
            result = subprocess.run([sys.executable, str(ROOT/path), '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
