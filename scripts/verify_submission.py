"""Check that the public submission's documents, notebooks and metadata are complete.

This complements verify_outputs.py, which recomputes model evaluation contracts.
No raw CSV or fitted joblib files are needed for this artifact check.
"""
from pathlib import Path
import argparse
import json
import hashlib
import ast
import re
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=[
 'README.md','CONTRIBUTIONS.md','docs/FINAL_AUDIT.md','docs/integration_checkpoint.md',
 'reports/literature_review.md','reports/robustness_analysis.md','reports/error_analysis.md',
 'reports/ieee_report/README.md','reports/ieee_report/main.tex','reports/ieee_report/main.pdf',
 'reports/ieee_report/ranking.tex','reports/ieee_report/operating.tex',
 'reports/T085_Phase1_IEEE_Report.pdf','docs/T085_Phase1.pptx','docs/T085_Phase1.pdf',
 'overleaf_T085_phase1.zip','results/experiment_plan.json','results/run_metadata.json',
 'results/submission_summary.json','results/source_manifest.json','results/models/README.md','data/processed/split_indices.npz',
 'scripts/verify_outputs.py','scripts/execute_notebooks.py']


def check(root=ROOT):
    missing=[p for p in REQUIRED if not (root/p).is_file()]
    if missing:raise ValueError('Missing submission artifacts: '+', '.join(missing))
    for item in (root/'reports/ieee_report/figures').iterdir():
        if item.is_file() and item.suffix!='.png' and item.name!='.gitkeep':
            raise ValueError(f'Non-figure file remains in the figures directory: {item.name}')
    source=(root/'reports/ieee_report/main.tex').read_text()
    for image in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',source):
        if not (root/'reports/ieee_report'/image).is_file():raise ValueError('Broken LaTeX image path: '+image)
    for name in ['ranking','operating']:
        if f'% BEGIN GENERATED {name}' not in source:raise ValueError('Generated table marker missing: '+name)
    if (root/'reports/T085_Phase1_IEEE_Report.pdf').read_bytes()!=(root/'reports/ieee_report/main.pdf').read_bytes():
        raise ValueError('Submission PDF differs from the compiled report source PDF.')
    notebooks=sorted((root/'notebooks').glob('*.ipynb'))
    if len(notebooks)!=7:raise ValueError('Expected all seven Phase 1 notebooks.')
    for path in notebooks:
        notebook=json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                if cell['execution_count'] is None:raise ValueError('Unexecuted notebook cell: '+path.name)
                if any(out['output_type']=='error' for out in cell['outputs']):raise ValueError('Notebook error: '+path.name)
    with zipfile.ZipFile(root/'overleaf_T085_phase1.zip') as archive:
        if archive.testzip():raise ValueError('Corrupt Overleaf ZIP.')
        if archive.read('main.tex')!=(root/'reports/ieee_report/main.tex').read_bytes():
            raise ValueError('Overleaf source is stale.')
        for path in (root/'reports/ieee_report/figures').glob('*.png'):
            if archive.read('figures/'+path.name)!=path.read_bytes():raise ValueError('Overleaf figure is stale: '+path.name)
    with zipfile.ZipFile(root/'docs/T085_Phase1.pptx') as archive:
        if archive.testzip():raise ValueError('Corrupt editable presentation.')
        slides=[n for n in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
        if len(slides)!=15:raise ValueError('Expected the full 15-slide presentation.')
    metadata=json.loads((root/'results/run_metadata.json').read_text())
    if metadata.get('status')!='complete' or not metadata.get('robustness_completed'):
        raise ValueError('The recorded official run is not a complete robustness run.')
    summary=json.loads((root/'results/submission_summary.json').read_text())
    if summary['notebooks']!=7:raise ValueError('Notebook summary differs from the submission.')
    tests=sum(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_') for path in (root/'tests').glob('test_*.py') for node in ast.walk(ast.parse(path.read_text())))
    if summary['tests']!=tests:raise ValueError('Test-count summary is stale.')
    team=json.loads((root/'configs/team.json').read_text())
    if any(role['status']!='integrated and verified' for role in team['roles']):
        raise ValueError('Role status metadata is stale.')
    manifest=json.loads((root/'results/source_manifest.json').read_text())
    for path,expected in manifest['files'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Source changed after final verification: '+path)
    tracked=subprocess.check_output(['git','-C',str(root),'ls-files'],text=True).splitlines()
    if any((p.startswith('data/raw/') and p.endswith('.csv')) or p.endswith('.joblib') for p in tracked):
        raise ValueError('Raw CSV or serialized models should not be tracked in this public submission.')
    print('Submission artifacts verified: canonical report paths, matching PDFs/Overleaf source, seven executed notebooks, 15-slide deck, run metadata and role statuses.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:check()
    except (ValueError,KeyError,OSError,zipfile.BadZipFile) as exc:parser.exit(2,f'Submission verification failed: {exc}\n')
