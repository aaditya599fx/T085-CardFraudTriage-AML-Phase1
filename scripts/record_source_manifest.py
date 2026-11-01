"""Record hashes of the integrated source/config used to reproduce the final artifacts."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=[ROOT/'run_all.py',ROOT/'requirements.txt',ROOT/'configs/config.yaml']
    for directory in ['src','scripts','tests']:
        paths.extend(sorted((ROOT/directory).glob('*.py')))
    paths.extend(sorted((ROOT/'scripts').glob('*.mjs')))
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    target=ROOT/'results/source_manifest.json'
    target.write_text(json.dumps(dict(algorithm='sha256',
        description='Hashes of the integrated source and experiment configuration accompanying the verified official run.',
        files=files),indent=2)+'\n')
    print(f'Recorded {len(files)} source/configuration hashes.')


if __name__=='__main__':main()
