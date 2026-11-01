"""Package the compiled canonical IEEE PDF and matching Overleaf sources."""
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'reports/ieee_report'
    compiled=source/'main.pdf'
    if not compiled.is_file():
        raise SystemExit('Compile reports/ieee_report/main.tex before packaging the report.')
    shutil.copy2(compiled,ROOT/'reports/T085_Phase1_IEEE_Report.pdf')
    with zipfile.ZipFile(ROOT/'overleaf_T085_phase1.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix in ['.tex','.png','.bib','.md']:
                archive.write(path,path.relative_to(source))
    print('Packaged the compiled submission PDF and matching Overleaf source/figures.')


if __name__=='__main__':main()
