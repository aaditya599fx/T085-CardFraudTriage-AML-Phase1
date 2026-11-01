"""Regenerate IEEE report tables from the recorded experiment; compile with pdflatex or Tectonic."""
from pathlib import Path
import json
import shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/ieee_report'
OUT.mkdir(parents=True,exist_ok=True)
tables=ROOT/'results/tables'
r=pd.read_csv(tables/'ranking_metrics.csv'); op=pd.read_csv(tables/'operating_points.csv'); cap=pd.read_csv(tables/'review_capacity.csv')
labels={'gmm':'GMM','pca':'PCA reconstruction','isolation_forest':'Isolation Forest','approx_ocsvm':'Approx. OC-SVM'}
rows=[]
for _,x in r[r.split=='test'].iterrows():
 rows.append(f"{labels[x.model]} & {x.average_precision:.4f} & {x.pr_auc_trapezoidal:.4f} & {x.roc_auc:.4f} \\\\")
(OUT/'ranking.tex').write_text('\n'.join(rows))
rows=[]
for _,x in op[(op.model=='gmm')&(op.split=='test')].iterrows():
 rows.append(f"{x.fpr_budget*100:.1f} & {int(x.tp)} & {int(x.fp)} & {x.recall*100:.2f} & {x.fpr*100:.3f} & {x.precision*100:.2f} \\\\")
(OUT/'operating.tex').write_text('\n'.join(rows))
fig=OUT/'figures'; fig.mkdir(exist_ok=True)
for n in ['pr_curves','threshold_tradeoffs','gmm_selection','eda_class_imbalance','eda_amount_time','dimensionality_stability','seed_stability','error_analysis','gmm_confusion_matrices']:
 shutil.copy2(ROOT/f'results/figures/{n}.png',fig/f'{n}.png')
print('Updated report tables and figure assets from recorded results.')

import re
main=OUT/'main.tex'
s=main.read_text()
for name in ['ranking','operating']:
    pattern=r'% BEGIN GENERATED '+name+r'.*?% END GENERATED '+name
    replacement='% BEGIN GENERATED '+name+'\n'+(OUT/(name+'.tex')).read_text()+'\n% END GENERATED '+name
    s=re.sub(pattern,lambda m:replacement,s,flags=re.S)
main.write_text(s)
