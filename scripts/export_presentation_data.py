"""Export recorded tables and PR points for the editable presentation builder."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
R=Path(__file__).resolve().parents[1]
if str(R) not in sys.path:sys.path.insert(0,str(R))
from src.data import load_raw

def main():
    tables=['ranking_metrics','operating_points','review_capacity','gmm_grid','split_summary',
            'robustness_trials','seed_summary','error_group_summary']
    d={name:pd.read_csv(R/f'results/tables/{name}.csv').fillna('').to_dict('records') for name in tables}
    with np.load(R/'data/processed/split_indices.npz') as z: ids=z['test']
    frame=load_raw(R/'data/raw/creditcard.csv')
    y=frame.loc[ids,'Class'].to_numpy()
    d['pr']=[]
    for name in ['gmm','pca','isolation_forest','approx_ocsvm']:
        scores=np.load(R/f'results/scores/{name}_test.npz')['score']
        p,r,_=precision_recall_curve(y,scores,drop_intermediate=True)
        d['pr'].append(dict(name=name,p=p.tolist(),r=r.tolist()))
    d['metadata']=json.loads((R/'results/submission_summary.json').read_text())
    target=R/'results/presentation_data.json'
    target.write_text(json.dumps(d,indent=2))
    print(target)

if __name__=='__main__':main()
