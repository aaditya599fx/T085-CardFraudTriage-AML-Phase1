"""Freeze validation thresholds first, then run the same evaluation harness for any scorer."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from .metrics import evaluate_scores, fpr_threshold, bootstrap_ap


def evaluate(scores, parts, cfg, out):
    policies = []
    yv = parts['validation'].Class.to_numpy()
    for name, ss in scores.items():
        for budget in cfg['fpr_budgets']:
            policies.append(dict(model=name,fpr_budget=budget,
                threshold=fpr_threshold(yv,ss['validation'],budget)))
    thresholds = pd.DataFrame(policies)
    thresholds['threshold_hex'] = thresholds.threshold.map(float.hex)
    thresholds.to_csv(out/'tables/thresholds_frozen_on_validation.csv',index=False)
    summary, operating, capacity = [], [], []
    fig, axes = plt.subplots(1,2,figsize=(11,4.3))
    for name, ss in scores.items():
        policies=thresholds[thresholds.model==name].to_dict('records')
        for index,split in enumerate(['validation','test']):
            y,s=parts[split].Class.to_numpy(),ss[split]
            result=evaluate_scores(y,s,thresholds=[p['threshold'] for p in policies],
                review_fractions=cfg['review_fractions'],review_counts=cfg.get('review_counts',[100]))
            row=dict(model=name,split=split,**result['ranking'])
            if split=='test':row.update(bootstrap_ap(y,s,cfg['bootstrap_repetitions'],cfg['seed']))
            summary.append(row)
            p,r,_=precision_recall_curve(y,s)
            axes[index].step(r,p,where='post',label=f'{name} (AP={row["average_precision"]:.3f})')
            for policy,point in zip(policies,result['operating_points']):
                operating.append(dict(model=name,split=split,fpr_budget=policy['fpr_budget'],**point))
            for point in result['review_fractions']:
                capacity.append(dict(model=name,split=split,policy='fraction',**point))
            for point in result['review_counts']:
                capacity.append(dict(model=name,split=split,policy='count',capacity_fraction=np.nan,**point))
    for ax,split in zip(axes,['validation','test']):
        ax.axhline(parts[split].Class.mean(),color='gray',linestyle='--',label=f'{split} fraud prevalence')
        ax.set(title=split.capitalize(),xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'figures/pr_curves.png');plt.close(fig)
    pd.DataFrame(summary).to_csv(out/'tables/ranking_metrics.csv',index=False)
    op=pd.DataFrame(operating);op.to_csv(out/'tables/operating_points.csv',index=False)
    pd.DataFrame(capacity).to_csv(out/'tables/review_capacity.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for name in scores:
        rows=op[(op.model==name)&(op.split=='test')]
        axes[0].plot(rows.fpr_budget*100,rows.recall,marker='o',label=name)
        axes[1].plot(rows.fpr_budget*100,rows.fpr*100,marker='o',label=name)
    upper=max(cfg['fpr_budgets'])*100
    axes[1].plot([0,upper],[0,upper],linestyle='--',color='gray',label='Nominal budget')
    axes[0].set(xlabel='Validation FPR budget (%)',ylabel='Test recall',ylim=(0,1))
    axes[1].set(xlabel='Validation FPR budget (%)',ylabel='Realized test FPR (%)')
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'figures/threshold_tradeoffs.png');plt.close(fig)
    rows=op[(op.model=='gmm')&(op.split=='test')]
    fig,axes=plt.subplots(1,len(rows),figsize=(3*len(rows),3))
    for ax,(_,row) in zip(np.atleast_1d(axes),rows.iterrows()):
        matrix=np.array([[row.tn,row.fp],[row.fn,row.tp]],dtype=int)
        ax.imshow(matrix,cmap='Blues')
        for (i,j),v in np.ndenumerate(matrix):
            ax.text(j,i,f'{v:,}',ha='center',va='center',color='white' if v>matrix.max()/2 else 'black')
        ax.set_xticks([0,1],['Legit','Alert']);ax.set_yticks([0,1],['Legit','Fraud'])
        ax.set(xlabel='Prediction',ylabel='Actual',title=f'{row.fpr_budget*100:g}% validation FPR')
    fig.tight_layout();fig.savefig(out/'figures/gmm_confusion_matrices.png');plt.close(fig)
    return pd.DataFrame(summary)
