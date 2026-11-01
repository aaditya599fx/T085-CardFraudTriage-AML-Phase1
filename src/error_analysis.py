"""Post-freeze descriptive error analysis, with no feedback into model selection."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .features import engineering
from .metrics import load_thresholds


def analyze_errors(parts, scores, out, budget=.005):
    thresholds = load_thresholds(out/'tables/thresholds_frozen_on_validation.csv')
    t = thresholds[(thresholds.model=='gmm') & np.isclose(thresholds.fpr_budget,budget)].threshold.iloc[0]
    test = parts['test'].copy()
    s = scores['gmm']['test']
    test['anomaly_score'] = s
    test['flagged'] = s > t
    test['group'] = np.select([
        (test.Class==1)&test.flagged,(test.Class==1)&~test.flagged,
        (test.Class==0)&test.flagged],['caught_fraud','missed_fraud','false_positive'],default='true_negative')
    columns = ['Time','Amount','Class','anomaly_score','flagged','group']
    test[test.group=='missed_fraud'][columns].sort_values('anomaly_score').to_csv(out/'tables/missed_frauds.csv')
    test[test.group=='false_positive'][columns].sort_values('anomaly_score',ascending=False).head(20).to_csv(out/'tables/top_false_positives.csv')
    rows=[]
    for name,frame in test.groupby('group'):
        rows.append(dict(group=name,rows=len(frame),median_amount=float(frame.Amount.median()),
            amount_q25=float(frame.Amount.quantile(.25)),amount_q75=float(frame.Amount.quantile(.75)),
            median_time_hours=float(frame.Time.median()/3600),median_score=float(frame.anomaly_score.median())))
    pd.DataFrame(rows).to_csv(out/'tables/error_group_summary.csv',index=False)
    normal = parts['train'][parts['train'].Class==0]
    reference=engineering(normal)
    center=reference.mean(axis=0); scale=reference.std(axis=0)
    scale[scale==0]=1
    labels = [f'V{i}' for i in range(1,29)]+['log1p(Amount)']
    deviations=[]
    for name in ['caught_fraud','missed_fraud','false_positive','true_negative']:
        frame=test[test.group==name]
        if len(frame):
            for feature,value in zip(labels,np.abs((engineering(frame)-center)/scale).mean(axis=0)):
                deviations.append(dict(group=name,feature=feature,mean_abs_training_z=float(value)))
    pd.DataFrame(deviations).to_csv(out/'tables/error_feature_deviations.csv',index=False)
    op=pd.read_csv(out/'tables/operating_points.csv')
    op['false_positives_per_fraud']=np.divide(op.fp,op.tp,out=np.full(len(op),np.nan),where=op.tp.to_numpy()!=0)
    op['reviews_per_fraud']=np.divide(op.flagged,op.tp,out=np.full(len(op),np.nan),where=op.tp.to_numpy()!=0)
    op['no_frauds_caught']=op.tp==0
    op.to_csv(out/'tables/review_workload.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    groups=[test[test.group==g] for g in ['caught_fraud','missed_fraud','false_positive']]
    for frame,name in zip(groups,['Caught fraud','Missed fraud','False positive']):
        if len(frame):
            axes[0].hist(np.log1p(frame.Amount),bins=20,density=True,alpha=.45,label=name)
            signed_log_score=np.sign(frame.anomaly_score)*np.log1p(np.abs(frame.anomaly_score))
            axes[1].hist(signed_log_score,bins=20,density=True,alpha=.45,label=name)
    axes[0].set(xlabel='log(1 + Amount)',ylabel='Conditional density')
    axes[1].set(xlabel='Signed log(1 + |score|)',ylabel='Conditional density')
    axes[1].axvline(np.sign(t)*np.log1p(abs(t)),color='black',linestyle='--',label='Frozen threshold')
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'figures/error_analysis.png');plt.close(fig)
    return pd.DataFrame(rows)
