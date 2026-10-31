"""Exploratory plots use training rows; follow-up evaluation discloses test reuse."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'figure.dpi':140, 'savefig.bbox':'tight', 'axes.spines.top':False,
                     'axes.spines.right':False, 'font.size':10})

def plot_eda(train, out):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = train.Class.value_counts().sort_index()
    ax.bar(['Legitimate', 'Fraud'], counts, color=['#147d92','#d65345'])
    ax.set_yscale('log'); ax.set_ylabel('Training transactions (log scale)')
    for i,v in enumerate(counts): ax.text(i, v*1.15, f'{v:,}', ha='center')
    ax.set_ylim(1, counts.max()*3); fig.savefig(out/'eda_class_imbalance.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9,3.5))
    for label,color in [(0,'#147d92'),(1,'#d65345')]:
        subset = train[train.Class==label]
        axes[0].hist(np.log1p(subset.Amount), bins=40, density=True, alpha=.5,
                     label=['Legitimate','Fraud'][label], color=color)
        axes[1].hist(subset.Time/3600, bins=40, density=True, alpha=.5,
                     label=['Legitimate','Fraud'][label], color=color)
    axes[0].set(xlabel='log(1 + Amount)', ylabel='Density')
    axes[1].set(xlabel='Hours since first transaction', ylabel='Density')
    axes[0].legend(); axes[1].legend(); fig.tight_layout()
    fig.savefig(out/'eda_amount_time.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,6))
    corr = train.drop(columns=['Class','Time']).corr()
    im = ax.imshow(corr, vmin=-1,vmax=1,cmap='RdBu_r')
    ticks = list(range(0,len(corr),3))
    ax.set_xticks(ticks, corr.columns[ticks], rotation=90); ax.set_yticks(ticks,corr.columns[ticks])
    fig.colorbar(im, ax=ax,label='Pearson correlation (training data)')
    fig.tight_layout(); fig.savefig(out/'eda_correlation.png'); plt.close(fig)
    fig,axes=plt.subplots(7,4,figsize=(14,18))
    for index,ax in enumerate(axes.flat,1):
        feature=f'V{index}'
        lo,hi=train[feature].quantile([.01,.99])
        for label,color in [(0,'#147d92'),(1,'#d65345')]:
            ax.hist(train.loc[train.Class==label,feature],bins=30,range=(lo,hi),density=True,
                    alpha=.5,color=color,label=['Legitimate','Fraud'][label])
        ax.set_title(feature,fontsize=10)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Training V1-V28 distributions: central 98% display range per feature',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/'eda_v_distributions.png');plt.close(fig)
