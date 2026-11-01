"""Classical comparators using precisely the same legitimate training observations."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def fit_baselines(x, seed):
    models = {
        'isolation_forest': IsolationForest(n_estimators=200, max_samples=256, random_state=seed, n_jobs=2),
        'pca': PCA(n_components=15, svd_solver='full'),
        # Approximate RBF kernel avoids the quadratic memory cost of exact OC-SVM.
        'approx_ocsvm': make_pipeline(Nystroem(gamma=1/x.shape[1], n_components=128, random_state=seed),
            StandardScaler(), SGDOneClassSVM(nu=.01, random_state=seed, max_iter=2000, tol=1e-4)),
    }
    for name, model in models.items():
        print(f'Fitting {name} on {len(x)} legitimate rows', flush=True)
        model.fit(x)
    return models

def score(name, model, x):
    if name == 'pca':
        return np.mean((x - model.inverse_transform(model.transform(x)))**2, axis=1)
    return -model.score_samples(x)
