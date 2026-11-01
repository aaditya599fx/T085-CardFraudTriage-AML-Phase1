"""Fixed features, log amount, and scaling learned exclusively on legitimate training rows."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from .data import FEATURES

def engineering(frame):
    x = frame[FEATURES].copy()
    x['Amount'] = np.log1p(x['Amount'])
    return x.to_numpy(dtype='float64')

def prepare(parts):
    normal = parts['train'].Class.to_numpy() == 0
    _, pipeline = preprocess(parts['train'].loc[normal])
    scaler = pipeline.named_steps['scaler']
    x = {name: pipeline.transform(frame) for name, frame in parts.items()}
    return x['train'][normal], x, scaler



def preprocess(frame, pipeline=None, pca_dimensions=None):
    """Fit on a caller-provided legitimate training frame or transform with an existing pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.decomposition import PCA
    if pipeline is None:
        if frame.empty or (frame.Class != 0).any():
            raise ValueError('Preprocessing fit requires nonempty legitimate-only training rows.')
        steps = [('engineering', FunctionTransformer(engineering, validate=False)),
                 ('scaler', StandardScaler())]
        if pca_dimensions is not None:
            steps.append(('pca', PCA(n_components=pca_dimensions, svd_solver='full')))
        pipeline = Pipeline(steps).fit(frame)
    elif pca_dimensions is not None:
        raise ValueError('A fitted pipeline fixes dimensionality; do not request another PCA size.')
    return pipeline.transform(frame), pipeline
