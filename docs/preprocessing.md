# Feature engineering and leakage checklist

Farhana owns preprocessing review under the team's role assignment.

- Inputs are V1-V28 and log1p(Amount), 29 numerical features in that fixed order.
- Time defines chronological partitions and training EDA; the model excludes its extrapolating elapsed value.
- Class identifies normal fitting rows and evaluation labels; it is never a model feature.
- `preprocess(normal_training_frame)` fits a raw-frame Pipeline with feature engineering and StandardScaler.
- `preprocess(frame, pipeline=fitted_pipeline)` only transforms. Fitting rejects a frame containing fraudulent observations.
- Optional PCA uses full SVD and is fit only on legitimate training records. The dimensionality study keeps 5, 10 or 20 components and compares with the unreduced 29-feature matrix.
- Full pipelines and standalone scaling are saved to results/models locally. Reuse the same objects; do not refit on validation or test.
- Provider-created V1-V28 already contain an upstream PCA transform whose original fitting population cannot be audited here.

Verification: automated tests check legitimate-only statistics and preservation of fitting statistics when transforming shifted validation-like inputs. Notebook 02 also checks the actual fitted training means and transforms.
