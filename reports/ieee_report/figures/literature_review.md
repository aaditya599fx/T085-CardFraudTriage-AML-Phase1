# Literature review: Phase 1

## Scope and research question

Can a normal-density model support a useful fraud review queue when fraudulent records are scarce? We compare GMM likelihood with isolation, linear reconstruction and an approximate one-class boundary under the same chronological protocol. Density fit quality and fraud ranking are evaluated as distinct questions.

## Dempster, Laird and Rubin (1977)

[Maximum Likelihood from Incomplete Data via the EM Algorithm](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x), JRSS B, 39(1), pp. 1-22.

EM alternates latent-variable inference with parameter updates. In a GMM, component membership is latent and responsibilities give soft assignments. This supplies the theoretical basis for fitting our density model. Convergence of the fitting algorithm does not establish that the model family describes legitimate spending well or ranks fraud well. We therefore record convergence separately from held-out triage performance.

## Liu, Ting and Zhou (2008)

[Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf), IEEE ICDM, pp. 413-422, DOI: 10.1109/ICDM.2008.17.

Isolation Forest isolates observations through random recursive partitions. Its score provides a comparator that does not require a Gaussian density model. We fix 200 trees and 256 observations per tree. The full legitimate training matrix is supplied to the estimator, while each tree uses its prescribed random subsample. Disagreement with GMM is informative because the methods encode different notions of unusualness.

## Saito and Rehmsmeier (2015)

[The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://doi.org/10.1371/journal.pone.0118432), PLOS ONE, 10(3), e0118432.

Precision-recall analysis exposes the effect of false alarms when positives are rare. We use AP as the main ranking summary and additionally report trapezoidal PR AUC. ROC AUC remains secondary. Review capacity and validation-calibrated FPR policies translate rankings into workload. A model may achieve an apparently strong ROC AUC yet yield low precision in the small queue analysts can actually review.

## Implementation references

- [scikit-learn Gaussian mixture documentation](https://scikit-learn.org/stable/modules/mixture.html): EM implementation, covariance structures and model-selection support.
- [scikit-learn average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): non-interpolated AP. It is computed as the sum of recall increments times precision, which differs from trapezoidal integration.
- [scikit-learn SGDOneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDOneClassSVM.html): scalable linear one-class estimation; paired here with a fixed Nyström approximation to an RBF kernel.

## Phase 2 bridge

Sakurada and Yairi (2014), *Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction*, MLSDA, pp. 4-11, [DOI: 10.1145/2689746.2689747](https://doi.org/10.1145/2689746.2689747).

This is a suggested Phase 2 reading from the project description. The full text was not available through the research fetch, so no detailed claims about its experiments are used here. An autoencoder is future work. Phase 1's PCA reconstruction baseline supplies a linear reconstruction comparator, without claiming equivalence to a nonlinear autoencoder.

## Gap and study design

A full-covariance GMM estimates many parameters and can concentrate density around narrow directions. Covariance regularization and bounded component searches address numerical risk but cannot guarantee useful ranking. AIC/BIC select density complexity, not fraud recall. The expanded experiment records both criteria, covariance condition numbers, staged optimizer trials, and validation-only dimensionality, seed and sample-size sensitivity. The already inspected test partition is explicitly reused. PCA and Isolation Forest test whether simpler or non-density assumptions are more useful for this data. The approximate OC-SVM's weak result is preserved, rather than removed after seeing test scores.
