# Data card

Source: [ULB/Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Observed quality

The downloaded CSV has 284,807 records, 31 columns, 492 frauds, no missing or infinite cells, and 1,081 exact duplicate rows. Fraud prevalence is 0.17275%. Time ranges from 0 to 172,792 seconds. V1-V28 are anonymized provider PCA projections. Amount and elapsed Time retain their names. No account, merchant, or cardholder IDs are available.

## Partitions

| Partition | All rows | Fraud | Legitimate | Fraud rate |
|---|---:|---:|---:|---:|
| Train | 170,888 | 360 | 170,528 | 0.21066% |
| Validation | 56,958 | 57 | 56,901 | 0.10007% |
| Test | 56,961 | 75 | 56,886 | 0.13167% |

The row fractions approximate 60/20/20. Boundaries move to the right edge of tied timestamps: train ends at 120,396 seconds, validation spans 120,397-145,248, and test spans 145,249-172,792. This enforces strict time separation. Splitting by row count does not create equal-duration windows.

## Feature choices

Use V1-V28 and log(1+Amount), yielding 29 features. Amount is nonnegative and skewed, motivating a monotone log transform. Standard scaling uses legitimate training means and standard deviations. No labels appear in model inputs. Time is reserved for splitting and EDA because its elapsed value is nonstationary across windows.

Training labels are used only to identify legitimate observations; this is clean-normal novelty detection rather than learning without any label access. Mixed validation labels support operational calibration and diagnostic reporting. No oversampling, SMOTE, random train/test shuffle, or test-set scaling is applied.

## Duplicate policy and limitations

Retain duplicates because absent transaction IDs prevent distinguishing actual repeats from duplicate ingestion. Exact duplicates include Time, so they cannot span partitions under the tied-time split rule. Repeated observations can influence covariance and evaluation precision. A deduplication sensitivity study is future work, not a completed claim.

Provider PCA may encode transformations fitted before our access; we cannot audit or refit those transformations. Anonymization restricts business explanations and subgroup fairness analysis. Time is insufficient to identify per-account leakage. The dataset's narrow historical window cannot establish real deployment performance.
