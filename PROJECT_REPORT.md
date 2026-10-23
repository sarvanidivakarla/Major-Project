# CICIDS2018 Intrusion Detection Project Report

## 1. Project Objective

The project classifies network-flow records as binary outcomes:

- `0`: Benign traffic
- `1`: Attack traffic, including FTP-BruteForce and SSH-Bruteforce

The model family used in both experiments is a scaled Logistic Regression pipeline:

```text
StandardScaler -> LogisticRegression
```

The classifier uses `class_weight="balanced"`, a maximum of 200 iterations, a fixed probability threshold of `0.5`, and the attack class as the positive class.

## 2. Dataset and Preprocessing Work

The original `cic.csv` contained 1,048,575 network-flow records and 80 columns. Its labels were:

- Benign: 667,626
- FTP-BruteForce: 193,360
- SSH-Bruteforce: 187,589

The preprocessing pipeline in `preprocess_cic.py` was developed incrementally and now performs the following operations:

1. Reads the large CSV in chunks to limit memory usage.
2. Removes `Flow ID`, `Src IP`, and `Dst IP` when present.
3. Preserves `Timestamp` for chronological auditing and splitting.
4. Replaces numeric positive and negative infinity with missing values.
5. Handles empty strings and textual infinity values.
6. Removes rows containing missing or invalid values.
7. Removes columns with identical content while retaining the first copy and preserving `Label`.
8. Removes exact duplicate rows across chunk boundaries.
9. Writes the cleaned dataset to `data/cic_preprocessed.csv`.

The current timestamp-retaining preprocessed dataset contains:

- 819,126 rows
- 65 columns
- 63 model features, plus `Timestamp` and `Label`
- 15 duplicate-content columns removed
- 225,625 duplicate rows removed
- 3,824 invalid rows removed

The current dataset has no blank values, duplicate rows, or duplicate-content columns. `Timestamp` remains metadata and is not used as a model feature.

## 3. Version 1.0: Random Stratified Evaluation

Version 1.0 is implemented in `train_model.py`.

### Method

- Random stratified 80/20 train-test split
- 835,800 training records
- 208,951 testing records
- `Timestamp` and identifier metadata excluded from model features
- Logistic Regression fitted on the training partition
- StandardScaler fitted only on training features
- Threshold fixed at `0.5`

### Results

| Metric | Training | Testing |
|---|---:|---:|
| Accuracy | 99.9664% | 99.9646% |
| Precision | Not stored separately | 99.9043% |
| Recall / detection rate | 99.9987% | 99.9987% |
| F1-score | 99.9539% | 99.9515% |
| False positive rate | Not stored separately | 0.0550% |
| PR-AUC | Not stored separately | 99.9990% |

Testing confusion matrix:

| | Predicted Benign | Predicted Attack |
|---|---:|---:|
| Actual Benign | 132,689 | 73 |
| Actual Attack | 1 | 76,188 |

Additional measurements:

- Training time: approximately 7.60 seconds
- Inference time: approximately 0.00099 ms per record

### Version 1.0 Advantages

- Simple and fast baseline.
- Stratification preserves the overall class ratio in both partitions.
- StandardScaler and Logistic Regression are easy to interpret and deploy.
- Useful for initial pipeline validation.
- The Flask frontend supports manual prediction and random synthetic feature generation.

### Version 1.0 Disadvantages

- Random record-level splitting does not reflect future-traffic deployment.
- Duplicate or highly similar records can appear in both train and test sets.
- Earlier duplication analysis found 11,763 duplicate feature-vector patterns crossing the random train-test boundary.
- There is no validation partition for threshold or model-selection decisions.
- The very high scores may therefore be optimistic.
- The result is not evidence of performance on later chronological traffic.

## 4. Version 1.1: Chronological Temporal-Window Evaluation

Version 1.1 is implemented in `temporal_evaluation_v1_1.py`.

### Changes from Version 1.0

Version 1.1 keeps the Logistic Regression model and training logic conceptually unchanged. The main change is the evaluation strategy:

| Area | Version 1.0 | Version 1.1 |
|---|---|---|
| Split method | Random stratified records | Chronological windows |
| Partitions | Train/test | Train/validation/test |
| Windowing | None | Non-overlapping 30-minute windows |
| Boundary selection | Random row selection | Timestamp and window record counts |
| Label use in split | Stratification | Not used for boundaries |
| Timestamp | Excluded before modeling | Preserved for segmentation, excluded from features |
| Evaluation purpose | Baseline random estimate | Later-traffic temporal evaluation |

Windows use the half-open convention:

```text
[window_start, window_start + 30 minutes)
```

No window is divided across partitions.

### Temporal Audit

- Input rows: 819,126
- Valid timestamps: 819,126
- Invalid or missing timestamps: 0
- Suspicious timestamps before 2000: 5
- Suspicious timestamps excluded: 0
- Total 30-minute windows: 25

The five suspicious 1970 records were explicitly retained and assigned chronologically to the training partition. Almost all other records occur on 14 February 2018.

### Actual Allocation

| Partition | Records | Actual allocation | Windows | Benign | Attack |
|---|---:|---:|---:|---:|---:|
| Train | 603,385 | 73.6620% | 21 | 475,194 | 128,191 |
| Validation | 129,091 | 15.7596% | 2 | 104,958 | 24,133 |
| Test | 86,650 | 10.5783% | 2 | 82,306 | 4,344 |

The target was 70/15/15. Exact allocation was impossible without splitting a window, so complete-window boundaries were preserved.

### Temporal Boundaries

- Training: through window 21, ending at `2018-02-14 11:00:00`
- Validation: windows 22-23, from `2018-02-14 11:00:00` to `2018-02-14 12:00:00`
- Testing: windows 24-25, from `2018-02-14 12:00:00` to `2018-02-14 13:00:00`

### Results

| Metric | Training | Validation | Testing |
|---|---:|---:|---:|
| Accuracy | 99.9576% | 99.9667% | 99.9561% |
| Precision | 99.8046% | 99.8221% | 99.1328% |
| Recall | 99.9961% | 100.0000% | 100.0000% |
| F1-score | 99.9002% | 99.9110% | 99.5645% |
| False positive rate | 0.0528% | 0.0410% | 0.0462% |
| False negative rate | 0.0039% | 0.0000% | 0.0000% |
| ROC-AUC | 99.9993% | 99.9995% | 99.9981% |
| PR-AUC | 99.9929% | 99.9971% | 99.9243% |

Testing confusion matrix:

| | Predicted Benign | Predicted Attack |
|---|---:|---:|
| Actual Benign | 82,268 | 38 |
| Actual Attack | 0 | 4,344 |

Additional measurements:

- Training time: approximately 4.48 seconds
- Testing inference time: approximately 0.00075 ms per record
- Threshold: `0.5`

### Version 1.1 Advantages

- Better reflects deployment on later traffic.
- Prevents a single temporal window from being split across partitions.
- Includes a validation partition.
- Split boundaries are deterministic and independent of labels.
- Explicit timestamp auditing and integrity assertions reduce silent data leakage.
- Preserves timestamps for auditability while excluding them from model features.
- Produces a window-level audit for reproducibility.

### Version 1.1 Disadvantages

- The dataset has only 25 usable windows.
- The test partition contains only two windows.
- Almost all traffic comes from one meaningful calendar day, so this is not independent-day generalization.
- Complete-window allocation deviates from the requested 70/15/15 proportions.
- Class prevalence changes substantially: attacks are 21.25% of training records but only 5.01% of test records.
- The five suspicious 1970 timestamps remain in training and require domain confirmation.
- Removing duplicate rows may remove legitimate repeated network behavior as well as leakage-prone duplicates.
- Very high metrics may still reflect the limited source data and structured nature of this dataset.

## 5. Generated Files

### Source and application files

- `preprocess_cic.py`: cleaning, duplicate removal, and timestamp preservation.
- `train_model.py`: Version 1 random-split training and evaluation.
- `temporal_evaluation_v1_1.py`: Version 1.1 temporal-window evaluation.
- `app.py`: Flask prediction service.
- `templates/index.html`: manual prediction interface, random value generation, and metric display.
- `requirements.txt`: Python dependencies.
- `.gitignore`: ignores datasets, model binaries, caches, environments, and local Flask state.

### Version 1.1 artifacts

- `artifacts/v1_1_temporal/partition_summary.csv`
- `artifacts/v1_1_temporal/window_audit.csv`
- `artifacts/v1_1_temporal/evaluation_metrics.json`
- `artifacts/v1_1_temporal/execution.log`
- `artifacts/v1_1_temporal/logistic_attack_model_v1_1.joblib`

## 6. Overall Conclusion

Version 1 provides a strong random-split baseline, but its evaluation can be optimistic because similar or duplicate records can cross the train-test boundary. Version 1.1 improves the evaluation design by assigning complete chronological windows to contiguous train, validation, and test partitions.

The main contribution of Version 1.1 is a more realistic temporal evaluation protocol, not a claim that the model is inherently more robust. Because the available data covers almost one meaningful capture day, the results should not be described as independent-day or cross-environment generalization.