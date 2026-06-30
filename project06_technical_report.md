# Precision-Constrained Evaluation for Fraud Detection Under Extreme Class Imbalance: A Systematic Comparison of Five Imbalance-Handling Strategies

**Author:** Muhammed Keita
**Date:** 2026
**Repository:** https://github.com/muhammed-keita-ml/project-06-fraud-detection-pipeline
**Live System:** https://huggingface.co/spaces/muhammed-keita-ml/credit-card-fraud-detector
**Experiment Tracking:** https://dagshub.com/muhammed-keita-ml/project-06-fraud-detection-pipeline

---

## Abstract

Credit card fraud detection presents a canonical class imbalance problem: positive rates of 0.1-2% are typical in production financial systems, and standard evaluation metrics -- accuracy, ROC-AUC, F1 -- are systematically misleading at this scale of imbalance. This report documents the design, systematic experimentation, and production deployment of a fraud detection pipeline evaluated under a precision-constrained recall framework. Thirteen model/strategy combinations were evaluated across five imbalance-handling approaches (baseline, class weighting, random undersampling, SMOTE, and unsupervised Isolation Forest) and three algorithm families (Logistic Regression, Random Forest, XGBoost), with every run logged to MLflow on DagsHub. The winning configuration -- XGBoost with class weighting -- achieved PR-AUC 0.880 and Recall at 90% Precision (Recall@P90) of 0.837. A central finding is that six of thirteen configurations produced Recall@P90 = 0.000 despite acceptable F1 scores, demonstrating that standard metrics are insufficient for operational fraud detection evaluation. A counterintuitive secondary finding -- that the best uncorrected baseline nearly matched the best engineered imbalance strategy on PR-AUC -- raises open questions about when imbalance correction benefits strong learners. This work is motivated by the cost-sensitive learning framework of Elkan (2001) and the precision-recall evaluation methodology of Davis and Goadrich (2006), and raises open questions that motivate the cloud-native deployment infrastructure in Project 07.

---

## 1. Introduction

### 1.1 Problem Context

Fraud detection in financial systems is an asymmetric classification problem in two senses. First, the class distribution is severely imbalanced: in the dataset used here, 492 of 284,807 transactions are fraudulent -- a positive rate of 0.173%. Second, the misclassification costs are asymmetric: a false negative (missed fraud) results in financial loss and potentially a compromised customer account, while a false positive (legitimate transaction flagged as fraud) results in customer friction, declined payment, and analyst review cost.

These two asymmetries interact in ways that make standard evaluation frameworks inadequate. A model achieving 99.83% accuracy by predicting "legitimate" for every transaction catches zero fraud. A model with high raw recall may require flagging so many transactions that no analyst team can operate it at scale.

This motivates a precision-constrained evaluation framework: the relevant question is not "what is the maximum recall?" but "what is the maximum recall achievable while keeping precision at or above an operationally viable threshold?" This project uses Recall@P90 -- the highest recall achievable at precision >= 90% -- as its primary metric, alongside PR-AUC as a threshold-independent summary of the full precision-recall tradeoff.

### 1.2 Engineering and Research Questions Addressed

This project addresses two interrelated questions:

1. **Evaluation methodology:** Does the choice of evaluation metric materially affect which model and strategy combinations are identified as operationally viable for fraud detection?

2. **Imbalance strategy selection:** Among standard class imbalance handling techniques, which produce classifiers that remain viable under a precision constraint -- and do more sophisticated techniques (SMOTE, undersampling) outperform simpler ones (class weighting) on strong learners?

### 1.3 Contributions

- A precision-constrained evaluation framework implementing Recall@P90 as a first-class metric alongside PR-AUC, with explicit justification for the metric choice in terms of operational fraud detection constraints
- A systematic 13-run comparison across five imbalance-handling strategies and three algorithm families, with every run logged to MLflow on DagsHub
- The finding that six of thirteen configurations produced Recall@P90 = 0.000 despite acceptable F1 scores -- a result invisible to standard metric frameworks
- The counterintuitive finding that the best uncorrected baseline (Random Forest, no correction) nearly matched the best engineered strategy (XGBoost + class weighting) on PR-AUC, with higher Recall@P90
- A production-deployed Gradio inference demo and fully reproducible pipeline validated across two independent execution environments (Kaggle and local)
- A recall_at_precision implementation that returns 0.0 (rather than raising an exception) when no threshold achieves the precision constraint -- making operationally unusable models explicitly visible in comparison tables

---

## 2. Related Work

**Davis, J. and Goadrich, M. (2006).** The Relationship Between Precision-Recall and ROC Curves. ICML. Provides the foundational theoretical justification for preferring precision-recall evaluation under class imbalance. Demonstrates formally that PR curves give more informative performance pictures than ROC curves when the negative class substantially outnumbers the positive class -- directly motivating the PR-AUC primary metric used here.

**Elkan, C. (2001).** The Foundations of Cost-Sensitive Learning. IJCAI. Frames class imbalance handling as a special case of asymmetric misclassification costs. The precision constraint in this project is the operational expression of those costs: the cost of a false positive (analyst review, customer friction) establishes the precision floor below which the system becomes operationally unviable regardless of recall. This framing motivates class weighting as theoretically principled rather than merely pragmatically convenient.

**Breck, E., Cai, S., Nielsen, E., Salib, M., and Sculley, D. (2017).** The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. IEEE Big Data. Motivates the pipeline architecture and test suite design. The 10-test pytest suite for the production serving code (Project 07) maps directly to their data, model, and infrastructure test categories. Their framework also informed the decision to treat evaluation metric correctness as a first-class software engineering concern -- the recall_at_precision function is tested explicitly with a case that returns 0.0.

**Grinsztajn, L., Oyallon, E., and Varoquaux, G. (2022).** Why Tree-Based Models Still Outperform Deep Learning on Tabular Data. NeurIPS. Provides context for the counterintuitive finding that default Random Forest nearly matched the best engineered strategy. Their systematic benchmark across 45 tabular datasets demonstrates that tree-based ensembles handle structured data well across a range of conditions -- which predicts that strong learners may require less distributional correction than weaker linear ones.

**Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., et al. (2015).** Hidden Technical Debt in Machine Learning Systems. NeurIPS. Motivates the pipeline modularity and experiment tracking infrastructure. The separation of data loading, preprocessing, imbalance strategy application, model training, and evaluation into distinct src/ modules addresses their taxonomy of ML-specific technical debt.

---

## 3. Methodology

### 3.1 Dataset

- **Source:** ULB Credit Card Fraud Detection Dataset (Kaggle: mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 492 fraudulent (0.173% positive rate)
- **Features:** V1-V28 (PCA-anonymised components), Time (seconds since first transaction), Amount (USD)
- **Target:** Binary -- fraud (1) / legitimate (0)
- **Note:** The PCA anonymisation means feature importance and model explanations cannot be grounded in domain knowledge; evaluation is purely quantitative

### 3.2 Preprocessing Decisions

- **Train/test split:** 80/20 stratified on target. Stratification is essential at 0.173% positive rate -- a random split risks insufficient fraud cases in the test set for reliable metric estimation
- **Scaling:** StandardScaler fitted on training data only, applied to Time and Amount. V1-V28 are already PCA-transformed to zero mean and unit variance; scaling them again would be redundant
- **Scaler persistence:** The fitted scaler is saved alongside the model artifact for use at inference time, preventing training-serving skew in Time and Amount distributions
- **Reproducibility:** Random state fixed at 42 across all experiments

### 3.3 Imbalance Strategies

Five strategies were evaluated:

1. **Baseline:** No correction. Establishes the problem severity and the performance of uncorrected models
2. **Class weighting:** class_weight="balanced" for Logistic Regression and Random Forest; scale_pos_weight = n_negatives/n_positives = 577.29 for XGBoost. No change to training set composition
3. **Random undersampling:** Majority class downsampled to match minority class count using RandomUnderSampler. Training set reduced from ~227,845 to ~788 rows
4. **SMOTE:** Synthetic Minority Oversampling -- synthetic fraud examples generated by interpolating between existing minority-class neighbours in feature space. Training set expanded to ~455,282 rows (balanced)
5. **Isolation Forest:** Unsupervised anomaly detection. No fraud labels used during training. Included to establish the cost of removing label dependency -- relevant for production systems where confirmed fraud labels lag transactions

### 3.4 Model Families

- **Logistic Regression:** Linear baseline; max_iter=1000
- **Random Forest:** Tree ensemble; n_estimators=100
- **XGBoost:** Gradient boosting; eval_metric="aucpr" -- PR-AUC as the training objective metric, consistent with the evaluation framework

### 3.5 Evaluation Framework

Primary metrics:

- **PR-AUC** (average_precision_score): Threshold-independent summary of the precision-recall tradeoff. Preferred over ROC-AUC under class imbalance per Davis and Goadrich (2006)
- **Recall@P90:** Highest recall achievable at precision >= 0.90. Returns 0.0 if no threshold achieves the constraint -- this is the operationally significant signal, not a failure of the implementation

Secondary metrics (logged for reference, not used for model selection):

- Precision, Recall, F1, ROC-AUC, confusion matrix components (TP, TN, FP, FN)

### 3.6 Experiment Tracking

All 13 runs logged to MLflow tracking server on DagsHub. Strategy and model logged as parameters (not metrics) for filterability. The final winning model is registered in the MLflow Model Registry under fraud-xgboost-class-weighted at Production stage.

---

## 4. Results

| Rank | Strategy | Model | PR-AUC | Recall@P90 | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | class_weighted | xgboost | 0.8800 | 0.8367 | 0.8817 | 0.8367 | 0.8586 | 0.9682 |
| 2 | smote | xgboost | 0.8774 | 0.7959 | 0.7311 | 0.8878 | 0.8018 | 0.9792 |
| 3 | smote | random_forest | 0.8741 | 0.8061 | 0.8265 | 0.8265 | 0.8265 | 0.9644 |
| 4 | baseline | random_forest | 0.8734 | 0.8571 | 0.9412 | 0.8163 | 0.8743 | 0.9630 |
| 5 | class_weighted | random_forest | 0.8629 | 0.7857 | 0.9059 | 0.7857 | 0.8415 | 0.9573 |
| 6 | baseline | xgboost | 0.7973 | 0.7857 | 0.8667 | 0.7959 | 0.8298 | 0.9390 |
| 7 | baseline | logistic_regression | 0.7432 | 0.2143 | 0.8289 | 0.6429 | 0.7241 | 0.9559 |
| 8 | smote | logistic_regression | 0.7249 | 0.0000 | 0.0580 | 0.9184 | 0.1092 | 0.9699 |
| 9 | class_weighted | logistic_regression | 0.7159 | 0.0000 | 0.0610 | 0.9184 | 0.1144 | 0.9722 |
| 10 | random_undersampling | random_forest | 0.6953 | 0.0000 | 0.0423 | 0.9184 | 0.0809 | 0.9777 |
| 11 | random_undersampling | logistic_regression | 0.6778 | 0.0000 | 0.0384 | 0.9184 | 0.0738 | 0.9759 |
| 12 | random_undersampling | xgboost | 0.3750 | 0.0000 | 0.0333 | 0.9184 | 0.0644 | 0.9749 |
| 13 | unsupervised | isolation_forest | 0.2180 | 0.0000 | 0.3084 | 0.3367 | 0.3220 | 0.9543 |

**Deployed model:** XGBoost + class weighting -- selected on highest PR-AUC (0.880) under consistent experimental conditions.

### 4.1 Key Findings

**Finding 1: Six configurations are operationally unusable at the 90% precision bar, invisibly so under F1.** Rows 8-13 show Recall@P90 = 0.000 despite raw recall of ~0.92 and F1 scores ranging from 0.064 to 0.322. These models flag the majority of transactions as fraudulent -- precision collapses below any operationally viable level, and no threshold on the probability output can simultaneously achieve >=90% precision and non-zero recall. F1 does not expose this.

**Finding 2: The best uncorrected baseline (Random Forest, no correction) nearly matched the best engineered strategy on PR-AUC (0.873 vs 0.880) and exceeded it on Recall@P90 (0.857 vs 0.837).** The total improvement from the best imbalance strategy over the best uncorrected baseline is 0.007 PR-AUC points. Tree-based ensembles appear to handle this level of imbalance without distributional correction.

**Finding 3: Random undersampling collapsed performance on all three models at the 90% precision bar.** Despite achieving raw recall of ~0.92 -- the highest of any strategy -- all three undersampled models produced Recall@P90 = 0.000. Discarding ~99% of legitimate transactions during training removed the signal needed for the model to discriminate at high precision.

**Finding 4: Class weighting is the dominant strategy for XGBoost.** XGBoost + class weighting ranks first on PR-AUC. The scale_pos_weight parameter directly reweights the gradient boosting loss function for the minority class, which is a more principled correction than distributional resampling for this algorithm.

---

## 5. Discussion

### 5.1 Interpretation

The dominant finding has a concrete operational translation. In a system processing 1 million daily transactions at a 0.17% fraud rate (1,700 daily fraud cases), random undersampling + XGBoost achieves raw recall of 91.8% with precision of 3.3%. This means the system flags approximately 47,500 transactions per day to catch ~1,561 fraudulent ones, generating ~45,939 false positives. At 5 minutes per analyst review, this requires approximately 3,828 analyst-hours per day -- operationally infeasible.

F1 = 0.064 for this configuration does not convey this. PR-AUC = 0.375 is a warning sign, but without Recall@P90 as an explicit metric, the operational consequence is not visible. This supports Elkan's (2001) framing: any evaluation framework that does not enforce the precision constraint obscures the cost structure of the problem.

### 5.2 The Counterintuitive Baseline Finding

The baseline Random Forest achieving higher Recall@P90 than the best engineered strategy is consistent with Grinsztajn et al. (2022). A more precise hypothesis: the baseline Random Forest's high precision (0.941 at Recall@P90 = 0.857) suggests it has learned a tight, conservative decision boundary for fraud. Class weighting loosens this boundary to improve recall, at the cost of precision. In this dataset, the default boundary was already near-optimal for the precision-constrained objective.

### 5.3 Limitations

**PCA anonymisation prevents domain validation.** V1-V28 have no interpretable meaning. Feature importance can be computed but not validated against fraud domain knowledge.

**Single dataset.** The ULB dataset represents European cardholders over two days in 2013. Whether the strategy ranking generalises to different fraud rates, geographies, or transaction types is unknown.

**No temporal validation.** The 80/20 split is random. Fraud patterns evolve over time. A temporally stratified evaluation would provide a more realistic estimate of deployment performance and is the direct motivation for the monitoring infrastructure in Project 07.

**Default scaler statistics in containerised deployment.** In the Project 07 deployment, when the model is loaded from the MLflow Registry without local scaler artifacts, default scaler statistics are used for Time and Amount scaling rather than the fitted scaler. This introduces a modest approximation for transactions with unusual Amount or Time values.

---

## 6. Future Work and Research Directions

**Precision threshold sensitivity analysis.** This project uses P90 as a fixed threshold. A sensitivity analysis varying the threshold from P70 to P99 would characterise how strategy rankings change across operating regimes, and whether some strategies are robust to the threshold choice while others are brittle.

**SMOTE geometry on PCA-transformed features.** SMOTE interpolates between existing minority-class neighbours. V1-V28 are already linearly decorrelated PCA components. Whether interpolation in this transformed space generates samples within the true manifold of fraudulent transactions is unclear. The poor performance of SMOTE + Logistic Regression suggests the synthetic samples may land in regions that confuse the linear boundary.

**Calibration before threshold application.** Tree ensemble probabilities are often poorly calibrated. Platt scaling or isotonic regression applied post-training would make Recall@P90 more meaningful -- "90% precision" would correspond to the actual empirical precision, not just the model's uncalibrated score.

**Imbalance ratio as an experimental variable.** Whether the strategy rankings hold at 0.01% (more extreme imbalance) or 2% (less extreme) is unknown. A synthetic experiment varying the positive rate would characterise how the benefit of each imbalance strategy scales with imbalance severity.

**Integration with drift monitoring.** The Project 07 monitoring infrastructure exposes a fraud_probability_distribution Prometheus metric. Calibrating the relationship between this metric and actual Recall@P90 degradation would provide an early warning system for model performance degradation in production.

---

## 7. Conclusion

This project produced an XGBoost classifier with PR-AUC 0.880 and Recall@P90 0.837, demonstrating that 83.7% of fraud can be caught at 90%+ precision from anonymised transaction features with a 0.173% positive rate.

More significantly, it demonstrated that six of thirteen standard configurations are operationally unusable under a precision constraint -- a result invisible to accuracy, ROC-AUC, and F1 -- and that the best uncorrected baseline nearly matched the best engineered strategy on both primary metrics. These findings have direct implications for evaluation methodology in applied fraud detection research and production system design.

The code is on GitHub. The 13 experiments are on DagsHub. The model is registered in the MLflow Model Registry. Every number in this report is reproducible.

---

## References

Breck, E., Cai, S., Nielsen, E., Salib, M., and Sculley, D. (2017). The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. IEEE International Conference on Big Data, 1123-1132.

Davis, J., and Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. Proceedings of the 23rd International Conference on Machine Learning (ICML), 233-240.

Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI), 973-978.

Grinsztajn, L., Oyallon, E., and Varoquaux, G. (2022). Why Tree-Based Models Still Outperform Deep Learning on Tabular Data. Advances in Neural Information Processing Systems, 35, 507-520.

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. Advances in Neural Information Processing Systems, 28.
