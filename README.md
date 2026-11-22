# 💳 Credit Scoring Model

A machine learning project that predicts whether a loan applicant is a good or bad credit risk using past financial data. Built as part of the **CodeAlpha Machine Learning Internship** (June 2026).

## 🔍 Overview

Three classifiers are trained on the German Credit Dataset and compared using ROC-AUC, precision, recall, and F1-score. Logistic Regression uses standardized features; the tree-based models use the raw features.

## 📁 Dataset

**German Credit Dataset** (UCI Machine Learning Repository)
- 1,000 loan applicants with 20 attributes covering financial status, credit history, employment, and personal details
- Target: Good credit (0) or Bad credit (1)
- Loaded directly from the UCI URL in the notebook

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy**: data handling
- **Scikit-learn**: preprocessing, models, evaluation
- **Matplotlib, Seaborn**: visualization

## 🔑 Method

1. **Load and encode.** Categorical columns are label-encoded.
2. **Feature engineering.**
   - `credit_per_duration` = credit amount / (duration + 1), the average monthly credit
   - `age_employment_ratio` = age / employment category
3. **Split.** 80/20 stratified train/test split (`random_state=42`).
4. **Scale.** `StandardScaler` is fit on the training set only, and used for Logistic Regression.
5. **Models:**
   - Logistic Regression (`max_iter=1000`)
   - Decision Tree (`max_depth=5`)
   - Random Forest (`n_estimators=100`)

## 📊 Results

Test set: 200 applicants (140 good, 60 bad).

| Model | ROC-AUC | Accuracy | Bad-credit Precision | Bad-credit Recall | Bad-credit F1 |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.791** | 0.77 | 0.63 | **0.53** | **0.58** |
| Decision Tree (max_depth=5) | 0.682 | 0.72 | 0.55 | 0.47 | 0.50 |
| Random Forest | 0.789 | **0.78** | **0.74** | 0.42 | 0.53 |

**Key takeaways**
- Logistic Regression had the highest ROC-AUC, but Random Forest was nearly tied (a 0.002 difference, which is within noise on 200 test rows).
- Random Forest was more precise on bad credit (0.74 vs 0.63) but missed more high-risk applicants (recall 0.42 vs 0.53). Which model is preferable depends on whether a missed bad loan costs more than wrongly rejecting a good applicant.
- Decision Tree scored lowest, which suggests the depth limit is restrictive.
- Top features in the Random Forest: `credit_amount`, `credit_per_duration`, `checking_account`, `age_employment_ratio`, and `duration`. The two credit-related features share importance because they are closely related.

## ⚠️ Limitations

- The test set has only 200 rows, so differences of about 0.01 between models are not meaningful.
- Label encoding gives nominal categories (e.g., loan purpose) an artificial order. One-hot encoding would suit Logistic Regression better.
- No cross-validation or hyperparameter tuning was performed.
- The class imbalance (about 70% good, 30% bad) was not addressed with class weights or resampling.
- The dataset was collected decades ago and may not reflect current lending.

## 🔮 Future Work

- One-hot encode categorical features
- Add cross-validation and hyperparameter tuning (GridSearchCV)
- Tune the decision threshold to balance bad-credit recall and precision
- Test class weighting or SMOTE for the class imbalance

## 🚀 Usage

```bash
git clone https://github.com/subhasmita4106/CodeAlpha_CreditScoring.git
cd CodeAlpha_CreditScoring
pip install pandas numpy scikit-learn matplotlib seaborn
```

Open `credit_scoring_model.ipynb` and run all cells. The dataset downloads automatically.

## 📚 References

- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

## 📄 License

MIT (see `LICENSE`).

## 👩‍💻 Author

**Subhasmita Panda**
[LinkedIn](https://linkedin.com/in/subhasmitapanda4106) · [GitHub](https://github.com/subhasmita4106)
