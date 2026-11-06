# Student Dropout and Academic Success Classifier

## a. Problem Statement
Universities lose students to dropout, and the earlier a struggling student is identified, the more time there is to help them. This project builds and compares five classification models that predict, for each student, one of three outcomes at the end of the normal course duration: **Dropout**, **Enrolled** or **Graduate**. The inputs are enrollment details, demographic and socio-economic factors, and first- and second-semester academic performance. This is a three-class (multi-class) problem with imbalanced classes. A Streamlit app lets anyone upload a test CSV, choose a trained model and inspect its performance.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository, *Predict Students' Dropout and Academic Success* (https://archive.ics.uci.edu/dataset/697)
- **Instances:** 4,424 students
- **Features:** 36 input features (mix of integer, real and categorical) plus 1 target column
- **Target column:** `Target` with classes Graduate (2,209), Dropout (1,421) and Enrolled (794), i.e. roughly 50% / 32% / 18%
- **Feature groups:** demographics (marital status, gender, age at enrollment, nationality), application details (application mode, order, course, attendance), parents' qualification and occupation, financial indicators (debtor, tuition fees up to date, scholarship), first- and second-semester curricular units (credited, enrolled, evaluated, approved, grade), and macro-economic indicators (unemployment rate, inflation rate, GDP).
- **Preprocessing:**
  - Raw file was semicolon-separated; saved as a comma-separated CSV with column names stripped of hidden whitespace.
  - Nine integer-coded columns that are really categories (marital status, application mode, course, previous qualification, nationality, and mother's/father's qualification and occupation) are one-hot encoded; remaining numeric columns use median imputation and standard scaling (scaling skipped for the decision tree).
  - Stratified 80/20 train/test split with `random_state=42`, so the test set (885 rows) keeps the class proportions.
  - Precision, Recall and F1 are weighted averages across the three classes; AUC is one-vs-rest weighted.

## c. GitHub Repository Link
<https://github.com/2026ak05119-cloud/student_dropout_classifier/tree/main>

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7672 | 0.8982 | 0.7534 | 0.7672 | 0.7563 | 0.6138 |
| Decision Tree | 0.7198 | 0.8272 | 0.7233 | 0.7198 | 0.7198 | 0.5433 |
| KNN | 0.7153 | 0.8377 | 0.6993 | 0.7153 | 0.6988 | 0.5238 |
| Naive Bayes | 0.6893 | 0.8302 | 0.6856 | 0.6893 | 0.6837 | 0.4865 |
| Support Vector Machine | 0.7605 | 0.8985 | 0.7487 | 0.7605 | 0.7484 | 0.6025 |

Settings: Logistic Regression (`max_iter=2000`), Decision Tree (`max_depth=8`), KNN (`n_neighbors=7`), Gaussian Naive Bayes (`var_smoothing=0.1`, chosen by 5-fold cross-validation on the training set), SVM (RBF kernel, `probability=True`).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best model on accuracy (0.7672), precision, recall, F1 (0.7563) and MCC (0.6138). A linear decision boundary on scaled and one-hot encoded features is enough for this data, and it is fast and easy to interpret. |
| Decision Tree | Accuracy 0.7198 and the lowest AUC (0.8272) among the strong models. A single tree with depth limited to 8 gives coarse probability estimates, which hurts AUC, and it is more sensitive to the particular training sample than the linear models. Its strength is interpretability. |
| KNN | Accuracy 0.7153 and weighted F1 0.6988, the lowest F1 of the first four models. Distance-based voting degrades with many one-hot columns and with the imbalanced classes, since the larger classes dominate the neighbourhood votes. |
| Naive Bayes | With default settings it scored only 0.2441 accuracy. After tuning `var_smoothing` to 0.1 with cross-validation on the training set it reached 0.6893 accuracy and 0.4865 MCC, the weakest of the five. Its assumption that features are independent is a poor fit here, because first- and second-semester measures are strongly related. |
| Support Vector Machine | Very close second: accuracy 0.7605, MCC 0.6025, and the highest AUC (0.8985, only 0.0003 above Logistic Regression). The RBF kernel captures non-linear structure, but it brings no real gain over the linear model on this data and is slower to train. |
| **Overall winner for this dataset** | **Logistic Regression.** It leads on accuracy, precision, recall, F1 and MCC, and ties for the best AUC. Because the Enrolled class is only about 18% of the data, weighted F1 and MCC matter more than accuracy alone, and it wins on both. SVM is a close runner-up. |

## Streamlit App
- **Live app:** <>
- **Features:** CSV upload (test data only), model selection dropdown, evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC), confusion matrix, classification report, and a comparison of all five models on the uploaded file. Models are loaded from the `model/` folder and are never retrained in the app.

## Repository Structure
```
├── app.py                                   # Streamlit app
├── metrics_utils.py                         # metric calculations shared by notebook and app
├── student_dropout_classification.ipynb     # training and evaluation notebook
├── requirements.txt
├── test_data.csv                            # held-out test set (885 rows) to upload in the app
├── model/                                   # saved models, label encoder, metadata, metrics table
└── README.md
```

## How to Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```
Then upload `test_data.csv` in the sidebar.
