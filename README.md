# Student Dropout and Academic Success Classifier

## a. Problem Statement
Some students drop out of college, and it would help a lot if the college could spot them early and support them. In this project I use student data (admission details, family background, fee status and marks from the first two semesters) to predict whether a student will **Drop out**, stay **Enrolled**, or **Graduate**. I trained five different models on the same data, compared them using six metrics, and built a small web app where anyone can upload test data, pick a model and see how well it performs.

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
<https://github.com/2026ak05119-cloud/student_dropout_classifier>

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
| Logistic Regression | Did the best overall. It has the highest accuracy (76.7%), F1 (0.756) and MCC (0.614). It is a simple model and trains very fast, but it works well on this data. |
| Decision Tree | Accuracy is 72%, which is okay, but its AUC (0.827) is the lowest of all five models. I limited the depth to 8 so it does not just memorise the training data. It is easy to explain, but less accurate than Logistic Regression and SVM. |
| KNN | Accuracy 71.5% and F1 0.699, so it sits in the middle. The data has many columns that became 0/1 columns after encoding, which probably makes "nearest neighbour" less meaningful and hurts KNN here. |
| Naive Bayes | The weakest model. With default settings its accuracy was only 24%. I fixed this by tuning one setting (var_smoothing = 0.1) using cross-validation on the training data only, which raised accuracy to 68.9%. It is still the lowest (MCC 0.487). It treats every feature as independent, which is not true here, for example first and second semester marks are closely related. |
| Support Vector Machine | Very close to Logistic Regression: accuracy 76.1%, MCC 0.603, and the highest AUC (0.8985, only 0.0003 more than Logistic Regression). It takes longer to train and gave no real improvement over the simpler model. |
| **Overall winner for this dataset** | **Logistic Regression.** It is best on accuracy, precision, recall, F1 and MCC. Only about 18% of students are in the "Enrolled" group, so I looked at F1 and MCC and not just accuracy, and Logistic Regression leads on those too. SVM is a close second. |

## Streamlit App
- **Live app:** <https://studentdropoutclassifier-5gxenasqqhyg2qurs4jyy4.streamlit.app/>
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
