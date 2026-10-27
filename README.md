# 🧠 AI-Based Mental Stress Detection

**An Academic Machine Learning System for Multi-Factor Stress Level Classification**  
*B.Tech Artificial Intelligence and Data Science Project*

---

## 📌 1. Project Overview

Mental stress is an increasingly prevalent challenge in modern academic and professional environments. The **AI-Based Mental Stress Detection** system is an educational and early-awareness machine learning platform designed to analyze 20 behavioral, psychological, physical, academic, and environmental factors to classify an individual's stress level into three distinct categories:

* **0 — Low Stress (🟢)**
* **1 — Moderate Stress (🟡)**
* **2 — High Stress (🔴)**

The primary objective is to facilitate proactive self-awareness and provide structured, non-clinical lifestyle guidance based on empirical survey responses.

> [!WARNING]
> **Important Educational & Non-Clinical Disclaimer:**  
> This system is intended for **educational and early-awareness purposes only. It is not a medical diagnosis.** The outputs represent statistical machine learning predictions based on survey data patterns and must not be used as clinical evaluations or psychiatric diagnostic instruments.

---

## 📊 2. Dataset Description

The system utilizes the verified **`StressLevelDataset.csv`** dataset located in the project root.

* **Total Records:** 1,100
* **Total Columns:** 21 (20 Input Features + 1 Target Variable)
* **Missing Values:** 0 (100% complete dataset)
* **Target Column:** `stress_level`

### Target Class Distribution

| Stress Level Class | Meaning | Record Count | Percentage |
| :--- | :--- | :---: | :---: |
| **Class 0** | Low Stress | 373 | 33.91% |
| **Class 1** | Moderate Stress | 358 | 32.55% |
| **Class 2** | High Stress | 369 | 33.55% |
| **Total** | | **1,100** | **100.0%** |

---

## 🗂️ 3. Input Features (20 Factors)

The model evaluates 20 survey features organized into four functional domains:

### 🧠 A. Mental & Emotional Factors (4 Features)
1. `anxiety_level`: Self-reported anxiety survey scale (0 – 21)
2. `self_esteem`: Perceived self-esteem and self-worth scale (0 – 30)
3. `depression`: Self-reported mood/depression factor scale (0 – 27)
4. `mental_health_history`: Binary indicator of personal/family mental health history (0 = No, 1 = Yes)

### 🩺 B. Physical Health Indicators (4 Features)
5. `headache`: Frequency and severity rating of tension headaches (0 – 5)
6. `blood_pressure`: Blood pressure rating in dataset (1 = Low/Normal, 2 = Moderate, 3 = High)
7. `sleep_quality`: Subjective rating of sleep restfulness and regularity (0 – 5)
8. `breathing_problem`: Frequency of stress-related shortness of breath (0 – 5)

### 🎓 C. Academic Factors (4 Features)
9. `academic_performance`: Self-rated academic standing and progress (0 – 5)
10. `study_load`: Perceived coursework volume and academic deadline pressure (0 – 5)
11. `teacher_student_relationship`: Quality of mentorship and rapport with teachers (0 – 5)
12. `future_career_concerns`: Level of anxiety regarding career opportunities and readiness (0 – 5)

### 🌐 D. Social & Environmental Factors (8 Features)
13. `social_support`: Support network available from family and friends (0 – 3)
14. `peer_pressure`: Conformity pressure and social expectations from peers (0 – 5)
15. `extracurricular_activities`: Engagement in sports, cultural activities, and hobbies (0 – 5)
16. `bullying`: Frequency/experience of interpersonal friction or bullying (0 – 5)
17. `noise_level`: Ambient noise disturbance in daily living/study area (0 – 5)
18. `living_conditions`: Perceived comfort, hygiene, and privacy of accommodation (0 – 5)
19. `safety`: Perceived physical and psychological safety in environment (0 – 5)
20. `basic_needs`: Degree to which essential daily supplies and sustenance are met (0 – 5)

---

## 🔬 4. Machine Learning Methodology

The predictive engine employs supervised machine learning trained on an 80/20 stratified split:

* **Training Set:** 880 samples (80%)
* **Testing Set:** 220 samples (20%)
* **Random State:** 42
* **Stratification:** Stratified by `y` (`stress_level`) to preserve class proportions

### Evaluated Algorithms

1. **Logistic Regression:** Multinomial classification with `StandardScaler` normalization and L2 regularization (`max_iter=1000`).
2. **Decision Tree Classifier:** Non-parametric recursive splitting (`random_state=42`).
3. **Random Forest Classifier (Best Model):** Ensemble of 200 bootstrapped decision trees with feature sub-sampling (`n_estimators=200`, `random_state=42`).
4. **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel with `StandardScaler` pipeline and calibrated probability estimates.
5. **XGBoost Classifier:** Extreme Gradient Boosting with 200 estimators, max depth 5, learning rate 0.05, and subsampling ratio 0.8.

---

## 🏆 5. Actual Baseline Model Results

The following benchmark results are obtained directly from the test split (`model_comparison.csv`):

| Algorithm | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **88.64%** | **88.74%** | **88.64%** | **88.61%** | **1 (Best)** |
| **Logistic Regression** | **88.18%** | **88.16%** | **88.18%** | **88.17%** | 2 |
| **XGBoost** | **88.18%** | **88.24%** | **88.18%** | **88.17%** | 3 |
| **SVM** | **87.73%** | **87.73%** | **87.73%** | **87.71%** | 4 |
| **Decision Tree** | **85.45%** | **85.59%** | **85.45%** | **85.42%** | 5 |

* **Best Model:** Random Forest Classifier
* **Best Test Accuracy:** **88.64%**
* **Model Artifact:** `best_stress_model.pkl`

---

## 📈 6. Advanced Evaluation Metrics

### A. Test Confusion Matrix (Random Forest, 220 Samples)

| Actual \ Predicted | Predicted Low (0) | Predicted Moderate (1) | Predicted High (2) | Class Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Low (0)** | **62** | 4 | 8 | 83.8% (62/74) |
| **Actual Moderate (1)** | 2 | **67** | 3 | 93.1% (67/72) |
| **Actual High (2)** | 4 | 4 | **66** | 89.2% (66/74) |

* **Total Correct Test Classifications:** 195 / 220 (88.64%)

### B. Multi-Class ROC-AUC (One-vs-Rest)
* **Random Forest Weighted ROC-AUC:** **0.9831**
  * Class 0 (Low): AUC = 0.9840
  * Class 1 (Moderate): AUC = 0.9792
  * Class 2 (High): AUC = 0.9862
* Demonstrates high separability across all three stress categories.

### C. 5-Fold Stratified Cross-Validation (Additional Experiment)
*Presented as an independent generalization experiment separate from the baseline 88.64% test split accuracy:*

* **Logistic Regression:** 89.09% (± 2.11%)
* **XGBoost:** 88.45% (± 1.17%)
* **Decision Tree:** 87.91% (± 2.20%)
* **Random Forest:** 87.55% (± 1.54%)
* **SVM:** 87.55% (± 1.96%)

### D. Random Forest Feature Importance Ranking
Calculated directly from `best_stress_model.pkl.feature_importances_`:

| Rank | Feature | Importance Weight |
| :---: | :--- | :---: |
| 1 | `blood_pressure` | 15.06% |
| 2 | `sleep_quality` | 7.45% |
| 3 | `academic_performance` | 7.16% |
| 4 | `teacher_student_relationship` | 6.84% |
| 5 | `depression` | 6.56% |
| 6 | `anxiety_level` | 6.30% |
| 7 | `headache` | 6.22% |
| 8 | `basic_needs` | 5.89% |
| 9 | `safety` | 5.84% |
| 10 | `future_career_concerns` | 5.61% |

*(Remaining features contribute the balanced balance across environmental and social domains).*

---

## 🏗️ 7. System & Pipeline Architecture

### End-User Prediction Pipeline
```
[ User Inputs (20 Survey Factors) ]
                ↓
[ Feature Order & Range Validation (feature_names.pkl) ]
                ↓
[ Trained Random Forest Classifier (best_stress_model.pkl) ]
                ↓
[ Multi-Class Probability Computation (predict_proba) ]
                ↓
[ Stress Classification (Low / Moderate / High) ]
                ↓
[ Educational Coping Awareness Guidance & Visual Charts ]
```

### ML Training & Benchmarking Pipeline
```
StressLevelDataset.csv (1,100 Rows, 20 Features)
                ↓
Feature / Target Separation (X, y)
                ↓
Stratified 80/20 Train-Test Split (Seed=42)
                ↓
 ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
 ↓              ↓              ↓              ↓              ↓
Logistic      Decision       Random         SVM           XGBoost
Regression    Tree           Forest
 ↓              ↓              ↓              ↓              ↓
 └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
                ↓
Standardized Evaluation (Accuracy, Precision, Recall, F1)
                ↓
Best Model Selection: Random Forest (88.64%)
                ↓
Serialized Artifacts (best_stress_model.pkl, feature_names.pkl)
                ↓
Offline Interactive Streamlit Dashboard (app.py)
```

---

## 💻 8. Installation & Setup

### Prerequisites
* Python 3.10 to 3.13
* Virtual environment (recommended)

### Installation Steps

1. **Clone or Navigate to the Repository:**
   ```bash
   cd "AI_Mental_Stress_Detection"
   ```

2. **Activate Virtual Environment:**
   * **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Evaluation Artifacts (Optional):**
   ```bash
   python generate_eval_artifacts.py
   ```

5. **Run the Automated Test Suite:**
   ```bash
   python test_app_pipeline.py
   ```

---

## 🚀 9. Running the Application

Launch the Streamlit dashboard locally:

```bash
streamlit run app.py
```

Once launched, access the application in your browser:
* **Local URL:** `http://localhost:8501`
* **Network URL:** Displayed in terminal output.

### Application Pages:
* **🏠 Dashboard:** High-level project KPIs, dataset distribution, and best-model highlight.
* **🧠 Stress Prediction:** Categorized inputs with quick demo preset buttons (Low, Moderate, High, Average), inference confidence, probability breakdown, and coping guidance.
* **📊 Model Performance:** Comparative benchmark table, confusion matrices, ROC curves, feature importances, and 5-fold cross-validation.
* **📁 Dataset Analysis:** Data preview, summary statistics, class counts, and correlation matrix.
* **ℹ️ About Project:** Academic objectives, system architecture, technology stack, and medical disclaimer.

---

## 📁 10. Repository Structure

```
AI_Mental_Stress_Detection/
├── .venv/                         # Python virtual environment
├── StressLevelDataset.csv         # Verified dataset (1,100 records, 20 features)
├── best_stress_model.pkl          # Trained Random Forest classifier (88.64% acc)
├── feature_names.pkl              # Ordered list of 20 feature names
├── model_comparison.csv           # Baseline evaluation metrics for 5 algorithms
├── eval_artifacts.pkl             # Authentic confusion matrices, ROC-AUC, feature importances
├── train_model.py                 # Baseline training and evaluation script
├── generate_eval_artifacts.py     # Evaluation artifact generator
├── test_app_pipeline.py           # Automated test suite
├── app.py                         # Complete 5-page Streamlit web application
├── requirements.txt               # Package dependencies
└── README.md                      # Comprehensive academic documentation
```

---

## 🔮 11. Future Scope & Enhancements

1. **Time-Series Tracking:** Incorporating longitudinal survey monitoring to track stress fluctuations across an academic semester.
2. **Wearable Sensor Integration:** Complementing self-reported survey metrics with non-invasive physiological signals (e.g., HRV, skin conductance).
3. **Advanced Explainable AI (XAI):** Integrating SHAP (SHapley Additive exPlanations) or LIME to provide granular, instance-level local feature explanations.
4. **Targeted Campus Resource Matching:** Directing students showing elevated stress indicators to specific campus counseling and wellness workshops.

---

## ⚖️ 12. Ethical Considerations & Disclaimer

This software is an engineering study on applying supervised machine learning algorithms to student survey responses. **It is not certified as a medical device or diagnostic application.** The predictions generated must never supersede consultation with a qualified medical professional, psychologist, or psychiatrist.
