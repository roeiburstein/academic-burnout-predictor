# Project Context: Predicting Academic Burnout in High School Students

## 📌 Project Overview
- **Title (Working)**: *Predicting Academic Burnout in High School Students Using Psychological Variables and Machine Learning*
- **Research Question**: Can we predict which students are at risk for academic burnout using behavioral and psychological data?
- **Objective**: Shift the paradigm from looking at isolated correlations to building, comparing, and evaluating robust predictive models. This project bridges educational psychology, statistics, and machine learning.

---

## 🧮 Core Methodological & Mathematical Concepts
This project combines psychological theory with quantitative data analysis. Key mathematical and scientific frameworks include:

### 1. Statistics & Data Analysis
- **Descriptive Statistics**: Mean, median, standard deviation, and data distributions (histograms/boxplots).
- **Inferential Statistics**: Correlation, $t$-tests, $p$-values, confidence intervals, and multiple regression.
- **Psychometrics**: Use of validated, standardized psychological scales (rather than self-authored questions) to measure stress, sleep quality, perfectionism, parental pressure, and motivation.

### 2. Machine Learning & Predictive Modeling
- **Algorithm Comparison**:
  - Linear Regression (for continuous burnout indices)
  - Logistic Regression (for binary classification: risk vs. no-risk)
  - Decision Trees
  - Random Forests (ensemble learning)
- **Evaluation Metrics**: Confusion matrices, Accuracy, Precision, Recall, F1-Score, and Probability estimates.

### 3. Alternative Cognitive/Behavioral Frameworks (Under consideration)
- **Decision-Making**: Risky decision-making under framing effects ("90% success" vs. "10% failure") compared against **Prospect Theory**.
- **Social Network Analysis**: Friendship network mapping represented as graphs to calculate degree centrality and clustering in relation to student stress.
- **Memory/Forgetting Curves**: Curve fitting (linear vs. exponential decay) for material recall over time.
- **AI & Metacognitive Calibration**: Calibration curves mapping student confidence (0–100%) against actual test accuracy.
- **Attention & Distraction**: Measuring the variance and distribution of reaction times under auditory/visual distractors.

---

## 📅 11-Phase Project Roadmap

```mermaid
gantt
    title Academic Burnout Project Timeline
    dateFormat  YYYY-MM
    section Literature & Stats
    Phase 1: Learn the Science & Lit Review      :active, p1, 2026-06, 2026-07
    Phase 2: Learn Stats & Python ML            :p2, 2026-07, 2026-08
    section Design & Ethics
    Phase 3: Survey Construction (Scales)        :p3, 2026-08, 2026-09
    Phase 4: Ethical Planning & Permissions      :p4, 2026-08, 2026-09
    section Data Collection & Prep
    Phase 5: Data Collection (200-500 students) :p5, 2026-09, 2026-11
    Phase 6: Data Cleaning & Preprocessing      :p6, 2026-10, 2026-11
    section Modeling & Analysis
    Phase 7: Exploratory Data Analysis (EDA)    :p7, 2026-11, 2026-12
    Phase 8: Model Training & Evaluation         :p8, 2026-12, 2027-01
    Phase 9: Results Interpretation              :p9, 2027-01, 2027-02
    section Writing & Presentation
    Phase 10: Academic Thesis Writing           :p10, 2027-02, 2027-04
    Phase 11: Public Presentation/Website       :p11, 2027-04, 2027-06
```

### Phase Details

| Phase | Timeframe | Focus | Deliverables & Tasks |
|---|---|---|---|
| **1. Learn the Science** | June–July | Psychology Lit Review | Read 20–30 peer-reviewed papers on burnout, stress, sleep, self-efficacy, etc. Catalog in a structured spreadsheet. |
| **2. Learn Statistics** | July | Math & Python ML Foundations | Study regression, t-tests, and machine learning basics (scikit-learn: Logistic Regression, Decision Trees, Random Forests). |
| **3. Build the Survey** | August | Survey Design | Select validated psychometric scales to measure burnout (outcome) and predictors (sleep, motivation, parental pressure). |
| **4. Ethical Safeguards**| August | Research Ethics | Design anonymous, voluntary consent flows. Secure teacher/academic supervision. |
| **5. Data Collection** | Sept–Oct | Sourcing Data | Distribute online survey to gather 200–500 student responses from schools and clubs. |
| **6. Clean the Data** | October | Data Preprocessing | Handle duplicates, missing values, incomplete surveys, and impossible outliers. |
| **7. Stats Analysis** | November | Exploratory Analysis (EDA) | Generate correlation matrices, scatterplots, histograms, and boxplots. |
| **8. Build Models** | December | Machine Learning | Train and evaluate ML models (split train/test sets). Compare F1-scores, accuracy, precision, and recall. |
| **9. Interpret Results** | January | Feature Importance | Evaluate variable contributions (e.g., comparing sleep vs. parental pressure vs. homework hours). |
| **10. Write Thesis** | Feb–March | Research Paper | Author standard scientific paper: Abstract, Intro, Lit Review, Methods, Results, Discussion, APA References. |
| **11. Disseminate** | Spring/Summer | Presentation & Outreach | Present findings via research posters, academic competitions, or a personal website. |

---

## ✨ Exceptional Project Extension: Interactive Risk Calculator
To elevate this project beyond standard academic requirements, we plan to design and build a web application:
- **Interactive Predictor UI**: A dashboard where users can input parameters (sleep, study hours, parental pressure, stress, social support).
- **Interactive Risk Engine**: Leverages the trained machine learning model to estimate burnout risk.
- **Ethical Framing**: Emphasizes that the tool is a **research and educational demonstration**, not a diagnostic clinical tool. Discusses the distinction between population-level risk patterns and individual psychiatric diagnosis.

---

## 🛠️ Skills & Experience Acquired
- Research methodology & scientific literature synthesis.
- Survey construction using validated psychometric scales.
- Academic ethics (informed consent, anonymity, data protection).
- Data processing, cleansing, and visualization in Python.
- Advanced machine learning implementation and model evaluation.
- High-level scientific reporting (APA standards).
