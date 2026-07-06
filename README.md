# Undergraduate Burnout Predictive Modeling & Risk Estimator

An end-to-end data science and machine learning research project exploring the psychological and academic behavioral factors that contribute to academic burnout in undergraduate/university students.

The project features a clean data processing and training pipeline in Python, coupled with a responsive, explainable Web interface that estimates individual student burnout risk dynamically using the trained model assets.

---

## 📁 Project Structure

```
Emma-project/
├── README.md                  # Project overview and usage guidelines
├── requirements.txt           # Python dependencies
├── run_pipeline.sh            # Automated pipeline script (EDA -> training)
├── .gitignore                 # Excludes large CSVs, caches, environments
├── context.md                 # Project timeline, context, and outline
├── data/                      # Data storage directory (git-ignored)
│   ├── student_mental_health_burnout.csv  # Raw student records
│   └── cleaned_burnout_data.csv           # Cleaned and processed dataset
├── src/                       # Python data science scripts
│   ├── eda.py                 # Cleaning, stats calculations (t-tests), plotting
│   └── train.py               # Model cross-validation, tuning, and asset export
└── web/                       # Web application files
    ├── index.html             # Interactive risk calculator UI
    ├── analysis.html          # Research findings & insights presentation
    ├── css/
    │   └── styles.css         # Shared CSS design tokens, nav, footer
    ├── js/
    │   ├── calculator.js      # Dynamic model coefficients loading & risk engine
    │   └── lightbox.js        # Responsive image gallery lightbox handler
    ├── model_assets.json      # Model coefficients and scaling parameters (exported)
    ├── stats_data.json        # Inferential statistics results (exported)
    └── plots/                 # Visual plots, ROC, and PR curves (generated)
```

---

## 🚀 Setting Up the Environment

### 1. Requirements & Dependencies
This project requires Python 3.9+ and the package dependencies listed in `requirements.txt`. Install them using `pip`:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Running the Data Science Pipeline

We provide an automated shell script to run the full pipeline in one command. It executes the exploratory analysis, runs statistical tests, tunes the machine learning classifiers, and updates all web assets:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### What happens under the hood?
1. **`src/eda.py`**:
   - Loads the raw data.
   - Cleans records (drops duplicates and missing values).
   - Injects a covariance structure into a latent score to assign categories (`Low`, `Medium`, `High` burnout) for synthetic data demonstration.
   - Performs independent two-sample t-tests and Cohen's d effect sizes for all numerical factors comparing the Low and High burnout cohorts.
   - Performs Chi-squared tests of independence for categorical fields.
   - Writes computed statistics to `web/stats_data.json`.
   - Generates distribution, correlation, and boxplots, saving them to `web/plots/`.
2. **`src/train.py`**:
   - Encodes categorical variables and scales numerical features.
   - Performs hyperparameter grid optimization using Stratified CV on Logistic Regression, Decision Tree, and Random Forest.
   - Addresses class imbalance using class weights (`class_weight='balanced'`).
   - Calculates Stratified 5-Fold Cross-Validation metrics and computes Permutation Feature Importances.
   - Generates diagnostic performance charts (ROC Curve and Precision-Recall Curve), saving them to `web/plots/`.
   - Exports the intercept, standardizer scaling parameters, best parameters, and test performance metrics to `web/model_assets.json`.

---

## 🌐 Running the Web Application

The frontend dashboard fetches the exported model coefficients and statistics files dynamically at runtime. 

To run the web application locally, start a local HTTP server from the root of the project:

```bash
# Start server using Python
python3 -m http.server --directory web 8000
```

Now open your browser and navigate to:
```
http://localhost:8000
```

---

## 🧠 Methodology & Academic Notes

### 1. The Synthetic Data Circularity Limitation
The raw dataset contains initial variables with near-zero correlations. To provide a realistic study structure, `eda.py` injects a mathematical covariance formula. Consequently, the model coefficients (e.g., Academic Pressure = +0.59) represent a successful recovery of this pre-defined formula rather than raw empirical discovery. A discussion of this limitation is included in the Research & Insights page.

### 2. Standardized Scaling
Input variables are scaled dynamically using parameters exported by `train.py` before running model evaluations in JavaScript:
$$x_{scaled} = \frac{x - \mu}{\sigma}$$
This aligns the calculator inputs with the scaled features the ML model was trained on, avoiding manual scaling mismatch errors.
