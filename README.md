# YallaLayer
### Turn Customer Behavior Into Marketing Wins.

YallaLayer is an AI-powered e-commerce marketing intelligence platform that transforms online shopping session data into meaningful business insights.

Inspired by the mission of **Yalla Start Academy** to support emerging brands through digital marketing education, consultancy, and practical growth strategies, YallaLayer explores how data analysis and machine learning can help businesses understand customer behavior, evaluate conversion opportunities, and make more informed marketing decisions.

## Live Demo

**Run the platform locally:**

```bash
streamlit run app.py
```

The dashboard will open in your browser.

**Public Live Demo:** Coming soon.

A public deployment link will be added once the platform is hosted online.

## The Idea

Emerging businesses need more than raw analytics. They need to understand what influences customer engagement, which visitor behaviors are associated with purchases, and where potential growth opportunities exist.

YallaLayer addresses this challenge through three current capabilities:

- **Marketing Intelligence:** Explore visitor behavior, conversion patterns, and performance across different session segments.
- **Purchase Prediction:** Estimate the likelihood that a browsing session will result in a purchase using a trained machine learning model.
- **Model Explainability:** Understand which features influence model predictions through SHAP-based analysis.

## The Yalla Start Academy Connection

Yalla Start Academy provides marketing consultancy and support to emerging brands, alongside digital marketing courses that help entrepreneurs and businesses strengthen their online presence.

YallaLayer translates this vision into a data-driven experience.

Instead of stopping at displaying numbers, the project aims to connect customer behavior analysis with machine learning insights and, in its next phase, practical marketing recommendations.

**The goal:** Help businesses move from understanding customer behavior to identifying smarter opportunities for growth.

## Platform Features

### 1. Marketing Intelligence

Explore conversion patterns across:

- Monthly performance
- Visitor types
- Traffic source codes
- Weekend vs. weekday sessions
- Product-related browsing activity
- Bounce rates and exit rates

The platform also includes statistical validation to help distinguish meaningful differences from patterns that may occur by chance.

### 2. Purchase Prediction

A Gradient Boosting model estimates purchase probability for an e-commerce browsing session.

The selected operating threshold prioritizes recall, helping identify more potential purchasing sessions while making the precision trade-off visible.

### 3. Model Explainability

SHAP analysis provides insight into the features contributing to model behavior, including PageValues, Month, ExitRates, and VisitorType.

This supports model interpretation rather than treating predictions as unexplained outputs.

## Coming Next: Marketing Recommendations & Action Center

**The next planned phase is to turn analytical findings into practical marketing actions.**

This module will connect observed customer behavior and model insights to prioritized recommendations that emerging brands can investigate and test.

Planned capabilities include:

- Identifying potential conversion improvement opportunities.
- Suggesting experiments for product-page engagement and navigation.
- Highlighting seasonal patterns that may inform campaign planning.
- Prioritizing marketing opportunities using behavioral and predictive insights.
- Presenting suggested actions with supporting evidence and limitations.

The goal is to move from **understanding what is happening** toward **deciding what to test next**.

These recommendations are planned capabilities, not features currently implemented. They will be framed as evidence-informed hypotheses rather than guaranteed outcomes.

## Machine Learning Development Methodology

### Reference Book

The project methodology was guided by:

**Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**  
*Aurélien Géron*

In particular, we used the principles of **Appendix A — Machine Learning Project Checklist** as a structured guide throughout the project.

Rather than jumping directly into model training, we followed an adapted version of the eight-step machine learning workflow:

| Step | Machine Learning Workflow | How We Applied It in YallaLayer |
|---|---|---|
| 1 | Frame the Problem and Look at the Big Picture | Defined the business objective: understand purchase behavior and support better marketing decisions. |
| 2 | Get the Data | Acquired the Online Shoppers Purchasing Intention dataset from the UCI Machine Learning Repository. |
| 3 | Explore the Data to Gain Insights | Performed EDA to investigate distributions, relationships, conversion patterns, and data quality. |
| 4 | Prepare the Data | Validated the data, handled duplicate-session considerations, prepared features, and created training and test splits. |
| 5 | Explore Different Models | Compared candidate machine learning models using cross-validation and appropriate evaluation metrics. |
| 6 | Fine-Tune the Models | Selected and tuned the model, evaluated decision thresholds, and used SHAP for explainability. |
| 7 | Present the Solution | Developed an interactive Streamlit dashboard for marketing intelligence, purchase prediction, and model explainability. |
| 8 | Launch, Monitor, and Maintain | Prepared the application and reproducible workflow. Public deployment and ongoing production monitoring remain future work. |

This checklist was adapted to the project's scope and available dataset. It served as a development framework, not as a claim that every production deployment activity has already been completed.

## Dataset

The project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

| Dataset detail | Value |
|---|---:|
| Total sessions | 12,330 |
| Features | 17 |
| Target | Revenue |
| Training sessions | 9,864 |
| Test sessions | 2,466 |
| Missing values | 0 |
| Duplicate rows identified | 125 |

Each record represents a browsing session, not necessarily a unique customer.

## Machine Learning Results

The selected model is Gradient Boosting, evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| Average Precision (PR-AUC) | 72.70% |
| ROC-AUC | 92.29% |
| Recall at selected threshold | 84.03% |
| Precision at selected threshold | 51.52% |
| Selected threshold | 0.1309 |

The selected threshold prioritizes identifying potential purchasing sessions. This increases recall but also produces more false positives. The appropriate threshold depends on the business cost of missed opportunities versus unnecessary follow-up.

## Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Explainability:** SHAP
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Statistical Analysis:** Statsmodels
- **Dashboard:** Streamlit
- **Model Serialization:** Joblib
- **Development:** Jupyter Notebook, Git, GitHub

## Project Structure

```text
YallaLayer/
├── app.py
├── pages/
│   ├── 2_Marketing_Intelligence.py
│   ├── 3_Purchase_Prediction.py
│   ├── 4_Model_Explainability.py
│   └── 5_About_and_Methodology.py
├── data/
│   ├── raw/
│   ├── splits/
│   └── exploration/
├── models/
│   └── yallalayer_model.joblib
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── reports/
│   ├── figures/
│   ├── marketing/
│   └── metrics/
├── scripts/
│   ├── get_data.py
│   ├── Yalla_analysis.py
│   ├── evaluate_model.py
│   ├── explain_model.py
│   ├── test_marketing_significance.py
│   └── validate_marketing.py
├── src/
│   └── yallalayer/
│       ├── dashboard_utils.py
│       └── train.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ArwaHassan22/YallaLayer.git
cd YallaLayer
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser.

**Note:** Raw datasets and local data splits are excluded from the repository. Reproduce the data preparation workflow to obtain the required data. The dashboard also requires the trained model and generated report artifacts.

## Methodology & Responsible Interpretation

YallaLayer is designed to support informed analysis, not to claim causal marketing impact.

Important considerations:

- Browsing sessions should not be interpreted as unique customers.
- TrafficType values are numeric codes and are not assumed to represent verified marketing channels.
- The dataset does not provide campaign spending, so campaign ROI cannot be calculated.
- Observational patterns do not prove that one behavior causes a purchase.
- PageValues must be available at the intended prediction time before the model can be used operationally.
- Predictions are estimates and should not be treated as guaranteed purchase outcomes.

## Project Context

YallaLayer was developed as a project aligned with **Yalla Start Academy's** focus on helping emerging brands grow through practical digital marketing knowledge and support.

It explores how machine learning and customer behavior analysis can contribute to smarter, evidence-informed marketing decisions.

---

**YallaLayer — From Customer Behavior to Smarter Marketing Decisions.**