# 🏦 Universal Bank — Personal Loan Intelligence Hub

A production-grade Streamlit dashboard for marketing analytics, ML modelling, and hyper-personalised loan campaign targeting.

---

## 🚀 Live App

Deploy instantly on [Streamlit Cloud](https://streamlit.io/cloud) by connecting your GitHub repo.

---

## 📋 Features

| Section | Description |
|---|---|
| 📊 **Descriptive Analytics** | KPIs, income distributions, acceptance rates by education, family, products |
| 🔍 **Exploratory Deep Dive** | Correlation heatmap, scatter plots, segment interaction analysis |
| 🤖 **ML Models & Performance** | Decision Tree, Random Forest, Gradient Boosted Tree — accuracy, precision, recall, F1, ROC-AUC, confusion matrices |
| 🎯 **Prescriptive Strategy** | Customer propensity scoring, budget optimisation simulator, persona-based campaign playbook |
| 📁 **Predict New Customers** | Upload CSV → get loan probability scores → download results |

---

## 📦 Setup

### Local Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/universal-bank-loan-dashboard.git
cd universal-bank-loan-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as main file
4. Click **Deploy** — done!

---

## 📁 File Structure

```
universal-bank-loan-dashboard/
├── app.py                  ← Main Streamlit application
├── UniversalBank.csv       ← Training dataset (5,000 customers)
├── sample_test_data.csv    ← Sample file to test predictions
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## 🔬 Models Used

| Model | Notes |
|---|---|
| **Decision Tree** | max_depth=6, class_weight=balanced |
| **Random Forest** | 200 estimators, max_depth=8 |
| **Gradient Boosted Tree** | 200 estimators, lr=0.1, max_depth=4 |

All models trained with **SMOTE oversampling** on the minority class (loan acceptors) to handle the 90/10 class imbalance.

---

## 📊 Dataset Column Reference

| Column | Description |
|---|---|
| Age | Customer age (years) |
| Experience | Professional experience (years) |
| Income | Annual income ($000) |
| ZIP Code | Home ZIP code (dropped in modelling) |
| Family | Family size (1–4) |
| CCAvg | Monthly credit card spend ($000) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced |
| Mortgage | Mortgage value ($000) |
| Personal Loan | **Target** — 1=Accepted, 0=Not Accepted |
| Securities Account | Has bank securities account (0/1) |
| CD Account | Has certificate of deposit account (0/1) |
| Online | Uses internet banking (0/1) |
| CreditCard | Has bank-issued credit card (0/1) |

---

## 🏛️ Built For

**Universal Bank · Marketing Department**  
Campaign: Personal Loan Cross-Sell  
Objective: Maximise acceptance rate within a reduced budget using AI-powered targeting.

---

*Built with Streamlit · Plotly · Scikit-learn · Imbalanced-learn*
