import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank · Loan Intelligence Hub",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #f0f9ff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #93c5fd;
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(56,189,248,0.2);
    border: 1px solid rgba(56,189,248,0.4);
    color: #7dd3fc;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.25rem;
}
.section-sub {
    font-size: 0.88rem;
    color: #64748b;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

.kpi-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
.kpi-label { font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.4rem; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1; }
.kpi-delta { font-size: 0.8rem; margin-top: 0.4rem; color: #10b981; font-weight: 500; }
.kpi-delta.red { color: #ef4444; }

.insight-box {
    background: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0 1.4rem 0;
    font-size: 0.87rem;
    color: #0c4a6e;
    line-height: 1.65;
}
.insight-box.amber {
    background: #fffbeb;
    border-left-color: #f59e0b;
    color: #78350f;
}
.insight-box.green {
    background: #f0fdf4;
    border-left-color: #22c55e;
    color: #14532d;
}
.insight-box.purple {
    background: #faf5ff;
    border-left-color: #a855f7;
    color: #4a044e;
}

.divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 2rem 0;
}

.tab-section {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
}

.model-card {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border-radius: 14px;
    padding: 1.4rem;
    color: white;
    text-align: center;
}
.model-card .model-name { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #7dd3fc; margin-bottom: 0.6rem; }
.model-card .model-score { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; color: #f0f9ff; }
.model-card .model-label { font-size: 0.75rem; color: #94a3b8; }

.download-btn {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    padding: 0.7rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    text-decoration: none;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("UniversalBank.csv")

    # Fix negative Experience
    df["Experience"] = df["Experience"].clip(lower=0)

    # Drop ID and ZIP Code
    df_model = df.drop(columns=["ID", "ZIP Code"])

    # Features & Target
    X = df_model.drop(columns=["Personal Loan"])
    y = df_model["Personal Loan"]

    return df, df_model, X, y


@st.cache_data
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance with SMOTE on training set only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced", n_jobs=-1),
        "Gradient Boosted Tree": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        trained[name] = model

        y_pred_train = model.predict(X_train_res)
        y_pred_test  = model.predict(X_test)
        y_prob_test  = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "train_acc": accuracy_score(y_train_res, y_pred_train),
            "test_acc":  accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test, zero_division=0),
            "recall":    recall_score(y_test, y_pred_test, zero_division=0),
            "f1":        f1_score(y_test, y_pred_test, zero_division=0),
            "cm":        confusion_matrix(y_test, y_pred_test),
            "fpr":       roc_curve(y_test, y_prob_test)[0],
            "tpr":       roc_curve(y_test, y_prob_test)[1],
            "auc":       auc(*roc_curve(y_test, y_prob_test)[:2]),
            "y_pred":    y_pred_test,
            "y_prob":    y_prob_test,
        }

    return trained, results, X_train, X_test, y_train, y_test


df, df_model, X, y = load_and_preprocess()
trained_models, model_results, X_train, X_test, y_train, y_test = train_models(X, y)

# Color palette
PALETTE = {
    "primary":   "#0ea5e9",
    "secondary": "#6366f1",
    "accent":    "#f59e0b",
    "success":   "#10b981",
    "danger":    "#ef4444",
    "dark":      "#0f172a",
    "models":    ["#0ea5e9", "#10b981", "#f59e0b"],
}

PLOTLY_LAYOUT = dict(
    font_family="Inter",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=20),
    title_font=dict(size=15, family="Syne", color="#0f172a"),
)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <div class="badge">🏦 Universal Bank · Marketing Intelligence</div>
  <h1>Personal Loan Intelligence Hub</h1>
  <p>From Descriptive Analytics → Predictive Modelling → Prescriptive Campaign Strategy</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["📊 Descriptive Analytics",
         "🔍 Exploratory Deep Dive",
         "🤖 ML Models & Performance",
         "🎯 Prescriptive Strategy",
         "📁 Predict New Customers"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset Summary**")
    st.markdown(f"- **Customers:** {len(df):,}")
    st.markdown(f"- **Features:** {X.shape[1]}")
    st.markdown(f"- **Loan Acceptors:** {y.sum():,} ({y.mean()*100:.1f}%)")
    st.markdown("---")
    st.caption("Universal Bank · Loan Campaign Analytics v1.0")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Descriptive Analytics":

    st.markdown('<div class="section-title">📊 Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A bird\'s eye view of the customer base — who they are, what they earn, and how they behave with financial products.</div>', unsafe_allow_html=True)

    # KPI Row
    loan_rate      = df["Personal Loan"].mean() * 100
    avg_income     = df["Income"].mean()
    avg_ccavg      = df["CCAvg"].mean()
    cd_loan_rate   = df[df["CD Account"]==1]["Personal Loan"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Total Customers",   f"{len(df):,}",           "Full dataset size",                 ""),
        (c2, "Loan Acceptors",    f"{df['Personal Loan'].sum():,}", f"{loan_rate:.1f}% acceptance rate", ""),
        (c3, "Avg Annual Income", f"${avg_income:.0f}K",    "Across all customers",              ""),
        (c4, "Avg CC Spending",   f"${avg_ccavg:.2f}K",     "Monthly credit card avg",           ""),
        (c5, "CD Account Loan Rate", f"{cd_loan_rate:.1f}%","Among CD account holders",          ""),
    ]
    for col, label, val, delta, cls in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-delta {cls}">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 — Target Distribution + Age Distribution
    col1, col2 = st.columns(2)

    with col1:
        counts = df["Personal Loan"].value_counts().reset_index()
        counts.columns = ["Loan", "Count"]
        counts["Label"] = counts["Loan"].map({0: "Not Accepted", 1: "Accepted"})
        counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)

        fig = go.Figure(go.Pie(
            labels=counts["Label"],
            values=counts["Count"],
            hole=0.55,
            marker_colors=[PALETTE["danger"], PALETTE["success"]],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            textfont_size=13,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Personal Loan — Acceptance Distribution",
                          legend=dict(orientation="h", y=-0.05))
        fig.add_annotation(text=f"<b>{loan_rate:.1f}%</b><br>Accepted", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="#0f172a"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">Only <b>9.6% of customers</b> accepted the personal loan during the last campaign. This heavy class imbalance means a targeted, data-driven approach is critical — blasting the entire base wastes 90% of your budget.</div>', unsafe_allow_html=True)

    with col2:
        fig = px.histogram(df, x="Age", nbins=30, color_discrete_sequence=[PALETTE["primary"]],
                           title="Customer Age Distribution",
                           labels={"Age": "Age (years)", "count": "Number of Customers"})
        fig.update_traces(marker_line_color="white", marker_line_width=0.5)
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">Customers are broadly spread between ages <b>23–67</b>, with a near-uniform distribution peaking around <b>45 years</b>. This wide spread means age alone is not a decisive filter — combine it with income and education for sharper targeting.</div>', unsafe_allow_html=True)

    # Row 2 — Income + Education
    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(df, x="Income", color="Personal Loan",
                           barmode="overlay", nbins=40,
                           color_discrete_map={0: "#cbd5e1", 1: PALETTE["success"]},
                           title="Income Distribution by Loan Acceptance",
                           labels={"Income": "Annual Income ($000)", "Personal Loan": "Loan Accepted"})
        fig.update_traces(marker_line_width=0.3, marker_line_color="white", opacity=0.85)
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        newnames = {0: "Not Accepted", 1: "Accepted"}
        fig.for_each_trace(lambda t: t.update(name=newnames.get(int(t.name), t.name)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box amber"><b>Income is the #1 differentiator.</b> Loan acceptors have a mean income of <b>$144.7K</b> vs <b>$66.2K</b> for non-acceptors — a 2.2× gap. Almost no one earning under $60K accepted the loan. Income-based segmentation alone can eliminate ~60% of low-probability prospects.</div>', unsafe_allow_html=True)

    with col4:
        edu_map = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"}
        df_edu = df.copy()
        df_edu["Education Label"] = df_edu["Education"].map(edu_map)
        edu_rate = df_edu.groupby("Education Label")["Personal Loan"].agg(["mean","count"]).reset_index()
        edu_rate.columns = ["Education", "Acceptance Rate", "Count"]
        edu_rate["Acceptance Rate %"] = (edu_rate["Acceptance Rate"] * 100).round(1)

        fig = px.bar(edu_rate, x="Education", y="Acceptance Rate %",
                     text="Acceptance Rate %",
                     color="Acceptance Rate %",
                     color_continuous_scale=["#bfdbfe","#0284c7"],
                     title="Loan Acceptance Rate by Education Level",
                     labels={"Education": "Education Level"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, showlegend=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", ticksuffix="%")
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">Graduate and Advanced/Professional customers accept loans at <b>3× the rate of undergraduates</b> (13% vs 4.4%). Higher education correlates with greater financial literacy and confidence in managing loan products — prioritise this segment.</div>', unsafe_allow_html=True)

    # Row 3 — Family + CD Account
    col5, col6 = st.columns(2)

    with col5:
        fam_rate = df.groupby("Family")["Personal Loan"].mean().reset_index()
        fam_rate["Acceptance Rate %"] = (fam_rate["Personal Loan"] * 100).round(1)
        fig = px.bar(fam_rate, x="Family", y="Acceptance Rate %",
                     text="Acceptance Rate %",
                     color="Acceptance Rate %",
                     color_continuous_scale=["#d1fae5","#059669"],
                     title="Acceptance Rate by Family Size",
                     labels={"Family": "Family Size"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, showlegend=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", ticksuffix="%")
        fig.update_xaxes(showgrid=False, tickmode="linear")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box green">Families with <b>3–4 members</b> show the highest loan acceptance rate. Larger families likely have greater financial needs (education, home improvements, emergencies), making them more receptive to personal loan offers.</div>', unsafe_allow_html=True)

    with col6:
        product_data = {
            "Product": ["Securities Account", "CD Account", "Online Banking", "Credit Card (Bank)"],
            "Has Product": [df["Securities Account"].mean()*100, df["CD Account"].mean()*100,
                            df["Online"].mean()*100, df["CreditCard"].mean()*100],
            "Loan Rate (with)": [
                df[df["Securities Account"]==1]["Personal Loan"].mean()*100,
                df[df["CD Account"]==1]["Personal Loan"].mean()*100,
                df[df["Online"]==1]["Personal Loan"].mean()*100,
                df[df["CreditCard"]==1]["Personal Loan"].mean()*100,
            ],
        }
        df_prod = pd.DataFrame(product_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="% Holding Product", x=df_prod["Product"], y=df_prod["Has Product"],
                             marker_color="#bfdbfe", text=df_prod["Has Product"].round(1),
                             texttemplate="%{text:.1f}%", textposition="outside"))
        fig.add_trace(go.Bar(name="Loan Acceptance Rate (holders)", x=df_prod["Product"], y=df_prod["Loan Rate (with)"],
                             marker_color=PALETTE["primary"], text=df_prod["Loan Rate (with)"].round(1),
                             texttemplate="%{text:.1f}%", textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                          title="Cross-Product Holding vs Loan Acceptance",
                          yaxis_ticksuffix="%",
                          legend=dict(orientation="h", y=-0.12))
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box purple"><b>CD Account holders are 6× more likely</b> to accept a personal loan (46.4%) despite making up only 6% of customers. These engaged, savings-oriented customers are your highest-value cross-sell targets.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · EXPLORATORY DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Exploratory Deep Dive":

    st.markdown('<div class="section-title">🔍 Exploratory Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Deeper cross-dimensional analysis to uncover hidden patterns and segment interactions that drive loan acceptance.</div>', unsafe_allow_html=True)

    # Correlation Heatmap
    st.markdown("#### Feature Correlation Matrix")
    corr = df_model.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}",
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Pairwise Feature Correlation Heatmap",
                      height=520, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box">Income shows the <b>strongest positive correlation with Personal Loan (0.50)</b>. Age and Experience are near-perfectly correlated (0.99), confirming redundancy. CCAvg also correlates moderately with both Income and Loan acceptance — spending behaviour mirrors financial capacity.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Income vs CCAvg scatter
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Income vs CC Spending by Loan Status")
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig = px.scatter(sample, x="Income", y="CCAvg",
                         color=sample["Personal Loan"].map({0:"Not Accepted",1:"Accepted"}),
                         color_discrete_map={"Not Accepted":"#cbd5e1","Accepted":PALETTE["success"]},
                         opacity=0.65, size_max=7,
                         labels={"Income":"Annual Income ($000)","CCAvg":"Monthly CC Spend ($000)"},
                         title="Income vs Credit Card Spending (Sample 2,000)")
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">Loan acceptors cluster in the <b>upper-right quadrant</b> — high income AND high credit card spending. Customers spending &gt;$3K/month on credit cards with income &gt;$100K are the sweet spot for targeted loan campaigns.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Mortgage Distribution by Loan Status")
        df_m = df.copy()
        df_m["Loan Label"] = df_m["Personal Loan"].map({0:"Not Accepted",1:"Accepted"})
        df_m_nonzero = df_m[df_m["Mortgage"] > 0]
        fig = px.box(df_m_nonzero, x="Loan Label", y="Mortgage",
                     color="Loan Label",
                     color_discrete_map={"Not Accepted":"#cbd5e1","Accepted":PALETTE["success"]},
                     points="outliers",
                     title="Mortgage Value Among Customers with Mortgages",
                     labels={"Mortgage":"Mortgage Value ($000)","Loan Label":""})
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box amber">Among customers who have mortgages, loan acceptors show <b>similar median mortgage values</b> to non-acceptors. This suggests mortgage holders are not inherently more or less likely to take a loan — income remains the dominant factor even within this group.</div>', unsafe_allow_html=True)

    # Education × Income heatmap
    st.markdown("#### Loan Acceptance Rate — Education × Income Bands")
    df_ei = df.copy()
    df_ei["Income Band"] = pd.cut(df_ei["Income"], bins=[0,50,80,120,160,230],
                                   labels=["<50K","50–80K","80–120K","120–160K","160K+"])
    df_ei["Edu Label"] = df_ei["Education"].map({1:"Undergrad",2:"Graduate",3:"Adv/Prof"})
    pivot = df_ei.groupby(["Edu Label","Income Band"], observed=True)["Personal Loan"].mean().unstack() * 100

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index,
        colorscale="Blues", zmin=0, zmax=100,
        text=pivot.values.round(1),
        texttemplate="%{text:.1f}%",
        hovertemplate="Education: %{y}<br>Income: %{x}<br>Acceptance: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Acceptance Rate (%) — Education × Income Band",
                      height=320, xaxis_title="Income Band", yaxis_title="Education Level")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box green">The highest acceptance rates occur at the intersection of <b>Graduate/Adv. Education + Income $120K+</b>, with rates exceeding 50% in some bands. This is your <b>primary target persona</b> — educated, high-income customers who understand and can afford loan products.</div>', unsafe_allow_html=True)

    # CCAvg bins
    st.markdown("#### Acceptance Rate by Credit Card Spending Bands")
    df_cc = df.copy()
    df_cc["CC Band"] = pd.cut(df_cc["CCAvg"], bins=[-0.1,0.5,1.5,3.0,5.0,10.1],
                               labels=["<$0.5K","$0.5–1.5K","$1.5–3K","$3–5K","$5K+"])
    cc_rate = df_cc.groupby("CC Band", observed=True)["Personal Loan"].agg(["mean","count"]).reset_index()
    cc_rate.columns = ["CC Spend Band","Rate","Count"]
    cc_rate["Rate %"] = (cc_rate["Rate"]*100).round(1)

    fig = px.bar(cc_rate, x="CC Spend Band", y="Rate %", text="Rate %",
                 color="Rate %", color_continuous_scale=["#e0f2fe","#0284c7"],
                 title="Loan Acceptance Rate by Monthly CC Spending Band",
                 labels={"CC Spend Band":"Monthly CC Spending","Rate %":"Acceptance Rate (%)"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, height=380)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", ticksuffix="%")
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box purple">Customers spending <b>$3K+ per month</b> on credit cards show dramatically higher loan acceptance rates — up to 4× compared to low spenders. High CC spending is a reliable proxy for financial engagement and credit comfort.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · ML MODELS & PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 ML Models & Performance":

    st.markdown('<div class="section-title">🤖 ML Models & Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Three classification algorithms trained with SMOTE oversampling to address class imbalance. Evaluate accuracy, precision, recall, F1, ROC-AUC and confusion matrices.</div>', unsafe_allow_html=True)

    model_colors = dict(zip(model_results.keys(), PALETTE["models"]))

    # ── Model Score Cards
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for i, (name, res) in enumerate(model_results.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="model-card">
              <div class="model-name">{name}</div>
              <div class="model-score">{res['test_acc']*100:.1f}%</div>
              <div class="model-label">Test Accuracy</div>
              <div style="margin-top:0.8rem; font-size:0.82rem; color:#94a3b8;">
                AUC: <span style="color:#7dd3fc; font-weight:700;">{res['auc']:.3f}</span> &nbsp;|&nbsp;
                F1: <span style="color:#7dd3fc; font-weight:700;">{res['f1']:.3f}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics Table
    st.markdown("#### 📋 Full Model Comparison Table")
    rows = []
    for name, res in model_results.items():
        rows.append({
            "Model": name,
            "Train Accuracy": f"{res['train_acc']*100:.2f}%",
            "Test Accuracy":  f"{res['test_acc']*100:.2f}%",
            "Precision":      f"{res['precision']*100:.2f}%",
            "Recall":         f"{res['recall']*100:.2f}%",
            "F1 Score":       f"{res['f1']*100:.2f}%",
            "ROC-AUC":        f"{res['auc']:.4f}",
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)
    st.markdown('<div class="insight-box">Gradient Boosted Trees typically achieve the best generalisation (highest AUC & F1). High training accuracy with lower test accuracy in Decision Tree indicates mild overfitting — use ensemble methods for production scoring. All models were trained on SMOTE-balanced data to handle the 90/10 class imbalance.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Single ROC Curve
    st.markdown("#### 📈 ROC Curves — All Models (Single Chart)")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color="#cbd5e1", width=1.5),
                                  name="Random Classifier (AUC=0.50)", showlegend=True))
    for name, res in model_results.items():
        fig_roc.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"], mode="lines",
            line=dict(color=model_colors[name], width=2.5),
            name=f"{name} (AUC = {res['auc']:.3f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))
    fig_roc.update_layout(
        **PLOTLY_LAYOUT,
        title="ROC Curve Comparison — Decision Tree vs Random Forest vs Gradient Boosted Tree",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR / Recall)",
        height=480,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0,1]),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0,1]),
        legend=dict(x=0.55, y=0.15, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#e2e8f0", borderwidth=1)
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown('<div class="insight-box">The ROC curve shows the trade-off between catching true loan acceptors (TPR) vs incorrectly flagging non-acceptors (FPR). A model with AUC &gt; 0.90 means it can rank-order customers effectively — send offers to the top-ranked customers for maximum ROI on your reduced marketing budget.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Confusion Matrices
    st.markdown("#### 🔢 Confusion Matrices — All Models")
    cm_cols = st.columns(3)
    for i, (name, res) in enumerate(model_results.items()):
        with cm_cols[i]:
            cm = res["cm"]
            total = cm.sum()
            # Labels with counts AND percentages
            z_text = [
                [f"{cm[0,0]}<br>({cm[0,0]/total*100:.1f}%)", f"{cm[0,1]}<br>({cm[0,1]/total*100:.1f}%)"],
                [f"{cm[1,0]}<br>({cm[1,0]/total*100:.1f}%)", f"{cm[1,1]}<br>({cm[1,1]/total*100:.1f}%)"],
            ]
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted: No Loan","Predicted: Loan"],
                y=["Actual: No Loan","Actual: Loan"],
                colorscale=[[0,"#f0f9ff"],[0.5,"#38bdf8"],[1.0,"#0c4a6e"]],
                text=z_text,
                texttemplate="%{text}",
                showscale=False,
                hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Count: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                **PLOTLY_LAYOUT,
                title=f"{name}",
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(side="bottom"),
            )
            fig_cm.update_xaxes(tickangle=-15)
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="insight-box amber"><b>Reading the matrix:</b> Top-left = True Negatives (correctly identified non-takers). Bottom-right = True Positives (correctly identified loan takers). Bottom-left = False Negatives (missed opportunities — customers who would take but weren\'t targeted). Top-right = False Positives (wasted campaign spend). For marketing, <b>minimising False Negatives</b> is key — you\'d rather send an extra offer than miss a willing customer.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Feature Importance (RF & GBT)
    st.markdown("#### 🏆 Feature Importance — Random Forest & Gradient Boosted Tree")
    col_fi1, col_fi2 = st.columns(2)
    for col_fi, mname in zip([col_fi1, col_fi2], ["Random Forest", "Gradient Boosted Tree"]):
        model = trained_models[mname]
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
        fig_fi = px.bar(importances, orientation="h",
                        color=importances.values,
                        color_continuous_scale=["#bfdbfe","#0284c7"],
                        title=f"Feature Importance — {mname}",
                        labels={"value":"Importance Score","index":"Feature"})
        fig_fi.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, height=400)
        fig_fi.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig_fi.update_yaxes(showgrid=False)
        with col_fi:
            st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown('<div class="insight-box green"><b>Income consistently ranks as the #1 most important feature</b> across both ensemble models, followed by CCAvg and Education. CD Account — despite being rare — carries disproportionate predictive weight. These are your campaign targeting pillars.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 · PRESCRIPTIVE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Prescriptive Strategy":

    st.markdown('<div class="section-title">🎯 Prescriptive Strategy</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Data-driven recommendations to maximise personal loan acceptance while operating within a reduced marketing budget.</div>', unsafe_allow_html=True)

    # Customer Scoring with Best Model
    best_model_name = max(model_results, key=lambda k: model_results[k]["auc"])
    best_model = trained_models[best_model_name]

    df_score = df_model.copy()
    X_full = df_score.drop(columns=["Personal Loan"])
    df_score["Loan Probability"] = best_model.predict_proba(X_full)[:, 1]
    df_score["Loan Prediction"] = best_model.predict(X_full)
    df_score["Risk Tier"] = pd.cut(df_score["Loan Probability"],
                                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                    labels=["Very Low","Low","Medium","High","Very High"])

    st.markdown(f'<div class="insight-box green">All customers have been scored using the best-performing model: <b>{best_model_name}</b>. Each customer now has a loan acceptance probability, allowing you to rank and target with surgical precision.</div>', unsafe_allow_html=True)

    # Tier Summary
    tier_summary = df_score.groupby("Risk Tier", observed=True).agg(
        Customers=("Loan Probability","count"),
        Actual_Loan_Rate=("Personal Loan","mean"),
        Avg_Income=("Income","mean"),
        Avg_CCAvg=("CCAvg","mean"),
    ).reset_index()
    tier_summary["Actual Acceptance %"] = (tier_summary["Actual_Loan_Rate"]*100).round(1)
    tier_summary["Avg Income $K"] = tier_summary["Avg_Income"].round(1)
    tier_summary["Avg CC Spend $K"] = tier_summary["Avg_CCAvg"].round(2)
    tier_summary = tier_summary.drop(columns=["Actual_Loan_Rate","Avg_Income","Avg_CCAvg"])

    st.markdown("#### 🗂️ Customer Propensity Tier Summary")
    st.dataframe(tier_summary.set_index("Risk Tier"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        tier_counts = tier_summary[["Risk Tier","Customers"]].copy()
        fig = px.pie(tier_counts, names="Risk Tier", values="Customers", hole=0.45,
                     color="Risk Tier",
                     color_discrete_map={
                         "Very Low":"#fee2e2","Low":"#fef9c3","Medium":"#dbeafe",
                         "High":"#bbf7d0","Very High":"#059669"},
                     title="Customer Distribution by Propensity Tier")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(tier_summary, x="Risk Tier", y="Actual Acceptance %",
                     text="Actual Acceptance %",
                     color="Actual Acceptance %",
                     color_continuous_scale=["#dcfce7","#16a34a"],
                     title="Actual Loan Acceptance Rate by Propensity Tier",
                     labels={"Risk Tier":"Propensity Tier","Actual Acceptance %":"Actual Acceptance Rate (%)"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", ticksuffix="%")
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Budget Optimisation
    st.markdown("#### 💰 Budget Optimisation Simulator")
    st.markdown("Simulate how focusing your budget on high-propensity tiers improves campaign ROI.")

    total_budget = st.slider("Total Campaign Budget ($)", 10000, 500000, 100000, step=5000,
                              format="$%d")
    cost_per_contact = st.slider("Cost per Customer Contact ($)", 5, 100, 25, step=5)

    max_contacts = total_budget // cost_per_contact

    tier_data = df_score.groupby("Risk Tier", observed=True).agg(
        count=("Loan Probability","count"),
        acceptance_rate=("Personal Loan","mean")
    ).reset_index()
    tier_data = tier_data.sort_values("acceptance_rate", ascending=False)
    tier_data["cumulative_contacts"] = tier_data["count"].cumsum()
    tier_data["expected_loans"] = (tier_data["count"] * tier_data["acceptance_rate"]).round(0)

    budget_results = []
    # Strategy A: Random (baseline)
    overall_rate = df["Personal Loan"].mean()
    random_loans = int(max_contacts * overall_rate)
    budget_results.append({"Strategy":"Random (Baseline)", "Contacts":int(max_contacts),
                            "Expected Loans":random_loans,
                            "Conversion Rate":f"{overall_rate*100:.1f}%",
                            "Est. ROI Multiplier":"1.0×"})

    # Strategy B: Top 2 tiers
    top2 = tier_data[tier_data["Risk Tier"].isin(["Very High","High"])]
    top2_contacts = min(int(max_contacts), int(top2["count"].sum()))
    top2_rate = (top2["count"] * top2["acceptance_rate"]).sum() / top2["count"].sum() if top2["count"].sum() > 0 else 0
    top2_loans = int(top2_contacts * top2_rate)
    multiplier = top2_rate / overall_rate if overall_rate > 0 else 1
    budget_results.append({"Strategy":"Targeted: High + Very High Propensity", "Contacts":top2_contacts,
                            "Expected Loans":top2_loans,
                            "Conversion Rate":f"{top2_rate*100:.1f}%",
                            "Est. ROI Multiplier":f"{multiplier:.1f}×"})

    # Strategy C: Top tier only
    top1 = tier_data[tier_data["Risk Tier"]=="Very High"]
    if len(top1) > 0:
        top1_contacts = min(int(max_contacts), int(top1["count"].sum()))
        top1_rate = float(top1["acceptance_rate"].values[0])
        top1_loans = int(top1_contacts * top1_rate)
        m1 = top1_rate / overall_rate if overall_rate > 0 else 1
        budget_results.append({"Strategy":"Ultra-Targeted: Very High Propensity Only", "Contacts":top1_contacts,
                                "Expected Loans":top1_loans,
                                "Conversion Rate":f"{top1_rate*100:.1f}%",
                                "Est. ROI Multiplier":f"{m1:.1f}×"})

    st.dataframe(pd.DataFrame(budget_results).set_index("Strategy"), use_container_width=True)
    st.markdown('<div class="insight-box amber">By targeting <b>only High + Very High propensity customers</b>, you can achieve the same number of loan conversions while contacting 60–80% fewer people — stretching your halved budget significantly further.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Persona Cards
    st.markdown("#### 👤 Target Personas for Hyper-Personalised Campaigns")
    personas = [
        ("🏆 Persona 1: The Affluent Professional",
         "Income >$120K | Graduate/Advanced degree | Family size 3–4 | CC Spend >$3K/month",
         "Loan acceptance rate: ~55–65%",
         "Message: 'Unlock your next milestone — competitive rates for high earners.' Offer premium loan packaging with fast approval and wealth manager referral.",
         "green"),
        ("💼 Persona 2: The CD Saver",
         "Has CD Account | Any income band | Already trusts the bank",
         "Loan acceptance rate: ~46%",
         "Message: 'As a valued savings customer, here's an exclusive loan offer.' Leverage existing trust. Offer loyalty rate discounts.",
         "primary"),
        ("🎓 Persona 3: The Educated Mid-Income",
         "Income $60–120K | Graduate education | Online banking user | Family size 2–3",
         "Loan acceptance rate: ~15–25%",
         "Message: 'Smart borrowing for smart planners — flexible terms designed for you.' Emphasise flexibility, EMI calculators, and digital-first experience.",
         "purple"),
    ]
    for title, desc, rate, action, style in personas:
        clr = {"green":"#f0fdf4","primary":"#f0f9ff","purple":"#faf5ff"}[style]
        brd = {"green":"#16a34a","primary":"#0ea5e9","purple":"#a855f7"}[style]
        txt = {"green":"#14532d","primary":"#0c4a6e","purple":"#4a044e"}[style]
        st.markdown(f"""
        <div style="background:{clr}; border-left:4px solid {brd}; border-radius:0 12px 12px 0;
                    padding:1.2rem 1.4rem; margin-bottom:1rem;">
          <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.05rem; color:{txt}; margin-bottom:0.4rem;">{title}</div>
          <div style="font-size:0.85rem; color:{txt}; opacity:0.85; margin-bottom:0.3rem;">📌 {desc}</div>
          <div style="font-size:0.85rem; color:{txt}; font-weight:600; margin-bottom:0.5rem;">📈 {rate}</div>
          <div style="font-size:0.85rem; color:{txt}; line-height:1.6;">💡 <em>{action}</em></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 · PREDICT NEW CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📁 Predict New Customers":

    st.markdown('<div class="section-title">📁 Predict New Customers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a CSV file of new customers (same column format as the training data) and download predictions with loan acceptance probability and propensity tier.</div>', unsafe_allow_html=True)

    best_model_name = max(model_results, key=lambda k: model_results[k]["auc"])
    best_model = trained_models[best_model_name]

    st.markdown(f"**Active Model:** `{best_model_name}` (AUC = {model_results[best_model_name]['auc']:.3f})")

    st.markdown("---")
    st.markdown("#### 📤 Upload New Customer Data")
    st.markdown("The file should have the same columns as the original dataset (with or without `Personal Loan` column).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(new_df):,} rows and {new_df.shape[1]} columns.")
            st.dataframe(new_df.head(5), use_container_width=True)

            # Preprocess
            df_pred = new_df.copy()
            drop_cols = [c for c in ["ID", "ZIP Code", "Personal Loan"] if c in df_pred.columns]
            df_pred_clean = df_pred.drop(columns=drop_cols, errors="ignore")

            # Fix experience
            if "Experience" in df_pred_clean.columns:
                df_pred_clean["Experience"] = df_pred_clean["Experience"].clip(lower=0)

            # Ensure column alignment
            expected_cols = X.columns.tolist()
            missing = [c for c in expected_cols if c not in df_pred_clean.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df_pred_clean = df_pred_clean[expected_cols]

                proba = best_model.predict_proba(df_pred_clean)[:, 1]
                pred  = best_model.predict(df_pred_clean)

                result_df = new_df.copy()
                result_df["Predicted_Loan_Probability"] = proba.round(4)
                result_df["Predicted_Loan (1=Yes,0=No)"] = pred
                result_df["Propensity_Tier"] = pd.cut(
                    proba, bins=[0,0.2,0.4,0.6,0.8,1.0],
                    labels=["Very Low","Low","Medium","High","Very High"]
                ).astype(str)

                st.markdown("#### 📊 Prediction Results Preview")
                st.dataframe(result_df.head(20), use_container_width=True)

                summary_cols = st.columns(4)
                metrics_pred = [
                    ("Total Customers", f"{len(result_df):,}"),
                    ("Predicted Loan = Yes", f"{pred.sum():,} ({pred.mean()*100:.1f}%)"),
                    ("Avg Probability", f"{proba.mean()*100:.1f}%"),
                    ("High/Very High Tier", f"{(result_df['Propensity_Tier'].isin(['High','Very High'])).sum():,}"),
                ]
                for col_s, (lbl, val) in zip(summary_cols, metrics_pred):
                    with col_s:
                        st.markdown(f"""
                        <div class="kpi-card">
                          <div class="kpi-label">{lbl}</div>
                          <div class="kpi-value" style="font-size:1.5rem;">{val}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv_out,
                    file_name="UniversalBank_Predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.info("👆 Upload a CSV file above to get predictions. A sample test file (`sample_test_data.csv`) is included in the project folder.")

        st.markdown("#### 📋 Expected Column Format")
        sample_schema = pd.DataFrame({
            "Column": ["Age","Experience","Income","ZIP Code","Family","CCAvg","Education",
                       "Mortgage","Securities Account","CD Account","Online","CreditCard"],
            "Type": ["int","int","int","int","int","float","int (1/2/3)","int","int (0/1)","int (0/1)","int (0/1)","int (0/1)"],
            "Description": [
                "Age in years (23–67)",
                "Years of experience",
                "Annual income $000",
                "ZIP code",
                "Family size (1–4)",
                "Avg monthly CC spend $000",
                "1=Undergrad, 2=Grad, 3=Adv",
                "Mortgage value $000",
                "Has securities account",
                "Has CD account",
                "Uses online banking",
                "Has bank credit card",
            ]
        })
        st.dataframe(sample_schema.set_index("Column"), use_container_width=True)
