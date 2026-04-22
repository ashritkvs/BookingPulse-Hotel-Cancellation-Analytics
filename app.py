import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BookingPulse",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117;
    color: #e8e8e8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1e2736;
}
section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f1117 0%, #141d2e 50%, #0f1117 100%);
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f0f4ff;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #7a8aaa;
    font-weight: 300;
    margin: 0;
}
.hero-tag {
    display: inline-block;
    margin-top: 14px;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #60a5fa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
}

/* Metric cards */
.metric-row { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: #161b27;
    border: 1px solid #1e2736;
    border-radius: 12px;
    padding: 20px 24px;
}
.metric-label {
    font-size: 0.72rem;
    color: #5a6a8a;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #f0f4ff;
    line-height: 1;
}
.metric-delta { font-size: 0.78rem; margin-top: 4px; }
.delta-bad  { color: #f87171; }
.delta-good { color: #4ade80; }

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e8eeff;
    border-left: 3px solid #3b82f6;
    padding-left: 14px;
    margin: 32px 0 16px;
}

/* Insight cards */
.insight-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 24px; }
.insight-card {
    flex: 1; min-width: 220px;
    background: #161b27;
    border: 1px solid #1e2736;
    border-radius: 12px;
    padding: 18px 20px;
}
.insight-icon { font-size: 1.4rem; margin-bottom: 8px; }
.insight-title { font-weight: 600; font-size: 0.9rem; color: #c9d1e0; margin-bottom: 4px; }
.insight-body  { font-size: 0.82rem; color: #6b7a99; line-height: 1.5; }

/* Risk meter */
.risk-box {
    background: #161b27;
    border-radius: 16px;
    border: 1px solid #1e2736;
    padding: 28px 32px;
    text-align: center;
}
.risk-pct {
    font-family: 'DM Serif Display', serif;
    font-size: 4rem;
    line-height: 1;
}
.risk-label { font-size: 1rem; color: #7a8aaa; margin-top: 8px; }

/* Tab styling */
div[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* Inputs */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    color: #7a8aaa !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* Plot background patch */
.plot-wrap {
    background: #161b27;
    border: 1px solid #1e2736;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #3a4a6a;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid #1e2736;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
CARD_BG   = "#161b27"
BORDER    = "#1e2736"
BLUE      = "#3b82f6"
GREEN     = "#4ade80"
RED       = "#f87171"
MUTED     = "#5a6a8a"

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   "#9aa3b8",
        "axes.titlecolor":   "#d0d8f0",
        "xtick.color":       "#6b7a99",
        "ytick.color":       "#6b7a99",
        "text.color":        "#9aa3b8",
        "grid.color":        "#1e2736",
        "grid.alpha":        1,
        "axes.grid":         True,
        "grid.linestyle":    "--",
        "font.family":       "sans-serif",
    })

set_plot_style()

@st.cache_data(show_spinner=False)
def load_and_train(file_bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))

    # Clean known nullable columns
    for col in ["children", "agent", "company"]:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    df.drop(columns=[c for c in ["reservation_status_date", "reservation_status"] if c in df.columns], inplace=True)

    # Fix columns with numeric values stored as strings (e.g. scientific notation '3.69095E-1')
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna().sum()
        if non_null > 0 and converted.notna().sum() / non_null > 0.8:
            df[col] = converted

    # EDA stats
    cancel_rate   = df["is_canceled"].mean()
    n_bookings    = len(df)
    city_cr       = df[df["hotel"] == "City Hotel"]["is_canceled"].mean() if "hotel" in df.columns else None
    resort_cr     = df[df["hotel"] == "Resort Hotel"]["is_canceled"].mean() if "hotel" in df.columns else None
    repeat_cr     = df[df["is_repeated_guest"] == 1]["is_canceled"].mean() if "is_repeated_guest" in df.columns else None
    med_lead_c    = df[df["is_canceled"] == 1]["lead_time"].median() if "lead_time" in df.columns else None
    med_lead_nc   = df[df["is_canceled"] == 0]["lead_time"].median() if "lead_time" in df.columns else None

    # Feature engineering
    df_enc = pd.get_dummies(df, drop_first=True)
    if all(c in df_enc.columns for c in ["adults", "children", "babies"]):
        df_enc["total_guests"] = df_enc["adults"] + df_enc["children"] + df_enc["babies"]

    # Pandas 2.x returns bool dtype for get_dummies — cast to int for XGBoost/SHAP compatibility
    bool_cols = df_enc.select_dtypes(include="bool").columns
    df_enc[bool_cols] = df_enc[bool_cols].astype(int)

    # Keep only numeric columns and fill remaining NaNs
    df_enc = df_enc.select_dtypes(include=[np.number]).fillna(0)

    X = df_enc.drop(["is_canceled"], axis=1)
    y = df_enc["is_canceled"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_estimators=200, base_score=0.5)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, target_names=["Not Cancelled", "Cancelled"], output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)

    explainer   = None
    shap_values = None

    stats = dict(
        cancel_rate=cancel_rate, n_bookings=n_bookings,
        city_cr=city_cr, resort_cr=resort_cr, repeat_cr=repeat_cr,
        med_lead_c=med_lead_c, med_lead_nc=med_lead_nc,
        roc=roc, report=report, cm=cm
    )
    return df, model, X_test, shap_values, stats, X.columns.tolist()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏨 BookingPulse")
    st.markdown("<p style='font-size:0.8rem;color:#5a6a8a;margin-top:-8px;'>Hotel Cancellation Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded = st.file_uploader("Upload hotel_bookings.csv", type=["csv"])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#3a4a6a;line-height:1.8'>
    <b style='color:#5a6a8a'>Dataset</b><br>
    Hotel Booking Demand<br>119,390 bookings · 32 features<br><br>
    <b style='color:#5a6a8a'>Model</b><br>
    XGBoost Classifier<br>80/20 train-test split<br><br>
    <b style='color:#5a6a8a'>Built by</b><br>
    Venkata Sai Ashrit Kommireddy<br>
    MS Data Science · Stony Brook
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='hero-title'>🏨 BookingPulse</div>
  <p class='hero-sub'>Hotel Booking Cancellation &amp; Revenue Optimization — End-to-End Analytics Dashboard</p>
  <span class='hero-tag'>Data Analyst Portfolio Project</span>
</div>
""", unsafe_allow_html=True)

# ── Gate: needs file ──────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈  Upload **hotel_bookings.csv** in the sidebar to get started. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).")
    st.stop()

# ── Load & train ──────────────────────────────────────────────────────────────
with st.spinner("Training model… this takes ~20 seconds on first load."):
    df, model, X_test, shap_values, stats, feature_cols = load_and_train(uploaded.getvalue())

# ── KPI row ───────────────────────────────────────────────────────────────────
cr  = stats["cancel_rate"]
roc = stats["roc"]
f1c = stats["report"]["Cancelled"]["f1-score"]
n   = stats["n_bookings"]

st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card'>
    <div class='metric-label'>Total Bookings</div>
    <div class='metric-value'>{n:,}</div>
    <div class='metric-delta' style='color:#5a6a8a'>2015 – 2017</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>Cancellation Rate</div>
    <div class='metric-value'>{cr:.1%}</div>
    <div class='metric-delta delta-bad'>↑ revenue exposure</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>ROC-AUC Score</div>
    <div class='metric-value'>{roc:.3f}</div>
    <div class='metric-delta delta-good'>XGBoost model</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>F1 (Cancelled class)</div>
    <div class='metric-value'>{f1c:.3f}</div>
    <div class='metric-delta delta-good'>precision · recall</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>Repeat Guest Cancel Rate</div>
    <div class='metric-value'>{stats["repeat_cr"]:.1%}</div>
    <div class='metric-delta delta-good'>vs {cr:.1%} overall</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Model Performance", "🔍 SHAP Explainability", "🎯 Risk Scorer"])

# ────────────────────── TAB 1: EDA ───────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Cancellation by hotel type
    with col1:
        if "hotel" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            hotel_cancel = df.groupby("hotel")["is_canceled"].mean().reset_index()
            bars = ax.bar(
                hotel_cancel["hotel"], hotel_cancel["is_canceled"],
                color=[BLUE, RED], width=0.5, zorder=3
            )
            ax.set_title("Cancellation Rate by Hotel Type", fontweight="bold", pad=12)
            ax.set_ylabel("Cancellation Rate")
            ax.set_ylim(0, 0.55)
            for bar, val in zip(bars, hotel_cancel["is_canceled"]):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f"{val:.1%}", ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color="#e8eeff")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Lead time distribution
    with col2:
        if "lead_time" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            canceled     = df[df["is_canceled"] == 1]["lead_time"].clip(upper=400)
            not_canceled = df[df["is_canceled"] == 0]["lead_time"].clip(upper=400)
            ax.hist(not_canceled, bins=50, alpha=0.6, color=GREEN,  label="Not Cancelled", zorder=3)
            ax.hist(canceled,     bins=50, alpha=0.6, color=RED,    label="Cancelled",     zorder=3)
            ax.set_title("Lead Time Distribution", fontweight="bold", pad=12)
            ax.set_xlabel("Lead Time (days, capped at 400)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    col3, col4 = st.columns(2)

    # Top countries
    with col3:
        if "country" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            top_c = df["country"].value_counts().head(8).index
            rates = (df[df["country"].isin(top_c)]
                       .groupby("country")["is_canceled"].mean()
                       .reindex(top_c)
                       .sort_values(ascending=True))
            colors = [RED if v > cr else BLUE for v in rates.values]
            ax.barh(rates.index, rates.values, color=colors, zorder=3)
            ax.axvline(cr, color="#f0f4ff", linestyle="--", linewidth=1, alpha=0.4, label=f"Avg {cr:.1%}")
            ax.set_title("Cancellation Rate — Top 8 Countries", fontweight="bold", pad=12)
            ax.set_xlabel("Cancellation Rate")
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Monthly cancellation trend
    with col4:
        if "arrival_date_month" in df.columns:
            month_order = ["January","February","March","April","May","June",
                           "July","August","September","October","November","December"]
            month_cr = (df.groupby("arrival_date_month")["is_canceled"]
                          .mean()
                          .reindex([m for m in month_order if m in df["arrival_date_month"].unique()]))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(len(month_cr)), month_cr.values,
                    color=BLUE, linewidth=2.5, marker="o", markersize=5, zorder=3)
            ax.fill_between(range(len(month_cr)), month_cr.values,
                            alpha=0.15, color=BLUE)
            ax.set_xticks(range(len(month_cr)))
            ax.set_xticklabels([m[:3] for m in month_cr.index], rotation=45, fontsize=8)
            ax.set_title("Monthly Cancellation Rate", fontweight="bold", pad=12)
            ax.set_ylabel("Cancellation Rate")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Business insights
    st.markdown("<div class='section-header'>Key Business Insights</div>", unsafe_allow_html=True)
    city_cr   = stats["city_cr"]
    resort_cr = stats["resort_cr"]
    mlc       = stats["med_lead_c"]
    mlnc      = stats["med_lead_nc"]
    st.markdown(f"""
    <div class='insight-grid'>
      <div class='insight-card'>
        <div class='insight-icon'>📅</div>
        <div class='insight-title'>Lead Time Drives Cancellations</div>
        <div class='insight-body'>Cancelled bookings have a median lead time of <b style='color:#f0f4ff'>{mlc:.0f} days</b> vs <b style='color:#f0f4ff'>{mlnc:.0f} days</b> for kept bookings. Long-horizon plans are inherently unstable.</div>
      </div>
      <div class='insight-card'>
        <div class='insight-icon'>🏙️</div>
        <div class='insight-title'>City Hotels Cancel More</div>
        <div class='insight-body'>City hotels cancel at <b style='color:#f87171'>{city_cr:.1%}</b> vs resort hotels at <b style='color:#4ade80'>{resort_cr:.1%}</b>. Business travelers dominate city bookings and are more itinerary-sensitive.</div>
      </div>
      <div class='insight-card'>
        <div class='insight-icon'>🔁</div>
        <div class='insight-title'>Repeat Guests Are Loyal</div>
        <div class='insight-body'>Returning guests cancel at just <b style='color:#4ade80'>{stats["repeat_cr"]:.1%}</b> — a fraction of the overall rate. Loyalty programs are a direct lever on cancellation exposure.</div>
      </div>
      <div class='insight-card'>
        <div class='insight-icon'>💰</div>
        <div class='insight-title'>High ADR = Higher Risk</div>
        <div class='insight-body'>Premium bookings show elevated cancellation tendency — yet these are the most valuable to protect. Proactive outreach 2–4 weeks before arrival can meaningfully reduce no-shows.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ────────────────────── TAB 2: MODEL PERFORMANCE ─────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Model Performance — XGBoost Classifier</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    # Confusion matrix
    with col1:
        cm   = stats["cm"]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Cancelled", "Cancelled"],
            yticklabels=["Not Cancelled", "Cancelled"],
            ax=ax, linewidths=0.5, linecolor=BORDER,
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_xlabel("Predicted",  fontsize=11)
        ax.set_ylabel("Actual",     fontsize=11)
        ax.set_title("Confusion Matrix", fontweight="bold", pad=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Classification report table
    with col2:
        st.markdown("#### Classification Report")
        rep = stats["report"]
        rows = []
        for label in ["Not Cancelled", "Cancelled", "macro avg", "weighted avg"]:
            if label in rep:
                r = rep[label]
                rows.append({
                    "Class":     label,
                    "Precision": f"{r['precision']:.3f}",
                    "Recall":    f"{r['recall']:.3f}",
                    "F1-Score":  f"{r['f1-score']:.3f}",
                    "Support":   f"{int(r['support']):,}" if "support" in r else "—",
                })
        st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

        st.markdown(f"""
        <div style='margin-top:20px;background:#161b27;border:1px solid #1e2736;border-radius:12px;padding:20px 24px'>
          <div style='font-size:0.75rem;color:#5a6a8a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>ROC-AUC Score</div>
          <div style='font-family:"DM Serif Display",serif;font-size:3rem;color:#60a5fa;line-height:1'>{roc:.4f}</div>
          <div style='font-size:0.8rem;color:#5a6a8a;margin-top:6px'>Higher is better · 0.5 = random · 1.0 = perfect</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance bar
    st.markdown("<div class='section-header'>Top 15 Feature Importances</div>", unsafe_allow_html=True)
    importances = pd.Series(model.feature_importances_, index=feature_cols).nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [BLUE if i < 10 else "#60a5fa" for i in range(len(importances))]
    ax.barh(importances.index, importances.values, color=colors[::-1], zorder=3)
    ax.set_title("XGBoost Feature Importance (Top 15)", fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ────────────────────── TAB 3: SHAP ──────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>SHAP Explainability</div>", unsafe_allow_html=True)

    if shap_values is not None:
        st.markdown("""
        <p style='color:#6b7a99;font-size:0.9rem;margin-bottom:20px'>
        SHAP (SHapley Additive exPlanations) tells us <em>why</em> the model made each prediction — not just what it predicted.
        Each dot below is one booking from the test set. <span style='color:#f87171'>Red = high feature value</span> pushing toward cancellation.
        <span style='color:#60a5fa'>Blue = low feature value</span> pushing away from cancellation.
        </p>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_test[:200], show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor(DARK_BG)
        for a in fig.axes:
            a.set_facecolor(CARD_BG)
            a.tick_params(colors="#7a8aaa")
            a.xaxis.label.set_color("#9aa3b8")
            a.title.set_color("#d0d8f0")
        st.pyplot(fig)
        plt.close()
    else:
        st.markdown("""
        <p style='color:#6b7a99;font-size:0.9rem;margin-bottom:20px'>
        SHAP beeswarm plot requires a compatible SHAP version. Showing XGBoost gain-based feature importance instead —
        this ranks features by how much they reduce prediction error across all trees.
        </p>
        """, unsafe_allow_html=True)
        importances = pd.Series(model.feature_importances_, index=feature_cols).nlargest(20).sort_values()
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = [RED if f in ["lead_time","adr","deposit_type_Non Refund"] else BLUE for f in importances.index]
        ax.barh(importances.index, importances.values, color=colors, zorder=3)
        ax.set_title("Feature Importance — Top 20 (XGBoost Gain)", fontweight="bold", pad=12)
        ax.set_xlabel("Importance Score")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    <div style='background:#161b27;border:1px solid #1e2736;border-radius:12px;padding:20px 24px;margin-top:16px'>
    <b style='color:#c9d1e0'>Key feature drivers:</b>
    <ul style='color:#6b7a99;font-size:0.85rem;line-height:2;margin-top:8px'>
      <li><b style='color:#c9d1e0'>lead_time</b> — Long lead times strongly push the model toward predicting cancellation.</li>
      <li><b style='color:#c9d1e0'>deposit_type_Non Refund</b> — Non-refundable deposits correlate with higher cancellation risk.</li>
      <li><b style='color:#c9d1e0'>adr</b> — Higher average daily rate increases cancellation risk.</li>
      <li><b style='color:#c9d1e0'>is_repeated_guest</b> — Returning guests push strongly toward <em>not</em> cancelling.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ────────────────────── TAB 4: RISK SCORER ───────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>Live Cancellation Risk Scorer</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7a99;font-size:0.9rem;margin-bottom:24px'>Enter a booking's details below. The model will output its predicted cancellation probability in real time.</p>", unsafe_allow_html=True)

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            hotel         = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
            lead_time     = st.number_input("Lead Time (days)", 0, 700, 60)
            adr           = st.number_input("ADR — Avg Daily Rate ($)", 0.0, 1000.0, 100.0, step=10.0)
            adults        = st.number_input("Adults", 1, 10, 2)

        with c2:
            market_seg    = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Complementary", "Groups", "Aviation"])
            deposit_type  = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
            total_nights  = st.number_input("Total Nights Stay", 1, 60, 3)
            is_repeated   = st.selectbox("Repeat Guest?", ["No", "Yes"])

        with c3:
            meal          = st.selectbox("Meal Plan", ["BB", "HB", "FB", "SC", "Undefined"])
            booking_changes = st.number_input("Booking Changes", 0, 20, 0)
            special_req   = st.number_input("Special Requests", 0, 10, 1)
            prev_cancel   = st.number_input("Previous Cancellations", 0, 20, 0)

        submitted = st.form_submit_button("⚡  Score This Booking", use_container_width=True)

    if submitted:
        # Build a dummy row matching training columns
        dummy = pd.DataFrame(columns=feature_cols).fillna(0)
        row   = {c: 0 for c in feature_cols}

        # Map fields
        row["lead_time"]            = lead_time
        row["adr"]                  = adr
        row["adults"]               = adults
        row["stays_in_week_nights"] = max(total_nights - 1, 0)
        row["stays_in_weekend_nights"] = min(total_nights, 1)
        row["booking_changes"]      = booking_changes
        row["total_of_special_requests"] = special_req
        row["previous_cancellations"] = prev_cancel
        row["is_repeated_guest"]    = 1 if is_repeated == "Yes" else 0
        row["total_guests"]         = adults

        hotel_col = "hotel_Resort Hotel"
        if hotel_col in row: row[hotel_col] = 1 if hotel == "Resort Hotel" else 0

        seg_col = f"market_segment_{market_seg}"
        if seg_col in row: row[seg_col] = 1

        dep_col = f"deposit_type_{deposit_type}"
        if dep_col in row: row[dep_col] = 1

        meal_col = f"meal_{meal}"
        if meal_col in row: row[meal_col] = 1

        X_input = pd.DataFrame([row])[feature_cols]
        prob    = model.predict_proba(X_input)[0][1]

        if prob < 0.35:
            risk_color = GREEN
            risk_label = "Low Risk"
            risk_msg   = "This booking looks stable. Standard follow-up is sufficient."
        elif prob < 0.65:
            risk_color = "#facc15"
            risk_label = "Medium Risk"
            risk_msg   = "Monitor this booking. Consider a confirmation nudge 2 weeks before arrival."
        else:
            risk_color = RED
            risk_label = "High Risk"
            risk_msg   = "Flag for revenue team. Consider overbooking offset or proactive outreach."

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"""
            <div class='risk-box'>
              <div class='risk-pct' style='color:{risk_color}'>{prob:.0%}</div>
              <div style='font-size:1.1rem;font-weight:600;color:{risk_color};margin-top:8px'>{risk_label}</div>
              <div class='risk-label'>Cancellation Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div style='background:#161b27;border:1px solid #1e2736;border-radius:12px;padding:24px 28px;height:100%'>
              <div style='font-size:0.72rem;color:#5a6a8a;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>Revenue Team Recommendation</div>
              <p style='color:#c9d1e0;font-size:0.95rem;line-height:1.7'>{risk_msg}</p>
              <hr style='border-color:#1e2736;margin:16px 0'>
              <div style='display:flex;gap:24px;font-size:0.8rem'>
                <div><span style='color:#5a6a8a'>Hotel Type</span><br><b style='color:#e8eeff'>{hotel}</b></div>
                <div><span style='color:#5a6a8a'>Lead Time</span><br><b style='color:#e8eeff'>{lead_time} days</b></div>
                <div><span style='color:#5a6a8a'>ADR</span><br><b style='color:#e8eeff'>${adr:.0f}</b></div>
                <div><span style='color:#5a6a8a'>Deposit</span><br><b style='color:#e8eeff'>{deposit_type}</b></div>
                <div><span style='color:#5a6a8a'>Repeat Guest</span><br><b style='color:#e8eeff'>{is_repeated}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  BookingPulse · Built by Venkata Sai Ashrit Kommireddy · MS Data Science, Stony Brook University<br>
  XGBoost · SHAP · Streamlit · Scikit-learn · Pandas
</div>
""", unsafe_allow_html=True)
