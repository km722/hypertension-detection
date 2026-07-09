"""
Streamlit presentation app for the hypertension dissertation.
Sections: Hero, Global Burden, Methodology, Results, Risk Calculator.

Run from project root:
    .venv/bin/streamlit run app/presentation_app.py
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hypertension Dissertation", page_icon="🩺", layout="wide")

# model path relative to this file's directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "PanelA_RenalDemo_logreg_tuned.joblib")

PANEL_A_FEATURES = [
    "ACR_mg_g_log", "BMXBMI", "LBXSAL", "LBXSBU", "LBXSCR_log", "LBXSUA",
    "RIDAGEYR", "URXUCR", "URXUMA", "eGFR",
    "race_mexican_american", "race_non_hispanic_asian", "race_non_hispanic_black",
    "race_non_hispanic_white", "race_other_hispanic", "race_other_race",
    "sex_female",
]

SCALER_PARAMS = {
    "ACR_mg_g_log": (2.4146918514, 0.8893573963),
    "BMXBMI":       (29.4655442671, 6.2164955299),
    "LBXSAL":       (4.1862312530, 0.3063669957),
    "LBXSBU":       (14.6884373488, 4.4491076742),
    "LBXSCR_log":   (0.6174236160, 0.1083172085),
    "LBXSUA":       (5.4274117078, 1.2609203396),
    "RIDAGEYR":     (50.9422351234, 17.4960539179),
    "URXUCR":       (122.9545234640, 72.3198846188),
    "URXUMA":       (19.4132133527, 28.3741709041),
    "eGFR":         (94.5556372210, 20.6325129685),
}

RACE_MAP = {
    "Non-Hispanic White": "race_non_hispanic_white",
    "Non-Hispanic Black":  "race_non_hispanic_black",
    "Mexican American":    "race_mexican_american",
    "Non-Hispanic Asian":  "race_non_hispanic_asian",
    "Other Hispanic":      "race_other_hispanic",
    "Other Race":          "race_other_race",
}

FEATURE_LABELS = {
    "ACR_mg_g_log": "ACR (log)", "BMXBMI": "BMI", "LBXSAL": "Serum Albumin",
    "LBXSBU": "Blood Urea Nitrogen", "LBXSCR_log": "Serum Creatinine (log)",
    "LBXSUA": "Uric Acid", "RIDAGEYR": "Age", "URXUCR": "Urine Creatinine",
    "URXUMA": "Urine Albumin", "eGFR": "eGFR",
    "race_mexican_american": "Race (Mexican Am.)",
    "race_non_hispanic_asian": "Race (NH Asian)",
    "race_non_hispanic_black": "Race (NH Black)",
    "race_non_hispanic_white": "Race (NH White)",
    "race_other_hispanic": "Race (Other Hispanic)",
    "race_other_race": "Race (Other)",
    "sex_female": "Sex (Female)",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Source+Sans+Pro:wght@400;600;700&display=swap');

* { font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif; }
.stApp { background: linear-gradient(180deg, #fef9f9 0%, #fff 30%, #fdfafa 70%, #fef5f5 100%); }
.main   { background: transparent; }

/* Hero banner */
.hero {
    padding: 4rem 3.5rem; border-radius: 32px;
    background: linear-gradient(135deg, #6B0F1A, #8B0000 25%, #B31B1B 50%, #C41E3A 75%, #DC143C);
    color: white; position: relative; overflow: hidden;
    box-shadow: 0 30px 80px rgba(107,15,26,.45), 0 15px 35px rgba(0,0,0,.15);
}
.hero h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 48px !important; font-weight: 800 !important; line-height: 1.2 !important;
    color: white !important; margin-bottom: 1.5rem; position: relative; z-index: 1;
}
.hero-subtitle { font-size: 20px !important; font-weight: 600 !important; color: white !important; position: relative; z-index: 1; }
.hero-meta     { font-size: 17px !important; color: rgba(255,255,255,.92) !important; position: relative; z-index: 1; }
.hero-tagline  { font-size: 18px !important; margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,.2); font-style: italic; position: relative; z-index: 1; }

/* Cards */
.glass-card {
    padding: 2.5rem; border-radius: 24px; background: rgba(255,255,255,.98);
    border: 1px solid rgba(220,20,60,.08); margin-bottom: 2rem;
    box-shadow: 0 15px 40px rgba(139,0,0,.06), 0 5px 15px rgba(0,0,0,.03);
    text-align: center;
}
.stat-card { margin-bottom: 2rem; }
.stat-card {
    text-align: center; padding: 3rem 2rem; border-radius: 24px;
    background: linear-gradient(145deg, #fff, #fff8f8);
    border: 1px solid rgba(139,0,0,.1);
    box-shadow: 0 12px 35px rgba(139,0,0,.08), 0 4px 12px rgba(0,0,0,.03);
}
.stat-number {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 72px !important; font-weight: 800 !important;
    background: linear-gradient(135deg, #6B0F1A, #B31B1B, #DC143C);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1.1;
}
.stat-label { color: #333 !important; font-size: 20px !important; margin-top: 1rem; font-weight: 600 !important; }

/* Section headers */
.section-header {
    display: inline-flex; align-items: center; gap: .8rem;
    padding: 1rem 2.5rem; border-radius: 35px;
    background: linear-gradient(135deg, #6B0F1A, #8B0000 50%, #B31B1B);
    color: #fff !important; font-weight: 700; font-size: 30px !important;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(107,15,26,.4);
}
/* Center section headers */
.section-header { display: block; text-align: center; margin-bottom: 2.5rem; }

/* Badges & risk pills */
.badge {
    display: inline-flex; height: 44px; padding: 0 1.5rem; border-radius: 22px;
    align-items: center; font-weight: 700; font-size: 15px;
    background: linear-gradient(135deg, #fff0f0, #ffe0e0); color: #8B0000;
    margin-right: .8rem; margin-bottom: .6rem;
    border: 2px solid rgba(139,0,0,.15);
}
.risk-pill-low      { background: linear-gradient(135deg, #d4edda, #b8e0c0); color: #155724; border-color: rgba(21,87,36,.25); }
.risk-pill-moderate { background: linear-gradient(135deg, #fff3cd, #ffe69c); color: #856404; border-color: rgba(133,100,4,.25); }
.risk-pill-high     { background: linear-gradient(135deg, #DC143C, #8B0000); color: #fff; border-color: transparent; font-weight: 800; }

/* Typography */
h2, h3 { color: #6B0F1A !important; font-weight: 700; }
ul { padding-left: 1.5rem; line-height: 2; text-align: left; }
ul li { margin-bottom: .5rem; color: #333; font-size: 17px; }

/* Input labels */
.stNumberInput label p, .stSelectbox label p { color: #8B0000 !important; font-weight: 700 !important; font-size: 1.2rem !important; }
/* Number inputs */
.stNumberInput input,
.stNumberInput [data-baseweb="input"] input,
[data-testid="stNumberInput"] input {
    background-color: #8B0000 !important; color: #fff !important;
    border-color: #6B0F1A !important; border-radius: 12px !important;
    -webkit-text-fill-color: #fff !important;
}
[data-testid="stNumberInput"] [data-baseweb="input"],
.stNumberInput [data-baseweb="input"] {
    background-color: #8B0000 !important; border-color: #6B0F1A !important;
}
/* Select boxes */
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] > div {
    background-color: #8B0000 !important; color: #fff !important;
    border-color: #6B0F1A !important; border-radius: 12px !important;
}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div[class*="value"] {
    color: #fff !important; -webkit-text-fill-color: #fff !important;
}
.stSelectbox svg { fill: #fff !important; }
/* +/- buttons */
.stNumberInput button,
[data-testid="stNumberInput"] button {
    background-color: #6B0F1A !important; color: #fff !important;
    border: none !important;
}

.stDivider { margin: 3rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Derived-metric helpers ───────────────────────────────────────────────────

def compute_bmi(height_cm, weight_kg):
    if height_cm <= 0 or weight_kg <= 0:
        return None
    return weight_kg / (height_cm / 100) ** 2


def compute_egfr(creatinine, age, sex):
    if creatinine <= 0 or age <= 0:
        return None
    k     = 0.7 if sex == "Female" else 0.9
    alpha = -0.241 if sex == "Female" else -0.302
    return (142
            * min(creatinine / k, 1) ** alpha
            * max(creatinine / k, 1) ** -1.200
            * 0.9938 ** age
            * (1.012 if sex == "Female" else 1))


def compute_acr(albumin_mg_l, urine_creatinine_mg_dl):
    if albumin_mg_l <= 0 or urine_creatinine_mg_dl <= 0:
        return None
    return (albumin_mg_l * 100) / urine_creatinine_mg_dl


# ── ML prediction helpers ────────────────────────────────────────────────────

def _build_feature_vector(age, sex, race, bmi, scr, uric, albumin, bun,
                          u_alb, u_cr, egfr, acr):
    bmi  = bmi  if bmi  is not None else 27.0
    egfr = egfr if egfr is not None else 90.0
    acr  = acr  if acr  is not None else 15.0

    raw = {
        "ACR_mg_g_log": np.log1p(acr), "BMXBMI": bmi, "LBXSAL": albumin,
        "LBXSBU": bun, "LBXSCR_log": np.log1p(scr), "LBXSUA": uric,
        "RIDAGEYR": age, "URXUCR": u_cr, "URXUMA": u_alb, "eGFR": egfr,
    }
    z = {f: (v - SCALER_PARAMS[f][0]) / SCALER_PARAMS[f][1] for f, v in raw.items()}

    sex_val = 1 if sex == "Female" else 0
    race_col = RACE_MAP.get(race, "race_other_race")
    race_vals = {c: int(c == race_col) for c in RACE_MAP.values()}

    vec = []
    for f in PANEL_A_FEATURES:
        if f in z:          vec.append(z[f])
        elif f == "sex_female": vec.append(sex_val)
        elif f in race_vals:    vec.append(race_vals[f])
        else:                   vec.append(0.0)
    return np.array(vec).reshape(1, -1)


def predict_htn_risk(age, sex, race, bmi, scr, uric, albumin, bun,
                     u_alb, u_cr, egfr, acr):
    X = _build_feature_vector(age, sex, race, bmi, scr, uric, albumin, bun,
                              u_alb, u_cr, egfr, acr)
    return float(load_model().predict_proba(X)[0, 1])


def compute_contributions(age, sex, race, bmi, scr, uric, albumin, bun,
                          u_alb, u_cr, egfr, acr):
    model = load_model()
    X = _build_feature_vector(age, sex, race, bmi, scr, uric, albumin, bun,
                              u_alb, u_cr, egfr, acr)
    pairs = [(FEATURE_LABELS.get(f, f), model.coef_[0][i] * X[0, i])
             for i, f in enumerate(PANEL_A_FEATURES)]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs


def rule_based_score(bmi, egfr, acr, age, sex, race, uric_acid):
    score, notes = 0, []
    if bmi is not None:
        if   bmi >= 35: score += 3; notes.append("Severe obesity (BMI >= 35) (+3)")
        elif bmi >= 30: score += 2; notes.append("Obesity (BMI >= 30) (+2)")
    if egfr is not None:
        if   egfr < 45: score += 4; notes.append("eGFR < 45 — major kidney impairment (+4)")
        elif egfr < 60: score += 3; notes.append("eGFR 45-60 — reduced kidney function (+3)")
    if acr is not None:
        if   acr >= 300: score += 4; notes.append("ACR > 300 — heavy proteinuria (+4)")
        elif acr >= 30:  score += 2; notes.append("ACR 30-300 — moderate proteinuria (+2)")
    if uric_acid >= 7: score += 2; notes.append("Uric acid > 7 mg/dL (+2)")
    if   age >= 60: score += 2; notes.append("Age >= 60 (+2)")
    elif age >= 45: score += 1; notes.append("Age >= 45 (+1)")
    if sex == "Male": score += 1; notes.append("Male sex (+1)")
    if race == "Non-Hispanic Black": score += 1; notes.append("NH Black ancestry (+1)")
    level = "High" if score >= 10 else ("Moderate" if score >= 5 else "Low")
    return level, score, notes


# ── Figures ──────────────────────────────────────────────────────────────────

def build_pipeline_figure():
    steps = [
        ("NHANES 2015-2018<br>cohort",
         "Merge two cycles;<br>adults >= 20; exclude<br>pregnancy; n = 10,335"),
        ("Outcome &<br>derived fields",
         "HTN (ACC/AHA 2017);<br>mean BP; CKD-EPI eGFR;<br>ACR mg/g; BMI"),
        ("Preprocessing",
         "Screen implausibles;<br>impute; winsorise 5/95;<br>log1p ACR & creatinine;<br>z-score numerics"),
        ("Feature panels<br>& baselines",
         "Panel A/B/C;<br>LR, SVM, DT, kNN;<br>temporal validation<br>(train 15-16, test 17-18)"),
        ("Ensembles &<br>evaluation",
         "RF, XGB, LGBM<br>(Bayesian-tuned);<br>stacking/voting;<br>SHAP + calibration"),
    ]
    n = len(steps)
    xs = list(range(n))
    colours = ["#8B0000", "#A01515", "#B31B1B", "#C92828", "#DC143C"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[0]*n, mode="lines", hoverinfo="skip",
                             line=dict(color="#DC143C", width=4)))
    for i in range(n):
        fig.add_trace(go.Scatter(
            x=[xs[i]], y=[0], mode="markers", hoverinfo="text",
            hovertext=steps[i][0].replace("<br>", " "),
            marker=dict(size=26, color=colours[i], line=dict(color="#fff", width=2.5))))

    annotations = []
    for i, (title, detail) in enumerate(steps):
        annotations.append(dict(x=xs[i], y=0.55, text=title, showarrow=False,
                                font=dict(size=13, color="#B31B1B", weight=600), xanchor="center"))
        annotations.append(dict(x=xs[i], y=-0.45, text=detail, showarrow=False,
                                font=dict(size=10, color="#444"), yanchor="top", xanchor="center"))

    # Animation frames
    frames = []
    for active in range(n):
        fd = [go.Scatter(x=xs, y=[0]*n, mode="lines", line=dict(color="#DC143C", width=4))]
        for i in range(n):
            sz, lc, lw = (38, "#FFD700", 4) if i == active else (26, "#fff", 2.5)
            fd.append(go.Scatter(x=[xs[i]], y=[0], mode="markers",
                                 marker=dict(size=sz, color=colours[i] if i != active else "#DC143C",
                                             line=dict(color=lc, width=lw))))
        frames.append(go.Frame(data=fd, name=str(active)))
    # "complete" frame
    fd = [go.Scatter(x=xs, y=[0]*n, mode="lines", line=dict(color="#DC143C", width=4))]
    for i in range(n):
        fd.append(go.Scatter(x=[xs[i]], y=[0], mode="markers",
                             marker=dict(size=32, color="#DC143C", line=dict(color="#FFD700", width=3))))
    frames.append(go.Frame(data=fd, name="done"))
    fig.frames = frames

    slider_steps = [
        dict(args=[[str(i)], dict(frame=dict(duration=400, redraw=True), mode="immediate",
                                  transition=dict(duration=400))],
             label=str(i+1), method="animate")
        for i in range(n)
    ] + [dict(args=[["done"], dict(frame=dict(duration=400, redraw=True), mode="immediate",
                                   transition=dict(duration=400))],
              label="done", method="animate")]

    fig.update_layout(
        margin=dict(l=80, r=80, t=60, b=120), height=420,
        xaxis=dict(visible=False, range=[-0.6, xs[-1]+0.6]),
        yaxis=dict(visible=False, range=[-1.6, 1.0]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        annotations=annotations, showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.0, xanchor="left", y=-0.18,
            font=dict(size=13, color="#6B0F1A"),
            bgcolor="rgba(255,240,240,.95)", bordercolor="#8B0000", borderwidth=1,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=800, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=400, easing="cubic-in-out"),
                                      mode="immediate")]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, yanchor="top", xanchor="left",
            currentvalue=dict(font=dict(size=13, color="#8B0000", weight=700),
                              prefix="Step: ", visible=True, xanchor="center"),
            font=dict(size=12, color="#333"),
            activebgcolor="#DC143C", bgcolor="#e0c0c0",
            bordercolor="#8B0000", borderwidth=1, tickcolor="#8B0000",
            transition=dict(duration=400, easing="cubic-in-out"),
            pad=dict(b=10, t=40), len=0.9, x=0.05, y=-0.08,
            steps=slider_steps,
        )],
    )
    return fig


def build_importance_figure():
    features = ["Age", "ACR (log)", "BMI", "Sex (Female)", "Race (NH Black)",
                "Race (NH White)", "Urine Albumin", "Race (Mexican Am.)", "eGFR", "Uric Acid"]
    values = [0.916, 0.673, 0.554, 0.360, 0.356, 0.317, 0.221, 0.217, 0.208, 0.176]
    n = len(features)
    colours = [f"rgb({107 + i*(220-107)//(n-1)}, {15 + i*5//(n-1)}, {26 + i*34//(n-1)})"
               for i in range(n)]
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker=dict(color=colours, line=dict(color="#8B0000", width=1.5)),
        text=[f"{v:.3f}" for v in values], textposition="outside",
        textfont=dict(size=11, color="#333", weight=600)))
    fig.update_layout(
        height=420, margin=dict(l=40, r=50, t=20, b=40),
        xaxis_title="Absolute LR Coefficient",
        xaxis=dict(title_font=dict(size=13, color="#8B0000"), tickfont=dict(size=11, color="#333")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#333")),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def build_waterfall_figure():
    baseline = 0.669  # model intercept (log-odds)
    contribs = [("ACR (log)", 0.848), ("Age", 0.579), ("Race (NH Black)", 0.356),
                ("BMI", 0.313), ("Uric Acid", 0.276), ("eGFR", 0.264),
                ("Urine Albumin", -0.160), ("Serum Albumin", -0.097)]
    labels   = [c[0] for c in contribs]
    measures = ["absolute"] + ["relative"] * len(contribs) + ["total"]
    x_vals   = ["Baseline"] + labels + ["Prediction"]
    y_vals   = [baseline] + [c[1] for c in contribs] + [baseline + sum(c[1] for c in contribs)]

    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=x_vals, y=y_vals,
        connector=dict(line=dict(color="#8B0000")),
        increasing=dict(marker=dict(color="#DC143C")),
        decreasing=dict(marker=dict(color="#4a7ebb")),
        totals=dict(marker=dict(color="#0b6e4f")),
        text=[f"{v:+.2f}" for v in y_vals], textposition="outside",
        textfont=dict(size=12, color="#333")))
    fig.update_layout(
        title=dict(text="Contribution waterfall for a synthetic patient",
                   font=dict(size=16, color="#6B0F1A")),
        yaxis_title="Risk contribution", xaxis_title="Features",
        height=460, margin=dict(l=40, r=40, t=60, b=60),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(size=12, color="#333"), title_font=dict(size=13, color="#8B0000")),
        yaxis=dict(tickfont=dict(size=11, color="#333"), title_font=dict(size=13, color="#8B0000")))
    return fig


# ── Sections ─────────────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>Ensemble Machine Learning for Hypertension Detection and Risk Stratification Using Renal and Haematological Biomarkers</h1>
        <p class="hero-subtitle"><strong>Author:</strong> Khaled Mazahreh &nbsp;|&nbsp; <strong>Supervisor:</strong> Dr. Maheen Hasib &nbsp;|&nbsp; <strong>Second Reader:</strong> Dr. Eric Aidoo</p>
        <p class="hero-meta"><strong>Degree:</strong> BSc. Data Sciences &nbsp;|&nbsp; <strong>Institution:</strong> Heriot Watt University &nbsp;|&nbsp; <strong>Date:</strong> 2026</p>
        <p class="hero-tagline">Using kidney and blood markers with explainable machine learning to detect high blood pressure before it causes permanent organ damage.</p>
    </div>""", unsafe_allow_html=True)


def render_burden():
    st.markdown('<h2 class="section-header">Global Burden &amp; Motivation</h2>', unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    for col, (num, label) in zip(cols, [
        ("1.4 B", "people living with hypertension worldwide"),
        ("10 M",  "preventable deaths annually from elevated BP"),
        ("~20 %", "of diagnosed patients achieve durable control"),
    ]):
        col.markdown(f'<div class="stat-card"><p class="stat-number">{num}</p>'
                     f'<p class="stat-label">{label}</p></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="margin-top:2rem;">
        <p style="font-size:1.05rem;line-height:1.8;margin:0;color:#333;">
        Traditional screening misses <strong>early kidney damage and blood vessel changes</strong>.
        This research uses the <strong>kidney-heart connection</strong> to build a practical screening
        tool from blood and urine markers available in routine clinical panels.
        </p>
    </div>""", unsafe_allow_html=True)


def render_methodology():
    st.markdown('<h2 class="section-header">Methodology</h2>', unsafe_allow_html=True)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <ul style="margin:0;">
            <li><strong>Data:</strong> Merged NHANES 2015-2016 and 2017-2018 adults (&ge;20); pregnancy excluded; n &asymp; 10,335.</li>
            <li><strong>Outcome:</strong> ACC/AHA 2017 hypertension (SBP &ge;130, DBP &ge;80, self-report, or meds). CKD-EPI 2021 eGFR; ACR (mg/g); BMI.</li>
            <li><strong>Preprocessing:</strong> Implausible values &rarr; missing; median/mode imputation; winsorised 5th/95th; log1p on ACR and creatinine; z-score standardisation; one-hot demographics.</li>
            <li><strong>Feature panels:</strong> <em>Panel A</em> (renal + demographics, 17 features), <em>Panel B</em> (CBC + demographics), <em>Panel C</em> (combined). Panel A dominates; Panel C adds &lt; 0.002 AUC.</li>
            <li><strong>Baselines:</strong> LR, Decision Tree, kNN, Linear SVM &mdash; trained on 2015-2016, validated on 2017-2018 (temporal split). LR and SVM led at AUC 0.818-0.819.</li>
            <li><strong>Ensembles:</strong> Random Forest, XGBoost, LightGBM (Bayesian-tuned, Optuna TPE, 100 trials); soft voting and stacking. Best: Stacking AUC 0.823 &mdash; only +0.4% over LR.</li>
            <li><strong>Recommendation:</strong> <strong style="color:#8B0000;">Panel A Logistic Regression</strong> (AUC 0.818, Brier 0.167) &mdash; 98.4% of best-ensemble discrimination with full interpretability.</li>
            <li><strong>Explainability:</strong> SHAP (TreeSHAP) + LR coefficients confirm age, ACR, and BMI as top predictors. Rankings concordant across models (&rho; &gt; 0.89).</li>
        </ul>
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(build_pipeline_figure(), use_container_width=True, config={"displayModeBar": False})


def render_results():
    st.markdown('<h2 class="section-header">Results &amp; Insights</h2>', unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    for col, (label, value) in zip(cols, [("AUC", "0.818"), ("Brier Score", "0.167"), ("Key Biomarkers", "10")]):
        col.markdown(f'<div class="stat-card"><p class="stat-label">{label}</p>'
                     f'<p class="stat-number">{value}</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-top:0;color:#6B0F1A;font-size:22px;">Waterfall: How Each Biomarker Shifts Risk</h4>
        <p style="margin:0 0 1rem;color:#333;font-size:17px;line-height:1.7;">
            Synthetic patient: 62 y/o male, BMI 33, eGFR 68, ACR 33, uric acid 7.4, albumin 3.8, NH Black.
        </p>
        <ul style="margin:0 0 1rem;color:#444;font-size:16px;line-height:1.8;">
            <li><strong>Baseline (0.67):</strong> Model intercept in log-odds space.</li>
            <li><strong>Red bars:</strong> ACR, age, race, BMI, uric acid, and eGFR decline push risk up.</li>
            <li><strong>Blue bars:</strong> Urine albumin and serum albumin pull risk down.</li>
            <li><strong>Prediction (3.07):</strong> High log-odds, corresponding to ~95.5% hypertension probability.</li>
        </ul>
    </div>""", unsafe_allow_html=True)

    st.plotly_chart(build_waterfall_figure(), use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(build_importance_figure(), use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown("""
        <div class="glass-card">
            <ul style="margin:0;">
                <li><strong>Age</strong> (|&beta;| = 0.92) and <strong>ACR</strong> (|&beta;| = 0.67) are the strongest predictors.</li>
                <li><strong>BMI</strong> (|&beta;| = 0.55) captures obesity-driven haemodynamic stress and is modifiable.</li>
                <li><strong>Sex &amp; Race</strong> encode demographic risk; female sex is protective, NH Black race increases risk.</li>
                <li><strong>eGFR</strong> (|&beta;| = 0.21) confirms the kidney-BP link; higher filtration is protective.</li>
            </ul>
        </div>""", unsafe_allow_html=True)


def render_calculator():
    header_l, header_r = st.columns([1.1, 0.9])
    with header_l:
        st.markdown('<h2 class="section-header">Prototype Risk Calculator</h2>', unsafe_allow_html=True)
    with header_r:
        st.markdown("""
        <div class="glass-card" style="padding:1.5rem 2rem;">
            <p style="margin:0;color:#333;font-size:1.05rem;line-height:1.7;">
                Powered by <strong style="color:#8B0000;">Panel A Logistic Regression</strong>
                (AUC 0.818, Brier 0.167). Enter biomarkers below to get a calibrated
                hypertension probability and per-feature breakdown.
            </p>
        </div>""", unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])
    with left:
        age  = st.number_input("Age (years)", 18, 90, 62, 1)
        sex  = st.selectbox("Sex", ["Female", "Male"], index=1)
        race = st.selectbox("Race / Ethnicity", list(RACE_MAP.keys()), index=1)
        height = st.number_input("Height (cm)", 120.0, 220.0, 175.0, 0.5)
        weight = st.number_input("Weight (kg)", 40.0, 200.0, 101.0, 0.5)
        scr    = st.number_input("Serum Creatinine (mg/dL)", 0.3, 8.0, 1.2, 0.1)
        uric   = st.number_input("Uric Acid (mg/dL)", 2.0, 13.0, 7.4, 0.1)
        alb    = st.number_input("Serum Albumin (g/dL)", 2.0, 6.0, 3.8, 0.1)
        bun    = st.number_input("Blood Urea Nitrogen (mg/dL)", 1.0, 60.0, 22.0, 0.5)
        u_alb  = st.number_input("Urine Albumin (mg/L)", 1.0, 1000.0, 33.0, 1.0)
        u_cr   = st.number_input("Urine Creatinine (mg/dL)", 5.0, 400.0, 100.0, 1.0)

    bmi  = compute_bmi(height, weight)
    egfr = compute_egfr(scr, age, sex)
    acr  = compute_acr(u_alb, u_cr)

    with right:
        # Derived metrics
        st.markdown("#### Derived Metrics")
        badges = ""
        if bmi:  badges += f'<span class="badge">BMI: {bmi:.1f} kg/m2</span>'
        if egfr: badges += f'<span class="badge">eGFR: {egfr:.0f} mL/min/1.73m2</span>'
        if acr:  badges += f'<span class="badge">ACR: {acr:.0f} mg/g</span>'
        st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)

        # Prediction
        inputs = dict(age=age, sex=sex, race=race, bmi=bmi, scr=scr, uric=uric,
                      albumin=alb, bun=bun, u_alb=u_alb, u_cr=u_cr, egfr=egfr, acr=acr)
        prob = predict_htn_risk(**inputs)
        pct  = prob * 100
        level = "Low" if prob < 0.40 else ("Moderate" if prob <= 0.60 else "High")
        pill  = {"Low": "risk-pill-low", "Moderate": "risk-pill-moderate", "High": "risk-pill-high"}[level]

        st.markdown(f"""
        <div class="glass-card" style="background:rgba(255,255,255,.95);">
            <p style="margin:0;color:#5c6784;font-size:.9rem;">ML Predicted Probability (Panel A LR)</p>
            <div style="display:flex;align-items:center;gap:1rem;margin-top:.3rem;">
                <div class="badge {pill}" style="font-size:1.1rem;">{level} risk</div>
                <span style="font-size:2rem;color:#072b58;">{pct:.1f}%</span>
            </div>
            <p style="margin-top:.8rem;color:#4a4f63;">
                Probability of meeting ACC/AHA 2017 hypertension definition.
            </p>
        </div>""", unsafe_allow_html=True)

        # Contribution chart
        contribs = compute_contributions(**inputs)
        top = contribs[:8]
        labels = [c[0] for c in top][::-1]
        vals   = [c[1] for c in top][::-1]

        fig = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker=dict(color=["#DC143C" if v > 0 else "#4a7ebb" for v in vals],
                        line=dict(color="#333", width=0.5)),
            text=[f"{v:+.2f}" for v in vals], textposition="outside",
            textfont=dict(size=11, color="#333")))
        fig.update_layout(
            title=dict(text="Top Feature Contributions", font=dict(size=15, color="#6B0F1A")),
            height=340, margin=dict(l=10, r=40, t=40, b=30),
            xaxis=dict(title="Log-odds contribution", zeroline=True, zerolinecolor="#999",
                       zerolinewidth=1.5, tickfont=dict(size=11, color="#333"),
                       title_font=dict(size=13, color="#8B0000")),
            yaxis=dict(tickfont=dict(size=12, color="#333")),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Rule-based reference score"):
            ref_level, ref_score, ref_notes = rule_based_score(bmi, egfr, acr, age, sex, race, uric)
            st.markdown(f"**{ref_level}** risk — score **{ref_score}**")
            for n in ref_notes:
                st.markdown(f"- {n}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    render_hero()
    st.divider()
    render_burden()
    st.divider()
    render_methodology()
    st.divider()
    render_results()
    st.divider()
    render_calculator()


if __name__ == "__main__":
    main()
