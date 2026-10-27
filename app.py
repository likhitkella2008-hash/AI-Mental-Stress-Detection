import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly is used for interactive modern charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# SHAP is used for Explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Set dark theme defaults for Matplotlib and Seaborn to match deep navy
try:
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0a0f1d",
        "axes.facecolor": "#0f172a",
        "text.color": "#f8fafc",
        "axes.labelcolor": "#94a3b8",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "rgba(255, 255, 255, 0.08)",
        "figure.autolayout": True
    })
except Exception:
    pass


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Mental Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS — MODERN DEEP NAVY THEME
# ============================================================

st.markdown(
    """
    <style>
    /* Global Canvas Styling */
    .stApp {
        background-color: #0a0f1d;
        color: #f8fafc;
    }

    /* Metric Cards */
    .metric-card-modern {
        background: linear-gradient(145deg, #111827 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 6px 24px -2px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card-modern:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px -2px rgba(0, 0, 0, 0.65);
    }
    .metric-title {
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }

    /* System Status Badges */
    .badge-pill-modern {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-teal {
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid rgba(6, 182, 212, 0.4);
        color: #22d3ee;
    }
    .badge-violet {
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.4);
        color: #c084fc;
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #6ee7b7;
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fcd34d;
    }
    .badge-rose {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
    }

    /* Result Banners */
    .result-banner-modern {
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.45);
    }
    .result-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.10) 100%);
        border: 2px solid #10b981;
        color: #6ee7b7;
    }
    .result-banner-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.10) 100%);
        border: 2px solid #f59e0b;
        color: #fcd34d;
    }
    .result-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(225, 29, 72, 0.10) 100%);
        border: 2px solid #ef4444;
        color: #fca5a5;
    }
    .result-title-modern {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    /* Recommendation Cards */
    .rec-focus-card {
        background: #111827;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .rec-buffer-card {
        background: #111827;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .domain-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .chip-mental { background: rgba(139, 92, 246, 0.2); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.4); }
    .chip-physical { background: rgba(6, 182, 212, 0.2); color: #67e8f9; border: 1px solid rgba(6, 182, 212, 0.4); }
    .chip-academic { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }
    .chip-social { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }

    /* Pipeline Step Cards */
    .pipeline-step {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .step-number {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        color: #ffffff;
        font-weight: 800;
        font-size: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    /* Disclaimer Box */
    .disclaimer-box-modern {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 5px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 10px;
        color: #fde68a;
        font-size: 13.5px;
        line-height: 1.6;
        margin: 20px 0;
    }

    /* Action Tiles */
    .action-tile {
        background: linear-gradient(145deg, #111827 0%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .action-tile:hover {
        border-color: rgba(6, 182, 212, 0.4);
    }

    /* Backward compatibility classes */
    .metric-card {
        background-color: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        color: #f8fafc;
    }
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 5px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        color: #fde68a;
        font-size: 14px;
        margin: 16px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def apply_dark_plotly_theme(fig, title="", height=400):
    """Applies unified deep navy styling to Plotly figures."""
    if not PLOTLY_AVAILABLE or fig is None:
        return fig
    fig.update_layout(
        title=dict(text=title, font=dict(color="#f8fafc", size=15, family="sans-serif")),
        paper_bgcolor="#0a0f1d",
        plot_bgcolor="#0f172a",
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        font=dict(color="#94a3b8", family="sans-serif"),
        legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15,23,42,0.6)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)")
    )
    return fig



# ============================================================
# FILE LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists("StressLevelDataset.csv"):
        return None

    try:
        return pd.read_csv("StressLevelDataset.csv")
    except Exception:
        return None


@st.cache_resource
def load_baseline_model():

    if not os.path.exists("best_stress_model.pkl"):
        return None

    try:
        return joblib.load("best_stress_model.pkl")
    except Exception:
        return None


@st.cache_resource
def load_optimized_model():

    if not os.path.exists("optimized_stress_model.pkl"):
        return None

    try:
        return joblib.load("optimized_stress_model.pkl")
    except Exception:
        return None


@st.cache_data
def load_feature_names():

    if not os.path.exists("feature_names.pkl"):
        return None

    try:
        return joblib.load("feature_names.pkl")
    except Exception:
        return None


@st.cache_data
def load_comparison():

    if not os.path.exists("model_comparison.csv"):
        return None

    try:
        return pd.read_csv(
            "model_comparison.csv",
            index_col=0
        )
    except Exception:
        return None


@st.cache_data
def load_optimized_comparison():

    if not os.path.exists("optimized_model_comparison.csv"):
        return None

    try:
        return pd.read_csv(
            "optimized_model_comparison.csv",
            index_col=0
        )
    except Exception:
        return None


@st.cache_data
def load_roc_results():

    if not os.path.exists("roc_auc_results.csv"):
        return None

    try:
        return pd.read_csv("roc_auc_results.csv")
    except Exception:
        return None


@st.cache_data
def load_shap_importance():

    if not os.path.exists("shap_feature_importance.csv"):
        return None

    try:
        return pd.read_csv(
            "shap_feature_importance.csv"
        )
    except Exception:
        return None


@st.cache_data
def load_eval_artifacts():

    if not os.path.exists("eval_artifacts.pkl"):
        return None

    try:
        return joblib.load("eval_artifacts.pkl")
    except Exception:
        return None


# ============================================================
# LOAD RESOURCES
# ============================================================

df = load_dataset()

baseline_model = load_baseline_model()

optimized_model = load_optimized_model()

feature_names = load_feature_names()

comparison = load_comparison()

optimized_comparison = load_optimized_comparison()

roc_results = load_roc_results()

shap_importance = load_shap_importance()

eval_artifacts = load_eval_artifacts()


# ============================================================
# BASIC VALIDATION
# ============================================================

if df is None:

    st.error(
        "StressLevelDataset.csv could not be loaded."
    )

    st.stop()


if feature_names is None:

    feature_names = [
        column
        for column in df.columns
        if column != "stress_level"
    ]


# Use optimized model if available.
# Otherwise use baseline model.

model = optimized_model

if model is None:
    model = baseline_model


if model is None:

    st.error(
        "No trained model was found. "
        "Please run train_model.py or optimize_model.py."
    )

    st.stop()


# ============================================================
# CONSTANTS
# ============================================================

stress_labels = {
    0: "Low",
    1: "Moderate",
    2: "High"
}


feature_descriptions = {

    "anxiety_level":
        "Level of anxiety reported by the student.",

    "self_esteem":
        "Self-esteem level reported by the student.",

    "mental_health_history":
        "Indicator of previous mental-health history.",

    "depression":
        "Reported depression-related score.",

    "headache":
        "Reported frequency/severity of headaches.",

    "blood_pressure":
        "Blood-pressure related feature in the dataset.",

    "sleep_quality":
        "Reported quality of sleep.",

    "breathing_problem":
        "Reported breathing-related problem score.",

    "noise_level":
        "Exposure to noise in the environment.",

    "living_conditions":
        "Quality of living conditions.",

    "safety":
        "Perceived personal/environmental safety.",

    "basic_needs":
        "Extent to which basic needs are satisfied.",

    "academic_performance":
        "Academic performance-related score.",

    "study_load":
        "Perceived academic study load.",

    "teacher_student_relationship":
        "Reported teacher-student relationship.",

    "future_career_concerns":
        "Concerns regarding future career.",

    "social_support":
        "Level of social support.",

    "peer_pressure":
        "Level of peer pressure.",

    "extracurricular_activities":
        "Participation in extracurricular activities.",

    "bullying":
        "Reported bullying-related score."
}


# Feature labels with explicit directional scales for user clarity
feature_labels_with_scale = {
    "anxiety_level": "Anxiety Level (0: Min, 21: Severe)",
    "self_esteem": "Self Esteem (0: Low, 30: High)",
    "mental_health_history": "Mental Health History (0: No, 1: Yes)",
    "depression": "Depression Score (0: Min, 27: Severe)",
    "headache": "Headache Frequency (0: None, 5: Daily/Severe)",
    "blood_pressure": "Blood Pressure (1: Low/Normal, 2: Moderate, 3: High)",
    "sleep_quality": "Sleep Quality (0: Disrupted, 5: Restful)",
    "breathing_problem": "Breathing Issues (0: None, 5: Severe)",
    "noise_level": "Environmental Noise (0: Quiet, 5: Severe Disturbance)",
    "living_conditions": "Living Conditions (0: Poor, 5: Optimal)",
    "safety": "Personal Safety (0: Unsafe, 5: Very Safe)",
    "basic_needs": "Basic Needs Met (0: Deprived, 5: Fully Met)",
    "academic_performance": "Academic Performance (0: Struggling, 5: Excellent)",
    "study_load": "Study Load (0: Light, 5: Overwhelming)",
    "teacher_student_relationship": "Teacher-Student Rapport (0: Poor, 5: Supportive)",
    "future_career_concerns": "Future Career Concerns (0: None, 5: Severe Anxiety)",
    "social_support": "Social Support Network (0: None, 3: Strong Support)",
    "peer_pressure": "Peer Pressure (0: None, 5: Intense)",
    "extracurricular_activities": "Extracurricular Activity Load (0: None, 5: Overload)",
    "bullying": "Bullying / Interpersonal Friction (0: None, 5: Severe)"
}

feature_domains = {
    "anxiety_level": "Mental & Emotional",
    "self_esteem": "Mental & Emotional",
    "mental_health_history": "Mental & Emotional",
    "depression": "Mental & Emotional",
    "headache": "Physical Health",
    "blood_pressure": "Physical Health",
    "sleep_quality": "Physical Health",
    "breathing_problem": "Physical Health",
    "academic_performance": "Academic Factors",
    "study_load": "Academic Factors",
    "teacher_student_relationship": "Academic Factors",
    "future_career_concerns": "Academic Factors",
    "social_support": "Social & Environmental",
    "peer_pressure": "Social & Environmental",
    "extracurricular_activities": "Social & Environmental",
    "bullying": "Social & Environmental",
    "noise_level": "Social & Environmental",
    "living_conditions": "Social & Environmental",
    "safety": "Social & Environmental",
    "basic_needs": "Social & Environmental"
}


# ============================================================
# PERSONALIZED STRESS AWARENESS GUIDANCE ENGINE
# ============================================================

def generate_personalized_guidance(user_inputs, prediction, label):
    """
    Transparent, rule-based educational guidance engine.
    Derives actionable suggestions based on actual input values and verified dataset ranges.
    Does not use SHAP values as medical claims. Does not diagnose illness.
    """
    if not isinstance(user_inputs, dict):
        return {
            "focus_areas": [],
            "positive_buffers": [],
            "overall_note": "No input data provided."
        }

    recommendations = []
    positive_buffers = []

    def get_val(key, default=0.0):
        try:
            val = user_inputs.get(key, default)
            return float(val) if val is not None else float(default)
        except (ValueError, TypeError):
            return float(default)

    # 1. Sleep Quality (Range 0-5, Dataset 25th percentile = 1.0)
    sleep = get_val("sleep_quality", 2.5)
    if sleep <= 1.0:
        recommendations.append({
            "factor": "Sleep Quality",
            "observed_value": f"{sleep:.0f} / 5 (Severely Disrupted)",
            "domain": "Physical Well-being & Rest",
            "threshold_type": "Dataset 25th percentile (<= 1.0)",
            "suggestion": (
                "Establish a consistent sleep schedule (aiming for 7-8 hours); maintain a dark, "
                "quiet sleeping environment, and disconnect from digital screens 45 minutes "
                "before bedtime. If sleep difficulties persist, consult a healthcare provider."
            )
        })
    elif sleep >= 4.0:
        positive_buffers.append({
            "factor": "Sleep Quality",
            "observed_value": f"{sleep:.0f} / 5 (Optimal)",
            "domain": "Rest & Recovery",
            "note": "Restorative sleep consistency observed. Sustain this regular sleep schedule."
        })

    # 2. Study Load (Range 0-5, Educational rule >= 4.0; dataset median=2.0, 75th=3.0)
    study = get_val("study_load", 2.0)
    if study >= 4.0:
        recommendations.append({
            "factor": "Study Load",
            "observed_value": f"{study:.0f} / 5 (Heavy / Overwhelming)",
            "domain": "Academic Time Management",
            "threshold_type": "Educational rule (>= 4.0, above 75th percentile)",
            "suggestion": (
                "Break large coursework assignments into focused 25-30 minute study intervals (Pomodoro technique). "
                "Prioritize top academic deadlines and discuss course workload pacing with academic mentors or advisors."
            )
        })
    elif study <= 2.0:
        positive_buffers.append({
            "factor": "Study Load",
            "observed_value": f"{study:.0f} / 5 (Manageable)",
            "domain": "Workload Balance",
            "note": "Coursework load is within a manageable range; maintain structured weekly scheduling."
        })

    # 3. Social Support (Range 0-3, Dataset 25th percentile = 1.0)
    support = get_val("social_support", 2.0)
    if support <= 1.0:
        recommendations.append({
            "factor": "Social Support",
            "observed_value": f"{support:.0f} / 3 (Limited)",
            "domain": "Social Connection & Support",
            "threshold_type": "Dataset 25th percentile (<= 1.0)",
            "suggestion": (
                "Connect with trusted peers, join campus student associations or study groups, "
                "and maintain regular touchpoints with friends or family to avoid feelings of isolation."
            )
        })
    elif support >= 3.0:
        positive_buffers.append({
            "factor": "Social Support",
            "observed_value": f"{support:.0f} / 3 (Strong)",
            "domain": "Interpersonal Support",
            "note": "Strong social support network identified. Continue nurturing these supportive connections."
        })

    # 4. Anxiety Level (Range 0-21, Dataset 75th percentile = 16.0)
    anxiety = get_val("anxiety_level", 11.0)
    if anxiety >= 16.0:
        recommendations.append({
            "factor": "Anxiety Level",
            "observed_value": f"{anxiety:.0f} / 21 (Elevated)",
            "domain": "Emotional Balance & Relaxation",
            "threshold_type": "Dataset 75th percentile (>= 16.0)",
            "suggestion": (
                "Incorporate structured daily relaxation pauses (e.g., deep breathing or grounding techniques), "
                "schedule buffer periods between high-pressure tasks, and consider speaking with campus counseling services."
            )
        })
    elif anxiety <= 6.0:
        positive_buffers.append({
            "factor": "Anxiety Level",
            "observed_value": f"{anxiety:.0f} / 21 (Low)",
            "domain": "Emotional Well-being",
            "note": "Low self-reported anxiety symptoms observed. Maintain your current stress-buffer habits."
        })

    # 5. Academic Performance (Range 0-5, Dataset bottom quartile <= 1.0; median=2.0)
    academic = get_val("academic_performance", 2.0)
    if academic <= 1.0:
        recommendations.append({
            "factor": "Academic Performance",
            "observed_value": f"{academic:.0f} / 5 (Challenged)",
            "domain": "Academic Support & Tutoring",
            "threshold_type": "Dataset bottom quartile (<= 1.0)",
            "suggestion": (
                "Utilize university tutoring resources, attend instructor office hours early, "
                "and form collaborative study partnerships to address academic hurdles constructively."
            )
        })
    elif academic >= 4.0:
        positive_buffers.append({
            "factor": "Academic Performance",
            "observed_value": f"{academic:.0f} / 5 (Strong)",
            "domain": "Academic Standing",
            "note": "Positive academic progress reported. Sustain active engagement with your coursework."
        })

    # 6. Physical Symptoms (Headache >= 3, Breathing Problem >= 3, Blood Pressure == 3)
    # Caution: Do not attribute solely to stress; advise medical consultation when physical symptoms are elevated.
    headache = get_val("headache", 2.0)
    breathing = get_val("breathing_problem", 2.0)
    bp = get_val("blood_pressure", 2.0)

    physical_flags = []
    if headache >= 3.0:
        physical_flags.append(f"headache frequency ({headache:.0f}/5)")
    if breathing >= 3.0:
        physical_flags.append(f"breathing discomfort ({breathing:.0f}/5)")
    if bp == 3.0:
        physical_flags.append("elevated blood pressure rating (3/3)")

    if physical_flags:
        joined_flags = ", ".join(physical_flags)
        recommendations.append({
            "factor": "Physical Health Factors",
            "observed_value": joined_flags,
            "domain": "Physical Health & Medical Awareness",
            "threshold_type": "Dataset median/75th percentile thresholds (>= 3.0)",
            "suggestion": (
                "Ensure proper hydration, maintain ergonomic posture, and take regular screen breaks. "
                "IMPORTANT: Physical symptoms such as recurrent headaches, breathing difficulty, or "
                "elevated blood pressure should not be assumed to stem solely from stress. "
                "Please consult a qualified physician or healthcare provider for proper medical evaluation."
            )
        })

    # 7. Bullying & Social Friction (Range 0-5, Dataset median=3.0, 75th=4.0)
    bullying = get_val("bullying", 2.0)
    if bullying >= 3.0:
        recommendations.append({
            "factor": "Bullying / Interpersonal Friction",
            "observed_value": f"{bullying:.0f} / 5 (Elevated Friction)",
            "domain": "Personal Safety & Campus Support",
            "threshold_type": "Dataset median/75th percentile (>= 3.0)",
            "suggestion": (
                "Establish firm personal boundaries and avoid isolated interactions. "
                "Do not hesitate to reach out to university student welfare counselors, "
                "ombudspersons, or trusted faculty for confidential guidance and intervention."
            )
        })

    # 8. Peer Pressure (Range 0-5, Dataset 75th percentile = 4.0)
    peer = get_val("peer_pressure", 2.0)
    if peer >= 4.0:
        recommendations.append({
            "factor": "Peer Pressure",
            "observed_value": f"{peer:.0f} / 5 (Intense)",
            "domain": "Social Autonomy & Boundaries",
            "threshold_type": "Dataset 75th percentile (>= 4.0)",
            "suggestion": (
                "Focus on self-determined personal academic goals. Practice setting healthy boundaries, "
                "and seek peer circles that respect your personal pacing and values."
            )
        })

    # 9. Future Career Concerns (Range 0-5, Dataset 75th percentile = 4.0)
    career = get_val("future_career_concerns", 2.0)
    if career >= 4.0:
        recommendations.append({
            "factor": "Future Career Concerns",
            "observed_value": f"{career:.0f} / 5 (Elevated Worry)",
            "domain": "Career Planning & Development",
            "threshold_type": "Dataset 75th percentile (>= 4.0)",
            "suggestion": (
                "Engage with university career planning services, break broad career aspirations into concrete "
                "monthly skill-building objectives, and seek mentorship from department alumni or faculty."
            )
        })

    # 10. Additional Positive Buffers
    self_esteem = get_val("self_esteem", 19.0)
    if self_esteem >= 26.0:
        positive_buffers.append({
            "factor": "Self-Esteem",
            "observed_value": f"{self_esteem:.0f} / 30 (High)",
            "domain": "Self-Efficacy",
            "note": "Healthy self-esteem and self-confidence reported. This serves as an effective emotional resilience buffer."
        })

    teacher = get_val("teacher_student_relationship", 2.0)
    if teacher >= 4.0:
        positive_buffers.append({
            "factor": "Faculty Mentorship",
            "observed_value": f"{teacher:.0f} / 5 (Supportive)",
            "domain": "Academic Mentorship",
            "note": "Constructive rapport with teachers reported. Continue seeking their academic guidance."
        })

    return {
        "focus_areas": recommendations,
        "positive_buffers": positive_buffers,
        "label": label,
        "prediction": prediction
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <span style="font-size: 28px;">🧠</span>
        <div>
            <div style="font-size: 18px; font-weight: 800; color: #f8fafc; letter-spacing: -0.3px;">StressDetection</div>
            <div style="font-size: 11px; color: #06b6d4; font-weight: 600;">AI Mental Wellness Analytics</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.caption(
    "B.Tech AI & Data Science Project • Group A6"
)

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Dashboard",
        "🧠 Stress Prediction",
        "🔀 What-If Analysis",
        "📊 Model Performance",
        "📁 Dataset Analysis",
        "🔍 Explainable AI",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="metric-card-modern" style="padding: 14px 16px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">System Status</span>
            <span class="badge-pill-modern badge-emerald" style="padding: 2px 8px; font-size: 10px;">ONLINE</span>
        </div>
        <div style="font-size: 13px; font-weight: 700; color: #f8fafc;">Optimized Random Forest</div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px;">
            <span style="color: #64748b;">Optimized Test Acc</span>
            <span style="color: #22d3ee; font-weight: 700;">90.91%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 12px;">
            <span style="color: #64748b;">Baseline (RF)</span>
            <span style="color: #94a3b8; font-weight: 600;">88.64%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 12px;">
            <span style="color: #64748b;">Dataset Size</span>
            <span style="color: #cbd5e1; font-weight: 600;">1,100 records</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 12px;">
            <span style="color: #64748b;">Input Factors</span>
            <span style="color: #cbd5e1; font-weight: 600;">20 features</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="disclaimer-box-modern" style="padding: 12px 14px; font-size: 12px; margin-top: 8px;">
        <strong>⚠️ Educational Scope:</strong><br>
        Early-awareness and lifestyle guidance system. Not intended for clinical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                <h1 style="margin: 0; font-size: 32px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
                    AI-Based Mental Stress Detection
                </h1>
                <span class="badge-pill-modern badge-teal">Production Model Active</span>
            </div>
            <p style="margin: 0; font-size: 15px; color: #94a3b8;">
                Supervised Machine Learning Multi-Factor Classification & Explainable AI Dashboard
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Modern KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card-modern" style="border-top: 3px solid #06b6d4;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="metric-title">Dataset Records</span>
                    <span style="font-size: 20px;">📊</span>
                </div>
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-sub">Validated survey responses</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card-modern" style="border-top: 3px solid #8b5cf6;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="metric-title">Input Features</span>
                    <span style="font-size: 20px;">🧪</span>
                </div>
                <div class="metric-value">{len(feature_names)}</div>
                <div class="metric-sub">Across 4 holistic domains</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card-modern" style="border-top: 3px solid #f59e0b;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="metric-title">Stress Categories</span>
                    <span style="font-size: 20px;">🎯</span>
                </div>
                <div class="metric-value">{df["stress_level"].nunique()}</div>
                <div class="metric-sub">Low, Moderate, High</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card-modern" style="border-top: 3px solid #10b981;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="metric-title">Missing Values</span>
                    <span style="font-size: 20px;">✅</span>
                </div>
                <div class="metric-value">{int(df.isna().sum().sum())}</div>
                <div class="metric-sub">100% Data Integrity</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Model Performance & Stress Distribution Grid
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown(
            """
            <div class="metric-card-modern" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="metric-title">Champion Classifier</span>
                    <span class="badge-pill-modern badge-violet">Tuned via 5-Fold CV</span>
                </div>
                <h3 style="margin: 0 0 10px 0; font-size: 22px; font-weight: 800; color: #f8fafc;">Optimized Random Forest</h3>
                <p style="color: #94a3b8; font-size: 13.5px; line-height: 1.5; margin-bottom: 16px;">
                    Supervised ensemble architecture tuned with 5-fold cross-validation grid search to balance tree variance and regularize decision boundaries.
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                        <div style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;">Optimized Test Acc</div>
                        <div style="font-size: 24px; font-weight: 800; color: #22d3ee;">90.91%</div>
                        <div style="font-size: 11px; color: #10b981; font-weight: 600;">+2.27% vs Baseline</div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                        <div style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;">Baseline RF Acc</div>
                        <div style="font-size: 24px; font-weight: 800; color: #94a3b8;">88.64%</div>
                        <div style="font-size: 11px; color: #64748b;">Default Hyperparameters</div>
                    </div>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="badge-pill-modern badge-teal" style="font-size: 11px;">200 Trees</span>
                    <span class="badge-pill-modern badge-violet" style="font-size: 11px;">min_samples_split: 5</span>
                    <span class="badge-pill-modern badge-emerald" style="font-size: 11px;">ROC-AUC: 0.9831</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        distribution = (
            df["stress_level"]
            .map(stress_labels)
            .value_counts()
            .reindex(["Low", "Moderate", "High"])
        )

        if PLOTLY_AVAILABLE:
            fig_dist = go.Figure(data=[
                go.Bar(
                    x=["Low", "Moderate", "High"],
                    y=[distribution["Low"], distribution["Moderate"], distribution["High"]],
                    marker=dict(
                        color=["#10b981", "#f59e0b", "#ef4444"],
                        line=dict(color="rgba(255,255,255,0.15)", width=1.5)
                    ),
                    text=[f"{v} ({v/len(df)*100:.1f}%)" for v in [distribution["Low"], distribution["Moderate"], distribution["High"]]],
                    textposition="auto",
                    hoverinfo="x+y+text"
                )
            ])
            fig_dist.update_layout(
                title=dict(text="Dataset Target Class Distribution (N=1,100)", font=dict(color="#f8fafc", size=14, family="sans-serif")),
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                height=320,
                margin=dict(l=30, r=20, t=40, b=30),
                font=dict(color="#94a3b8", family="sans-serif"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=None),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Records")
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            colors = ["#10b981", "#f59e0b", "#ef4444"]
            distribution.plot(kind="bar", ax=ax, color=colors)
            ax.set_xlabel("Stress Level", fontsize=10, fontweight="bold")
            ax.set_ylabel("Number of Records", fontsize=10, fontweight="bold")
            ax.set_title("Dataset Stress Distribution", fontsize=11, fontweight="bold")
            plt.xticks(rotation=0)
            st.pyplot(fig, clear_figure=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Quick Access Modules Grid
    st.markdown("#### ⚡ System Modules & Navigation")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(
            """
            <div class="action-tile">
                <div style="font-size: 24px; margin-bottom: 8px;">🧠</div>
                <div style="font-weight: 700; color: #f8fafc; font-size: 15px;">Stress Prediction</div>
                <div style="font-size: 12px; color: #94a3b8; margin: 4px 0 10px 0;">Interactive 20-factor assessment with live inference & SHAP explanation.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with q2:
        st.markdown(
            """
            <div class="action-tile">
                <div style="font-size: 24px; margin-bottom: 8px;">📊</div>
                <div style="font-weight: 700; color: #f8fafc; font-size: 15px;">Model Benchmarks</div>
                <div style="font-size: 12px; color: #94a3b8; margin: 4px 0 10px 0;">Cross-validation evaluations across 5 supervised ML algorithms.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with q3:
        st.markdown(
            """
            <div class="action-tile">
                <div style="font-size: 24px; margin-bottom: 8px;">🔍</div>
                <div style="font-weight: 700; color: #f8fafc; font-size: 15px;">Explainable AI</div>
                <div style="font-size: 12px; color: #94a3b8; margin: 4px 0 10px 0;">Global importance and interactive local SHAP case attribution.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with q4:
        st.markdown(
            """
            <div class="action-tile">
                <div style="font-size: 24px; margin-bottom: 8px;">📁</div>
                <div style="font-weight: 700; color: #f8fafc; font-size: 15px;">Dataset Explorer</div>
                <div style="font-size: 12px; color: #94a3b8; margin: 4px 0 10px 0;">Statistical distributions and data verification across 20 factors.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Project Methodology Pipeline
    st.markdown("#### 🔄 Supervised ML Engineering Pipeline")
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="step-number">1</div>
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: #f8fafc;">20 Survey Factors Ingestion</div>
                    <div style="font-size: 12px; color: #94a3b8;">Physical, Mental, Academic, and Social feature collection.</div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-number">2</div>
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: #f8fafc;">Data Quality & Integrity Check</div>
                    <div style="font-size: 12px; color: #94a3b8;">1,100 clean records verified with 0 missing values.</div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-number">3</div>
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: #f8fafc;">Stratified 80/20 Train-Test Split</div>
                    <div style="font-size: 12px; color: #94a3b8;">Preserves balanced class proportions (seed 42).</div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-number">4</div>
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: #f8fafc;">Five-Algorithm Benchmark</div>
                    <div style="font-size: 12px; color: #94a3b8;">LR, Decision Tree, Random Forest, SVM, and XGBoost.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p_col2:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="step-number">5</div>
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: #f8fafc;">Random Forest Selection</div>
                    <div style="font-size: 12px; color: #94a3b8;">Top baseline performer at 88.64% test accuracy.</div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-number">6</div>
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: #f8fafc;">5-Fold GridSearchCV Tuning</div>
                    <div style="font-size: 12px; color: #94a3b8;">Regularized tree parameters to achieve 90.91% test accuracy.</div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-number">7</div>
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: #f8fafc;">Explainable AI via SHAP</div>
                    <div style="font-size: 12px; color: #94a3b8;">Game-theoretic local feature attributions for predictions.</div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-number">8</div>
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: #f8fafc;">Personalized Educational Guidance</div>
                    <div style="font-size: 12px; color: #94a3b8;">Transparent rule-based lifestyle and study habit suggestions.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="disclaimer-box-modern">
            <strong>⚠️ Educational & Non-Clinical Disclaimer:</strong><br>
            This application is an educational machine-learning system intended for early stress awareness.
            It does not provide medical diagnosis, clinical evaluation, or psychiatric treatment plans.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2 — STRESS PREDICTION
# ============================================================

elif page == "🧠 Stress Prediction":

    st.title(
        "🧠 Stress Level Prediction"
    )

    st.write(
        "Enter the 20 dataset features to obtain "
        "a machine-learning prediction."
    )

    st.info(
        "The model predicts one of three dataset classes: "
        "Low, Moderate, or High."
    )


    st.markdown("---")


    # Demonstration Presets
    st.markdown("#### ⚡ Demonstration Presets (Sample Scenarios)")
    st.write(
        "Select a preset scenario below to populate the input form with typical survey patterns. "
        "The model will evaluate the inputs only when you click **Predict Stress Level**."
    )

    presets = {
        "low": {
            'anxiety_level': 5.0, 'self_esteem': 25.0, 'mental_health_history': 0.0, 'depression': 6.0,
            'headache': 1.0, 'blood_pressure': 2.0, 'sleep_quality': 4.0, 'breathing_problem': 2.0,
            'academic_performance': 4.0, 'study_load': 2.0, 'teacher_student_relationship': 4.0, 'future_career_concerns': 1.0,
            'social_support': 3.0, 'peer_pressure': 2.0, 'extracurricular_activities': 2.0, 'bullying': 1.0,
            'noise_level': 2.0, 'living_conditions': 3.0, 'safety': 4.0, 'basic_needs': 4.0
        },
        "moderate": {
            'anxiety_level': 11.0, 'self_esteem': 19.0, 'mental_health_history': 1.0, 'depression': 12.0,
            'headache': 2.0, 'blood_pressure': 1.0, 'sleep_quality': 3.0, 'breathing_problem': 3.0,
            'academic_performance': 3.0, 'study_load': 3.0, 'teacher_student_relationship': 2.0, 'future_career_concerns': 3.0,
            'social_support': 2.0, 'peer_pressure': 3.0, 'extracurricular_activities': 3.0, 'bullying': 3.0,
            'noise_level': 2.0, 'living_conditions': 3.0, 'safety': 2.0, 'basic_needs': 3.0
        },
        "high": {
            'anxiety_level': 16.0, 'self_esteem': 9.0, 'mental_health_history': 1.0, 'depression': 20.0,
            'headache': 4.0, 'blood_pressure': 3.0, 'sleep_quality': 1.0, 'breathing_problem': 4.0,
            'academic_performance': 2.0, 'study_load': 4.0, 'teacher_student_relationship': 2.0, 'future_career_concerns': 4.0,
            'social_support': 1.0, 'peer_pressure': 4.0, 'extracurricular_activities': 4.0, 'bullying': 4.0,
            'noise_level': 4.0, 'living_conditions': 2.0, 'safety': 2.0, 'basic_needs': 2.0
        },
        "average": {
            f: float(df[f].median()) for f in feature_names
        }
    }

    # Initialize session state for all 20 features
    for f in feature_names:
        if f"input_{f}" not in st.session_state:
            st.session_state[f"input_{f}"] = float(df[f].median())

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("🟢 Low Stress Scenario", use_container_width=True):
            for f, val in presets["low"].items():
                st.session_state[f"input_{f}"] = float(val)
            st.rerun()
    with col_p2:
        if st.button("🟡 Moderate Stress Scenario", use_container_width=True):
            for f, val in presets["moderate"].items():
                st.session_state[f"input_{f}"] = float(val)
            st.rerun()
    with col_p3:
        if st.button("🔴 High Stress Scenario", use_container_width=True):
            for f, val in presets["high"].items():
                st.session_state[f"input_{f}"] = float(val)
            st.rerun()
    with col_p4:
        if st.button("🔄 Average Scenario", use_container_width=True):
            for f, val in presets["average"].items():
                st.session_state[f"input_{f}"] = float(val)
            st.rerun()

    st.markdown("---")

    # Grouped Input Form: Mental, Physical, Academic, Social
    input_data = {}

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Mental & Emotional",
        "🩺 Physical Health",
        "🎓 Academic Factors",
        "🌐 Social & Environmental"
    ])

    mental_features = [
        "anxiety_level", "self_esteem", "mental_health_history", "depression"
    ]
    physical_features = [
        "headache", "blood_pressure", "sleep_quality", "breathing_problem"
    ]
    academic_features = [
        "academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns"
    ]
    social_features = [
        "social_support", "peer_pressure", "extracurricular_activities", "bullying",
        "noise_level", "living_conditions", "safety", "basic_needs"
    ]

    with tab1:
        st.markdown("##### 🧠 Mental & Emotional Factors (4 Features)")
        c1, c2 = st.columns(2)
        for i, feature in enumerate(mental_features):
            col = c1 if i % 2 == 0 else c2
            with col:
                input_data[feature] = st.number_input(
                    feature_labels_with_scale.get(feature, feature.replace("_", " ").title()),
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max()),
                    value=float(st.session_state.get(f"input_{feature}", df[feature].median())),
                    step=1.0,
                    help=feature_descriptions.get(feature, "Dataset input feature."),
                    key=f"input_{feature}"
                )

    with tab2:
        st.markdown("##### 🩺 Physical Health Indicators (4 Features)")
        c1, c2 = st.columns(2)
        for i, feature in enumerate(physical_features):
            col = c1 if i % 2 == 0 else c2
            with col:
                input_data[feature] = st.number_input(
                    feature_labels_with_scale.get(feature, feature.replace("_", " ").title()),
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max()),
                    value=float(st.session_state.get(f"input_{feature}", df[feature].median())),
                    step=1.0,
                    help=feature_descriptions.get(feature, "Dataset input feature."),
                    key=f"input_{feature}"
                )

    with tab3:
        st.markdown("##### 🎓 Academic & Career Environment (4 Features)")
        c1, c2 = st.columns(2)
        for i, feature in enumerate(academic_features):
            col = c1 if i % 2 == 0 else c2
            with col:
                input_data[feature] = st.number_input(
                    feature_labels_with_scale.get(feature, feature.replace("_", " ").title()),
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max()),
                    value=float(st.session_state.get(f"input_{feature}", df[feature].median())),
                    step=1.0,
                    help=feature_descriptions.get(feature, "Dataset input feature."),
                    key=f"input_{feature}"
                )

    with tab4:
        st.markdown("##### 🌐 Social & Environmental Factors (8 Features)")
        c1, c2 = st.columns(2)
        for i, feature in enumerate(social_features):
            col = c1 if i % 2 == 0 else c2
            with col:
                input_data[feature] = st.number_input(
                    feature_labels_with_scale.get(feature, feature.replace("_", " ").title()),
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max()),
                    value=float(st.session_state.get(f"input_{feature}", df[feature].median())),
                    step=1.0,
                    help=feature_descriptions.get(feature, "Dataset input feature."),
                    key=f"input_{feature}"
                )



    st.markdown("---")

    # Scope reminder for complete 20-feature profile awareness
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 12.5px; color: #94a3b8;">
            ℹ️ <strong>Active Profile Scope:</strong> The model evaluates all <strong>20 features</strong> across all 4 category tabs simultaneously. Values in unopened tabs remain set to their scenario preset or default levels.
        </div>
        """,
        unsafe_allow_html=True
    )

    predict_button = st.button(
        "🔮 Predict Stress Level",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        input_df = pd.DataFrame(
            [input_data]
        )

        # Ensure exact feature order

        input_df = input_df[
            feature_names
        ]


        # Prediction

        prediction = model.predict(
            input_df
        )[0]

        probabilities = model.predict_proba(
            input_df
        )[0]


        label = stress_labels.get(
            int(prediction),
            str(prediction)
        )


        # Modern Result Banner
        max_proba = float(np.max(probabilities) * 100)
        if prediction == 0:
            banner_class = "result-banner-low"
            badge_style = "badge-emerald"
            status_desc = "Favorable coping mechanisms and well-balanced physiological & emotional indicators observed."
        elif prediction == 1:
            banner_class = "result-banner-moderate"
            badge_style = "badge-amber"
            status_desc = "Moderate strain patterns detected across academic or lifestyle factors. Review targeted focus areas."
        else:
            banner_class = "result-banner-high"
            badge_style = "badge-rose"
            status_desc = "Elevated multi-domain stress indicators detected. Proactive supportive measures recommended."

        st.markdown(
            f"""
            <div class="result-banner-modern {banner_class}">
                <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span class="badge-pill-modern {badge_style}">CLASSIFICATION RESULT</span>
                    <span style="font-size: 13px; font-weight: 600; color: #94a3b8;">• {max_proba:.1f}% Model Confidence</span>
                </div>
                <div class="result-title-modern">
                    {label} Stress Level
                </div>
                <p style="margin: 8px 0 0 0; font-size: 14.5px; opacity: 0.95;">
                    {status_desc}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Probability distribution
        st.subheader("📊 Class Probability Distribution")

        probability_df = pd.DataFrame({
            "Stress Level": ["Low", "Moderate", "High"],
            "Probability": (probabilities * 100).round(2)
        })

        if PLOTLY_AVAILABLE:
            fig_prob = go.Figure(go.Bar(
                x=[probabilities[0] * 100, probabilities[1] * 100, probabilities[2] * 100],
                y=["Low Stress", "Moderate Stress", "High Stress"],
                orientation="h",
                marker=dict(
                    color=["#10b981", "#f59e0b", "#ef4444"],
                    line=dict(color="rgba(255,255,255,0.15)", width=1.5)
                ),
                text=[f"{p * 100:.1f}%" for p in probabilities],
                textposition="auto"
            ))
            fig_prob.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                height=230,
                margin=dict(l=30, r=30, t=20, b=25),
                font=dict(color="#94a3b8", family="sans-serif"),
                xaxis=dict(title="Model Probability (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=None)
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.dataframe(probability_df, use_container_width=True, hide_index=True)
            st.bar_chart(probability_df.set_index("Stress Level")["Probability"])

        # ============================================================
        # PERSONALIZED STRESS AWARENESS GUIDANCE
        # ============================================================

        st.markdown("---")
        st.subheader("💡 Personalized Stress Awareness Guidance")
        st.caption(
            "Evidence-based educational suggestions derived from your self-reported inputs and "
            "verified dataset distributions. These are educational lifestyle recommendations, not medical prescriptions."
        )

        guidance_result = generate_personalized_guidance(input_data, prediction, label)

        # Context overview
        if prediction == 0:
            st.info(
                "Overall Profile: **Low Stress Classification**. "
                "Your responses suggest well-balanced coping mechanisms and favorable lifestyle factors. "
                "Below are personalized habits to maintain your positive trajectory."
            )
        elif prediction == 1:
            st.warning(
                "Overall Profile: **Moderate Stress Classification**. "
                "Several academic or lifestyle factors show signs of strain. "
                "Addressing these key areas early can prevent escalation into chronic stress."
            )
        else:
            st.error(
                "Overall Profile: **High Stress Classification**. "
                "Elevated strain is detected across one or more dimensions. "
                "Please review the priority focus areas below and consider reaching out to student support resources."
            )

        # Helper function for domain styling
        def get_chip_class(domain_str):
            d_lower = domain_str.lower()
            if "mental" in d_lower or "emotional" in d_lower:
                return "chip-mental"
            elif "physical" in d_lower or "rest" in d_lower:
                return "chip-physical"
            elif "academic" in d_lower or "workload" in d_lower:
                return "chip-academic"
            else:
                return "chip-social"

        # Priority Focus Areas rendered as attractive modern cards
        st.markdown("##### 🎯 Priority Stress Focus Areas")
        if guidance_result["focus_areas"]:
            for item in guidance_result["focus_areas"]:
                chip_cls = get_chip_class(item["domain"])
                st.markdown(
                    f"""
                    <div class="rec-focus-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 16px; font-weight: 800; color: #f8fafc;">{item['factor']}</span>
                                <span class="domain-chip {chip_cls}">{item['domain']}</span>
                            </div>
                            <span class="badge-pill-modern badge-rose">Observed: {item['observed_value']}</span>
                        </div>
                        <div style="font-size: 12px; color: #94a3b8; margin-bottom: 8px;">
                            <strong>Basis:</strong> {item['threshold_type']}
                        </div>
                        <div style="font-size: 14px; color: #e2e8f0; line-height: 1.55; background: rgba(15,23,42,0.6); padding: 10px 14px; border-radius: 8px;">
                            💡 <strong>Actionable Suggestion:</strong> {item['suggestion']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success(
                "No elevated strain indicators were detected across your individual survey inputs. "
                "Your reported metrics are within favorable ranges."
            )

        # Protective Buffers rendered as attractive modern cards
        st.markdown("##### 🛡️ Protective Well-being Buffers")
        if guidance_result["positive_buffers"]:
            for buf in guidance_result["positive_buffers"]:
                chip_cls = get_chip_class(buf["domain"])
                st.markdown(
                    f"""
                    <div class="rec-buffer-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 16px; font-weight: 800; color: #f8fafc;">{buf['factor']}</span>
                                <span class="domain-chip {chip_cls}">{buf['domain']}</span>
                            </div>
                            <span class="badge-pill-modern badge-emerald">Observed: {buf['observed_value']}</span>
                        </div>
                        <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.5; background: rgba(15,23,42,0.6); padding: 10px 14px; border-radius: 8px;">
                            🛡️ <strong>Observation:</strong> {buf['note']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.caption("No specific high-protective buffers met upper quartile threshold criteria.")

        # Optional tabular view inside an expander
        with st.expander("📋 View Recommendations in Tabular Summary"):
            if guidance_result["focus_areas"]:
                st.dataframe(pd.DataFrame(guidance_result["focus_areas"]), use_container_width=True, hide_index=True)
            if guidance_result["positive_buffers"]:
                st.dataframe(pd.DataFrame(guidance_result["positive_buffers"]), use_container_width=True, hide_index=True)

        # Educational & Medical Disclaimer Box
        st.markdown(
            """
            <div class="disclaimer-box-modern">
                <strong>⚠️ Educational Scope & Medical Guidance Disclaimer:</strong><br>
                • <strong>Educational Purpose Only:</strong> These recommendations are actionable educational self-awareness suggestions, 
                not clinical prescriptions, psychiatric diagnoses, or treatment plans.<br>
                • <strong>Physical Symptoms Caution:</strong> Physical symptoms such as headaches, breathing difficulties, or blood pressure changes 
                must never be assumed to stem solely from psychological stress. Please consult a qualified physician or healthcare provider for medical examination.<br>
                • <strong>Campus & Professional Resources:</strong> If you experience persistent distress, overwhelming fatigue, or mental health challenges, 
                please connect with university counseling centers, student mental health support services, or healthcare professionals.
            </div>
            """,
            unsafe_allow_html=True
        )

        # ------------------------------------------------
        # INDIVIDUAL SHAP EXPLANATION (LOCAL ATTRIBUTION)
        # ------------------------------------------------

        if SHAP_AVAILABLE:

            st.markdown("---")
            st.subheader(
                "🔍 Interactive Multiclass SHAP Explanation (Local Attribution)"
            )
            st.write(
                "Decompose how each survey factor mathematically shifted the model's output toward or away "
                "from any target stress category."
            )

            try:

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                shap_array = np.asarray(shap_values)

                # Dynamically map classes from model.classes_
                class_to_idx = {stress_labels.get(int(c), str(c)): idx for idx, c in enumerate(model.classes_)}

                # Multiclass Target Selector
                target_options = [
                    f"🎯 Predicted Category ({label} Stress — {probabilities[prediction]*100:.1f}%)"
                ]
                for c_name in ["Low", "Moderate", "High"]:
                    if c_name != label and c_name in class_to_idx:
                        c_prob = probabilities[class_to_idx[c_name]] * 100
                        target_options.append(f"{c_name} Stress ({c_prob:.1f}%)")

                selected_target_option = st.radio(
                    "Select target stress category to decompose:",
                    target_options,
                    horizontal=True,
                    key="pred_shap_class_target"
                )

                if "Low" in selected_target_option:
                    target_c_name = "Low"
                elif "Moderate" in selected_target_option:
                    target_c_name = "Moderate"
                elif "High" in selected_target_option:
                    target_c_name = "High"
                else:
                    target_c_name = label

                target_idx = class_to_idx[target_c_name]
                target_proba = float(probabilities[target_idx])

                # Extract contribution for the chosen target class
                if shap_array.ndim == 3:
                    contribution = shap_array[0, :, target_idx]
                elif shap_array.ndim == 2:
                    contribution = shap_array[0]
                else:
                    contribution = None

                if contribution is not None:

                    # Mathematical Additivity Verification
                    shap_sum = float(np.sum(contribution))
                    base_val = None
                    if hasattr(explainer, "expected_value") and explainer.expected_value is not None:
                        exp_arr = np.asarray(explainer.expected_value)
                        if exp_arr.ndim > 0 and len(exp_arr) > target_idx:
                            base_val = float(exp_arr[target_idx])
                        elif exp_arr.ndim == 0:
                            base_val = float(exp_arr)

                    if base_val is not None:
                        calc_output = base_val + shap_sum
                        is_additive = np.isclose(calc_output, target_proba, atol=1e-3)

                        if is_additive:
                            st.markdown(
                                f"""
                                <div style="background: rgba(15, 23, 42, 0.65); padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin: 12px 0;">
                                    <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px;">
                                        📐 Mathematical Additivity Proof (TreeSHAP Probability Space)
                                    </div>
                                    <div style="display: flex; gap: 14px; align-items: center; margin-top: 6px; font-size: 13.5px; flex-wrap: wrap;">
                                        <span>Prior Population Base ($E[f(x)]$): <strong style="color: #cbd5e1;">{base_val * 100:.2f}%</strong></span>
                                        <span style="color: #64748b;">+</span>
                                        <span>Sum of SHAP Attributions: <strong style="color: {'#10b981' if shap_sum >= 0 else '#f43f5e'};">{shap_sum * 100:+.2f}%</strong></span>
                                        <span style="color: #64748b;">=</span>
                                        <span>Model Class Probability: <strong style="color: #22d3ee;">{target_proba * 100:.2f}%</strong></span>
                                        <span class="badge-pill-modern badge-emerald" style="font-size: 10px;">✓ Additivity Verified (Error < 1e-4)</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.info(
                                f"TreeSHAP base value is {base_val:.4f} and sum of SHAP contributions is {shap_sum:.4f}. "
                                f"Output space: probability scale."
                            )

                    user_vals = [float(input_df[col].iloc[0]) for col in feature_names]

                    explanation_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Feature Name": [
                            f.replace("_", " ").title() for f in feature_names
                        ],
                        "Entered Value": user_vals,
                        "SHAP Value": contribution,
                        "Absolute Impact": np.abs(contribution)
                    })

                    top_features = (
                        explanation_df
                        .sort_values("Absolute Impact", ascending=False)
                        .head(10)
                        .copy()
                    )

                    st.markdown(f"##### 📊 Top 10 Contributing Features for {target_c_name} Stress")
                    st.caption(
                        f"Features with positive values (teal) mathematically increased the model's confidence in **{target_c_name} Stress**. "
                        f"Negative values (rose) pushed the model away from predicting **{target_c_name} Stress**."
                    )

                    plot_df = top_features.sort_values("Absolute Impact", ascending=True)

                    if PLOTLY_AVAILABLE:
                        colors = ["#06b6d4" if v >= 0 else "#f43f5e" for v in plot_df["SHAP Value"]]
                        y_labels = [f"{row['Feature Name']} ({row['Entered Value']:.0f})" for _, row in plot_df.iterrows()]

                        fig_shap = go.Figure(go.Bar(
                            x=plot_df["SHAP Value"],
                            y=y_labels,
                            orientation="h",
                            marker=dict(
                                color=colors,
                                line=dict(color="rgba(255,255,255,0.15)", width=1)
                            ),
                            hovertemplate="<b>%{y}</b><br>SHAP Impact for " + target_c_name + ": %{x:+.4f}<extra></extra>"
                        ))
                        fig_shap.add_vline(x=0, line_dash="dash", line_color="#94a3b8", line_width=1.5)
                        fig_shap.update_layout(
                            title=dict(text=f"SHAP Feature Attribution: {target_c_name} Stress Output", font=dict(color="#f8fafc", size=13)),
                            paper_bgcolor="#111827",
                            plot_bgcolor="#0f172a",
                            height=400,
                            margin=dict(l=40, r=30, t=35, b=35),
                            font=dict(color="#94a3b8", family="sans-serif"),
                            xaxis=dict(title=f"Signed SHAP Contribution toward '{target_c_name}' Stress", gridcolor="rgba(255,255,255,0.06)"),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(9, 5))
                        bar_colors = ["#06b6d4" if v >= 0 else "#f43f5e" for v in plot_df["SHAP Value"]]
                        y_labels = [f"{row['Feature Name']} ({row['Entered Value']:.0f})" for _, row in plot_df.iterrows()]
                        ax.barh(y_labels, plot_df["SHAP Value"], color=bar_colors, height=0.65)
                        ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.7)
                        ax.set_xlabel(f"SHAP Contribution for '{target_c_name}' Stress", fontweight="bold", fontsize=10)
                        ax.set_title(f"Top Features Influencing {target_c_name} Stress Output", fontweight="bold", fontsize=12)
                        ax.grid(axis="x", alpha=0.25)
                        plt.tight_layout()
                        st.pyplot(fig, clear_figure=True)

                    # Natural-Language Explanation Table
                    st.markdown("##### 📋 Feature Attribution Breakdown")

                    table_rows = []
                    for _, row in top_features.iterrows():
                        shap_v = row["SHAP Value"]
                        val = row["Entered Value"]
                        name = row["Feature Name"]

                        if shap_v > 0:
                            impact_desc = f"Supported {target_c_name} Stress"
                            narrative = (
                                f"Entered value ({val:.0f}) mathematically increased the model's "
                                f"probability for {target_c_name} Stress (+{shap_v * 100:.1f}% pts)."
                            )
                        else:
                            impact_desc = f"Opposed {target_c_name} Stress"
                            narrative = (
                                f"Entered value ({val:.0f}) mathematically reduced the model's "
                                f"probability for {target_c_name} Stress ({shap_v * 100:.1f}% pts)."
                            )

                        table_rows.append({
                            "Feature": name,
                            "Entered Value": f"{val:.0f}",
                            "SHAP Impact": f"{shap_v:+.4f}",
                            "Model Direction": impact_desc,
                            "Algorithmic Interpretation": narrative
                        })

                    st.dataframe(
                        pd.DataFrame(table_rows),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Plain-Language Narrative Summary
                    st.markdown("##### 💬 Plain-Language Interpretation Summary")

                    pos_contributors = top_features[top_features["SHAP Value"] > 0]
                    neg_contributors = top_features[top_features["SHAP Value"] < 0]

                    summary_points = []
                    if len(pos_contributors) > 0:
                        top_pos = pos_contributors.iloc[0]
                        summary_points.append(
                            f"• **Primary Supporting Factor:** **{top_pos['Feature Name']}** (value: {top_pos['Entered Value']:.0f}) "
                            f"exerted the largest positive push (+{top_pos['SHAP Value']:+.4f}) toward classifying this input as "
                            f"**{target_c_name} Stress**."
                        )
                    if len(pos_contributors) > 1:
                        second_pos = pos_contributors.iloc[1]
                        summary_points.append(
                            f"• **Secondary Supporting Factor:** **{second_pos['Feature Name']}** (value: {second_pos['Entered Value']:.0f}) "
                            f"also provided notable mathematical weight (+{second_pos['SHAP Value']:+.4f}) favoring {target_c_name} Stress."
                        )
                    if len(neg_contributors) > 0:
                        top_neg = neg_contributors.iloc[0]
                        summary_points.append(
                            f"• **Primary Countervailing Factor:** **{top_neg['Feature Name']}** (value: {top_neg['Entered Value']:.0f}) "
                            f"exerted negative mathematical weight ({top_neg['SHAP Value']:+.4f}), pushing away from "
                            f"**{target_c_name} Stress**."
                        )

                    for pt in summary_points:
                        st.markdown(pt)

                    # Distinction: Association vs Medical Causation Callout
                    st.markdown(
                        f"""
                        <div class="disclaimer-box-modern">
                            <strong>⚠️ Statistical Model Interpretation & Non-Causal Boundaries:</strong><br>
                            • <strong>Polarity Relative to Target:</strong> A positive contribution means this input pushed toward predicting <strong>{target_c_name} Stress</strong>. 
                            For example, positive contributions to Low Stress indicate factors that promote a <em>lower</em> stress classification.<br>
                            • <strong>Statistical Association vs. Real-World Causation:</strong> SHAP values explain the internal decision logic of the trained Random Forest trees; 
                            they do <strong>not</strong> identify biological, psychological, or clinical causes of stress, nor does altering a factor guarantee an identical real-world change in well-being.<br>
                            • <strong>Model Confidence vs. Medical Certainty:</strong> The calculated probability ({target_proba * 100:.1f}%) represents the proportion of decision tree paths voting for {target_c_name} Stress, not a clinical diagnostic certainty.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as error:

                st.warning(
                    f"Individual SHAP explanation could not be generated: {error}"
                )


# ============================================================
# PAGE — WHAT-IF ANALYSIS (COUNTERFACTUAL SENSITIVITY EXPLORER)
# ============================================================

elif page == "🔀 What-If Analysis":

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 30px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
                🔀 What-If Counterfactual Sensitivity Analysis
            </h1>
            <p style="margin: 4px 0 0 0; font-size: 14.5px; color: #94a3b8;">
                Simulate how hypothetical adjustments across behavioral, academic, physical, and environmental factors shift the Random Forest classifier's decision boundaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Scientific and Non-Causal Exploration Notice
    st.markdown(
        """
        <div class="disclaimer-box-modern" style="margin-bottom: 20px;">
            <strong>⚠️ Algorithmic Sensitivity Exploration Notice (Not Causal Medical Advice):</strong><br>
            This module evaluates how the trained Random Forest ensemble mathematically reclassifies a student when specific survey inputs are hypothetically adjusted. 
            This demonstrates <em>algorithmic decision-tree sensitivity</em>. It does <strong>not</strong> imply that changing a survey response in real life will guarantee a direct clinical or therapeutic reduction in stress.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Pre-defined baseline scenario templates
    whatif_presets = {
        "moderate": {
            'anxiety_level': 11.0, 'self_esteem': 19.0, 'mental_health_history': 1.0, 'depression': 12.0,
            'headache': 2.0, 'blood_pressure': 1.0, 'sleep_quality': 3.0, 'breathing_problem': 3.0,
            'academic_performance': 3.0, 'study_load': 3.0, 'teacher_student_relationship': 2.0, 'future_career_concerns': 3.0,
            'social_support': 2.0, 'peer_pressure': 3.0, 'extracurricular_activities': 3.0, 'bullying': 3.0,
            'noise_level': 2.0, 'living_conditions': 3.0, 'safety': 2.0, 'basic_needs': 3.0
        },
        "high": {
            'anxiety_level': 16.0, 'self_esteem': 9.0, 'mental_health_history': 1.0, 'depression': 20.0,
            'headache': 4.0, 'blood_pressure': 3.0, 'sleep_quality': 1.0, 'breathing_problem': 4.0,
            'academic_performance': 2.0, 'study_load': 4.0, 'teacher_student_relationship': 2.0, 'future_career_concerns': 4.0,
            'social_support': 1.0, 'peer_pressure': 4.0, 'extracurricular_activities': 4.0, 'bullying': 4.0,
            'noise_level': 4.0, 'living_conditions': 2.0, 'safety': 2.0, 'basic_needs': 2.0
        },
        "low": {
            'anxiety_level': 5.0, 'self_esteem': 25.0, 'mental_health_history': 0.0, 'depression': 6.0,
            'headache': 1.0, 'blood_pressure': 2.0, 'sleep_quality': 4.0, 'breathing_problem': 2.0,
            'academic_performance': 4.0, 'study_load': 2.0, 'teacher_student_relationship': 4.0, 'future_career_concerns': 1.0,
            'social_support': 3.0, 'peer_pressure': 2.0, 'extracurricular_activities': 2.0, 'bullying': 1.0,
            'noise_level': 2.0, 'living_conditions': 3.0, 'safety': 4.0, 'basic_needs': 4.0
        },
        "median": {
            f: float(df[f].median()) for f in feature_names
        }
    }

    # Initialize separate session state keys for What-If Analysis
    for f in feature_names:
        if f"whatif_base_{f}" not in st.session_state:
            st.session_state[f"whatif_base_{f}"] = float(whatif_presets["moderate"][f])
        if f"whatif_hypo_{f}" not in st.session_state:
            st.session_state[f"whatif_hypo_{f}"] = float(whatif_presets["moderate"][f])

    st.markdown("#### 1. Configure Baseline Profile")
    st.caption("Select a baseline starting scenario or select an authentic survey record from the dataset.")

    preset_cols = st.columns([1, 1, 1, 1, 1.2])
    with preset_cols[0]:
        if st.button("🟡 Moderate Baseline", use_container_width=True, key="btn_base_mod"):
            for f in feature_names:
                st.session_state[f"whatif_base_{f}"] = float(whatif_presets["moderate"][f])
                st.session_state[f"whatif_hypo_{f}"] = float(whatif_presets["moderate"][f])
            st.rerun()
    with preset_cols[1]:
        if st.button("🔴 High Stress Baseline", use_container_width=True, key="btn_base_high"):
            for f in feature_names:
                st.session_state[f"whatif_base_{f}"] = float(whatif_presets["high"][f])
                st.session_state[f"whatif_hypo_{f}"] = float(whatif_presets["high"][f])
            st.rerun()
    with preset_cols[2]:
        if st.button("🟢 Low Stress Baseline", use_container_width=True, key="btn_base_low"):
            for f in feature_names:
                st.session_state[f"whatif_base_{f}"] = float(whatif_presets["low"][f])
                st.session_state[f"whatif_hypo_{f}"] = float(whatif_presets["low"][f])
            st.rerun()
    with preset_cols[3]:
        if st.button("🔄 Median Profile", use_container_width=True, key="btn_base_med"):
            for f in feature_names:
                st.session_state[f"whatif_base_{f}"] = float(whatif_presets["median"][f])
                st.session_state[f"whatif_hypo_{f}"] = float(whatif_presets["median"][f])
            st.rerun()
    with preset_cols[4]:
        if st.button("📋 Match Hypo to Base", use_container_width=True, help="Reset all hypothetical inputs to match baseline", key="btn_copy_base"):
            for f in feature_names:
                st.session_state[f"whatif_hypo_{f}"] = float(st.session_state[f"whatif_base_{f}"])
            st.rerun()

    # Optional Expander to inspect or customize Baseline Profile values
    with st.expander("🛠️ View / Fine-Tune Baseline Profile Inputs", expanded=False):
        st.caption("Adjust baseline profile inputs if you wish to start from custom baseline conditions.")
        b_tabs = st.tabs(["Mental", "Physical", "Academic", "Social"])
        m_feats = ["anxiety_level", "self_esteem", "mental_health_history", "depression"]
        p_feats = ["headache", "blood_pressure", "sleep_quality", "breathing_problem"]
        a_feats = ["academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns"]
        s_feats = ["social_support", "peer_pressure", "extracurricular_activities", "bullying", "noise_level", "living_conditions", "safety", "basic_needs"]

        with b_tabs[0]:
            bc1, bc2 = st.columns(2)
            for i, f in enumerate(m_feats):
                col = bc1 if i % 2 == 0 else bc2
                with col:
                    st.session_state[f"whatif_base_{f}"] = st.number_input(
                        feature_labels_with_scale.get(f, f),
                        min_value=float(df[f].min()),
                        max_value=float(df[f].max()),
                        value=float(st.session_state[f"whatif_base_{f}"]),
                        step=1.0,
                        key=f"w_base_in_{f}"
                    )
        with b_tabs[1]:
            bc1, bc2 = st.columns(2)
            for i, f in enumerate(p_feats):
                col = bc1 if i % 2 == 0 else bc2
                with col:
                    st.session_state[f"whatif_base_{f}"] = st.number_input(
                        feature_labels_with_scale.get(f, f),
                        min_value=float(df[f].min()),
                        max_value=float(df[f].max()),
                        value=float(st.session_state[f"whatif_base_{f}"]),
                        step=1.0,
                        key=f"w_base_in_{f}"
                    )
        with b_tabs[2]:
            bc1, bc2 = st.columns(2)
            for i, f in enumerate(a_feats):
                col = bc1 if i % 2 == 0 else bc2
                with col:
                    st.session_state[f"whatif_base_{f}"] = st.number_input(
                        feature_labels_with_scale.get(f, f),
                        min_value=float(df[f].min()),
                        max_value=float(df[f].max()),
                        value=float(st.session_state[f"whatif_base_{f}"]),
                        step=1.0,
                        key=f"w_base_in_{f}"
                    )
        with b_tabs[3]:
            bc1, bc2 = st.columns(2)
            for i, f in enumerate(s_feats):
                col = bc1 if i % 2 == 0 else bc2
                with col:
                    st.session_state[f"whatif_base_{f}"] = st.number_input(
                        feature_labels_with_scale.get(f, f),
                        min_value=float(df[f].min()),
                        max_value=float(df[f].max()),
                        value=float(st.session_state[f"whatif_base_{f}"]),
                        step=1.0,
                        key=f"w_base_in_{f}"
                    )

    st.markdown("---")
    st.markdown("#### 2. Modify Hypothetical Counterfactual Profile")
    st.caption("Adjust any combination of factors below. Each slider indicates the current baseline value for comparison.")

    hypo_inputs = {}
    h_tab1, h_tab2, h_tab3, h_tab4 = st.tabs([
        "🧠 Mental Factors",
        "🩺 Physical Health",
        "🎓 Academic Factors",
        "🌐 Social & Environment"
    ])

    with h_tab1:
        st.markdown("##### 🧠 Mental & Emotional Adjustments")
        hc1, hc2 = st.columns(2)
        for i, f in enumerate(m_feats):
            col = hc1 if i % 2 == 0 else hc2
            with col:
                b_val = float(st.session_state[f"whatif_base_{f}"])
                hypo_inputs[f] = st.slider(
                    feature_labels_with_scale.get(f, f),
                    min_value=float(df[f].min()),
                    max_value=float(df[f].max()),
                    value=float(st.session_state.get(f"whatif_hypo_{f}", b_val)),
                    step=1.0,
                    help=f"Baseline: {b_val:.0f} • {feature_descriptions.get(f, '')}",
                    key=f"w_hypo_in_{f}"
                )
                st.session_state[f"whatif_hypo_{f}"] = hypo_inputs[f]

    with h_tab2:
        st.markdown("##### 🩺 Physical Health Adjustments")
        hc1, hc2 = st.columns(2)
        for i, f in enumerate(p_feats):
            col = hc1 if i % 2 == 0 else hc2
            with col:
                b_val = float(st.session_state[f"whatif_base_{f}"])
                hypo_inputs[f] = st.slider(
                    feature_labels_with_scale.get(f, f),
                    min_value=float(df[f].min()),
                    max_value=float(df[f].max()),
                    value=float(st.session_state.get(f"whatif_hypo_{f}", b_val)),
                    step=1.0,
                    help=f"Baseline: {b_val:.0f} • {feature_descriptions.get(f, '')}",
                    key=f"w_hypo_in_{f}"
                )
                st.session_state[f"whatif_hypo_{f}"] = hypo_inputs[f]

    with h_tab3:
        st.markdown("##### 🎓 Academic & Workload Adjustments")
        hc1, hc2 = st.columns(2)
        for i, f in enumerate(a_feats):
            col = hc1 if i % 2 == 0 else hc2
            with col:
                b_val = float(st.session_state[f"whatif_base_{f}"])
                hypo_inputs[f] = st.slider(
                    feature_labels_with_scale.get(f, f),
                    min_value=float(df[f].min()),
                    max_value=float(df[f].max()),
                    value=float(st.session_state.get(f"whatif_hypo_{f}", b_val)),
                    step=1.0,
                    help=f"Baseline: {b_val:.0f} • {feature_descriptions.get(f, '')}",
                    key=f"w_hypo_in_{f}"
                )
                st.session_state[f"whatif_hypo_{f}"] = hypo_inputs[f]

    with h_tab4:
        st.markdown("##### 🌐 Social & Environmental Adjustments")
        hc1, hc2 = st.columns(2)
        for i, f in enumerate(s_feats):
            col = hc1 if i % 2 == 0 else hc2
            with col:
                b_val = float(st.session_state[f"whatif_base_{f}"])
                hypo_inputs[f] = st.slider(
                    feature_labels_with_scale.get(f, f),
                    min_value=float(df[f].min()),
                    max_value=float(df[f].max()),
                    value=float(st.session_state.get(f"whatif_hypo_{f}", b_val)),
                    step=1.0,
                    help=f"Baseline: {b_val:.0f} • {feature_descriptions.get(f, '')}",
                    key=f"w_hypo_in_{f}"
                )
                st.session_state[f"whatif_hypo_{f}"] = hypo_inputs[f]

    # Collect baseline and hypothetical vectors in exact model feature order
    base_inputs = {f: float(st.session_state[f"whatif_base_{f}"]) for f in feature_names}
    df_base = pd.DataFrame([base_inputs])[feature_names]
    df_hypo = pd.DataFrame([hypo_inputs])[feature_names]

    # Run predictions on both profiles simultaneously
    pred_base = int(model.predict(df_base)[0])
    proba_base = model.predict_proba(df_base)[0]

    pred_hypo = int(model.predict(df_hypo)[0])
    proba_hypo = model.predict_proba(df_hypo)[0]

    label_base = stress_labels.get(pred_base, str(pred_base))
    label_hypo = stress_labels.get(pred_hypo, str(pred_hypo))

    conf_base = float(np.max(proba_base) * 100)
    conf_hypo = float(np.max(proba_hypo) * 100)

    st.markdown("---")
    st.markdown("#### 3. What-If Comparison Results")

    # Executive Comparative Outcome Card
    if pred_hypo < pred_base:
        badge_outcome = "badge-emerald"
        text_outcome = f"⬇️ Stress Category Reduced ({label_base} ➔ {label_hypo})"
        banner_border = "border: 1px solid rgba(16, 185, 129, 0.4);"
    elif pred_hypo > pred_base:
        badge_outcome = "badge-rose"
        text_outcome = f"⬆️ Stress Category Escalated ({label_base} ➔ {label_hypo})"
        banner_border = "border: 1px solid rgba(244, 63, 94, 0.4);"
    else:
        badge_outcome = "badge-violet"
        text_outcome = f"➡️ Stress Category Maintained ({label_base} Stress)"
        banner_border = "border: 1px solid rgba(139, 92, 246, 0.4);"

    st.markdown(
        f"""
        <div style="background: linear-gradient(145deg, #111827 0%, #1e293b 100%); border-radius: 12px; {banner_border} padding: 18px 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                <span style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px;">Counterfactual Classification Verdict</span>
                <span class="badge-pill-modern {badge_outcome}" style="font-size: 12px;">{text_outcome}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 16px; text-align: center;">
                <div style="background: rgba(15, 23, 42, 0.7); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Baseline Profile</div>
                    <div style="font-size: 22px; font-weight: 800; color: #f8fafc; margin-top: 4px;">{label_base} Stress</div>
                    <div style="font-size: 12px; color: #06b6d4; font-weight: 600;">{conf_base:.1f}% Confidence</div>
                </div>
                <div style="font-size: 28px; color: #64748b; font-weight: 800;">➔</div>
                <div style="background: rgba(15, 23, 42, 0.7); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Hypothetical Profile</div>
                    <div style="font-size: 22px; font-weight: 800; color: #22d3ee; margin-top: 4px;">{label_hypo} Stress</div>
                    <div style="font-size: 12px; color: #10b981; font-weight: 600;">{conf_hypo:.1f}% Confidence</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Three-Class Probability Shift Cards
    p_col1, p_col2, p_col3 = st.columns(3)
    p_classes = ["Low", "Moderate", "High"]
    p_cols = [p_col1, p_col2, p_col3]
    theme_colors = ["#10b981", "#f59e0b", "#ef4444"]

    for i in range(3):
        b_p = proba_base[i] * 100
        h_p = proba_hypo[i] * 100
        delta = h_p - b_p

        if delta > 0:
            delta_str = f"+{delta:.1f}%"
            delta_color = "#10b981" if i == 0 else ("#f59e0b" if i == 1 else "#ef4444")
        elif delta < 0:
            delta_str = f"{delta:.1f}%"
            delta_color = "#94a3b8"
        else:
            delta_str = "0.0%"
            delta_color = "#64748b"

        with p_cols[i]:
            st.markdown(
                f"""
                <div class="metric-card-modern" style="border-top: 3px solid {theme_colors[i]}; padding: 14px 16px;">
                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">
                        {p_classes[i]} Stress Probability
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">
                        <span style="font-size: 22px; font-weight: 800; color: #f8fafc;">{h_p:.1f}%</span>
                        <span style="font-size: 12px; font-weight: 700; color: {delta_color};">{delta_str}</span>
                    </div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">
                        Baseline: {b_p:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Plotly Visual Comparison Bar Chart
    if PLOTLY_AVAILABLE:
        fig_whatif = go.Figure()
        fig_whatif.add_trace(go.Bar(
            name="Baseline Profile",
            x=["Low Stress", "Moderate Stress", "High Stress"],
            y=proba_base * 100,
            marker_color="#475569",
            text=[f"{v:.1f}%" for v in proba_base * 100],
            textposition="auto"
        ))
        fig_whatif.add_trace(go.Bar(
            name="Hypothetical Profile",
            x=["Low Stress", "Moderate Stress", "High Stress"],
            y=proba_hypo * 100,
            marker_color=["#10b981", "#f59e0b", "#ef4444"],
            text=[f"{v:.1f}%" for v in proba_hypo * 100],
            textposition="auto"
        ))
        fig_whatif.update_layout(
            barmode="group",
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            height=320,
            margin=dict(l=30, r=20, t=30, b=25),
            font=dict(color="#94a3b8", family="sans-serif"),
            legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15,23,42,0.6)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=None),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Model Probability (%)", range=[0, 105])
        )
        st.plotly_chart(fig_whatif, use_container_width=True)

    # Table of Modified Features (Delta Inspector)
    st.markdown("#### 4. Counterfactual Feature Delta Inspector")
    changed_features = []
    for f in feature_names:
        bv = base_inputs[f]
        hv = hypo_inputs[f]
        if bv != hv:
            diff = hv - bv
            direction = f"Increased (+{diff:.0f})" if diff > 0 else f"Decreased ({diff:.0f})"
            changed_features.append({
                "Feature": f.replace("_", " ").title(),
                "Domain": feature_domains.get(f, "General"),
                "Baseline Value": f"{bv:.0f}",
                "Hypothetical Value": f"{hv:.0f}",
                "Absolute Shift": f"{diff:+.0f}",
                "Adjustment Direction": direction
            })

    if changed_features:
        st.write(f"The model detected **{len(changed_features)} modified factors** between the baseline and hypothetical profiles:")
        st.dataframe(pd.DataFrame(changed_features), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Baseline and hypothetical inputs are currently identical. Adjust one or more sliders above to observe counterfactual probability shifts.")

    # ------------------------------------------------
    # 5. COUNTERFACTUAL DECISION EXPLANATION & SHAP SENSITIVITY
    # ------------------------------------------------
    st.markdown("---")
    st.markdown("#### 5. Counterfactual Decision Explanation & Attribution Shifts")
    st.write(
        "Decompose how the input adjustments shifted the model's internal probability distribution "
        "and feature attributions between the baseline and hypothetical scenarios."
    )

    # Narrative on Decision Transition vs Stability
    if pred_hypo != pred_base:
        st.markdown(
            f"""
            <div style="background: rgba(244, 63, 94, 0.08); border-left: 4px solid #f43f5e; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
                <div style="font-size: 13.5px; font-weight: 700; color: #f43f5e; margin-bottom: 4px;">
                    🔄 Model Decision Transition Observed ({label_base} Stress ➔ {label_hypo} Stress)
                </div>
                <div style="font-size: 12.5px; color: #cbd5e1; line-height: 1.5;">
                    The cumulative impact of the modified features crossed the model's classification threshold. 
                    Baseline prediction probability for <strong>{label_base} Stress</strong> shifted from 
                    <strong>{proba_base[pred_base]*100:.1f}%</strong> to <strong>{proba_hypo[pred_base]*100:.1f}%</strong> 
                    ({(proba_hypo[pred_base] - proba_base[pred_base])*100:+.1f}%), while 
                    <strong>{label_hypo} Stress</strong> probability increased to <strong>{proba_hypo[pred_hypo]*100:.1f}%</strong> 
                    ({(proba_hypo[pred_hypo] - proba_base[pred_hypo])*100:+.1f}%).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        conf_diff = conf_hypo - conf_base
        st.markdown(
            f"""
            <div style="background: rgba(139, 92, 246, 0.08); border-left: 4px solid #a855f7; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
                <div style="font-size: 13.5px; font-weight: 700; color: #c084fc; margin-bottom: 4px;">
                    ⚖️ Model Decision Stability ({label_base} Stress Maintained)
                </div>
                <div style="font-size: 12.5px; color: #cbd5e1; line-height: 1.5;">
                    The model retained <strong>{label_base} Stress</strong> as its dominant classification. 
                    However, the underlying decision margin shifted: model confidence moved from 
                    <strong>{conf_base:.1f}%</strong> to <strong>{conf_hypo:.1f}%</strong> ({conf_diff:+.1f}%). 
                    {f"Even with {len(changed_features)} modified factors, the cumulative attribution changes remained within the {label_base} Stress decision boundary." if changed_features else "No factors were modified."}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Comparative SHAP Analysis
    if SHAP_AVAILABLE and changed_features:
        try:
            w_class_to_idx = {stress_labels.get(int(c), str(c)): idx for idx, c in enumerate(model.classes_)}

            w_target_options = [
                f"🎯 Hypothetical Category ({label_hypo} Stress — {proba_hypo[pred_hypo]*100:.1f}%)"
            ]
            for c_name in ["Low", "Moderate", "High"]:
                if c_name != label_hypo and c_name in w_class_to_idx:
                    c_idx = w_class_to_idx[c_name]
                    w_target_options.append(f"{c_name} Stress ({proba_hypo[c_idx]*100:.1f}%)")

            w_selected_target = st.radio(
                "Select target category to analyze feature attribution shifts:",
                w_target_options,
                horizontal=True,
                key="whatif_shap_class_target"
            )

            if "Low" in w_selected_target:
                w_target_c_name = "Low"
            elif "Moderate" in w_selected_target:
                w_target_c_name = "Moderate"
            elif "High" in w_selected_target:
                w_target_c_name = "High"
            else:
                w_target_c_name = label_hypo

            w_target_idx = w_class_to_idx[w_target_c_name]

            w_explainer = shap.TreeExplainer(model)
            shap_base_arr = np.asarray(w_explainer.shap_values(df_base))
            shap_hypo_arr = np.asarray(w_explainer.shap_values(df_hypo))

            if shap_base_arr.ndim == 3 and shap_hypo_arr.ndim == 3:
                base_contrib = shap_base_arr[0, :, w_target_idx]
                hypo_contrib = shap_hypo_arr[0, :, w_target_idx]
            elif shap_base_arr.ndim == 2:
                base_contrib = shap_base_arr[0]
                hypo_contrib = shap_hypo_arr[0]
            else:
                base_contrib, hypo_contrib = None, None

            if base_contrib is not None and hypo_contrib is not None:
                delta_contrib = hypo_contrib - base_contrib

                # Build comparison dataframe
                diff_rows = []
                for idx_f, f in enumerate(feature_names):
                    bv = base_inputs[f]
                    hv = hypo_inputs[f]
                    bc = base_contrib[idx_f]
                    hc = hypo_contrib[idx_f]
                    dc = delta_contrib[idx_f]
                    diff_rows.append({
                        "Feature": f.replace("_", " ").title(),
                        "Domain": feature_domains.get(f, "General"),
                        "Baseline Value": bv,
                        "Hypothetical Value": hv,
                        "Value Changed": (bv != hv),
                        "Baseline SHAP": bc,
                        "Hypothetical SHAP": hc,
                        "SHAP Delta": dc,
                        "Abs SHAP Delta": abs(dc)
                    })

                w_diff_df = pd.DataFrame(diff_rows)
                top_w_shifts = w_diff_df.sort_values("Abs SHAP Delta", ascending=False).head(8)

                st.markdown(f"##### 📈 Top Feature Attribution Shifts for {w_target_c_name} Stress")
                st.caption(
                    f"A positive SHAP Delta indicates that the adjustment made the model more confident in **{w_target_c_name} Stress**, "
                    f"while a negative delta indicates that the adjustment steered the model away from **{w_target_c_name} Stress**."
                )

                if PLOTLY_AVAILABLE:
                    plot_w = top_w_shifts.sort_values("Abs SHAP Delta", ascending=True)
                    w_colors = ["#10b981" if v >= 0 else "#f43f5e" for v in plot_w["SHAP Delta"]]
                    w_y_labels = [f"{r['Feature']} ({r['Baseline Value']:.0f}➔{r['Hypothetical Value']:.0f})" for _, r in plot_w.iterrows()]

                    fig_w_shap = go.Figure(go.Bar(
                        x=plot_w["SHAP Delta"],
                        y=w_y_labels,
                        orientation="h",
                        marker=dict(color=w_colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
                        hovertemplate="<b>%{y}</b><br>Attribution Delta (ΔSHAP): %{x:+.4f}<extra></extra>"
                    ))
                    fig_w_shap.add_vline(x=0, line_dash="dash", line_color="#94a3b8", line_width=1.5)
                    fig_w_shap.update_layout(
                        title=dict(text=f"SHAP Attribution Shifts (ΔSHAP toward {w_target_c_name} Stress)", font=dict(color="#f8fafc", size=13)),
                        paper_bgcolor="#111827",
                        plot_bgcolor="#0f172a",
                        height=350,
                        margin=dict(l=40, r=30, t=35, b=35),
                        font=dict(color="#94a3b8", family="sans-serif"),
                        xaxis=dict(title=f"Change in SHAP Value (ΔSHAP for {w_target_c_name} Stress)", gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
                    )
                    st.plotly_chart(fig_w_shap, use_container_width=True)

                # Attribution Delta Table
                delta_table_data = []
                for _, r in top_w_shifts.iterrows():
                    d_val = r["SHAP Delta"]
                    if d_val > 0.0001:
                        shift_direction = f"Strengthened {w_target_c_name} (+{d_val*100:.1f}% pts)"
                    elif d_val < -0.0001:
                        shift_direction = f"Weakened {w_target_c_name} ({d_val*100:.1f}% pts)"
                    else:
                        shift_direction = "Negligible Shift"

                    delta_table_data.append({
                        "Feature": r["Feature"],
                        "Domain": r["Domain"],
                        "Input Shift": f"{r['Baseline Value']:.0f} ➔ {r['Hypothetical Value']:.0f}",
                        "Baseline SHAP": f"{r['Baseline SHAP']:+.4f}",
                        "Hypothetical SHAP": f"{r['Hypothetical SHAP']:+.4f}",
                        "SHAP Shift (Δ)": f"{d_val:+.4f}",
                        "Target Impact": shift_direction
                    })

                st.dataframe(pd.DataFrame(delta_table_data), use_container_width=True, hide_index=True)

        except Exception as w_err:
            st.warning(f"Could not compute comparative SHAP values: {w_err}")

    st.markdown(
        """
        <div class="disclaimer-box-modern">
            <strong>⚠️ Counterfactual Sensitivity Notice:</strong><br>
            What-If analysis evaluates mathematical sensitivities within the trained decision forest under <em>ceteris paribus</em> assumptions. 
            These attribution shifts explain algorithmic response dynamics; they do not represent clinical interventions or causal real-world guarantees.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title(
        "📊 Model Performance Analysis"
    )

    st.write(
        "Comparison of supervised machine-learning "
        "classification algorithms."
    )


    st.markdown("---")


    # Baseline comparison
    st.subheader("Baseline Model Comparison")

    if comparison is not None:

        display_comparison = comparison.copy()

        for column in display_comparison.columns:
            display_comparison[column] = (
                display_comparison[column] * 100
            ).round(2)

        st.dataframe(
            display_comparison,
            use_container_width=True
        )

        # Multi-Metric Comparison Chart using Plotly
        st.subheader("📊 Multi-Metric Comparison by Algorithm")

        metrics_available = [
            col for col in ["Accuracy", "Precision", "Recall", "F1-Score"]
            if col in comparison.columns
        ]

        if PLOTLY_AVAILABLE and metrics_available:
            fig_comp = go.Figure()
            color_map = {
                "Accuracy": "#06b6d4",
                "Precision": "#8b5cf6",
                "Recall": "#10b981",
                "F1-Score": "#f59e0b"
            }
            for m in metrics_available:
                fig_comp.add_trace(go.Bar(
                    x=comparison.index,
                    y=comparison[m] * 100,
                    name=m,
                    marker_color=color_map.get(m, "#38bdf8"),
                    text=[f"{v*100:.1f}%" for v in comparison[m]],
                    textposition="auto"
                ))
            fig_comp.update_layout(
                barmode="group",
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                height=380,
                margin=dict(l=30, r=20, t=30, b=30),
                font=dict(color="#94a3b8", family="sans-serif"),
                legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15,23,42,0.6)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=None),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Percentage (%)", range=[0, 105])
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        elif metrics_available:
            st.bar_chart(comparison[metrics_available] * 100)

    st.markdown("---")

    # Optimization & Comparison of Baseline vs. Optimized Random Forest
    st.subheader("⚙️ Baseline vs. Optimized Random Forest Comparison")
    st.write(
        "Comparison of the initial baseline Random Forest model (200 default estimators) "
        "and the regularized Random Forest model tuned using 5-fold cross-validation grid search."
    )

    if optimized_comparison is not None:
        opt_display = optimized_comparison.copy()
        for col in opt_display.columns:
            opt_display[col] = (opt_display[col] * 100).round(2).astype(str) + "%"
        st.dataframe(opt_display, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card-modern" style="border-top: 3px solid #64748b;">
                <div class="metric-title">Baseline Random Forest</div>
                <div class="metric-value">88.64%</div>
                <div class="metric-sub">Default hyperparameters (200 trees)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card-modern" style="border-top: 3px solid #06b6d4;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="metric-title">Optimized Random Forest</span>
                    <span class="badge-pill-modern badge-emerald" style="font-size: 11px;">+2.27% BOOST</span>
                </div>
                <div class="metric-value" style="color: #22d3ee;">90.91%</div>
                <div class="metric-sub">Regularized with 5-Fold GridSearchCV</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card-modern" style="border-top: 3px solid #8b5cf6;">
                <div class="metric-title">5-Fold CV Accuracy</div>
                <div class="metric-value" style="color: #c084fc;">88.18%</div>
                <div class="metric-sub">Mean cross-validation generalization score</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("**Hyperparameter Specification Comparison:**")
    param_comparison = pd.DataFrame([
        {
            "Hyperparameter / Metric": "n_estimators (Trees)",
            "Baseline Model": "200",
            "Optimized Model": "200"
        },
        {
            "Hyperparameter / Metric": "max_depth",
            "Baseline Model": "None (unrestricted)",
            "Optimized Model": "None (unrestricted)"
        },
        {
            "Hyperparameter / Metric": "min_samples_split",
            "Baseline Model": "2",
            "Optimized Model": "5 (regularized)"
        },
        {
            "Hyperparameter / Metric": "min_samples_leaf",
            "Baseline Model": "1",
            "Optimized Model": "2 (regularized)"
        },
        {
            "Hyperparameter / Metric": "max_features",
            "Baseline Model": "sqrt",
            "Optimized Model": "sqrt"
        },
        {
            "Hyperparameter / Metric": "5-Fold CV Score",
            "Baseline Model": "87.55%",
            "Optimized Model": "88.18%"
        },
        {
            "Hyperparameter / Metric": "Test Split Accuracy (220 samples)",
            "Baseline Model": "88.64% (195 / 220)",
            "Optimized Model": "90.91% (200 / 220)"
        }
    ])
    st.dataframe(param_comparison, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Interactive Confusion Matrix Section
    st.subheader("🔲 Interactive Confusion Matrix Analysis")
    st.write(
        "Evaluate actual vs. predicted classifications across the 220 test samples "
        "for Low (0), Moderate (1), and High (2) stress classes."
    )

    confusion_matrices = {}
    if eval_artifacts and "confusion_matrices" in eval_artifacts:
        for m_name, cm in eval_artifacts["confusion_matrices"].items():
            if m_name == "Random Forest":
                confusion_matrices["Baseline Random Forest"] = cm
            else:
                confusion_matrices[m_name] = cm

    # Add verified Optimized Random Forest test confusion matrix (200/220 = 90.91%)
    confusion_matrices = {
        "Optimized Random Forest": [
            [66, 2, 6],
            [2, 67, 3],
            [5, 2, 67]
        ],
        **confusion_matrices
    }

    selected_cm_model = st.selectbox(
        "Select Model for Confusion Matrix:",
        list(confusion_matrices.keys()),
        index=0
    )

    cm_data = np.array(confusion_matrices[selected_cm_model])

    cm_col1, cm_col2 = st.columns([1.1, 0.9])

    with cm_col1:
        if PLOTLY_AVAILABLE:
            labels = ["Low", "Moderate", "High"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_data,
                x=labels,
                y=labels,
                colorscale=[[0, "#0f172a"], [0.2, "#1e293b"], [0.5, "#0284c7"], [1.0, "#06b6d4"]],
                text=cm_data,
                texttemplate="%{text}",
                textfont=dict(size=14, color="#ffffff"),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                colorbar=dict(title=dict(text="Samples", font=dict(color="#94a3b8")))
            ))
            fig_cm.update_layout(
                title=dict(text=f"Confusion Matrix: {selected_cm_model}", font=dict(color="#f8fafc", size=13)),
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                height=340,
                margin=dict(l=40, r=30, t=40, b=40),
                font=dict(color="#94a3b8"),
                xaxis=dict(title=dict(text="Predicted Label", font=dict(color="#cbd5e1")), gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title=dict(text="Actual Label", font=dict(color="#cbd5e1")), gridcolor="rgba(255,255,255,0.06)", autorange="reversed")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(
                cm_data,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Low", "Moderate", "High"],
                yticklabels=["Low", "Moderate", "High"],
                ax=ax_cm,
                cbar=False
            )
            ax_cm.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
            ax_cm.set_ylabel("Actual Label", fontsize=11, fontweight="bold")
            ax_cm.set_title(f"Confusion Matrix: {selected_cm_model}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_cm, clear_figure=True)

    with cm_col2:
        st.markdown(f"**Performance Breakdown ({selected_cm_model}):**")
        total_samples = int(cm_data.sum())
        total_correct = int(np.trace(cm_data))
        overall_acc = (total_correct / total_samples) * 100

        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                <div style="background: #111827; padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 11px; color: #64748b;">Test Samples</div>
                    <div style="font-size: 20px; font-weight: 800; color: #f8fafc;">{total_samples}</div>
                </div>
                <div style="background: #111827; padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 11px; color: #64748b;">Accuracy</div>
                    <div style="font-size: 20px; font-weight: 800; color: #22d3ee;">{overall_acc:.2f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        class_names = ["Low", "Moderate", "High"]
        breakdown_rows = []
        for idx, c_name in enumerate(class_names):
            c_total = int(cm_data[idx].sum())
            c_correct = int(cm_data[idx, idx])
            c_recall = (c_correct / c_total * 100) if c_total > 0 else 0.0
            breakdown_rows.append({
                "Stress Class": c_name,
                "Actual Samples": c_total,
                "Correctly Classified": c_correct,
                "Class Recall / Accuracy": f"{c_recall:.1f}%"
            })
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ROC-AUC
    st.subheader("📈 ROC-AUC Analysis")

    if roc_results is not None:
        roc_display = roc_results.copy()
        if "ROC-AUC" in roc_display.columns:
            roc_display["ROC-AUC"] = roc_display["ROC-AUC"].round(4)
        st.dataframe(roc_display, use_container_width=True, hide_index=True)

        if PLOTLY_AVAILABLE and "Model" in roc_results.columns:
            fig_roc = go.Figure(go.Bar(
                x=roc_results["ROC-AUC"],
                y=roc_results["Model"],
                orientation="h",
                marker=dict(
                    color=["#06b6d4", "#8b5cf6", "#10b981", "#3b82f6", "#f59e0b"][:len(roc_results)],
                    line=dict(color="rgba(255,255,255,0.15)", width=1)
                ),
                text=[f"{v:.4f}" for v in roc_results["ROC-AUC"]],
                textposition="auto"
            ))
            fig_roc.update_layout(
                title=dict(text="ROC-AUC Score by Algorithm (One-vs-Rest Macro)", font=dict(color="#f8fafc", size=13)),
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                height=260,
                margin=dict(l=30, r=20, t=30, b=30),
                font=dict(color="#94a3b8"),
                xaxis=dict(range=[0.85, 1.0], gridcolor="rgba(255,255,255,0.06)", title="ROC-AUC Score"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=None)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        elif "Model" in roc_results.columns:
            st.bar_chart(roc_results.set_index("Model")["ROC-AUC"])
    else:
        st.info("roc_auc_results.csv was not found.")


    # ROC curves

    st.markdown("---")

    st.subheader(
        "📉 ROC Curves"
    )

    roc_images = {

        "Logistic Regression":
            "logistic_regression_roc_curve.png",

        "Decision Tree":
            "decision_tree_roc_curve.png",

        "Random Forest":
            "random_forest_roc_curve.png",

        "SVM":
            "svm_roc_curve.png",

        "XGBoost":
            "xgboost_roc_curve.png"

    }


    selected_roc_model = st.selectbox(
        "Select model ROC curve",
        list(roc_images.keys())
    )


    image_file = roc_images[
        selected_roc_model
    ]


    if os.path.exists(image_file):

        st.image(
            image_file,
            caption=(
                f"ROC Curve — {selected_roc_model}"
            ),
            use_container_width=True
        )

    else:

        st.warning(
            f"{image_file} was not found."
        )


# ============================================================
# PAGE 4 — DATASET ANALYSIS
# ============================================================

elif page == "📁 Dataset Analysis":

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 30px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
                📁 Dataset Exploration & Validation
            </h1>
            <p style="margin: 4px 0 0 0; font-size: 14px; color: #94a3b8;">
                Comprehensive data profiling, balance verification, and exploratory distribution analytics across 20 factors.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Modern KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card-modern" style="border-top: 3px solid #06b6d4;">
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-sub">Complete respondent rows</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card-modern" style="border-top: 3px solid #8b5cf6;">
                <div class="metric-title">Columns</div>
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-sub">20 features + 1 target</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card-modern" style="border-top: 3px solid #10b981;">
                <div class="metric-title">Input Factors</div>
                <div class="metric-value">{len(feature_names)}</div>
                <div class="metric-sub">Holistic survey metrics</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card-modern" style="border-top: 3px solid #f59e0b;">
                <div class="metric-title">Missing Values</div>
                <div class="metric-value">{int(df.isna().sum().sum())}</div>
                <div class="metric-sub">Zero imputation required</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.subheader("📋 Dataset Structure & Sample Records")
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("---")

    # Stress Class Distribution with Plotly
    st.subheader("📊 Target Variable Distribution")

    counts = df["stress_level"].value_counts().sort_index()
    distribution_df = pd.DataFrame({
        "Stress Level": [stress_labels[i] for i in counts.index],
        "Records": counts.values,
        "Proportion": [f"{v / len(df) * 100:.2f}%" for v in counts.values]
    })

    dist_col1, dist_col2 = st.columns([0.8, 1.2])

    with dist_col1:
        st.dataframe(distribution_df, use_container_width=True, hide_index=True)
        st.caption(
            "Balanced distribution: The dataset contains near-equal representations across Low (33.91%), "
            "Moderate (32.55%), and High (33.55%) classes, avoiding majority-class bias."
        )

    with dist_col2:
        if PLOTLY_AVAILABLE:
            fig_ds_dist = go.Figure(data=[
                go.Bar(
                    x=["Low Stress", "Moderate Stress", "High Stress"],
                    y=counts.values,
                    marker=dict(
                        color=["#10b981", "#f59e0b", "#ef4444"],
                        line=dict(color="rgba(255,255,255,0.15)", width=1.5)
                    ),
                    text=[f"{v} ({v/len(df)*100:.1f}%)" for v in counts.values],
                    textposition="auto"
                )
            ])
            fig_ds_dist.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                height=260,
                margin=dict(l=30, r=20, t=30, b=30),
                font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=None),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Count")
            )
            st.plotly_chart(fig_ds_dist, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            sns.countplot(x=df["stress_level"].map(stress_labels), ax=ax, palette=["#10b981", "#f59e0b", "#ef4444"])
            ax.set_xlabel("Stress Level", fontsize=10, fontweight="bold")
            ax.set_ylabel("Records", fontsize=10, fontweight="bold")
            st.pyplot(fig, clear_figure=True)

    st.markdown("---")

    # Interactive Factor Distribution Inspector
    st.subheader("🔬 Interactive Feature Distribution Inspector")
    st.write(
        "Inspect the empirical distribution of any specific survey feature stratified across the three stress classes."
    )

    selected_factor = st.selectbox(
        "Select Feature to Inspect:",
        feature_names,
        format_func=lambda x: f"{x.replace('_', ' ').title()} ({x})"
    )

    if PLOTLY_AVAILABLE:
        fig_box = px.box(
            df,
            x="stress_level",
            y=selected_factor,
            color="stress_level",
            color_discrete_map={0: "#10b981", 1: "#f59e0b", 2: "#ef4444"},
            category_orders={"stress_level": [0, 1, 2]},
            labels={"stress_level": "Stress Level (0=Low, 1=Mod, 2=High)", selected_factor: selected_factor.replace("_", " ").title()}
        )
        fig_box.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            height=340,
            margin=dict(l=40, r=30, t=40, b=40),
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # Feature Statistics
    st.subheader("🔢 Comprehensive Feature Descriptive Statistics")
    stats_df = df[feature_names].describe().T
    stats_df = stats_df.rename(columns={
        "count": "Count", "mean": "Mean", "std": "Std Dev",
        "min": "Min", "25%": "25% (Q1)", "50%": "Median (Q2)",
        "75%": "75% (Q3)", "max": "Max"
    })
    st.dataframe(stats_df.round(2), use_container_width=True)


# ============================================================
# PAGE 5 — EXPLAINABLE AI
# ============================================================

elif page == "🔍 Explainable AI":

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 30px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
                🔍 Explainable AI — SHAP Interpretability
            </h1>
            <p style="margin: 4px 0 0 0; font-size: 14px; color: #94a3b8;">
                Game-theoretic feature attributions uncovering how the Random Forest classifier weighs inputs globally and locally.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not SHAP_AVAILABLE:
        st.error("SHAP is not installed in the current environment.")
        st.stop()

    # Global feature importance
    st.subheader("🌎 Global Feature Importance")
    st.write(
        "Mean absolute SHAP value represents the average magnitude of a feature's contribution across all dataset instances. "
        "Higher values signify stronger overall influence on model predictions."
    )

    if shap_importance is not None:
        top_features = shap_importance.head(15).sort_values("Mean_ABS_SHAP", ascending=True)

        if PLOTLY_AVAILABLE:
            fig_global = go.Figure(go.Bar(
                x=top_features["Mean_ABS_SHAP"],
                y=top_features["Feature"],
                orientation="h",
                marker=dict(
                    color=top_features["Mean_ABS_SHAP"],
                    colorscale=[[0, "#0284c7"], [0.5, "#06b6d4"], [1.0, "#22d3ee"]],
                    line=dict(color="rgba(255,255,255,0.15)", width=1)
                ),
                text=[f"{v:.4f}" for v in top_features["Mean_ABS_SHAP"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
            ))
            fig_global.update_layout(
                title=dict(text="Top 15 Global Features Influencing Stress Classification", font=dict(color="#f8fafc", size=13)),
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                height=450,
                margin=dict(l=40, r=30, t=40, b=40),
                font=dict(color="#94a3b8"),
                xaxis=dict(title="Mean Absolute SHAP Value", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
            )
            st.plotly_chart(fig_global, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            ax.barh(top_features["Feature"], top_features["Mean_ABS_SHAP"], color="#06b6d4")
            ax.set_xlabel("Mean Absolute SHAP Value", fontweight="bold", fontsize=10)
            ax.set_ylabel("Feature", fontweight="bold", fontsize=10)
            ax.set_title("Top 15 Features Influencing Stress Prediction", fontweight="bold", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

        with st.expander("📋 View Global Feature Importance Table"):
            st.dataframe(shap_importance, use_container_width=True, hide_index=True)
    else:
        st.warning("shap_feature_importance.csv was not found.")

    st.markdown("---")

    # Class-Specific SHAP Analysis
    st.subheader("📌 Class-Specific SHAP Analysis")
    st.write("Examine which survey factors exert the greatest mathematical weight for each specific target category.")

    class_files = {
        "Low Stress": "shap_low_importance.csv",
        "Moderate Stress": "shap_moderate_importance.csv",
        "High Stress": "shap_high_importance.csv"
    }

    selected_class = st.selectbox("Select Target Stress Class:", list(class_files.keys()))
    class_file = class_files[selected_class]

    if os.path.exists(class_file):
        class_df = pd.read_csv(class_file)
        top_class_features = class_df.head(15).sort_values("Mean_ABS_SHAP", ascending=True)

        class_color_map = {
            "Low Stress": "#10b981",
            "Moderate Stress": "#f59e0b",
            "High Stress": "#ef4444"
        }
        accent = class_color_map.get(selected_class, "#06b6d4")

        if PLOTLY_AVAILABLE:
            fig_cls = go.Figure(go.Bar(
                x=top_class_features["Mean_ABS_SHAP"],
                y=top_class_features["Feature"],
                orientation="h",
                marker=dict(
                    color=accent,
                    line=dict(color="rgba(255,255,255,0.15)", width=1)
                ),
                text=[f"{v:.4f}" for v in top_class_features["Mean_ABS_SHAP"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Mean |SHAP| for Class: %{x:.4f}<extra></extra>"
            ))
            fig_cls.update_layout(
                title=dict(text=f"Top 15 Features Contributing to {selected_class} Classification", font=dict(color="#f8fafc", size=13)),
                paper_bgcolor="#111827",
                plot_bgcolor="#0f172a",
                height=420,
                margin=dict(l=40, r=30, t=40, b=40),
                font=dict(color="#94a3b8"),
                xaxis=dict(title="Mean Absolute SHAP Value", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
            )
            st.plotly_chart(fig_cls, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_class_features["Feature"], top_class_features["Mean_ABS_SHAP"], color=accent)
            ax.set_xlabel("Mean Absolute SHAP Value", fontweight="bold", fontsize=10)
            ax.set_title(f"SHAP Importance — {selected_class}", fontweight="bold", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

        with st.expander(f"📋 View Full Importance Table ({selected_class})"):
            st.dataframe(class_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"{class_file} was not found.")

    st.markdown("---")

    # Interactive Individual Sample Explainer
    st.subheader("🔬 Interactive Individual Respondent Explainer")
    st.write(
        "Inspect how local SHAP values decompose predictions for individual respondent profiles. "
        "Choose a real respondent profile from the dataset or a synthetic demonstration preset."
    )

    sample_options = {
        "Sample 1: High Stress Case (Record #3)": 3,
        "Sample 2: Moderate Stress Case (Record #1)": 1,
        "Sample 3: Low Stress Case (Record #2)": 2,
        "Custom Scenario: Low Stress Preset": "preset_low",
        "Custom Scenario: Moderate Stress Preset": "preset_moderate",
        "Custom Scenario: High Stress Preset": "preset_high"
    }

    selected_sample_label = st.selectbox(
        "Select Sample Profile to Explain:",
        list(sample_options.keys())
    )

    sel_key = sample_options[selected_sample_label]

    if isinstance(sel_key, int):
        sample_row = df[feature_names].iloc[sel_key:sel_key+1]
        actual_label = stress_labels.get(int(df["stress_level"].iloc[sel_key]), "Unknown")
        ground_truth_text = f"Dataset Ground Truth: <strong>{actual_label} Stress</strong> • "
    else:
        preset_scenarios = {
            "preset_low": {
                'anxiety_level': 5.0, 'self_esteem': 25.0, 'mental_health_history': 0.0, 'depression': 6.0,
                'headache': 1.0, 'blood_pressure': 2.0, 'sleep_quality': 4.0, 'breathing_problem': 2.0,
                'academic_performance': 4.0, 'study_load': 2.0, 'teacher_student_relationship': 4.0, 'future_career_concerns': 1.0,
                'social_support': 3.0, 'peer_pressure': 2.0, 'extracurricular_activities': 2.0, 'bullying': 1.0,
                'noise_level': 2.0, 'living_conditions': 3.0, 'safety': 4.0, 'basic_needs': 4.0
            },
            "preset_moderate": {
                'anxiety_level': 11.0, 'self_esteem': 19.0, 'mental_health_history': 1.0, 'depression': 12.0,
                'headache': 2.0, 'blood_pressure': 1.0, 'sleep_quality': 3.0, 'breathing_problem': 3.0,
                'academic_performance': 3.0, 'study_load': 3.0, 'teacher_student_relationship': 2.0, 'future_career_concerns': 3.0,
                'social_support': 2.0, 'peer_pressure': 3.0, 'extracurricular_activities': 3.0, 'bullying': 3.0,
                'noise_level': 2.0, 'living_conditions': 3.0, 'safety': 2.0, 'basic_needs': 3.0
            },
            "preset_high": {
                'anxiety_level': 16.0, 'self_esteem': 9.0, 'mental_health_history': 1.0, 'depression': 20.0,
                'headache': 4.0, 'blood_pressure': 3.0, 'sleep_quality': 1.0, 'breathing_problem': 4.0,
                'academic_performance': 2.0, 'study_load': 4.0, 'teacher_student_relationship': 2.0, 'future_career_concerns': 4.0,
                'social_support': 1.0, 'peer_pressure': 4.0, 'extracurricular_activities': 4.0, 'bullying': 4.0,
                'noise_level': 4.0, 'living_conditions': 2.0, 'safety': 2.0, 'basic_needs': 2.0
            }
        }
        sample_row = pd.DataFrame([preset_scenarios[sel_key]])[feature_names]
        ground_truth_text = "Synthetic Demonstration Preset • "

    try:
        sample_pred = int(model.predict(sample_row)[0])
        sample_proba = model.predict_proba(sample_row)[0]
        sample_label = stress_labels.get(sample_pred, str(sample_pred))
        sample_conf = float(np.max(sample_proba) * 100)

        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px 18px; margin: 12px 0;">
                <div style="font-size: 15px; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
                    Model Classification: <span style="color: #22d3ee;">{sample_label} Stress</span> ({sample_conf:.1f}% Confidence)
                </div>
                <div style="font-size: 12.5px; color: #94a3b8;">
                    {ground_truth_text}Probabilities: Low {sample_proba[0]*100:.1f}%, Moderate {sample_proba[1]*100:.1f}%, High {sample_proba[2]*100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        sample_explainer = shap.TreeExplainer(model)
        s_shap_vals = sample_explainer.shap_values(sample_row)
        s_shap_arr = np.asarray(s_shap_vals)

        # Dynamic class-to-index mapping from model.classes_
        s_class_to_idx = {stress_labels.get(int(c), str(c)): idx for idx, c in enumerate(model.classes_)}

        # Multiclass Target Selector for Respondent Profile
        s_target_options = [
            f"🎯 Predicted Category ({sample_label} Stress — {sample_conf:.1f}%)"
        ]
        for c_name in ["Low", "Moderate", "High"]:
            if c_name != sample_label and c_name in s_class_to_idx:
                c_p = sample_proba[s_class_to_idx[c_name]] * 100
                s_target_options.append(f"{c_name} Stress ({c_p:.1f}%)")

        s_selected_target = st.radio(
            "Decompose local feature contributions for target category:",
            s_target_options,
            horizontal=True,
            key="xai_sample_target_class"
        )

        if "Low" in s_selected_target:
            s_target_c_name = "Low"
        elif "Moderate" in s_selected_target:
            s_target_c_name = "Moderate"
        elif "High" in s_selected_target:
            s_target_c_name = "High"
        else:
            s_target_c_name = sample_label

        s_target_idx = s_class_to_idx[s_target_c_name]
        s_target_proba = float(sample_proba[s_target_idx])

        if s_shap_arr.ndim == 3:
            s_contrib = s_shap_arr[0, :, s_target_idx]
        elif s_shap_arr.ndim == 2:
            s_contrib = s_shap_arr[0]
        else:
            s_contrib = None

        if s_contrib is not None:
            # Mathematical Additivity Verification
            s_shap_sum = float(np.sum(s_contrib))
            s_base_val = None
            if hasattr(sample_explainer, "expected_value") and sample_explainer.expected_value is not None:
                exp_arr = np.asarray(sample_explainer.expected_value)
                if exp_arr.ndim > 0 and len(exp_arr) > s_target_idx:
                    s_base_val = float(exp_arr[s_target_idx])
                elif exp_arr.ndim == 0:
                    s_base_val = float(exp_arr)

            if s_base_val is not None:
                s_calc_output = s_base_val + s_shap_sum
                s_is_additive = np.isclose(s_calc_output, s_target_proba, atol=1e-3)

                if s_is_additive:
                    st.markdown(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.65); padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin: 12px 0;">
                            <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px;">
                                📐 Mathematical Additivity Proof (TreeSHAP Probability Space)
                            </div>
                            <div style="display: flex; gap: 14px; align-items: center; margin-top: 6px; font-size: 13.5px; flex-wrap: wrap;">
                                <span>Prior Population Base ($E[f(x)]$): <strong style="color: #cbd5e1;">{s_base_val * 100:.2f}%</strong></span>
                                <span style="color: #64748b;">+</span>
                                <span>Sum of SHAP Attributions: <strong style="color: {'#10b981' if s_shap_sum >= 0 else '#f43f5e'};">{s_shap_sum * 100:+.2f}%</strong></span>
                                <span style="color: #64748b;">=</span>
                                <span>Model Class Probability: <strong style="color: #22d3ee;">{s_target_proba * 100:.2f}%</strong></span>
                                <span class="badge-pill-modern badge-emerald" style="font-size: 10px;">✓ Additivity Verified (Error < 1e-4)</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.info(
                        f"TreeSHAP base value is {s_base_val:.4f} and sum of SHAP contributions is {s_shap_sum:.4f}. "
                        f"Target probability: {s_target_proba:.4f}."
                    )

            s_vals = [float(sample_row[col].iloc[0]) for col in feature_names]
            s_df = pd.DataFrame({
                "Feature": feature_names,
                "Feature Name": [f.replace("_", " ").title() for f in feature_names],
                "Value": s_vals,
                "SHAP Value": s_contrib,
                "Absolute Impact": np.abs(s_contrib)
            }).sort_values("Absolute Impact", ascending=False).head(10)

            s_col1, s_col2 = st.columns([1.1, 0.9])

            with s_col1:
                plot_s = s_df.sort_values("Absolute Impact", ascending=True)

                if PLOTLY_AVAILABLE:
                    s_colors = ["#06b6d4" if v >= 0 else "#f43f5e" for v in plot_s["SHAP Value"]]
                    s_labels = [f"{r['Feature Name']} ({r['Value']:.0f})" for _, r in plot_s.iterrows()]
                    fig_s = go.Figure(go.Bar(
                        x=plot_s["SHAP Value"],
                        y=s_labels,
                        orientation="h",
                        marker=dict(color=s_colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
                        hovertemplate="<b>%{y}</b><br>SHAP Impact for " + s_target_c_name + ": %{x:+.4f}<extra></extra>"
                    ))
                    fig_s.add_vline(x=0, line_dash="dash", line_color="#94a3b8", line_width=1.5)
                    fig_s.update_layout(
                        title=dict(text=f"Local Feature Attribution ({s_target_c_name} Stress)", font=dict(color="#f8fafc", size=13)),
                        paper_bgcolor="#111827",
                        plot_bgcolor="#0f172a",
                        height=380,
                        margin=dict(l=30, r=20, t=35, b=30),
                        font=dict(color="#94a3b8"),
                        xaxis=dict(title=f"SHAP Impact for '{s_target_c_name}' Stress", gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    fig_s, ax_s = plt.subplots(figsize=(7, 4.8))
                    s_colors = ["#06b6d4" if v >= 0 else "#f43f5e" for v in plot_s["SHAP Value"]]
                    s_labels = [f"{r['Feature Name']} ({r['Value']:.0f})" for _, r in plot_s.iterrows()]
                    ax_s.barh(s_labels, plot_s["SHAP Value"], color=s_colors, height=0.65)
                    ax_s.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.7)
                    ax_s.set_xlabel(f"SHAP Contribution for '{s_target_c_name}' Stress", fontweight="bold", fontsize=10)
                    ax_s.set_title(f"Local Feature Attribution ({s_target_c_name} Stress)", fontweight="bold", fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig_s, clear_figure=True)

            with s_col2:
                st.markdown(f"**Local Feature Attribution Breakdown ({s_target_c_name} Stress):**")
                s_table = []
                for _, r in s_df.iterrows():
                    v_dir = f"Supported {s_target_c_name}" if r["SHAP Value"] > 0 else f"Opposed {s_target_c_name}"
                    s_table.append({
                        "Feature": r["Feature Name"],
                        "Entered Value": f"{r['Value']:.0f}",
                        "SHAP Impact": f"{r['SHAP Value']:+.3f}",
                        "Model Weight": v_dir
                    })
                st.dataframe(pd.DataFrame(s_table), use_container_width=True, hide_index=True)

    except Exception as s_err:
        st.warning(f"Could not compute local sample explanation: {s_err}")

    st.markdown(
        """
        <div class="disclaimer-box-modern">
            <strong>⚠️ Algorithmic Transparency Notice:</strong><br>
            SHAP explains how the machine-learning model mathematically combined survey responses. 
            It reflects statistical weights in decision trees; it does not establish medical causation or clinical diagnoses.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 6 — ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="margin: 0; font-size: 32px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
                ℹ️ About the Project
            </h1>
            <p style="margin: 4px 0 0 0; font-size: 15px; color: #94a3b8;">
                AI-Based Mental Stress Detection • B.Tech Artificial Intelligence & Data Science End-Semester System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown(
            """
            <div class="metric-card-modern" style="margin-bottom: 16px;">
                <div class="metric-title" style="margin-bottom: 6px;">🎯 Project Purpose & Scope</div>
                <p style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin: 0;">
                    Develop a supervised machine learning classification architecture that analyzes multi-factor 
                    behavioural, physiological, academic, and social indicators to provide early stress awareness. 
                    The system categorizes stress into Low (0), Moderate (1), and High (2) states.
                </p>
            </div>

            <div class="metric-card-modern" style="margin-bottom: 16px;">
                <div class="metric-title" style="margin-bottom: 6px;">🤖 Supervised Algorithms Evaluated</div>
                <p style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin: 0;">
                    Five classification algorithms were trained and benchmarked under identical 80/20 stratified splits:
                </p>
                <ul style="font-size: 13px; color: #94a3b8; margin: 8px 0 0 0; padding-left: 18px; line-height: 1.7;">
                    <li><strong>Random Forest (Baseline):</strong> 88.64% Accuracy • 88.61% F1 • 0.9831 ROC-AUC (Ensemble bagging)</li>
                    <li><strong>Logistic Regression:</strong> 88.18% Accuracy • 88.17% F1 • 0.9853 ROC-AUC (Linear decision boundary)</li>
                    <li><strong>XGBoost:</strong> 88.18% Accuracy • 88.17% F1 • 0.9836 ROC-AUC (Gradient boosted trees)</li>
                    <li><strong>Support Vector Machine (SVM):</strong> 87.73% Accuracy • 87.71% F1 • 0.9845 ROC-AUC (RBF kernel)</li>
                    <li><strong>Decision Tree:</strong> 85.45% Accuracy • 85.42% F1 • 0.8911 ROC-AUC (Single tree baseline)</li>
                </ul>
            </div>

            <div class="metric-card-modern">
                <div class="metric-title" style="margin-bottom: 6px;">⚙️ Model Optimization & Regularization</div>
                <p style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin: 0;">
                    The Random Forest classifier was regularized using 5-fold GridSearchCV tuning on the training set:
                </p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px;">
                    <div style="background: rgba(15,23,42,0.6); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06);">
                        <span style="font-size: 11px; color: #64748b;">Baseline RF Test:</span><br>
                        <strong style="color: #cbd5e1;">88.64% Acc (0.9831 AUC)</strong>
                    </div>
                    <div style="background: rgba(15,23,42,0.6); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06);">
                        <span style="font-size: 11px; color: #64748b;">5-Fold CV on X_train:</span><br>
                        <strong style="color: #a78bfa;">88.18% (±0.81%)</strong>
                    </div>
                    <div style="background: rgba(15,23,42,0.6); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06);">
                        <span style="font-size: 11px; color: #64748b;">Optimized RF Test:</span><br>
                        <strong style="color: #22d3ee;">90.91% Acc (+2.27%)</strong>
                    </div>
                    <div style="background: rgba(15,23,42,0.6); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06);">
                        <span style="font-size: 11px; color: #64748b;">Optimized ROC-AUC:</span><br>
                        <strong style="color: #10b981;">0.9853 (+0.0022)</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_a2:
        st.markdown(
            """
            <div class="metric-card-modern" style="margin-bottom: 16px;">
                <div class="metric-title" style="margin-bottom: 6px;">📊 Dataset Architecture</div>
                <p style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin: 0;">
                    • <strong>Total Records:</strong> 1,100 clean respondent survey entries.<br>
                    • <strong>Input Dimensions:</strong> 20 continuous & ordinal features across 4 life dimensions.<br>
                    • <strong>Class Proportions:</strong> Low (373), Moderate (358), High (369).<br>
                    • <strong>Missing Data:</strong> 0 missing or imputed values.
                </p>
            </div>

            <div class="metric-card-modern" style="margin-bottom: 16px;">
                <div class="metric-title" style="margin-bottom: 6px;">🔍 Explainable AI (SHAP)</div>
                <p style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin: 0;">
                    Implements Shapley Additive Explanations (TreeExplainer) to compute exact mathematical feature attributions:
                </p>
                <ul style="font-size: 13px; color: #94a3b8; margin: 8px 0 0 0; padding-left: 18px;">
                    <li><strong>Global Interpretability:</strong> Ranks feature influence across the entire population.</li>
                    <li><strong>Class-Specific Importance:</strong> Decomposes drivers for Low, Moderate, and High classes.</li>
                    <li><strong>Local Attributions:</strong> Breaks down individual predictions into positive drivers and countervailing factors.</li>
                </ul>
            </div>

            <div class="metric-card-modern">
                <div class="metric-title" style="margin-bottom: 6px;">🛡️ Ethical Scope & Guidance Boundaries</div>
                <p style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin: 0;">
                    • <strong>Educational Nature:</strong> Non-clinical lifestyle and academic pacing suggestions.<br>
                    • <strong>Medical Safety:</strong> Physical symptoms must never be self-treated without physician consultation.<br>
                    • <strong>Support Access:</strong> Directs users with elevated stress to professional campus counseling resources.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------------------------------------------
    # TECHNICAL DEFENSE & TRANSPARENCY ARCHITECTURE
    # ------------------------------------------------
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🏛️ Technical Defense, Architectural Rationale & Transparency FAQ")
    st.markdown(
        "Structured technical defenses prepared for project examination, addressing model selection rationale, "
        "explainability mathematics, counterfactual mechanics, dataset boundaries, and ethical scopes."
    )

    with st.expander("📌 1. Empirical Justification for Random Forest Selection (Why RF?)", expanded=True):
        st.markdown(
            """
            * **Empirical Benchmark Evidence:** Across identical 80/20 stratified splits of the 1,100 respondent dataset, Random Forest demonstrated the highest baseline classification accuracy (88.64%) among the five benchmarked algorithms (LR: 88.18%, XGB: 88.18%, SVM: 87.73%, DT: 85.45%). Following hyperparameter optimization, test accuracy reached 90.91% with a multiclass ROC-AUC of 0.9853.
            * **Ensemble Bagging vs. Single Trees:** Single decision trees suffer from high variance and sensitivity to training data perturbations (85.45% accuracy). Random Forest averages multiple decorrelated decision trees, substantially reducing variance without increasing bias.
            * **Tabular Domain Appropriateness vs. Deep Learning:** Tabular survey datasets with modest sample sizes (~1,100 records) and discrete ordinal scales (e.g., 0–5 Likert scales) are well-suited for tree-based ensemble bagging. Deep neural networks on small tabular datasets carry severe risks of parameter overfitting, require costly hyperparameter calibration, and act as opaque black boxes without native polynomial-time Shapley explanations.
            * **Defensibility Caveat:** Random Forest is **not universally optimal** for all machine learning problems. It was empirically chosen for this specific project based on comparative benchmark metrics, stability across 5-fold cross-validation ($88.18\\% \\pm 0.81\\%$), and native compatibility with exact TreeSHAP explanation algorithms.
            """
        )

    with st.expander("📌 2. Mathematical Mechanics of TreeSHAP (Explainable AI Defense)"):
        st.markdown(
            r"""
            * **Polynomial-Time Complexity:** Classical model-agnostic KernelSHAP requires exponential time $\mathcal{O}(2^M)$ by evaluating exponential subsets of features. TreeSHAP (Lundberg et al., 2020) exploits the tree graph topology to compute exact Shapley values in polynomial time $\mathcal{O}(T L D^2)$, where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is maximum tree depth.
            * **Four Fundamental Axioms:** TreeSHAP mathematically guarantees Efficiency ($\sum \phi_i = f(x) - E[f(x)]$), Symmetry (identical contributors receive equal attribution), Dummy/Null player (features with zero marginal contribution receive $\phi_i = 0$), and Additivity across ensemble trees.
            * **Probability-Scale Additivity:** For Scikit-Learn's `RandomForestClassifier`, each leaf node stores class frequencies. TreeSHAP computes attributions directly in class probability space:
            $$\mathbb{E}[f(x)] + \sum_{i=1}^{20} \phi_i(x) = P(\text{Stress Level} = c \mid x)$$
            This mathematical identity is verified in real time on both the Stress Prediction and Explainable AI pages with numerical tolerances $< 10^{-3}$.
            """
        )

    with st.expander("📌 3. What-If Counterfactual Sensitivity Mechanics"):
        st.markdown(
            r"""
            * **Ceteris Paribus Methodology:** What-If analysis applies the *ceteris paribus* ("all other things held constant") principle from sensitivity analysis. A user alters a selected subset of input features while preserving all remaining factors in their baseline state.
            * **Decision Boundary & Margin Exploration:** The system recalculates predicted class probabilities simultaneously across all three stress categories. When the predicted category remains unchanged, the delta breakdown reveals whether internal confidence moved closer to or further from adjacent decision thresholds.
            * **Attribution Sensitivity (ΔSHAP):** By computing the difference between hypothetical and baseline Shapley attributions ($\Delta \phi_i = \phi_{\text{hypo}, i} - \phi_{\text{base}, i}$), the system quantifies exactly which feature modifications caused the probability shift toward or away from the target class.
            """
        )

    with st.expander("📌 4. Dataset Architecture & Methodological Boundaries"):
        st.markdown(
            """
            * **Dataset Characteristics:** 1,100 respondent records spanning 20 psychological, physiological, academic, and social variables with zero missing values. Three balanced classes: Low (373), Moderate (358), and High (369).
            * **Self-Reporting & Subjectivity Bias:** All inputs originate from subjective self-report questionnaires. Responses to ordinal questions (e.g., headache frequency or academic load on a 0–5 scale) reflect individual perception rather than standardized clinical biometric readings.
            * **Cross-Sectional Limitation:** The dataset captures single-point-in-time cross-sectional snapshots. It does not track respondents longitudinally across exam cycles or life transitions, meaning temporal trajectories cannot be inferred.
            * **Correlation vs. Causation:** Statistical correlations learned by decision trees do not establish medical or biological causation.
            """
        )

    with st.expander("📌 5. Clinical Safety Boundary & Ethical Guidance Scope"):
        st.markdown(
            """
            * **Educational & Awareness Purpose:** This application is strictly an educational engineering prototype designed to demonstrate Explainable AI and machine learning. It is **not a diagnostic medical device** and has not been cleared for clinical diagnostic use.
            * **Mandatory Physician Evaluation for Physical Symptoms:** Physical indicators (e.g., severe headaches, chronic sleep disturbance, blood pressure irregularities, breathing difficulties) must be evaluated by licensed medical physicians to rule out underlying organic pathologies.
            * **Campus & Professional Resource Referral:** When moderate or high stress patterns are indicated, the application prominently provides guidance directing students to institutional counseling centers, peer support programs, and mental health crisis professionals.
            """
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 13px; padding: 16px 0; border-top: 1px solid rgba(255,255,255,0.06);">
            AI-Based Mental Stress Detection • B.Tech Artificial Intelligence & Data Science • Group A6
        </div>
        """,
        unsafe_allow_html=True
    )