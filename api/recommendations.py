"""
api/recommendations.py
======================
Transparent, rule-based personalized educational wellness recommendations.
Provides domain-specific focus suggestions, protective buffer recognitions,
and medical awareness notices.
"""

def generate_personalized_guidance(user_inputs: dict, prediction: int, label: str) -> dict:
    """
    Transparent, rule-based educational guidance engine.
    Derives actionable suggestions based on actual input values and verified dataset ranges.
    Does not use SHAP values as medical claims. Does not diagnose illness.
    """
    if not isinstance(user_inputs, dict):
        return {
            "focus_areas": [],
            "positive_buffers": [],
            "label": label,
            "prediction": prediction,
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
                "Establish a consistent sleep schedule (7-8 hours); maintain a dark, "
                "quiet sleeping environment, and disconnect from digital screens 45 minutes "
                "before bedtime. If sleep difficulties persist, seek guidance from a healthcare provider."
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
        "prediction": prediction,
        "disclaimer": (
            "Guidance is educational and non-clinical. Physical symptoms require physician evaluation."
        )
    }
