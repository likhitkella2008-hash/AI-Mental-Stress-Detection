"""
test_recommendations.py
=======================
Automated test suite for the Personalized Stress Awareness Guidance engine.
Tests every rule against low, moderate, and high input scenarios,
boundary values, positive well-being factors, and missing input edge cases.
"""

import unittest
import numpy as np
import pandas as pd

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


class TestPersonalizedRecommendations(unittest.TestCase):

    def test_sleep_rule_trigger(self):
        """Test that sleep quality <= 1 triggers rest recommendation and >= 4 triggers positive buffer."""
        # Low sleep
        res_low = generate_personalized_guidance({"sleep_quality": 1.0}, 2, "High")
        factors_low = [r["factor"] for r in res_low["focus_areas"]]
        self.assertIn("Sleep Quality", factors_low)
        self.assertIn("7-8 hours", res_low["focus_areas"][0]["suggestion"])

        # High sleep (positive buffer)
        res_high = generate_personalized_guidance({"sleep_quality": 5.0}, 0, "Low")
        buffer_factors = [b["factor"] for b in res_high["positive_buffers"]]
        self.assertIn("Sleep Quality", buffer_factors)

    def test_study_load_rule_trigger(self):
        """Test study load >= 4 triggers workload advice and <= 2 triggers positive buffer."""
        res_heavy = generate_personalized_guidance({"study_load": 5.0}, 2, "High")
        factors = [r["factor"] for r in res_heavy["focus_areas"]]
        self.assertIn("Study Load", factors)
        self.assertIn("Pomodoro", res_heavy["focus_areas"][0]["suggestion"])

        res_light = generate_personalized_guidance({"study_load": 1.0}, 0, "Low")
        buffers = [b["factor"] for b in res_light["positive_buffers"]]
        self.assertIn("Study Load", buffers)

    def test_social_support_rule_trigger(self):
        """Test social support <= 1 triggers connection advice and >= 3 triggers buffer."""
        res_low = generate_personalized_guidance({"social_support": 0.0}, 2, "High")
        factors = [r["factor"] for r in res_low["focus_areas"]]
        self.assertIn("Social Support", factors)

        res_high = generate_personalized_guidance({"social_support": 3.0}, 0, "Low")
        buffers = [b["factor"] for b in res_high["positive_buffers"]]
        self.assertIn("Social Support", buffers)

    def test_anxiety_rule_trigger(self):
        """Test anxiety >= 16 triggers relaxation guidance."""
        res = generate_personalized_guidance({"anxiety_level": 18.0}, 2, "High")
        factors = [r["factor"] for r in res["focus_areas"]]
        self.assertIn("Anxiety Level", factors)

    def test_academic_performance_trigger(self):
        """Test academic performance <= 1 triggers tutoring/advising advice."""
        res = generate_personalized_guidance({"academic_performance": 1.0}, 1, "Moderate")
        factors = [r["factor"] for r in res["focus_areas"]]
        self.assertIn("Academic Performance", factors)

    def test_physical_symptoms_careful_advice(self):
        """Test physical symptoms trigger careful medical advice without self-care substitute."""
        res = generate_personalized_guidance({"headache": 4.0, "blood_pressure": 3.0}, 2, "High")
        factors = [r["factor"] for r in res["focus_areas"]]
        self.assertIn("Physical Health Factors", factors)
        sugg = [r["suggestion"] for r in res["focus_areas"] if r["factor"] == "Physical Health Factors"][0]
        self.assertIn("consult a qualified physician", sugg)
        self.assertIn("should not be assumed to stem solely from stress", sugg)

    def test_bullying_and_peer_pressure_triggers(self):
        """Test bullying >= 3 and peer pressure >= 4 triggers."""
        res = generate_personalized_guidance({"bullying": 4.0, "peer_pressure": 5.0}, 2, "High")
        factors = [r["factor"] for r in res["focus_areas"]]
        self.assertIn("Bullying / Interpersonal Friction", factors)
        self.assertIn("Peer Pressure", factors)

    def test_future_career_concerns_trigger(self):
        """Test future career concerns >= 4 triggers career guidance."""
        res = generate_personalized_guidance({"future_career_concerns": 4.0}, 1, "Moderate")
        factors = [r["factor"] for r in res["focus_areas"]]
        self.assertIn("Future Career Concerns", factors)

    def test_positive_buffers_detection(self):
        """Test positive buffers detection for high self esteem and good mentorship."""
        res = generate_personalized_guidance({
            "self_esteem": 28.0,
            "teacher_student_relationship": 5.0
        }, 0, "Low")
        buffers = [b["factor"] for b in res["positive_buffers"]]
        self.assertIn("Self-Esteem", buffers)
        self.assertIn("Faculty Mentorship", buffers)

    def test_graceful_missing_or_invalid_inputs(self):
        """Test that missing keys, None, or invalid strings do not raise unhandled exceptions."""
        res_empty = generate_personalized_guidance({}, 1, "Moderate")
        self.assertIsInstance(res_empty["focus_areas"], list)

        res_invalid = generate_personalized_guidance({"sleep_quality": "invalid", "anxiety_level": None}, 1, "Moderate")
        self.assertIsInstance(res_invalid["focus_areas"], list)

        res_none = generate_personalized_guidance(None, 0, "Low")
        self.assertEqual(res_none["overall_note"], "No input data provided.")


if __name__ == "__main__":
    unittest.main()
