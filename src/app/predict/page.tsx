"use client";

import { useState } from "react";
import { 
  Brain, 
  Sparkles, 
  RotateCcw, 
  Activity, 
  CheckCircle2, 
  AlertTriangle, 
  ShieldCheck, 
  Info,
  ChevronRight,
  TrendingUp,
  BarChart2
} from "lucide-react";

const PRESETS = {
  low: {
    name: "Low Stress Preset",
    color: "emerald",
    inputs: {
      anxiety_level: 5.0, self_esteem: 25.0, mental_health_history: 0.0, depression: 6.0,
      headache: 1.0, blood_pressure: 2.0, sleep_quality: 4.0, breathing_problem: 2.0,
      academic_performance: 4.0, study_load: 2.0, teacher_student_relationship: 4.0, future_career_concerns: 1.0,
      social_support: 3.0, peer_pressure: 2.0, extracurricular_activities: 2.0, bullying: 1.0,
      noise_level: 2.0, living_conditions: 3.0, safety: 4.0, basic_needs: 4.0
    }
  },
  moderate: {
    name: "Moderate Stress Preset",
    color: "amber",
    inputs: {
      anxiety_level: 11.0, self_esteem: 19.0, mental_health_history: 1.0, depression: 12.0,
      headache: 2.0, blood_pressure: 1.0, sleep_quality: 3.0, breathing_problem: 3.0,
      academic_performance: 3.0, study_load: 3.0, teacher_student_relationship: 2.0, future_career_concerns: 3.0,
      social_support: 2.0, peer_pressure: 3.0, extracurricular_activities: 3.0, bullying: 3.0,
      noise_level: 2.0, living_conditions: 3.0, safety: 2.0, basic_needs: 3.0
    }
  },
  high: {
    name: "High Stress Preset",
    color: "rose",
    inputs: {
      anxiety_level: 16.0, self_esteem: 9.0, mental_health_history: 1.0, depression: 20.0,
      headache: 4.0, blood_pressure: 3.0, sleep_quality: 1.0, breathing_problem: 4.0,
      academic_performance: 2.0, study_load: 4.0, teacher_student_relationship: 2.0, future_career_concerns: 4.0,
      social_support: 1.0, peer_pressure: 4.0, extracurricular_activities: 4.0, bullying: 4.0,
      noise_level: 4.0, living_conditions: 2.0, safety: 2.0, basic_needs: 2.0
    }
  },
  average: {
    name: "Population Median",
    color: "cyan",
    inputs: {
      anxiety_level: 11.0, self_esteem: 18.0, mental_health_history: 0.0, depression: 13.0,
      headache: 3.0, blood_pressure: 2.0, sleep_quality: 3.0, breathing_problem: 3.0,
      academic_performance: 3.0, study_load: 3.0, teacher_student_relationship: 3.0, future_career_concerns: 3.0,
      social_support: 2.0, peer_pressure: 3.0, extracurricular_activities: 3.0, bullying: 3.0,
      noise_level: 3.0, living_conditions: 3.0, safety: 3.0, basic_needs: 3.0
    }
  }
};

const FEATURE_CONFIGS = [
  {
    domain: "Mental & Emotional",
    features: [
      { id: "anxiety_level", label: "Anxiety Level", min: 0, max: 21, step: 1, scale: "0 to 21 (Higher = More Severe)" },
      { id: "self_esteem", label: "Self-Esteem", min: 0, max: 30, step: 1, scale: "0 to 30 (Higher = Stronger Buffer)" },
      { id: "mental_health_history", label: "Mental Health History", min: 0, max: 1, step: 1, scale: "0 = None, 1 = Past History" },
      { id: "depression", label: "Depression Score", min: 0, max: 27, step: 1, scale: "0 to 27 (Higher = More Severe)" },
    ]
  },
  {
    domain: "Physical Health",
    features: [
      { id: "headache", label: "Headache Frequency", min: 0, max: 5, step: 1, scale: "0 to 5 (Higher = More Frequent)" },
      { id: "blood_pressure", label: "Blood Pressure Rating", min: 1, max: 3, step: 1, scale: "1 = Low, 2 = Normal, 3 = Elevated" },
      { id: "sleep_quality", label: "Sleep Quality", min: 0, max: 5, step: 1, scale: "0 = Poor / Insomnia, 5 = Excellent" },
      { id: "breathing_problem", label: "Breathing Difficulty", min: 0, max: 5, step: 1, scale: "0 to 5 (Higher = More Frequent)" },
    ]
  },
  {
    domain: "Academic Factors",
    features: [
      { id: "academic_performance", label: "Academic Standing", min: 0, max: 5, step: 1, scale: "0 = Struggling, 5 = Excellent" },
      { id: "study_load", label: "Course Study Load", min: 0, max: 5, step: 1, scale: "0 = Light, 5 = Overwhelming" },
      { id: "teacher_student_relationship", label: "Teacher Rapport", min: 0, max: 5, step: 1, scale: "0 = Poor, 5 = Very Supportive" },
      { id: "future_career_concerns", label: "Career Anxiety", min: 0, max: 5, step: 1, scale: "0 = None, 5 = Extreme Worry" },
    ]
  },
  {
    domain: "Social & Environmental",
    features: [
      { id: "social_support", label: "Social Support Network", min: 0, max: 3, step: 1, scale: "0 = Isolated, 3 = Strong Support" },
      { id: "peer_pressure", label: "Peer Pressure", min: 0, max: 5, step: 1, scale: "0 = None, 5 = Intense" },
      { id: "extracurricular_activities", label: "Extracurriculars", min: 0, max: 5, step: 1, scale: "0 = None, 5 = High Activity" },
      { id: "bullying", label: "Bullying / Friction", min: 0, max: 5, step: 1, scale: "0 = None, 5 = Severe" },
      { id: "noise_level", label: "Ambient Noise", min: 0, max: 5, step: 1, scale: "0 = Quiet, 5 = Very Loud" },
      { id: "living_conditions", label: "Living Conditions", min: 0, max: 5, step: 1, scale: "0 = Inadequate, 5 = Very Comfortable" },
      { id: "safety", label: "Personal Safety", min: 0, max: 5, step: 1, scale: "0 = Unsafe, 5 = Completely Secure" },
      { id: "basic_needs", label: "Basic Needs Security", min: 0, max: 5, step: 1, scale: "0 = Deprived, 5 = Fully Met" },
    ]
  }
];

export default function PredictionPage() {
  const [inputs, setInputs] = useState<Record<string, number>>(PRESETS.moderate.inputs);
  const [selectedDomain, setSelectedDomain] = useState("Mental & Emotional");
  const [targetCategory, setTargetCategory] = useState<string>("Predicted");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: string, value: number) => {
    setInputs((prev) => ({ ...prev, [field]: value }));
  };

  const handlePreset = (presetKey: keyof typeof PRESETS) => {
    setInputs(PRESETS[presetKey].inputs);
  };

  const handlePredict = async (chosenTarget?: string) => {
    setLoading(true);
    setError(null);
    try {
      const payload: any = { ...inputs };
      if (chosenTarget && chosenTarget !== "Predicted") {
        payload.target_category = chosenTarget;
      } else if (targetCategory !== "Predicted") {
        payload.target_category = targetCategory;
      }

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Prediction request failed");
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTargetChange = (cat: string) => {
    setTargetCategory(cat);
    handlePredict(cat);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">
          <Brain className="w-4 h-4" />
          <span>Interactive Machine Learning Inference</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Multidimensional Stress Prediction &amp; SHAP Explainer
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Input 20 behavioral, physiological, academic, and social variables to evaluate stress state 
          using the trained Champion Random Forest model.
        </p>
      </div>

      {/* Preset Scenario Buttons */}
      <div className="metric-card-modern p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Quick Demonstration Presets:
          </span>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => handlePreset("low")}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 transition-colors"
            >
              🟢 Low Stress
            </button>
            <button
              onClick={() => handlePreset("moderate")}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 transition-colors"
            >
              🟡 Moderate Stress
            </button>
            <button
              onClick={() => handlePreset("high")}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 transition-colors"
            >
              🔴 High Stress
            </button>
            <button
              onClick={() => handlePreset("average")}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 transition-colors"
            >
              🔄 Dataset Median
            </button>
          </div>
        </div>
      </div>

      {/* 4-Domain Form */}
      <div className="metric-card-modern p-6">
        {/* Domain Tabs */}
        <div className="flex flex-wrap border-b border-white/10 gap-2 pb-3 mb-6">
          {FEATURE_CONFIGS.map((c) => (
            <button
              key={c.domain}
              onClick={() => setSelectedDomain(c.domain)}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                selectedDomain === c.domain
                  ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-white/5"
              }`}
            >
              {c.domain}
            </button>
          ))}
        </div>

        {/* Input sliders for active domain */}
        {FEATURE_CONFIGS.map((cfg) => {
          if (cfg.domain !== selectedDomain) return null;
          return (
            <div key={cfg.domain} className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {cfg.features.map((f) => {
                const val = inputs[f.id] ?? f.min;
                return (
                  <div key={f.id} className="bg-navy-950/60 p-4 rounded-xl border border-white/5 space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-bold text-slate-200">{f.label}</label>
                      <span className="text-xs font-mono font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                        {val}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={val}
                      onChange={(e) => handleInputChange(f.id, parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500">
                      <span>{f.scale}</span>
                      <span>Range: [{f.min} - {f.max}]</span>
                    </div>
                  </div>
                );
              })}
            </div>
          );
        })}

        {/* Predict Action */}
        <div className="mt-8 pt-6 border-t border-white/10 flex justify-end gap-3">
          <button
            onClick={() => handlePredict()}
            disabled={loading}
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 text-navy-950 font-bold text-xs uppercase tracking-wider hover:opacity-95 shadow-lg shadow-cyan-500/20 flex items-center gap-2 transition-all disabled:opacity-50"
          >
            {loading ? <Activity className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
            <span>{loading ? "Computing Prediction..." : "Run AI Stress Inference"}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Results View */}
      {result && result.prediction && (
        <div className="space-y-6">
          {/* Main Verdict Card */}
          <div
            className={`metric-card-modern p-6 border-l-4 ${
              result.prediction.predicted_label === "Low"
                ? "border-l-emerald-500 bg-gradient-to-r from-emerald-950/20 to-navy-900"
                : result.prediction.predicted_label === "Moderate"
                ? "border-l-amber-500 bg-gradient-to-r from-amber-950/20 to-navy-900"
                : "border-l-rose-500 bg-gradient-to-r from-rose-950/20 to-navy-900"
            }`}
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                  Model Classification Output
                </span>
                <h3 className="text-2xl font-extrabold text-white mt-1 flex items-center gap-2">
                  <span>{result.prediction.predicted_label} Stress State</span>
                  <span
                    className={`badge-pill ${
                      result.prediction.predicted_label === "Low"
                        ? "badge-emerald"
                        : result.prediction.predicted_label === "Moderate"
                        ? "badge-amber"
                        : "badge-rose"
                    }`}
                  >
                    Class {result.prediction.predicted_class}
                  </span>
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Model confidence: <strong className="text-cyan-400">{result.prediction.confidence_pct}%</strong>
                </p>
              </div>

              {/* Three-class probability pills */}
              <div className="flex gap-2">
                <div className="bg-navy-950/80 p-2.5 rounded-xl border border-white/5 text-center min-w-[80px]">
                  <span className="text-[10px] text-emerald-400 font-bold block">Low</span>
                  <span className="text-sm font-mono font-bold text-white">
                    {(result.prediction.probabilities.Low * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-navy-950/80 p-2.5 rounded-xl border border-white/5 text-center min-w-[80px]">
                  <span className="text-[10px] text-amber-400 font-bold block">Moderate</span>
                  <span className="text-sm font-mono font-bold text-white">
                    {(result.prediction.probabilities.Moderate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-navy-950/80 p-2.5 rounded-xl border border-white/5 text-center min-w-[80px]">
                  <span className="text-[10px] text-rose-400 font-bold block">High</span>
                  <span className="text-sm font-mono font-bold text-white">
                    {(result.prediction.probabilities.High * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Probability Progress Bar */}
            <div className="mt-5 space-y-1.5">
              <div className="flex justify-between text-[11px] text-slate-400 font-medium">
                <span>Model Output Distribution</span>
                <span>Sum = 100%</span>
              </div>
              <div className="h-2.5 w-full bg-slate-900 rounded-full overflow-hidden flex">
                <div 
                  className="bg-emerald-500 transition-all duration-500" 
                  style={{ width: `${result.prediction.probabilities.Low * 100}%` }} 
                  title={`Low: ${(result.prediction.probabilities.Low * 100).toFixed(1)}%`}
                />
                <div 
                  className="bg-amber-500 transition-all duration-500" 
                  style={{ width: `${result.prediction.probabilities.Moderate * 100}%` }} 
                  title={`Moderate: ${(result.prediction.probabilities.Moderate * 100).toFixed(1)}%`}
                />
                <div 
                  className="bg-rose-500 transition-all duration-500" 
                  style={{ width: `${result.prediction.probabilities.High * 100}%` }} 
                  title={`High: ${(result.prediction.probabilities.High * 100).toFixed(1)}%`}
                />
              </div>
            </div>
          </div>

          {/* Explainable AI (TreeSHAP Decomposition) */}
          {result.explanation && (
            <div className="metric-card-modern p-6 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h4 className="text-base font-bold text-white flex items-center gap-2">
                    <BarChart2 className="w-4 h-4 text-cyan-400" />
                    <span>Interactive Multiclass SHAP Explanation (Local Attribution)</span>
                  </h4>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Decompose how each survey factor shifted output probability toward or away from any target category.
                  </p>
                </div>

                {/* Target Category Selector */}
                <div className="flex items-center gap-1.5 bg-navy-950 p-1 rounded-xl border border-white/10">
                  {["Low", "Moderate", "High"].map((cat) => (
                    <button
                      key={cat}
                      onClick={() => handleTargetChange(cat)}
                      className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                        result.explanation.target_category === cat
                          ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                          : "text-slate-400 hover:text-white"
                      }`}
                    >
                      {cat} Stress
                    </button>
                  ))}
                </div>
              </div>

              {/* Mathematical Additivity Proof Card */}
              {result.explanation.is_additive && (
                <div className="bg-navy-950/80 p-4 rounded-xl border border-white/10 text-xs">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-slate-300 uppercase tracking-wider text-[10px]">
                      📐 Mathematical Additivity Proof (TreeSHAP Probability Space)
                    </span>
                    <span className="badge-pill badge-emerald text-[9px]">
                      ✓ Additivity Verified (Error &lt; 1e-4)
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-3 text-slate-300 text-xs font-mono">
                    <span>
                      Prior Base E[f(x)]: <strong className="text-slate-100">{result.explanation.base_value_pct}%</strong>
                    </span>
                    <span className="text-slate-500">+</span>
                    <span>
                      Sum of SHAP Attributions:{" "}
                      <strong className={result.explanation.shap_sum >= 0 ? "text-emerald-400" : "text-rose-400"}>
                        {result.explanation.shap_sum_pct >= 0 ? `+${result.explanation.shap_sum_pct}` : result.explanation.shap_sum_pct}%
                      </strong>
                    </span>
                    <span className="text-slate-500">=</span>
                    <span>
                      Model Class Probability: <strong className="text-cyan-400">{result.explanation.target_probability_pct}%</strong>
                    </span>
                  </div>
                </div>
              )}

              {/* Top 10 Contributing Factors Diverging Chart */}
              <div className="space-y-3 pt-2">
                <div className="flex justify-between items-center text-xs text-slate-400">
                  <span>Top Contributing Factors toward <strong>{result.explanation.target_category} Stress</strong></span>
                  <span className="flex items-center gap-3 text-[11px]">
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded bg-cyan-400" /> Supported</span>
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded bg-rose-400" /> Opposed</span>
                  </span>
                </div>

                <div className="space-y-2">
                  {result.explanation.top_features.map((feat: any) => {
                    const isPositive = feat.shap_value >= 0;
                    const maxImpact = Math.max(...result.explanation.top_features.map((f: any) => f.absolute_impact), 0.01);
                    const widthPct = Math.min(100, (feat.absolute_impact / maxImpact) * 100);

                    return (
                      <div key={feat.feature} className="bg-navy-950/50 p-2.5 rounded-lg border border-white/5 space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="font-semibold text-slate-200">
                            {feat.feature_name} <span className="text-slate-500 font-mono">({feat.entered_value})</span>
                          </span>
                          <span className={`font-mono font-bold ${isPositive ? "text-cyan-400" : "text-rose-400"}`}>
                            {isPositive ? `+${feat.impact_pct}%` : `${feat.impact_pct}%`}
                          </span>
                        </div>
                        <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden flex items-center">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${isPositive ? "bg-cyan-500" : "bg-rose-500"}`}
                            style={{ width: `${widthPct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Personalized Recommendations */}
          {result.recommendations && (
            <div className="space-y-6">
              {/* Focus Areas */}
              {result.recommendations.focus_areas && result.recommendations.focus_areas.length > 0 && (
                <div className="metric-card-modern p-6 space-y-4">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    <h4 className="text-base font-bold text-white">Priority Focus &amp; Pacing Areas</h4>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {result.recommendations.focus_areas.map((rec: any, idx: number) => (
                      <div key={idx} className="bg-navy-950/70 p-4 rounded-xl border border-white/5 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="badge-pill badge-amber text-[10px]">{rec.domain}</span>
                          <span className="text-xs font-mono text-slate-400">{rec.observed_value}</span>
                        </div>
                        <h5 className="text-xs font-bold text-slate-100">{rec.factor}</h5>
                        <p className="text-xs text-slate-300 leading-relaxed">{rec.suggestion}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Protective Buffers */}
              {result.recommendations.positive_buffers && result.recommendations.positive_buffers.length > 0 && (
                <div className="metric-card-modern p-6 space-y-4">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    <h4 className="text-base font-bold text-white">Identified Protective Well-Being Buffers</h4>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {result.recommendations.positive_buffers.map((buf: any, idx: number) => (
                      <div key={idx} className="bg-navy-950/70 p-4 rounded-xl border border-white/5 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="badge-pill badge-emerald text-[10px]">{buf.domain}</span>
                          <span className="text-xs font-mono text-emerald-400">{buf.observed_value}</span>
                        </div>
                        <h5 className="text-xs font-bold text-slate-100">{buf.factor}</h5>
                        <p className="text-xs text-slate-300 leading-relaxed">{buf.note}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
