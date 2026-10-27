"use client";

import { useState, useEffect } from "react";
import { Search, Sparkles, BarChart2, Layers, Activity, User, ShieldCheck } from "lucide-react";

const SAMPLE_PRESETS: Record<string, { label: string; sample: Record<string, number> }> = {
  high_case: {
    label: "Record #3 — High Stress Case",
    sample: {
      anxiety_level: 16.0, self_esteem: 9.0, mental_health_history: 1.0, depression: 20.0,
      headache: 4.0, blood_pressure: 3.0, sleep_quality: 1.0, breathing_problem: 4.0,
      academic_performance: 2.0, study_load: 4.0, teacher_student_relationship: 2.0, future_career_concerns: 4.0,
      social_support: 1.0, peer_pressure: 4.0, extracurricular_activities: 4.0, bullying: 4.0,
      noise_level: 4.0, living_conditions: 2.0, safety: 2.0, basic_needs: 2.0
    }
  },
  mod_case: {
    label: "Record #1 — Moderate Stress Case",
    sample: {
      anxiety_level: 11.0, self_esteem: 19.0, mental_health_history: 1.0, depression: 12.0,
      headache: 2.0, blood_pressure: 1.0, sleep_quality: 3.0, breathing_problem: 3.0,
      academic_performance: 3.0, study_load: 3.0, teacher_student_relationship: 2.0, future_career_concerns: 3.0,
      social_support: 2.0, peer_pressure: 3.0, extracurricular_activities: 3.0, bullying: 3.0,
      noise_level: 2.0, living_conditions: 3.0, safety: 2.0, basic_needs: 3.0
    }
  },
  low_case: {
    label: "Record #2 — Low Stress Case",
    sample: {
      anxiety_level: 5.0, self_esteem: 25.0, mental_health_history: 0.0, depression: 6.0,
      headache: 1.0, blood_pressure: 2.0, sleep_quality: 4.0, breathing_problem: 2.0,
      academic_performance: 4.0, study_load: 2.0, teacher_student_relationship: 4.0, future_career_concerns: 1.0,
      social_support: 3.0, peer_pressure: 2.0, extracurricular_activities: 2.0, bullying: 1.0,
      noise_level: 2.0, living_conditions: 3.0, safety: 4.0, basic_needs: 4.0
    }
  }
};

export default function ExplainabilityPage() {
  const [globalShap, setGlobalShap] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeClassTab, setActiveClassTab] = useState<"low_stress" | "moderate_stress" | "high_stress">("high_stress");
  const [selectedSampleKey, setSelectedSampleKey] = useState<string>("high_case");
  const [sampleResult, setSampleResult] = useState<any>(null);
  const [sampleTarget, setSampleTarget] = useState<string>("Predicted");
  const [sampleLoading, setSampleLoading] = useState(false);

  useEffect(() => {
    fetch("/api/shap/global")
      .then((res) => res.json())
      .then((d) => {
        if (d.success) setGlobalShap(d.data);
      })
      .finally(() => setLoading(false));
  }, []);

  const loadSamplePrediction = async (sampleKey: string, target?: string) => {
    setSampleLoading(true);
    try {
      const payload: any = { ...SAMPLE_PRESETS[sampleKey].sample };
      if (target && target !== "Predicted") {
        payload.target_category = target;
      }

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setSampleResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setSampleLoading(false);
    }
  };

  useEffect(() => {
    loadSamplePrediction(selectedSampleKey, sampleTarget);
  }, [selectedSampleKey]);

  const handleTargetSelect = (cat: string) => {
    setSampleTarget(cat);
    loadSamplePrediction(selectedSampleKey, cat);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh] text-slate-400 gap-2">
        <Activity className="w-5 h-5 animate-spin text-cyan-400" />
        <span>Loading SHAP explainability artifacts...</span>
      </div>
    );
  }

  const classData = globalShap?.[activeClassTab] || [];
  const maxClassVal = Math.max(...classData.map((d: any) => d.importance), 0.01);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-bold text-teal-400 uppercase tracking-wider mb-2">
          <Search className="w-4 h-4" />
          <span>Explainable Artificial Intelligence (XAI)</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          TreeSHAP Interpretability &amp; Feature Attribution
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Explore global feature importance rankings, class-specific drivers, and local respondent attributions
          using Lundberg &amp; Lee (2020) TreeSHAP axioms.
        </p>
      </div>

      {/* Global Feature Importance Top 15 */}
      <div className="metric-card-modern p-6 space-y-4">
        <div className="flex justify-between items-center flex-wrap gap-2">
          <div>
            <h3 className="text-base font-bold text-white">Global Feature Importance (Population Level)</h3>
            <p className="text-xs text-slate-400">Mean absolute SHAP value across all 1,100 survey records.</p>
          </div>
          <span className="badge-pill badge-cyan">TreeSHAP Global</span>
        </div>

        <div className="space-y-2.5 pt-2">
          {globalShap?.global?.slice(0, 15).map((item: any) => {
            const maxVal = globalShap.global[0]?.importance || 0.1;
            const pct = (item.importance / maxVal) * 100;
            return (
              <div key={item.feature} className="bg-navy-950/50 p-2.5 rounded-lg border border-white/5 space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-slate-200">{item.feature_name}</span>
                  <span className="font-mono text-cyan-400 font-bold">{item.importance}</span>
                </div>
                <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-cyan-500 to-teal-400 rounded-full" style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Class-Specific Importance */}
      <div className="metric-card-modern p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white">Class-Specific Feature Importance</h3>
            <p className="text-xs text-slate-400">Features driving specific stress state classifications.</p>
          </div>
          <div className="flex gap-1.5 bg-navy-950 p-1 rounded-xl border border-white/10">
            <button
              onClick={() => setActiveClassTab("low_stress")}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                activeClassTab === "low_stress" ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" : "text-slate-400 hover:text-white"
              }`}
            >
              🟢 Low Stress
            </button>
            <button
              onClick={() => setActiveClassTab("moderate_stress")}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                activeClassTab === "moderate_stress" ? "bg-amber-500/20 text-amber-300 border border-amber-500/30" : "text-slate-400 hover:text-white"
              }`}
            >
              🟡 Moderate Stress
            </button>
            <button
              onClick={() => setActiveClassTab("high_stress")}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                activeClassTab === "high_stress" ? "bg-rose-500/20 text-rose-300 border border-rose-500/30" : "text-slate-400 hover:text-white"
              }`}
            >
              🔴 High Stress
            </button>
          </div>
        </div>

        <div className="space-y-2.5 pt-2">
          {classData.slice(0, 10).map((item: any) => {
            const pct = (item.importance / maxClassVal) * 100;
            const barColor =
              activeClassTab === "low_stress"
                ? "from-emerald-500 to-teal-400"
                : activeClassTab === "moderate_stress"
                ? "from-amber-500 to-yellow-400"
                : "from-rose-500 to-red-400";
            return (
              <div key={item.feature} className="bg-navy-950/50 p-2.5 rounded-lg border border-white/5 space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-slate-200">{item.feature_name}</span>
                  <span className="font-mono text-slate-300 font-bold">{item.importance}</span>
                </div>
                <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden">
                  <div className={`h-full bg-gradient-to-r ${barColor} rounded-full`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Interactive Individual Respondent Profile Explainer */}
      <div className="metric-card-modern p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <User className="w-4 h-4 text-cyan-400" />
              <span>Interactive Individual Respondent Profile Explainer</span>
            </h3>
            <p className="text-xs text-slate-400">Inspect how TreeSHAP decomposes predictions for individual respondents.</p>
          </div>
          <select
            value={selectedSampleKey}
            onChange={(e) => setSelectedSampleKey(e.target.value)}
            className="bg-navy-950 text-xs text-slate-200 px-3 py-1.5 rounded-xl border border-white/10 focus:outline-none focus:border-cyan-500"
          >
            {Object.entries(SAMPLE_PRESETS).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
        </div>

        {sampleResult && sampleResult.prediction && (
          <div className="space-y-4">
            {/* Status card */}
            <div className="bg-navy-950/80 p-4 rounded-xl border border-white/10 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <span className="text-[10px] text-slate-400 uppercase font-bold">Model Classification</span>
                <div className="text-lg font-bold text-white mt-0.5">
                  <span className="text-cyan-400">{sampleResult.prediction.predicted_label} Stress</span>{" "}
                  ({sampleResult.prediction.confidence_pct}% Confidence)
                </div>
              </div>
              <div className="flex gap-2 text-xs font-mono">
                <span className="px-2.5 py-1 bg-slate-900 rounded border border-white/5 text-emerald-400">
                  Low: {(sampleResult.prediction.probabilities.Low * 100).toFixed(1)}%
                </span>
                <span className="px-2.5 py-1 bg-slate-900 rounded border border-white/5 text-amber-400">
                  Mod: {(sampleResult.prediction.probabilities.Moderate * 100).toFixed(1)}%
                </span>
                <span className="px-2.5 py-1 bg-slate-900 rounded border border-white/5 text-rose-400">
                  High: {(sampleResult.prediction.probabilities.High * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Target category selector for sample */}
            <div className="flex items-center justify-between flex-wrap gap-2 pt-2">
              <span className="text-xs font-bold text-slate-300">Decompose local attributions for target state:</span>
              <div className="flex gap-1.5 bg-navy-950 p-1 rounded-xl border border-white/10">
                {["Low", "Moderate", "High"].map((cat) => (
                  <button
                    key={cat}
                    onClick={() => handleTargetSelect(cat)}
                    className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                      sampleResult.explanation?.target_category === cat
                        ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {cat} Stress
                  </button>
                ))}
              </div>
            </div>

            {/* Additivity Card */}
            {sampleResult.explanation?.is_additive && (
              <div className="bg-navy-950/60 p-3.5 rounded-xl border border-white/5 text-xs font-mono flex flex-wrap items-center gap-3">
                <span>Prior Base E[f(x)]: <strong>{sampleResult.explanation.base_value_pct}%</strong></span>
                <span className="text-slate-500">+</span>
                <span>
                  Sum SHAP:{" "}
                  <strong className={sampleResult.explanation.shap_sum >= 0 ? "text-emerald-400" : "text-rose-400"}>
                    {sampleResult.explanation.shap_sum_pct >= 0 ? `+${sampleResult.explanation.shap_sum_pct}` : sampleResult.explanation.shap_sum_pct}%
                  </strong>
                </span>
                <span className="text-slate-500">=</span>
                <span>Probability: <strong className="text-cyan-400">{sampleResult.explanation.target_probability_pct}%</strong></span>
                <span className="badge-pill badge-emerald text-[9px] ml-auto">✓ Additivity Verified</span>
              </div>
            )}

            {/* Diverging Local Bar Chart */}
            <div className="space-y-2 pt-2">
              {sampleResult.explanation?.top_features?.map((f: any) => {
                const isPositive = f.shap_value >= 0;
                return (
                  <div key={f.feature} className="bg-navy-950/40 p-2.5 rounded-lg border border-white/5 flex items-center justify-between text-xs">
                    <div>
                      <span className="font-semibold text-slate-200">{f.feature_name}</span>
                      <span className="text-slate-500 ml-2 font-mono">({f.entered_value})</span>
                    </div>
                    <div className="font-mono font-bold">
                      <span className={isPositive ? "text-cyan-400" : "text-rose-400"}>
                        {isPositive ? `+${f.impact_pct}` : f.impact_pct}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Algorithmic Transparency Notice */}
      <div className="p-4 rounded-xl bg-navy-950/60 border border-white/10 text-xs text-slate-400 leading-relaxed">
        <strong>⚠️ Algorithmic Transparency Notice:</strong> SHAP explains how the machine-learning model mathematically combined survey responses.
        It reflects statistical weights in decision trees; it does not establish medical causation or clinical diagnoses.
      </div>
    </div>
  );
}
