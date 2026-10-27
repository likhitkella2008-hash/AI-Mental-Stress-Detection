"use client";

import { useState, useEffect } from "react";
import { 
  GitFork, 
  ArrowRight, 
  TrendingDown, 
  TrendingUp, 
  Minus, 
  RotateCcw, 
  Layers, 
  Activity, 
  AlertCircle,
  Sparkles,
  Info
} from "lucide-react";

const INITIAL_BASELINE = {
  anxiety_level: 12.0, self_esteem: 18.0, mental_health_history: 1.0, depression: 14.0,
  headache: 3.0, blood_pressure: 2.0, sleep_quality: 2.0, breathing_problem: 2.0,
  academic_performance: 3.0, study_load: 4.0, teacher_student_relationship: 2.0, future_career_concerns: 3.0,
  social_support: 1.0, peer_pressure: 3.0, extracurricular_activities: 2.0, bullying: 2.0,
  noise_level: 3.0, living_conditions: 2.0, safety: 3.0, basic_needs: 3.0
};

const FEATURE_METADATA: Record<string, { label: string; min: number; max: number; step: number; domain: string }> = {
  anxiety_level: { label: "Anxiety Level", min: 0, max: 21, step: 1, domain: "Mental & Emotional" },
  self_esteem: { label: "Self-Esteem", min: 0, max: 30, step: 1, domain: "Mental & Emotional" },
  mental_health_history: { label: "Mental Health History", min: 0, max: 1, step: 1, domain: "Mental & Emotional" },
  depression: { label: "Depression Score", min: 0, max: 27, step: 1, domain: "Mental & Emotional" },
  headache: { label: "Headache Frequency", min: 0, max: 5, step: 1, domain: "Physical Health" },
  blood_pressure: { label: "Blood Pressure Rating", min: 1, max: 3, step: 1, domain: "Physical Health" },
  sleep_quality: { label: "Sleep Quality", min: 0, max: 5, step: 1, domain: "Physical Health" },
  breathing_problem: { label: "Breathing Difficulty", min: 0, max: 5, step: 1, domain: "Physical Health" },
  academic_performance: { label: "Academic Standing", min: 0, max: 5, step: 1, domain: "Academic Factors" },
  study_load: { label: "Course Study Load", min: 0, max: 5, step: 1, domain: "Academic Factors" },
  teacher_student_relationship: { label: "Teacher Rapport", min: 0, max: 5, step: 1, domain: "Academic Factors" },
  future_career_concerns: { label: "Career Anxiety", min: 0, max: 5, step: 1, domain: "Academic Factors" },
  social_support: { label: "Social Support Network", min: 0, max: 3, step: 1, domain: "Social & Environmental" },
  peer_pressure: { label: "Peer Pressure", min: 0, max: 5, step: 1, domain: "Social & Environmental" },
  extracurricular_activities: { label: "Extracurriculars", min: 0, max: 5, step: 1, domain: "Social & Environmental" },
  bullying: { label: "Bullying / Friction", min: 0, max: 5, step: 1, domain: "Social & Environmental" },
  noise_level: { label: "Ambient Noise", min: 0, max: 5, step: 1, domain: "Social & Environmental" },
  living_conditions: { label: "Living Conditions", min: 0, max: 5, step: 1, domain: "Social & Environmental" },
  safety: { label: "Personal Safety", min: 0, max: 5, step: 1, domain: "Social & Environmental" },
  basic_needs: { label: "Basic Needs Security", min: 0, max: 5, step: 1, domain: "Social & Environmental" },
};

export default function WhatIfPage() {
  const [baseline, setBaseline] = useState<Record<string, number>>(INITIAL_BASELINE);
  const [hypothetical, setHypothetical] = useState<Record<string, number>>(INITIAL_BASELINE);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const runWhatIf = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/whatif", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ baseline, hypothetical }),
      });
      if (!res.ok) throw new Error("What-If computation failed");
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runWhatIf();
  }, [baseline, hypothetical]);

  const handleHypoChange = (feat: string, val: number) => {
    setHypothetical((prev) => ({ ...prev, [feat]: val }));
  };

  const applyPreset = (type: "lifestyle" | "academic" | "reset") => {
    if (type === "reset") {
      setHypothetical({ ...baseline });
    } else if (type === "lifestyle") {
      setHypothetical((prev) => ({
        ...prev,
        sleep_quality: 5.0,
        social_support: 3.0,
        anxiety_level: Math.max(0, prev.anxiety_level - 5),
        study_load: Math.max(0, prev.study_load - 1),
      }));
    } else if (type === "academic") {
      setHypothetical((prev) => ({
        ...prev,
        study_load: 5.0,
        peer_pressure: 5.0,
        sleep_quality: 1.0,
        future_career_concerns: 5.0,
      }));
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-bold text-violet-400 uppercase tracking-wider mb-2">
          <GitFork className="w-4 h-4" />
          <span>Counterfactual Sensitivity Analysis</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          What-If Stress Sensitivity &amp; Decision Boundary Explorer
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Perturb individual lifestyle, academic, or physiological factors while holding other conditions constant (<em>ceteris paribus</em>)
          to observe non-linear model probability shifts and attribution movements.
        </p>
      </div>

      {/* Preset Controls */}
      <div className="metric-card-modern p-4 flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
          Quick Counterfactual Adjustments:
        </span>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => applyPreset("lifestyle")}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 transition-colors"
          >
            + Enhance Rest &amp; Support
          </button>
          <button
            onClick={() => applyPreset("academic")}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 transition-colors"
          >
            + Heighten Academic Load
          </button>
          <button
            onClick={() => applyPreset("reset")}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-white/10 flex items-center gap-1.5 transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Reset to Baseline</span>
          </button>
        </div>
      </div>

      {/* Results Section */}
      {result && result.success && (
        <div className="space-y-6">
          {/* Executive Verdict Banner */}
          <div className="metric-card-modern p-6 bg-gradient-to-r from-navy-900 via-navy-850 to-navy-900 border border-white/10">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                Counterfactual Classification Verdict
              </span>
              <span
                className={`badge-pill ${
                  result.hypothetical_prediction.predicted_class < result.baseline_prediction.predicted_class
                    ? "badge-emerald"
                    : result.hypothetical_prediction.predicted_class > result.baseline_prediction.predicted_class
                    ? "badge-rose"
                    : "badge-violet"
                }`}
              >
                {result.hypothetical_prediction.predicted_class < result.baseline_prediction.predicted_class
                  ? "⬇️ Stress Reduced"
                  : result.hypothetical_prediction.predicted_class > result.baseline_prediction.predicted_class
                  ? "⬆️ Stress Escalated"
                  : "➡️ Stress Maintained"}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 items-center gap-4 text-center">
              <div className="bg-navy-950/70 p-4 rounded-xl border border-white/5">
                <span className="text-[11px] text-slate-400 uppercase font-bold">Baseline Profile</span>
                <div className="text-xl font-extrabold text-white mt-1">
                  {result.baseline_prediction.predicted_label} Stress
                </div>
                <span className="text-xs text-cyan-400 font-semibold">{result.baseline_prediction.confidence_pct}% Confidence</span>
              </div>

              <div className="flex justify-center text-slate-500 font-extrabold text-xl">➔</div>

              <div className="bg-navy-950/70 p-4 rounded-xl border border-cyan-500/30 bg-cyan-500/5">
                <span className="text-[11px] text-cyan-400 uppercase font-bold">Hypothetical Profile</span>
                <div className="text-xl font-extrabold text-cyan-300 mt-1">
                  {result.hypothetical_prediction.predicted_label} Stress
                </div>
                <span className="text-xs text-emerald-400 font-semibold">{result.hypothetical_prediction.confidence_pct}% Confidence</span>
              </div>
            </div>
          </div>

          {/* Three-class probability shift cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {["Low", "Moderate", "High"].map((cat) => {
              const shift = result.probability_shifts[cat];
              const isUp = shift.delta > 0;
              const isZero = shift.delta === 0;

              return (
                <div key={cat} className="metric-card-modern p-4 space-y-2">
                  <div className="flex justify-between items-center text-xs font-bold text-slate-400 uppercase">
                    <span>{cat} Stress Probability</span>
                    <span className={`font-mono ${isUp ? (cat === "Low" ? "text-emerald-400" : "text-rose-400") : "text-slate-400"}`}>
                      {isZero ? "0.0%" : isUp ? `+${shift.delta_pct}%` : `${shift.delta_pct}%`}
                    </span>
                  </div>
                  <div className="text-2xl font-extrabold text-white">
                    {(shift.hypo_prob * 100).toFixed(1)}%
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Baseline: {(shift.baseline_prob * 100).toFixed(1)}%
                  </div>
                </div>
              );
            })}
          </div>

          {/* Modified Factors Delta Inspector */}
          <div className="metric-card-modern p-6 space-y-4">
            <h4 className="text-base font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Counterfactual Feature Delta Inspector</span>
            </h4>

            {result.changed_features.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-navy-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-white/10">
                    <tr>
                      <th className="p-3">Feature</th>
                      <th className="p-3">Domain</th>
                      <th className="p-3">Baseline</th>
                      <th className="p-3">Hypothetical</th>
                      <th className="p-3">Shift</th>
                      <th className="p-3">Direction</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {result.changed_features.map((f: any) => (
                      <tr key={f.feature} className="hover:bg-white/5">
                        <td className="p-3 font-semibold text-white">{f.name}</td>
                        <td className="p-3 text-slate-400">{f.domain}</td>
                        <td className="p-3 font-mono">{f.baseline_value}</td>
                        <td className="p-3 font-mono text-cyan-400 font-bold">{f.hypothetical_value}</td>
                        <td className="p-3 font-mono font-bold text-emerald-400">{f.shift > 0 ? `+${f.shift}` : f.shift}</td>
                        <td className="p-3">
                          <span className={`badge-pill ${f.shift > 0 ? "badge-emerald" : "badge-amber"}`}>
                            {f.direction}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-navy-950/50 border border-white/5 text-xs text-slate-400">
                Baseline and hypothetical inputs are currently identical. Adjust any slider below to inspect counterfactual probability deltas.
              </div>
            )}
          </div>

          {/* Section 5: Counterfactual Decision Explanation & Attribution Shifts */}
          <div className="metric-card-modern p-6 space-y-4">
            <h4 className="text-base font-bold text-white">
              5. Counterfactual Decision Explanation &amp; Attribution Shifts
            </h4>
            <div className="p-4 rounded-xl bg-navy-950/80 border border-white/10 text-xs text-slate-300 leading-relaxed">
              <p>{result.summary}</p>
            </div>

            {result.shap_shifts && result.shap_shifts.top_shifts && (
              <div className="space-y-3 pt-2">
                <span className="text-xs text-slate-400 font-bold">Top Feature Attribution Shifts (&Delta;SHAP):</span>
                <div className="space-y-2">
                  {result.shap_shifts.top_shifts.map((s: any) => (
                    <div key={s.feature} className="bg-navy-950/60 p-3 rounded-lg border border-white/5 flex items-center justify-between text-xs">
                      <div>
                        <span className="font-semibold text-slate-200">{s.feature_name}</span>
                        <span className="text-[11px] text-slate-500 ml-2">({s.baseline_value} ➔ {s.hypothetical_value})</span>
                      </div>
                      <div className="font-mono font-bold">
                        <span className={s.shap_delta >= 0 ? "text-emerald-400" : "text-rose-400"}>
                          &Delta;SHAP: {s.shap_delta >= 0 ? `+${s.delta_pct}` : s.delta_pct}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="p-3 rounded-lg bg-navy-950/60 border border-white/5 text-[11px] text-slate-400">
              <strong>⚠️ Counterfactual Sensitivity Notice:</strong> What-If analysis illustrates mathematical sensitivities in the decision forest under <em>ceteris paribus</em> assumptions. It does not establish clinical interventions or causal real-world guarantees.
            </div>
          </div>
        </div>
      )}

      {/* Interactive Adjustment Sliders */}
      <div className="metric-card-modern p-6 space-y-6">
        <h3 className="text-base font-bold text-white">Hypothetical Factor Adjustments</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(FEATURE_METADATA).map(([key, meta]) => (
            <div key={key} className="bg-navy-950/60 p-4 rounded-xl border border-white/5 space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-slate-200">{meta.label}</span>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-500">Base: {baseline[key]}</span>
                  <span className="text-xs font-mono font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                    Hypo: {hypothetical[key]}
                  </span>
                </div>
              </div>
              <input
                type="range"
                min={meta.min}
                max={meta.max}
                step={meta.step}
                value={hypothetical[key]}
                onChange={(e) => handleHypoChange(key, parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>{meta.domain}</span>
                <span>Range: [{meta.min} - {meta.max}]</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
