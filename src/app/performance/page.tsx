"use client";

import { useState, useEffect } from "react";
import { BarChart3, TrendingUp, Award, Layers, Activity } from "lucide-react";

export default function PerformancePage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>("Random Forest");

  useEffect(() => {
    fetch("/api/performance")
      .then((res) => res.json())
      .then((d) => {
        if (d.success) setData(d.data);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh] text-slate-400 gap-2">
        <Activity className="w-5 h-5 animate-spin text-cyan-400" />
        <span>Loading verified benchmark data...</span>
      </div>
    );
  }

  const cm = data?.confusion_matrices?.[selectedModel] || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const aucScore = data?.roc_auc_scores?.[selectedModel] || 0.0;
  const cv = data?.cv_results?.[selectedModel];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2">
          <BarChart3 className="w-4 h-4" />
          <span>Empirical Evaluation Artifacts</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Supervised Model Performance &amp; Benchmark Analysis
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Comparative benchmarking of 5 supervised machine learning classifiers trained and evaluated under identical 80/20 stratified splits.
        </p>
      </div>

      {/* Champion Optimization Spotlight */}
      <div className="metric-card-modern p-6 bg-gradient-to-r from-navy-900 via-navy-850 to-navy-900 border border-white/10">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
              <Award className="w-6 h-6" />
            </div>
            <div>
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">Champion Optimization Result</span>
              <h3 className="text-xl font-bold text-white">Baseline vs. Optimized Random Forest</h3>
            </div>
          </div>
          <span className="badge-pill badge-cyan text-xs">
            +2.27% Accuracy Gain
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
          <div className="bg-navy-950/70 p-4 rounded-xl border border-white/5">
            <span className="text-[11px] text-slate-400 uppercase font-bold">Baseline Accuracy</span>
            <div className="text-2xl font-extrabold text-white mt-1">88.64%</div>
            <span className="text-[10px] text-slate-500">200 Estimators (default)</span>
          </div>
          <div className="bg-navy-950/70 p-4 rounded-xl border border-cyan-500/30 bg-cyan-500/5">
            <span className="text-[11px] text-cyan-400 uppercase font-bold">Optimized Accuracy</span>
            <div className="text-2xl font-extrabold text-cyan-300 mt-1">90.91%</div>
            <span className="text-[10px] text-emerald-400 font-semibold">+2.27% test improvement</span>
          </div>
          <div className="bg-navy-950/70 p-4 rounded-xl border border-white/5">
            <span className="text-[11px] text-slate-400 uppercase font-bold">Baseline ROC-AUC</span>
            <div className="text-2xl font-extrabold text-violet-400 mt-1">0.9831</div>
            <span className="text-[10px] text-slate-500">One-vs-Rest Macro</span>
          </div>
          <div className="bg-navy-950/70 p-4 rounded-xl border border-white/5">
            <span className="text-[11px] text-slate-400 uppercase font-bold">Optimized ROC-AUC</span>
            <div className="text-2xl font-extrabold text-emerald-400 mt-1">0.9853</div>
            <span className="text-[10px] text-slate-500">+0.0022 discriminability</span>
          </div>
        </div>
      </div>

      {/* 5-Model Benchmark Table */}
      <div className="metric-card-modern p-6 space-y-4">
        <h3 className="text-base font-bold text-white">Supervised Multi-Algorithm Benchmark Table</h3>
        <p className="text-xs text-slate-400">
          Evaluated on independent test partition of 220 samples with weighted precision, recall, and F1 metrics.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-navy-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-white/10">
              <tr>
                <th className="p-3.5">Algorithm</th>
                <th className="p-3.5">Accuracy</th>
                <th className="p-3.5">Precision</th>
                <th className="p-3.5">Recall</th>
                <th className="p-3.5">F1-Score</th>
                <th className="p-3.5">ROC-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 font-mono">
              {data?.baseline_comparison?.map((m: any) => (
                <tr key={m.model} className="hover:bg-white/5">
                  <td className="p-3.5 font-sans font-semibold text-white flex items-center gap-2">
                    <span>{m.model}</span>
                    {m.model === "Random Forest" && (
                      <span className="badge-pill badge-cyan text-[9px]">Champion</span>
                    )}
                  </td>
                  <td className="p-3.5 font-bold text-cyan-400">{m.accuracy}%</td>
                  <td className="p-3.5">{m.precision}%</td>
                  <td className="p-3.5">{m.recall}%</td>
                  <td className="p-3.5 text-emerald-400">{m.f1_score}%</td>
                  <td className="p-3.5 text-violet-400">
                    {data?.roc_auc_scores?.[m.model] ?? "0.98xx"}
                  </td>
                </tr>
              ))}
              {/* Highlight Optimized RF */}
              <tr className="bg-cyan-500/5 font-bold">
                <td className="p-3.5 font-sans font-bold text-cyan-300 flex items-center gap-2">
                  <span>Optimized Random Forest</span>
                  <span className="badge-pill badge-emerald text-[9px]">Tuned</span>
                </td>
                <td className="p-3.5 font-bold text-emerald-400">90.91%</td>
                <td className="p-3.5">90.95%</td>
                <td className="p-3.5">90.91%</td>
                <td className="p-3.5 text-emerald-400">90.92%</td>
                <td className="p-3.5 text-violet-400">0.9853</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Selector for Confusion Matrix & Details */}
      <div className="metric-card-modern p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white">Interactive Model Matrix Inspector</h3>
            <p className="text-xs text-slate-400">Select an algorithm to view its test confusion matrix and cross-validation stability.</p>
          </div>
          <div className="flex flex-wrap gap-1.5 bg-navy-950 p-1 rounded-xl border border-white/10">
            {Object.keys(data?.confusion_matrices || {}).map((name) => (
              <button
                key={name}
                onClick={() => setSelectedModel(name)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                  selectedModel === name
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Confusion Matrix Heatmap Representation */}
          <div className="bg-navy-950/60 p-5 rounded-xl border border-white/5 space-y-3">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-white">Test Confusion Matrix (220 Samples)</span>
              <span className="text-[11px] text-slate-400 font-mono">ROC-AUC: {aucScore}</span>
            </div>

            <div className="grid grid-cols-4 gap-2 text-center text-xs font-mono mt-3">
              <div className="p-2 text-slate-500 font-sans text-[10px]">Actual \ Pred</div>
              <div className="p-2 text-emerald-400 font-bold bg-navy-900 rounded">Pred Low</div>
              <div className="p-2 text-amber-400 font-bold bg-navy-900 rounded">Pred Mod</div>
              <div className="p-2 text-rose-400 font-bold bg-navy-900 rounded">Pred High</div>

              <div className="p-2 text-emerald-400 font-bold bg-navy-900 rounded flex items-center justify-center">True Low</div>
              <div className="p-3 bg-emerald-500/20 text-emerald-300 font-extrabold text-sm rounded border border-emerald-500/30">
                {cm[0]?.[0]}
              </div>
              <div className="p-3 bg-slate-800 text-slate-300 rounded">
                {cm[0]?.[1]}
              </div>
              <div className="p-3 bg-slate-800 text-slate-300 rounded">
                {cm[0]?.[2]}
              </div>

              <div className="p-2 text-amber-400 font-bold bg-navy-900 rounded flex items-center justify-center">True Mod</div>
              <div className="p-3 bg-slate-800 text-slate-300 rounded">
                {cm[1]?.[0]}
              </div>
              <div className="p-3 bg-amber-500/20 text-amber-300 font-extrabold text-sm rounded border border-amber-500/30">
                {cm[1]?.[1]}
              </div>
              <div className="p-3 bg-slate-800 text-slate-300 rounded">
                {cm[1]?.[2]}
              </div>

              <div className="p-2 text-rose-400 font-bold bg-navy-900 rounded flex items-center justify-center">True High</div>
              <div className="p-3 bg-slate-800 text-slate-300 rounded">
                {cm[2]?.[0]}
              </div>
              <div className="p-3 bg-slate-800 text-slate-300 rounded">
                {cm[2]?.[1]}
              </div>
              <div className="p-3 bg-rose-500/20 text-rose-300 font-extrabold text-sm rounded border border-rose-500/30">
                {cm[2]?.[2]}
              </div>
            </div>
          </div>

          {/* 5-Fold Cross Validation Summary */}
          <div className="bg-navy-950/60 p-5 rounded-xl border border-white/5 space-y-4">
            <span className="text-xs font-bold text-white block">5-Fold Stratified Cross-Validation (Training Partition)</span>
            
            {cv ? (
              <div className="space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400">Mean CV Accuracy:</span>
                  <span className="font-mono font-bold text-cyan-400 text-base">{cv.mean_accuracy}%</span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400">Standard Deviation (&plusmn;&sigma;):</span>
                  <span className="font-mono text-slate-300">&plusmn;{cv.std_accuracy}%</span>
                </div>
                <div className="pt-2">
                  <span className="text-[11px] text-slate-400 block mb-2 font-semibold">Individual Fold Scores:</span>
                  <div className="grid grid-cols-5 gap-1.5 text-center font-mono text-xs">
                    {cv.fold_scores?.map((score: number, idx: number) => (
                      <div key={idx} className="p-2 bg-navy-900 rounded border border-white/5">
                        <span className="text-[9px] text-slate-500 block">Fold {idx + 1}</span>
                        <span className="font-bold text-slate-200">{score}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-400">No cross-validation data for this model.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
