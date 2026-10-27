"use client";

import { useState, useEffect } from "react";
import { Database, Layers, CheckCircle2, Target, Activity } from "lucide-react";

export default function DatasetPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [selectedFeature, setSelectedFeature] = useState<string>("anxiety_level");

  useEffect(() => {
    fetch("/api/dataset")
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
        <span>Loading dataset distribution metrics...</span>
      </div>
    );
  }

  const stat = data?.feature_statistics?.[selectedFeature];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">
          <Database className="w-4 h-4" />
          <span>Exploratory Data Architecture</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Dataset Distribution &amp; Feature Characteristics
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Explore summary metrics, class distributions, and parametric statistics across all 1,100 clean respondent entries.
        </p>
      </div>

      {/* 4 KPI Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="metric-card-modern p-5 border-t-2 border-t-cyan-500">
          <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">Total Records</span>
          <div className="text-3xl font-extrabold text-white mt-2">1,100</div>
          <span className="text-[11px] text-slate-500">Clean survey entries</span>
        </div>
        <div className="metric-card-modern p-5 border-t-2 border-t-violet-500">
          <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">Input Features</span>
          <div className="text-3xl font-extrabold text-white mt-2">20</div>
          <span className="text-[11px] text-slate-500">Predictive dimensions</span>
        </div>
        <div className="metric-card-modern p-5 border-t-2 border-t-emerald-500">
          <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">Missing Values</span>
          <div className="text-3xl font-extrabold text-white mt-2">0</div>
          <span className="text-[11px] text-slate-500">100% complete data</span>
        </div>
        <div className="metric-card-modern p-5 border-t-2 border-t-teal-500">
          <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">Target Classes</span>
          <div className="text-3xl font-extrabold text-white mt-2">3 States</div>
          <span className="text-[11px] text-slate-500">Low, Moderate, High</span>
        </div>
      </div>

      {/* Interactive Feature Distribution Inspector */}
      <div className="metric-card-modern p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white">Interactive Feature Distribution Inspector</h3>
            <p className="text-xs text-slate-400">Select any feature to inspect its parametric bounds and quartiles.</p>
          </div>
          <select
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
            className="bg-navy-950 text-xs text-slate-200 px-3 py-1.5 rounded-xl border border-white/10 focus:outline-none focus:border-cyan-500"
          >
            {Object.keys(data?.feature_statistics || {}).map((f) => (
              <option key={f} value={f}>
                {data.feature_statistics[f].name}
              </option>
            ))}
          </select>
        </div>

        {stat && (
          <div className="bg-navy-950/60 p-5 rounded-xl border border-white/5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/10 pb-3">
              <div>
                <span className="text-xs text-cyan-400 font-bold uppercase tracking-wider">{stat.name}</span>
                <div className="text-lg font-bold text-white">
                  Mean: {stat.mean} <span className="text-xs text-slate-400 font-normal font-mono">(&plusmn;{stat.std})</span>
                </div>
              </div>
              <div className="text-xs text-slate-400 font-mono">
                Min: {stat.min} • Median: {stat.median} • Max: {stat.max}
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center text-xs font-mono">
              <div className="p-3 bg-navy-900 rounded border border-white/5">
                <span className="text-[10px] text-slate-500 block uppercase">Min (0%)</span>
                <span className="font-bold text-slate-200 text-sm">{stat.min}</span>
              </div>
              <div className="p-3 bg-navy-900 rounded border border-white/5">
                <span className="text-[10px] text-slate-500 block uppercase">25th %ile</span>
                <span className="font-bold text-cyan-400 text-sm">{stat.p25}</span>
              </div>
              <div className="p-3 bg-navy-900 rounded border border-cyan-500/30 bg-cyan-500/5">
                <span className="text-[10px] text-cyan-400 block uppercase font-bold">Median (50%)</span>
                <span className="font-bold text-cyan-300 text-sm">{stat.median}</span>
              </div>
              <div className="p-3 bg-navy-900 rounded border border-white/5">
                <span className="text-[10px] text-slate-500 block uppercase">75th %ile</span>
                <span className="font-bold text-cyan-400 text-sm">{stat.p75}</span>
              </div>
              <div className="p-3 bg-navy-900 rounded border border-white/5">
                <span className="text-[10px] text-slate-500 block uppercase">Max (100%)</span>
                <span className="font-bold text-slate-200 text-sm">{stat.max}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Complete Descriptive Statistics Table */}
      <div className="metric-card-modern p-6 space-y-4">
        <h3 className="text-base font-bold text-white">Complete Feature Parametric Statistics</h3>
        <p className="text-xs text-slate-400">
          Distribution summary across all 20 predictive features in the dataset.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-navy-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-white/10">
              <tr>
                <th className="p-3">Feature Name</th>
                <th className="p-3">Mean</th>
                <th className="p-3">Std Dev</th>
                <th className="p-3">Min</th>
                <th className="p-3">25%</th>
                <th className="p-3">Median</th>
                <th className="p-3">75%</th>
                <th className="p-3">Max</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 font-mono text-[11px]">
              {Object.entries(data?.feature_statistics || {}).map(([key, s]: [string, any]) => (
                <tr key={key} className="hover:bg-white/5">
                  <td className="p-3 font-sans font-semibold text-white">{s.name}</td>
                  <td className="p-3 font-bold text-cyan-400">{s.mean}</td>
                  <td className="p-3 text-slate-400">&plusmn;{s.std}</td>
                  <td className="p-3 text-slate-400">{s.min}</td>
                  <td className="p-3">{s.p25}</td>
                  <td className="p-3 text-cyan-300 font-bold">{s.median}</td>
                  <td className="p-3">{s.p75}</td>
                  <td className="p-3 text-slate-400">{s.max}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
