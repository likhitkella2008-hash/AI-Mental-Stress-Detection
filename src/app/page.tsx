import Link from "next/link";
import { 
  Database, 
  Layers, 
  Target, 
  CheckCircle2, 
  ArrowRight, 
  Zap, 
  Brain, 
  GitFork, 
  BarChart3, 
  Search,
  Sparkles,
  TrendingUp,
  Cpu
} from "lucide-react";

export default function DashboardPage() {
  return (
    <div className="space-y-10">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-navy-900 via-navy-850 to-navy-900 border border-white/10 p-8 shadow-2xl">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-80 h-80 rounded-full bg-cyan-500/10 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 -mb-16 w-80 h-80 rounded-full bg-violet-500/10 blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-4">
            <Sparkles className="w-3.5 h-3.5" />
            B.Tech End-Semester Engineering Project
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
            AI-Based Mental Stress Detection & Explainable Analytics
          </h1>
          <p className="mt-3 text-slate-300 text-sm sm:text-base leading-relaxed">
            A supervised machine learning architecture regularized via 5-fold cross-validation, 
            combining tree-ensemble bagging with exact TreeSHAP local attributions and counterfactual sensitivity analysis.
          </p>

          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              href="/predict"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 text-navy-950 font-bold text-xs uppercase tracking-wider hover:opacity-95 shadow-lg shadow-cyan-500/25 transition-all"
            >
              <Brain className="w-4 h-4" />
              <span>Launch Stress Prediction</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
            <Link
              href="/what-if"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-navy-800 hover:bg-navy-700 text-cyan-300 border border-cyan-500/30 font-semibold text-xs uppercase tracking-wider transition-all"
            >
              <GitFork className="w-4 h-4" />
              <span>Explore What-If Sensitivity</span>
            </Link>
          </div>
        </div>
      </div>

      {/* 4 KPI Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="metric-card-modern p-5 border-t-2 border-t-cyan-500">
          <div className="flex items-center justify-between text-slate-400 text-xs uppercase font-bold tracking-wider">
            <span>Survey Records</span>
            <Database className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="mt-3 text-3xl font-extrabold text-white tracking-tight">1,100</div>
          <div className="mt-1 text-xs text-slate-400">Validated respondent responses</div>
        </div>

        <div className="metric-card-modern p-5 border-t-2 border-t-violet-500">
          <div className="flex items-center justify-between text-slate-400 text-xs uppercase font-bold tracking-wider">
            <span>Input Factors</span>
            <Layers className="w-4 h-4 text-violet-400" />
          </div>
          <div className="mt-3 text-3xl font-extrabold text-white tracking-tight">20</div>
          <div className="mt-1 text-xs text-slate-400">Across 4 holistic life domains</div>
        </div>

        <div className="metric-card-modern p-5 border-t-2 border-t-emerald-500">
          <div className="flex items-center justify-between text-slate-400 text-xs uppercase font-bold tracking-wider">
            <span>Stress States</span>
            <Target className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-3 text-3xl font-extrabold text-white tracking-tight">3 Levels</div>
          <div className="mt-1 text-xs text-slate-400">Low (0), Moderate (1), High (2)</div>
        </div>

        <div className="metric-card-modern p-5 border-t-2 border-t-teal-500">
          <div className="flex items-center justify-between text-slate-400 text-xs uppercase font-bold tracking-wider">
            <span>Data Integrity</span>
            <CheckCircle2 className="w-4 h-4 text-teal-400" />
          </div>
          <div className="mt-3 text-3xl font-extrabold text-white tracking-tight">100%</div>
          <div className="mt-1 text-xs text-slate-400">0 missing or imputed values</div>
        </div>
      </div>

      {/* Champion Classifier Spotlight & Target Distribution Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Champion Model Card */}
        <div className="lg:col-span-2 metric-card-modern p-6 bg-gradient-to-br from-navy-850 to-navy-900 relative">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">Champion Classifier: Optimized Random Forest</h3>
                <p className="text-xs text-slate-400">Ensemble bagging regularized with 5-fold GridSearchCV</p>
              </div>
            </div>
            <span className="badge-pill badge-cyan">
              +2.27% Accuracy Boost
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
            <div className="bg-navy-950/70 p-3.5 rounded-xl border border-white/5">
              <span className="text-[11px] text-slate-400 uppercase font-bold">Baseline Test</span>
              <div className="text-xl font-extrabold text-slate-300 mt-0.5">88.64%</div>
              <span className="text-[10px] text-slate-500">200 Estimators</span>
            </div>
            <div className="bg-navy-950/70 p-3.5 rounded-xl border border-cyan-500/30 bg-cyan-500/5">
              <span className="text-[11px] text-cyan-400 uppercase font-bold">Optimized Test</span>
              <div className="text-xl font-extrabold text-cyan-300 mt-0.5">90.91%</div>
              <span className="text-[10px] text-emerald-400 font-semibold">+2.27% over baseline</span>
            </div>
            <div className="bg-navy-950/70 p-3.5 rounded-xl border border-white/5">
              <span className="text-[11px] text-slate-400 uppercase font-bold">Multiclass ROC-AUC</span>
              <div className="text-xl font-extrabold text-violet-400 mt-0.5">0.9853</div>
              <span className="text-[10px] text-slate-500">One-vs-Rest Macro</span>
            </div>
            <div className="bg-navy-950/70 p-3.5 rounded-xl border border-white/5">
              <span className="text-[11px] text-slate-400 uppercase font-bold">5-Fold CV</span>
              <div className="text-xl font-extrabold text-slate-200 mt-0.5">88.18%</div>
              <span className="text-[10px] text-slate-500">±0.81% on train</span>
            </div>
          </div>

          {/* Hyperparameter chips */}
          <div className="mt-5 pt-4 border-t border-white/10 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400 font-medium">Tuned Hyperparameters:</span>
            <span className="px-2.5 py-1 text-[11px] font-mono bg-slate-800 text-slate-300 rounded-md border border-white/10">n_estimators=200</span>
            <span className="px-2.5 py-1 text-[11px] font-mono bg-slate-800 text-slate-300 rounded-md border border-white/10">max_depth=10</span>
            <span className="px-2.5 py-1 text-[11px] font-mono bg-slate-800 text-slate-300 rounded-md border border-white/10">min_samples_split=2</span>
            <span className="px-2.5 py-1 text-[11px] font-mono bg-slate-800 text-slate-300 rounded-md border border-white/10">max_features=&apos;sqrt&apos;</span>
          </div>
        </div>

        {/* Dataset Distribution */}
        <div className="metric-card-modern p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">Class Balance</h3>
              <span className="badge-pill badge-emerald">Stratified 1:1:1</span>
            </div>
            <div className="space-y-3.5">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-emerald-400 font-semibold">🟢 Low Stress (Class 0)</span>
                  <span className="text-slate-300 font-mono">373 (33.9%)</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div className="bg-emerald-500 h-2 rounded-full" style={{ width: "33.9%" }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-amber-400 font-semibold">🟡 Moderate Stress (Class 1)</span>
                  <span className="text-slate-300 font-mono">358 (32.5%)</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div className="bg-amber-500 h-2 rounded-full" style={{ width: "32.5%" }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-rose-400 font-semibold">🔴 High Stress (Class 2)</span>
                  <span className="text-slate-300 font-mono">369 (33.6%)</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div className="bg-rose-500 h-2 rounded-full" style={{ width: "33.6%" }} />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-3 border-t border-white/10 text-xs text-slate-400">
            Balanced ground truth prevents majority-class bias across all algorithms.
          </div>
        </div>
      </div>

      {/* 4 Feature Exploration Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Link href="/predict" className="metric-card-modern p-5 group hover:border-cyan-500/50">
          <div className="w-10 h-10 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
            <Brain className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">Stress Prediction</h4>
          <p className="text-xs text-slate-400 mt-1">Multi-domain survey inputs with probability distributions &amp; TreeSHAP decomposition.</p>
        </Link>

        <Link href="/what-if" className="metric-card-modern p-5 group hover:border-violet-500/50">
          <div className="w-10 h-10 rounded-xl bg-violet-500/10 text-violet-400 border border-violet-500/30 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
            <GitFork className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-white group-hover:text-violet-300 transition-colors">What-If Analysis</h4>
          <p className="text-xs text-slate-400 mt-1">Counterfactual sensitivity testing with simultaneous inference &amp; &Delta;SHAP shifts.</p>
        </Link>

        <Link href="/performance" className="metric-card-modern p-5 group hover:border-emerald-500/50">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
            <BarChart3 className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-white group-hover:text-emerald-300 transition-colors">Model Benchmarks</h4>
          <p className="text-xs text-slate-400 mt-1">5-algorithm benchmark comparison, confusion matrix heatmap, and ROC curves.</p>
        </Link>

        <Link href="/explainability" className="metric-card-modern p-5 group hover:border-teal-500/50">
          <div className="w-10 h-10 rounded-xl bg-teal-500/10 text-teal-400 border border-teal-500/30 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
            <Search className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-white group-hover:text-teal-300 transition-colors">Explainable AI</h4>
          <p className="text-xs text-slate-400 mt-1">Global feature importance, class-specific drivers, and local respondent explainer.</p>
        </Link>
      </div>

      {/* 8-Stage Engineering Pipeline */}
      <div className="metric-card-modern p-6">
        <h3 className="text-base font-bold text-white mb-2">End-to-End Machine Learning System Pipeline</h3>
        <p className="text-xs text-slate-400 mb-6">
          Architectural lifecycle from raw respondent survey ingestion to explainability and educational awareness.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-3.5 rounded-xl bg-navy-950/60 border border-white/5">
            <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Step 01</span>
            <h5 className="text-xs font-bold text-slate-200 mt-1">Dataset Ingestion</h5>
            <p className="text-[11px] text-slate-400 mt-1">1,100 records collected via 20 standardized ordinal and continuous questions.</p>
          </div>
          <div className="p-3.5 rounded-xl bg-navy-950/60 border border-white/5">
            <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Step 02</span>
            <h5 className="text-xs font-bold text-slate-200 mt-1">Stratified 80/20 Split</h5>
            <p className="text-[11px] text-slate-400 mt-1">Preserves identical 3-class distribution across 880 train and 220 test records.</p>
          </div>
          <div className="p-3.5 rounded-xl bg-navy-950/60 border border-white/5">
            <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Step 03</span>
            <h5 className="text-xs font-bold text-cyan-400 mt-1">5-Model Benchmark</h5>
            <p className="text-[11px] text-slate-400 mt-1">Trained and compared Logistic Regression, Decision Tree, RF, SVM, and XGBoost.</p>
          </div>
          <div className="p-3.5 rounded-xl bg-navy-950/60 border border-white/5">
            <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Step 04</span>
            <h5 className="text-xs font-bold text-cyan-400 mt-1">GridSearchCV Tuning</h5>
            <p className="text-[11px] text-slate-400 mt-1">5-fold cross-validation on X_train tuned tree depth and estimators (90.91% test).</p>
          </div>
          <div className="p-3.5 rounded-xl bg-navy-950/60 border border-white/5">
            <span className="text-[10px] font-bold text-violet-400 uppercase tracking-wider">Step 05</span>
            <h5 className="text-xs font-bold text-slate-200 mt-1">TreeSHAP Mathematics</h5>
            <p className="text-[11px] text-slate-400 mt-1">Fast polynomial-time computation decomposing probability output into signed attributions.</p>
          </div>
          <div className="p-3.5 rounded-xl bg-navy-950/60 border border-white/5">
            <span className="text-[10px] font-bold text-violet-400 uppercase tracking-wider">Step 06</span>
            <h5 className="text-xs font-bold text-slate-200 mt-1">What-If Sensitivity</h5>
            <p className="text-[11px] text-slate-400 mt-1">Ceteris paribus feature adjustments evaluating non-linear decision boundary shifts.</p>
          </div>
          <div className="p-3.5 rounded-xl bg-navy-950/60 border border-white/5">
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Step 07</span>
            <h5 className="text-xs font-bold text-slate-200 mt-1">Educational Guidance</h5>
            <p className="text-[11px] text-slate-400 mt-1">Domain-aligned lifestyle and study suggestions derived from observed percentiles.</p>
          </div>
          <div className="p-3.5 rounded-xl bg-navy-950/60 border border-white/5">
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Step 08</span>
            <h5 className="text-xs font-bold text-slate-200 mt-1">Vercel Serverless</h5>
            <p className="text-[11px] text-slate-400 mt-1">Next.js React frontend coupled with lightweight Python serverless inference API.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
