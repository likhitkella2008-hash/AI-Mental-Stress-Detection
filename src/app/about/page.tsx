"use client";

import { useState } from "react";
import { Info, ShieldAlert, Cpu, Award, ChevronDown, ChevronUp, Layers, CheckCircle2 } from "lucide-react";

export default function AboutPage() {
  const [openFaq, setOpenFaq] = useState<number | null>(0);

  const toggleFaq = (idx: number) => {
    setOpenFaq(openFaq === idx ? null : idx);
  };

  const FAQS = [
    {
      title: "1. Empirical Justification for Random Forest Selection (Why RF?)",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            • <strong>Empirical Benchmark Evidence:</strong> Across identical 80/20 stratified splits of the 1,100 respondent dataset,
            Random Forest achieved the highest baseline classification accuracy (88.64%) among the five benchmarked algorithms
            (LR: 88.18%, XGB: 88.18%, SVM: 87.73%, DT: 85.45%). Following hyperparameter optimization, test accuracy reached 90.91% with a multiclass ROC-AUC of 0.9853.
          </p>
          <p>
            • <strong>Ensemble Bagging vs. Single Trees:</strong> Single decision trees suffer from high variance and sensitivity to training data perturbations (85.45% accuracy).
            Random Forest averages multiple decorrelated decision trees, substantially reducing variance without increasing bias.
          </p>
          <p>
            • <strong>Tabular Domain Appropriateness vs. Deep Learning:</strong> Tabular survey datasets with modest sample sizes (~1,100 records) and discrete ordinal scales
            (e.g., 0–5 Likert scales) are well-suited for tree-based ensemble bagging. Deep neural networks on small tabular datasets carry severe risks of parameter overfitting,
            require costly hyperparameter calibration, and act as opaque black boxes without native polynomial-time Shapley explanations.
          </p>
          <p>
            • <strong>Defensibility Caveat:</strong> Random Forest is <strong>not universally optimal</strong> for all machine learning problems. It was empirically chosen
            for this specific project based on comparative benchmark metrics, stability across 5-fold cross-validation (88.18% &plusmn; 0.81%), and native compatibility
            with exact TreeSHAP explanation algorithms.
          </p>
        </div>
      )
    },
    {
      title: "2. Mathematical Mechanics of TreeSHAP (Explainable AI Defense)",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            • <strong>Polynomial-Time Complexity:</strong> Classical model-agnostic KernelSHAP requires exponential time O(2^M) by evaluating exponential subsets of features.
            TreeSHAP (Lundberg et al., 2020) exploits the tree graph topology to compute exact Shapley values in polynomial time O(T L D^2), where T is tree count,
            L is maximum leaves, and D is tree depth.
          </p>
          <p>
            • <strong>Four Fundamental Axioms:</strong> TreeSHAP mathematically guarantees Efficiency (&sum; &phi;_i = f(x) - E[f(x)]), Symmetry (identical contributors receive equal attribution),
            Dummy/Null player (features with zero marginal contribution receive &phi;_i = 0), and Additivity across ensemble trees.
          </p>
          <p>
            • <strong>Probability-Scale Additivity:</strong> For Scikit-Learn&apos;s RandomForestClassifier, each leaf node stores class frequencies.
            TreeSHAP computes attributions directly in class probability space: E[f(x)] + &sum; &phi;_i(x) = P(Stress Level = c | x).
            This mathematical identity is verified in real time on the Stress Prediction and Explainable AI modules with numerical error &lt; 10^-3.
          </p>
        </div>
      )
    },
    {
      title: "3. What-If Counterfactual Sensitivity Mechanics",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            • <strong>Ceteris Paribus Methodology:</strong> What-If analysis applies the <em>ceteris paribus</em> (&quot;all other things held constant&quot;) principle from sensitivity analysis.
            A user alters a selected subset of input features while preserving all remaining factors in their baseline state.
          </p>
          <p>
            • <strong>Decision Boundary &amp; Margin Exploration:</strong> The system recalculates predicted class probabilities simultaneously across all three stress categories.
            When the predicted category remains unchanged, the delta breakdown reveals whether internal confidence moved closer to or further from adjacent decision thresholds.
          </p>
          <p>
            • <strong>Attribution Sensitivity (&Delta;SHAP):</strong> By computing the difference between hypothetical and baseline Shapley attributions
            (&Delta;&phi;_i = &phi;_hypo,i - &phi;_base,i), the system quantifies exactly which feature modifications caused the probability shift toward or away from the target class.
          </p>
        </div>
      )
    },
    {
      title: "4. Dataset Architecture & Methodological Boundaries",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            • <strong>Dataset Characteristics:</strong> 1,100 respondent records spanning 20 psychological, physiological, academic, and social variables with zero missing values.
            Three balanced classes: Low (373), Moderate (358), and High (369).
          </p>
          <p>
            • <strong>Self-Reporting &amp; Subjectivity Bias:</strong> All inputs originate from subjective self-report questionnaires. Responses to ordinal questions
            (e.g., headache frequency or academic load on a 0–5 scale) reflect individual perception rather than standardized clinical biometric readings.
          </p>
          <p>
            • <strong>Cross-Sectional Limitation:</strong> The dataset captures single-point-in-time cross-sectional snapshots. It does not track respondents longitudinally
            across exam cycles or life transitions, meaning temporal trajectories cannot be inferred.
          </p>
          <p>
            • <strong>Correlation vs. Causation:</strong> Statistical correlations learned by decision trees do not establish medical or biological causation.
          </p>
        </div>
      )
    },
    {
      title: "5. Clinical Safety Boundary & Ethical Guidance Scope",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            • <strong>Educational &amp; Awareness Purpose:</strong> This application is strictly an educational engineering prototype designed to demonstrate Explainable AI and machine learning.
            It is <strong>not a diagnostic medical device</strong> and has not been cleared for clinical diagnostic use.
          </p>
          <p>
            • <strong>Mandatory Physician Evaluation for Physical Symptoms:</strong> Physical indicators (severe headaches, chronic sleep disturbance, blood pressure irregularities, breathing difficulties)
            must be evaluated by licensed medical physicians to rule out underlying organic pathologies.
          </p>
          <p>
            • <strong>Campus &amp; Professional Resource Referral:</strong> When moderate or high stress patterns are indicated, the application prominently provides guidance
            directing students to institutional counseling centers, peer support programs, and mental health crisis professionals.
          </p>
        </div>
      )
    }
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-bold text-violet-400 uppercase tracking-wider mb-2">
          <Info className="w-4 h-4" />
          <span>Technical Transparency &amp; Defense</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          System Architecture, Mathematical Defense &amp; FAQ
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Technical defenses, algorithm rationale, and mathematical proofs prepared for B.Tech end-semester project defense.
        </p>
      </div>

      {/* 2-Column Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="metric-card-modern p-6 space-y-3">
          <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">🎯 Project Scope &amp; Purpose</span>
          <p className="text-xs text-slate-300 leading-relaxed">
            Develop a regularized supervised machine learning classification pipeline analyzing multi-factor behavioral,
            physiological, academic, and social indicators to provide early stress awareness across Low, Moderate, and High states.
          </p>
        </div>

        <div className="metric-card-modern p-6 space-y-3">
          <span className="text-xs font-bold text-violet-400 uppercase tracking-wider">🤖 Benchmarked Algorithms</span>
          <p className="text-xs text-slate-300 leading-relaxed">
            Five classification models trained under identical stratified 80/20 splits: Random Forest (88.64% baseline / 90.91% optimized),
            Logistic Regression (88.18%), XGBoost (88.18%), SVM (87.73%), and Decision Tree (85.45%).
          </p>
        </div>
      </div>

      {/* 5 Expandable Technical Defense Cards */}
      <div className="metric-card-modern p-6 space-y-4">
        <h3 className="text-base font-bold text-white mb-2">Technical Defense &amp; Transparency Questions (FAQ)</h3>
        <p className="text-xs text-slate-400 mb-4">
          Click any expander to inspect the technical rationale, mathematical guarantees, and scientific boundary conditions.
        </p>

        <div className="space-y-3">
          {FAQS.map((faq, idx) => {
            const isOpen = openFaq === idx;
            return (
              <div key={idx} className="bg-navy-950/70 rounded-xl border border-white/5 overflow-hidden transition-all">
                <button
                  onClick={() => toggleFaq(idx)}
                  className="w-full p-4 flex items-center justify-between text-left text-xs font-bold text-slate-200 hover:text-cyan-300 transition-colors"
                >
                  <span>{faq.title}</span>
                  {isOpen ? <ChevronUp className="w-4 h-4 text-cyan-400 shrink-0" /> : <ChevronDown className="w-4 h-4 text-slate-500 shrink-0" />}
                </button>
                {isOpen && <div className="p-4 pt-0 border-t border-white/5 mt-2">{faq.content}</div>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
