import { ShieldAlert } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-white/10 bg-navy-950/80 backdrop-blur-sm py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Educational Disclaimer */}
        <div className="bg-slate-900/60 border border-amber-500/20 rounded-xl p-4 flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-slate-300 leading-relaxed">
            <strong className="text-amber-400 font-semibold">Educational & Algorithmic Demonstration Scope:</strong>{" "}
            This project is developed as an academic end-semester machine learning system. It is <strong>not a medical diagnostic tool</strong>.
            Predictions, SHAP attributions, and What-If scenarios reflect mathematical sensitivities within decision trees, not medical causation.
            Physical symptoms (headaches, breathing problems, blood pressure changes) require clinical examination by licensed healthcare professionals.
          </div>
        </div>

        {/* Academic metadata */}
        <div className="flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 pt-2 border-t border-white/5">
          <p>
            AI-Based Mental Stress Detection • B.Tech Artificial Intelligence & Data Science • Group A6
          </p>
          <p className="mt-2 sm:mt-0">
            Powered by Next.js, React, Scikit-Learn Random Forest & TreeSHAP
          </p>
        </div>
      </div>
    </footer>
  );
}
