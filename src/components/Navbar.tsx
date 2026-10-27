"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  Activity, 
  Brain, 
  GitFork, 
  BarChart3, 
  Search, 
  Database, 
  Info,
  Menu,
  X
} from "lucide-react";
import { useState } from "react";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: Activity },
  { href: "/predict", label: "Prediction", icon: Brain },
  { href: "/what-if", label: "What-If Analysis", icon: GitFork },
  { href: "/performance", label: "Model Evaluation", icon: BarChart3 },
  { href: "/explainability", label: "Explainable AI", icon: Search },
  { href: "/dataset", label: "Dataset", icon: Database },
  { href: "/about", label: "Technical Defense", icon: Info },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 bg-navy-900/90 backdrop-blur-md border-b border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <Link href="/" className="flex items-center space-x-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-violet-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 group-hover:scale-105 transition-transform">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="text-sm font-bold tracking-tight text-white flex items-center gap-2">
                <span>AI Mental Stress Detection</span>
                <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] uppercase font-extrabold tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded-full">
                  v2.0
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Explainable Machine Learning Analytics Dashboard
              </p>
            </div>
          </Link>

          {/* Desktop Nav */}
          <nav className="hidden lg:flex items-center space-x-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
                    isActive
                      ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 shadow-sm"
                      : "text-slate-300 hover:text-white hover:bg-white/5"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Status Badge */}
          <div className="hidden md:flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1 bg-slate-900/80 rounded-full border border-white/10 text-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-slate-300 font-medium text-[11px]">RF Champion: <strong className="text-cyan-400">90.91%</strong></span>
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="lg:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 focus:outline-none"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-navy-850 border-b border-white/10 px-4 pt-2 pb-4 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`block px-3 py-2 rounded-md text-sm font-medium flex items-center gap-2 ${
                  isActive
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                    : "text-slate-300 hover:text-white hover:bg-white/5"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
}
