"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

export function Nav() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    onScroll();
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "backdrop-blur-xl bg-[#07080a]/70 border-b border-[var(--border)]"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="w-6 h-6 rounded-full bg-gradient-to-br from-[#7aa2ff] to-[#c792ff] opacity-90 group-hover:opacity-100 transition-opacity" />
          <span className="text-sm font-medium tracking-tight">
            did-the-ai-notice
          </span>
        </Link>
        <div className="flex items-center gap-7 text-sm text-[var(--ink-soft)]">
          <Link
            href="https://github.com/pj4533/introspection-autoresearch"
            target="_blank"
            className="hover:text-[var(--ink)] transition-colors"
          >
            github
          </Link>
        </div>
      </div>
    </nav>
  );
}
