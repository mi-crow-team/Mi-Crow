import "./globals.css";
import Link from "next/link";
import { ReactNode } from "react";

const nav = [
  { href: "/", label: "Home" },
  { href: "/activations", label: "Activations" },
  { href: "/training", label: "Training" },
  { href: "/inference", label: "Inference" },
];

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-amber-50 text-slate-900">
        <div className="border-b border-amber-100 bg-white/80 backdrop-blur">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-amber-400 text-slate-900 font-bold shadow-sm">
                A
              </div>
              <div>
                <div className="text-lg font-semibold text-slate-900 tracking-tight">Amber SAE Studio</div>
                <div className="text-xs text-slate-500">Train SAEs, inspect activations, and play with concepts.</div>
              </div>
            </div>
            <nav className="flex gap-1 text-sm">
              {nav.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-full px-3 py-1 text-slate-600 hover:text-slate-900 hover:bg-amber-100 transition"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </div>
        <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}

