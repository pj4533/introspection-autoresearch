import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans-inter",
  subsets: ["latin"],
  display: "swap",
});

const mono = JetBrains_Mono({
  variable: "--font-mono-jet",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Did the AI notice? — Mind-reading experiments on Gemma 3",
  description:
    "We secretly plant a thought inside an AI's mind and ask: did you notice? Sometimes — it does. A live research project reproducing and extending Macar et al. 2026 on Google's Gemma 3 12B, running on a Mac Studio.",
  openGraph: {
    title: "Did the AI notice?",
    description:
      "We plant thoughts inside an AI's mind. Sometimes it notices. Live research on Gemma 3 12B.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
      <body>
        <div className="noise" />
        {children}
      </body>
    </html>
  );
}
