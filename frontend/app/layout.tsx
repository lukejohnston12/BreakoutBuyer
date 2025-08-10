import "../styles/globals.css";
export const metadata = { title: "BreakoutBuyer", description: "NBA breakout predictor (early-career)" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-black text-slate-100">{children}</body>
    </html>
  );
}
