"use client";
import React from "react";
import Image from "next/image";
import Papa from "papaparse";
import { motion } from "framer-motion";
import { Download, PlayCircle, Search, Sparkles } from "lucide-react";

type Row = {
  PLAYER_ID?: number;
  DISPLAY_FIRST_LAST: string;
  AGE?: number;
  SEASON_EXP?: number;
  MPG?: number;
  PTS_36?: number;
  TS?: number;
  P_BREAKOUT_NEXT: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Page() {
  const [rows, setRows] = React.useState<Row[]>([]);
  const [query, setQuery] = React.useState("");
  const [minProb, setMinProb] = React.useState(60);
  const [minMPG, setMinMPG] = React.useState(18);
  const [running, setRunning] = React.useState(false);
  const [lastUpdated, setLastUpdated] = React.useState<string>("");
  const [statusText, setStatusText] = React.useState<string>("");
  const [fetching, setFetching] = React.useState(false);

  React.useEffect(() => { fetchLatest(); }, []);

  async function fetchLatest() {
    try {
      const ctrl = new AbortController();
      const id = setTimeout(() => ctrl.abort(), 120000); // 120s
      const r = await fetch(`${API_BASE}/api/candidates`, { signal: ctrl.signal, cache: "no-store" });
      clearTimeout(id);
      if (r.status === 202) {
        alert("Candidates not ready yet. Run the model first (or wait until it finishes).");
        return;
      }
      if (!r.ok) throw new Error(`GET /api/candidates ${r.status}`);
      const d = await r.json();
      setRows(d);
      setLastUpdated(new Date().toLocaleString());
    } catch (e:any) {
      alert(`Fetch Latest failed: ${e.message ?? e}`);
    }
  }

  async function runModel() {
    setRunning(true);
    const t0 = Date.now();
    try {
      // kick off build
      const r = await fetch(`${API_BASE}/api/run`, { method: "POST" });
      if (!r.ok) {
        const msg = await r.text().catch(()=> "");
        throw new Error(`POST /api/run ${r.status} ${msg}`);
      }
      // poll status until done
      let tries = 0;
      while (tries < 900) { // ~30 min max @ 2s
        await new Promise(res => setTimeout(res, 2000));
        const s = await fetch(`${API_BASE}/api/status`, { cache: "no-store" }).then(x=>x.json()).catch(()=> ({}));
        if (s?.phase === "done") break;
        tries++;
      }
      await fetchLatest();
      const secs = Math.round((Date.now()-t0)/1000);
      alert(`Model run complete in ${secs}s`);
    } catch (e:any) {
      alert(`Run failed: ${e.message ?? e}`);
    } finally {
      setRunning(false);
    }
  }

  React.useEffect(() => {
    if (!running) return;
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/status`, { cache: "no-store" });
        const s = await r.json();
        const base = s?.phase ?? "idle";
        const prog = (s?.done && s?.total) ? ` ${s.done}/${s.total}` : "";
        setStatusText(`${base}${prog}`);
      } catch { /* noop */ }
    }, 2000);
    return () => clearInterval(id);
  }, [running]);

  function downloadCsv() {
    const header = ["DISPLAY_FIRST_LAST","AGE","SEASON_EXP","MPG","PTS_36","TS","P_BREAKOUT_NEXT"];
    const body = filtered.map(r => header.map(k => (r as any)[k] ?? "").join(","));
    const csv = [header.join(","), ...body].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "breakoutbuyer_candidates.csv"; a.click();
    URL.revokeObjectURL(url);
  }

  const list = Array.isArray(rows) ? rows : [];
  const filtered = list
    .filter(r => (r.DISPLAY_FIRST_LAST || "").toLowerCase().includes(query.toLowerCase()))
    .filter(r => (r.P_BREAKOUT_NEXT ?? 0) * 100 >= minProb)
    .filter(r => (r.MPG ?? 0) >= minMPG)
    .sort((a,b)=> (b.P_BREAKOUT_NEXT ?? 0) - (a.P_BREAKOUT_NEXT ?? 0));

  return (
    <div className="min-h-screen w-full bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-black to-slate-950">
      <header className="sticky top-0 z-20 backdrop-blur bg-black/40 border-b border-cyan-500/40">
        <div className="mx-auto max-w-6xl px-4 py-4 flex items-center gap-3">
          <Image src="/nba-logo.svg" alt="NBA logo" width={36} height={36} className="h-9 w-9 rounded-xl" />
          <div>
            <h1 className="text-xl font-bold tracking-widest uppercase">BreakoutBuyer</h1>
            <p className="text-xs text-slate-300/80 -mt-1">Early-career NBA breakout predictor</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {lastUpdated && <span className="text-xs text-slate-400">Updated {lastUpdated}</span>}
            <button onClick={downloadCsv} className="inline-flex items-center gap-2 rounded-lg border border-cyan-500/50 px-3 py-2 text-sm">
              <Download className="h-4 w-4"/> Export
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        <section className="rounded-2xl border border-cyan-500/30 bg-slate-900/50 p-4 shadow-[0_0_40px_-15px_rgba(0,255,255,.35)]">
          <div className="flex flex-wrap items-end gap-4">
            <div className="relative">
              <input className="pl-9 pr-3 py-2 rounded-lg bg-slate-800/70 border border-cyan-500/30 outline-none"
                     placeholder="Search player…" value={query} onChange={e=>setQuery(e.target.value)} />
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400"/>
            </div>
            <label className="text-xs text-slate-300">Min Prob
              <input type="range" min={0} max={100} step={1} value={minProb}
                     onChange={e=>setMinProb(parseInt(e.target.value))}
                     className="ml-2 align-middle"/>
              <span className="ml-2 tabular-nums">{minProb}%</span>
            </label>
            <label className="text-xs text-slate-300">Min MPG
              <input type="range" min={0} max={36} step={1} value={minMPG}
                     onChange={e=>setMinMPG(parseInt(e.target.value))}
                     className="ml-2 align-middle"/>
              <span className="ml-2 tabular-nums">{minMPG}</span>
            </label>
            <button
              disabled={running}
              onClick={runModel}
              className="ml-auto inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-cyan-600 to-fuchsia-600 px-3 py-2 text-sm"
            >
              <PlayCircle className="h-4 w-4" />
              {running ? "Running…" : "Run Model"}
            </button>
            <button
              disabled={running || fetching}
              onClick={fetchLatest}
              className="inline-flex items-center gap-2 rounded-lg border border-cyan-500/50 px-3 py-2 text-sm"
            >
              <Sparkles className="h-4 w-4" />
              {fetching ? "Fetching…" : "Fetch Latest"}
            </button>
            {running && (
              <span className="text-xs rounded-md border border-cyan-500/40 px-2 py-1 text-cyan-300">
                {statusText || "running…"}
              </span>
            )}
          </div>

          <div className="mt-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {filtered.map((c,i)=> <RetroCard key={i} c={c} rank={i+1}/>)}
            {filtered.length===0 && <div className="text-slate-400 text-sm">No matches. Try relaxing filters.</div>}
          </div>
        </section>
      </main>
    </div>
  );
}

function RetroCard({ c, rank }: { c: Row; rank: number }) {
  const prob = c.P_BREAKOUT_NEXT ?? 0;
  const holo = `conic-gradient(from 180deg at 50% 50%, rgba(72,232,255,.35), rgba(245,133,255,.25), rgba(255,232,64,.25), rgba(72,232,255,.35))`;
  return (
    <motion.div initial={{opacity:0,y:8}} animate={{opacity:1,y:0}} transition={{duration:.25, delay: rank*0.02}} className="relative">
      <div className="absolute inset-0 rounded-3xl blur-xl opacity-50" style={{background: holo}} />
      <div className="relative rounded-3xl p-[2px] bg-gradient-to-br from-cyan-400/60 via-fuchsia-400/60 to-amber-400/60">
        <div className="rounded-[22px] h-full bg-slate-900/90">
          <div className="flex items-center justify-between px-4 pt-3">
            <span className="text-xs bg-cyan-600/70 text-white rounded px-2 py-0.5">#{rank}</span>
            <span className="text-xs border border-white/20 text-white/80 rounded px-2 py-0.5">{(prob*100).toFixed(0)}%</span>
          </div>
          <div className="px-4 mt-2">
            <div className="text-lg font-bold tracking-wide">{c.DISPLAY_FIRST_LAST}</div>
            <div className="text-xs text-slate-400">Age {c.AGE ?? "—"} · Exp {c.SEASON_EXP ?? "—"}</div>
          </div>
          <div className="mx-4 my-3 h-20 rounded-xl border border-white/10 overflow-hidden relative">
            <div className="absolute inset-0 bg-[linear-gradient(120deg,transparent,rgba(255,255,255,.15),transparent)] translate-x-[-60%] hover:translate-x-[60%] transition-transform duration-700" />
            <div className="absolute inset-0" style={{background: holo, opacity: .25}} />
            <div className="absolute inset-0 backdrop-blur-[2px]" />
          </div>
          <div className="px-4 pb-4">
            <div className="grid grid-cols-3 gap-3 text-sm">
              <Stat label="MPG" value={c.MPG?.toFixed(1) ?? "—"} />
              <Stat label="PTS/36" value={c.PTS_36?.toFixed(1) ?? "—"} />
              <Stat label="TS%" value={c.TS !== undefined ? `${(c.TS*100).toFixed(1)}%` : "—"} />
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
function Stat({label, value}:{label:string, value:string}) {
  return (
    <div className="p-3 rounded-xl bg-slate-800/60 border border-white/10">
      <div className="text-[10px] uppercase tracking-widest text-slate-400">{label}</div>
      <div className="text-base font-semibold tabular-nums">{value}</div>
    </div>
  );
}
