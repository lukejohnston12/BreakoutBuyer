import os, time, json
import datetime as dt
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import (
    playercareerstats,
    playergamelog,
    commonplayerinfo,
)
from nba_api.stats.endpoints import leaguedashplayerstats
from sklearn.ensemble import HistGradientBoostingClassifier

# --- Config ---
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
CANDIDATES_PATH = os.getenv("BREAKOUTBUYER_CANDIDATES", "/data/candidates_latest.csv")
MIN_SEASON = int(os.getenv("MIN_SEASON", 2018))
MAX_SEASON = int(os.getenv("MAX_SEASON", 2025))  # t+1 target
EARLY_MAX_EXP = int(os.getenv("EARLY_MAX_EXP", 3))  # focus rookies/sophs/yr3
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"    # 1 = active players only (faster)
FAST_PIPELINE = os.getenv("FAST_PIPELINE", "1") == "1"  # default ON for now
MAX_PLAYERS = int(os.getenv("MAX_PLAYERS", "0"))  # 0 = no cap
STATUS_PATH = os.getenv("BREAKOUTBUYER_STATUS", "/data/status.json")

CACHE_DIR = os.getenv("BREAKOUTBUYER_CACHE", "/data/cache")

def cache_read(key: str):
    try:
        path = os.path.join(CACHE_DIR, f"{key}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        pass
    return None

def cache_write(key: str, df: pd.DataFrame):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"{key}.parquet")
        df.to_parquet(path, index=False)
    except Exception:
        pass

# --- App & CORS ---
app = FastAPI(title="BreakoutBuyer API", version="0.2-early")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in FRONTEND_ORIGINS if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/api/ping")
def ping():
    return {"ok": True, "msg": "pong_v2"}

def write_status(**kw):
    try:
        os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
        kw["ts"] = dt.datetime.utcnow().isoformat() + "Z"
        with open(STATUS_PATH, "w") as f:
            json.dump(kw, f)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass

@app.get("/api/ping-nba")
def ping_nba():
    """
    Sanity check that outbound network + nba_api work.
    Returns the number of NBA teams (should be 30) or a 502 with the error.
    """
    try:
        teams = static_teams.get_teams()
        write_status(phase="ping_nba_ok", teams=len(teams))
        return {"ok": True, "teams": len(teams)}
    except Exception as e:
        msg = f"nba_api error: {str(e)[:200]}"
        write_status(phase="ping_nba_error", message=msg)
        raise HTTPException(status_code=502, detail=msg)

# --- Utils ---
def safe_call(fn, max_retries=5, sleep=0.6, **kwargs):
    last_err = None
    for i in range(max_retries):
        try:
            # Add a request timeout when supported
            if hasattr(fn, "__init__") and hasattr(fn.__init__, "__code__"):
                if "timeout" in fn.__init__.__code__.co_varnames:
                    kwargs.setdefault("timeout", 20)
            return fn(**kwargs)
        except Exception as e:
            last_err = e
            time.sleep(sleep * (1.6 ** i))
    raise last_err

def season_id(y:int)->str:
    return f"{y}-{str((y+1)%100).zfill(2)}"

def parse_min(m):
    if isinstance(m,(int,float)) and not pd.isna(m): return float(m)
    if isinstance(m,str) and ":" in m:
        a,b=m.split(":"); return int(a)+int(b)/60.0
    return np.nan

def pull_player_season_totals(pid:int, years:list[int])->pd.DataFrame:
    rows=[]
    for y in years:
        gl = safe_call(playergamelog.PlayerGameLog, player_id=pid, season=season_id(y)).get_data_frames()[0]
        if gl.empty: continue
        for c in ["MIN","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","REB","AST","STL","BLK","TOV","PF","PTS"]:
            gl[c]=pd.to_numeric(gl[c], errors="coerce")
        gl["MIN_float"]=gl["MIN"].apply(parse_min)
        agg=gl.groupby("SEASON_ID").agg({
            "GAME_ID":"nunique","MIN_float":"sum","FGM":"sum","FGA":"sum","FG3M":"sum","FG3A":"sum","FTM":"sum","FTA":"sum",
            "OREB":"sum","DREB":"sum","REB":"sum","AST":"sum","STL":"sum","BLK":"sum","TOV":"sum","PF":"sum","PTS":"sum"
        }).reset_index()
        agg.rename(columns={"GAME_ID":"G","MIN_float":"MIN"}, inplace=True)
        agg["SEASON"]=y; agg["PLAYER_ID"]=pid; rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def get_player_info(pid:int)->dict:
    info=safe_call(commonplayerinfo.CommonPlayerInfo, player_id=pid).get_data_frames()[0]
    return {
        "PERSON_ID":pid,
        "DISPLAY_FIRST_LAST":info.loc[0,"DISPLAY_FIRST_LAST"],
        "BIRTHDATE":info.loc[0,"BIRTHDATE"],
        "SEASON_EXP":pd.to_numeric(info.loc[0,"SEASON_EXP"], errors="coerce"),
        "DRAFT_NUMBER":pd.to_numeric(info.loc[0,"DRAFT_NUMBER"], errors="coerce"),
    }

# --- Core ---
def build_dataset_fast(min_season=MIN_SEASON, max_season=MAX_SEASON, early_max_exp=EARLY_MAX_EXP):
    """
    Fast path using per-season aggregates. Tries hard not to return empty.
    Emits detailed status at each phase.
    """
    try:
        min_season = int(min_season)
        max_season = int(max_season)
    except Exception:
        write_status(phase="error", message="FAST: bad season bounds")
        return pd.DataFrame()

    seasons = list(range(min_season, max_season))
    if not seasons:
        write_status(phase="error", message="FAST: no seasons in range")
        return pd.DataFrame()

    frames = []
    write_status(phase="FAST: fetch seasons", total=len(seasons))

    for i, yr in enumerate(seasons, 1):
        season_str = f"{yr}-{str(yr+1)[-2:]}"
        try:
            cached = cache_read(f"league_dash_{season_str}") if 'cache_read' in globals() else None
            if cached is not None and not cached.empty:
                df = cached
            else:
                df = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season_str,
                    measure_type_detailed_def="Base",
                    per_mode_detailed="PerGame"
                ).get_data_frames()[0]
                if 'cache_write' in globals():
                    cache_write(f"league_dash_{season_str}", df)
            df["SEASON"] = yr
            frames.append(df)
            write_status(phase="FAST: fetched", done=i, total=len(seasons), last_season=season_str)
        except Exception as e:
            write_status(phase="FAST: fetch error", done=i-1, total=len(seasons), last_season=season_str, error=str(e)[:180])

    if not frames:
        write_status(phase="error", message="FAST: no frames after fetch")
        return pd.DataFrame()

    all_s = pd.concat(frames, ignore_index=True)

    # Select core columns defensively
    wanted = [
        "PLAYER_ID","PLAYER_NAME","TEAM_ID","GP","MIN","PTS","AST","REB",
        "FGA","FTA","FG_PCT","FG3_PCT","FT_PCT","SEASON"
    ]
    cols = [c for c in wanted if c in all_s.columns]
    all_s = all_s[cols].copy()

    # Feature engineering
    all_s["MPG"] = pd.to_numeric(all_s.get("MIN", 0), errors="coerce").fillna(0.0)
    # Per-36 points (safe)
    with np.errstate(divide="ignore", invalid="ignore"):
        all_s["PTS_36"] = np.where(all_s["MPG"] > 0, all_s["PTS"] * (36.0 / all_s["MPG"]), 0.0)

    # True Shooting using per-game FGA/FTA (proxy)
    FGA = pd.to_numeric(all_s.get("FGA", 0), errors="coerce").fillna(0.0)
    FTA = pd.to_numeric(all_s.get("FTA", 0), errors="coerce").fillna(0.0)
    denom = (2.0 * (FGA + 0.44 * FTA)).replace(0, np.nan)
    TS = pd.to_numeric(all_s.get("PTS", 0), errors="coerce") / denom
    all_s["TS"] = TS.fillna(0.0).clip(0, 1.5)

    # Experience index by player
    all_s = all_s.sort_values(["PLAYER_ID","SEASON"])
    all_s["SEASON_EXP"] = all_s.groupby("PLAYER_ID")["SEASON"].rank(method="dense").astype(int) - 1

    # Breakout label (simple next-season lift)
    all_s["PTS_next"] = all_s.groupby("PLAYER_ID")["PTS"].shift(-1)
    all_s["TS_next"]  = all_s.groupby("PLAYER_ID")["TS"].shift(-1)
    all_s["BREAKOUT_NEXT"] = ((all_s["PTS_next"] >= all_s["PTS"].fillna(0) + 3.0) &
                              (all_s["TS_next"] >= all_s["TS"].fillna(0) + 0.015)).astype(int)

    # Train only on early-career rows
    train = all_s[ all_s["SEASON_EXP"] < early_max_exp ].copy()
    feats = ["MPG","PTS_36","TS","SEASON_EXP"]
    train = train.replace([np.inf,-np.inf], np.nan).fillna(0)

    write_status(phase="FAST: counts",
                 train_rows=int(len(train)),
                 uniq_players=int(train["PLAYER_ID"].nunique()),
                 seasons=len(seasons))

    # If very small, still train (fallback test_size)
    if len(train) < 40:
        test_size = 0.1 if len(train) >= 20 else 0.0
    else:
        test_size = 0.2

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X = train[feats].values
    y = train["BREAKOUT_NEXT"].values
    if test_size > 0:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)
    else:
        Xtr, ytr = X, y

    model = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced")
    write_status(phase="FAST: train")
    if len(Xtr) == 0:
        write_status(phase="error", message="FAST: no training rows")
        return pd.DataFrame()
    model.fit(Xtr, ytr)

    # Score most recent season’s early-career players
    latest = max(seasons)
    cur = all_s[(all_s["SEASON"] == latest) & (all_s["SEASON_EXP"] < early_max_exp)].copy()
    if cur.empty:
        # Widen once if empty
        cur = all_s[(all_s["SEASON"] == latest) & (all_s["SEASON_EXP"] <= early_max_exp)].copy()

    cur = cur.replace([np.inf,-np.inf], np.nan).fillna(0)
    if cur.empty:
        write_status(phase="error", message=f"FAST: no current rows for {latest}")
        return pd.DataFrame()

    write_status(phase="FAST: score", current_rows=int(len(cur)))
    cur["P_BREAKOUT_NEXT"] = model.predict_proba(cur[feats].values)[:, 1]

    ranked = cur[["PLAYER_ID","PLAYER_NAME","SEASON_EXP","MPG","PTS_36","TS","P_BREAKOUT_NEXT"]].copy()
    ranked = ranked.rename(columns={"PLAYER_NAME":"DISPLAY_FIRST_LAST"})
    ranked = ranked.sort_values("P_BREAKOUT_NEXT", ascending=False)

    write_status(phase="done", rows=int(len(ranked)))
    return ranked

def build_dataset(min_season=MIN_SEASON, max_season=MAX_SEASON):
    all_players = static_players.get_players()
    id2name = {p["id"]: p["full_name"] for p in all_players}

    # Choose discovery source
    if FAST_MODE:
        ids_source = static_players.get_active_players()
        write_status(phase="Discovering players (FAST)", total=len(ids_source))
    else:
        ids_source = all_players
        write_status(phase="Discovering players", total=len(ids_source))

    ids = []
    for i, p in enumerate(ids_source, 1):
        pid = p["id"]

        # Heartbeat every 5
        if i % 5 == 0:
            write_status(
                phase="Discovering players (FAST)" if FAST_MODE else "Discovering players",
                done=i, total=len(ids_source), last_name=p.get("full_name", "?")
            )

        try:
            cs = safe_call(playercareerstats.PlayerCareerStats, player_id=pid).get_data_frames()[0]
            if not cs.empty and cs["SEASON_ID"].str.contains(str(min_season)).any():
                ids.append(pid)
        except Exception:
            continue

    # Apply cap AFTER discovery, then report the capped total
    if MAX_PLAYERS and MAX_PLAYERS > 0:
        ids = ids[:MAX_PLAYERS]

    write_status(phase="Players listed", total=len(ids))
    seasons = list(range(min_season, max_season))
    stacks: list[pd.DataFrame] = []
    total = len(ids)
    t0 = time.time()

    for i, pid in enumerate(ids, 1):
        try:
            dfp = pull_player_season_totals(pid, seasons)
            if not dfp.empty:
                stacks.append(dfp)
        except Exception as e:
            write_status(
                phase="Pulling game logs (error)",
                done=i-1, total=total,
                last_id=pid, last_name=id2name.get(pid, "?"),
                error=str(e)[:180]
            )
            continue

        # heartbeat every 5 players (and at the end)
        if i % 5 == 0 or i == total:
            dt_s = max(time.time() - t0, 1e-6)
            rps = i / dt_s
            eta = int(max(0, (total - i) / rps)) if rps > 0 else None
            write_status(
                phase="Pulling game logs",
                done=i, total=total,
                last_id=pid, last_name=id2name.get(pid, "?"),
                rps=round(rps, 2), eta_sec=eta
            )
    if not stacks: return pd.DataFrame()
    ps=pd.concat(stacks, ignore_index=True)

    infos=[get_player_info(pid) for pid in sorted(set(ps["PLAYER_ID"]))] 
    info_df=pd.DataFrame(infos)
    df=ps.merge(info_df, left_on="PLAYER_ID", right_on="PERSON_ID", how="left")

    df["BIRTHDATE_dt"]=pd.to_datetime(df["BIRTHDATE"], errors="coerce")
    df["AGE"]=df.apply(lambda r: r["SEASON"] - (r["BIRTHDATE_dt"].year if pd.notnull(r["BIRTHDATE_dt"]) else np.nan), axis=1)

    # Features
    eps=1e-9
    df["MPG"]=df["MIN"]/(df["G"]+eps)
    for s in ["PTS","REB","AST","STL","BLK","TOV","FGA","FGM","FG3A","FG3M","FTA","FTM","OREB","DREB","REB"]:
        df[f"{s}_36"]=(df[s]/(df["MIN"]+eps))*36.0
    df["TS"]=df["PTS"]/(2*(df["FGA"]+0.44*df["FTA"]+eps))
    df["FG3_rate"]=df["FG3A"]/(df["FGA"]+eps)
    df["FT_rate"]=df["FTA"]/(df["FGA"]+eps)
    df["RoleLoad_per_min"]=(df["FGA"]+0.44*df["FTA"]+df["TOV"])/(df["MIN"]+eps)

    df=df.sort_values(["PLAYER_ID","SEASON"]).reset_index(drop=True)
    for c in ["PTS_36","REB_36","AST_36","TS","MPG","RoleLoad_per_min","FG3_rate","FT_rate"]:
        df[f"YoY_{c}"]=df.groupby("PLAYER_ID")[c].pct_change()

    # League baselines
    med=df.groupby("SEASON")[["TS","PTS_36","MPG"]].median().rename(columns=lambda c: f"LEAGUE_MED_{c}")
    df=df.merge(med, left_on="SEASON", right_index=True, how="left")

    # --- Strict breakout label (all-star adjacent)
    def label_breakout(g):
        labels=[]; seen=False
        for _,r in g.iterrows():
            if pd.isna(r.get("YoY_PTS_36")): labels.append(0); continue
            rook = (r["SEASON_EXP"] in [0,1])
            # Steeper for early career
            cond_min = r["MPG"]>=26 if rook else r["MPG"]>=24
            cond_pts = r["PTS_36"]>=20 if rook else r["PTS_36"]>=max(18, r["LEAGUE_MED_PTS_36"]+2)
            cond_eff = r["TS"]>=(r["LEAGUE_MED_TS"]+0.04) if rook else r["TS"]>=(r["LEAGUE_MED_TS"]+0.03)
            yoy_pts = (r["YoY_PTS_36"] or 0)>=0.25
            yoy_mpg = (r["YoY_MPG"] or 0)>=0.20
            yoy_ts  = (r["YoY_TS"] or 0)>=0.02
            is_b = cond_min and cond_pts and cond_eff and (yoy_pts and yoy_mpg) and yoy_ts if rook \
                   else cond_min and cond_pts and cond_eff and (yoy_pts or yoy_mpg) and yoy_ts
            # Only one breakout per player (prevents relabeling)
            if is_b and seen: is_b=False
            labels.append(1 if is_b else 0)
            if is_b: seen=True
        g["LABEL_BREAKOUT"]=labels; return g

    df=df.groupby("PLAYER_ID", group_keys=False).apply(label_breakout)
    df=df.sort_values(["PLAYER_ID","SEASON"]).reset_index(drop=True)

    # --- Early-career training frame
    df["HAS_PRIOR_BREAKOUT"] = df.groupby("PLAYER_ID")["LABEL_BREAKOUT"].cummax().shift(1).fillna(0)
    df["TARGET_NEXT_BREAKOUT"] = df.groupby("PLAYER_ID")["LABEL_BREAKOUT"].shift(-1)
    target_early_mask = df.groupby("PLAYER_ID")["SEASON_EXP"].shift(-1) <= EARLY_MAX_EXP

    train_pool_mask = (df["SEASON_EXP"] <= EARLY_MAX_EXP) & (df["HAS_PRIOR_BREAKOUT"] == 0)
    mask = train_pool_mask & df["TARGET_NEXT_BREAKOUT"].notna() & target_early_mask

    # Features used for training
    feature_cols = [
        "AGE","SEASON_EXP","MPG","PTS_36","REB_36","AST_36","TS","FG3_rate","FT_rate","RoleLoad_per_min",
        "YoY_PTS_36","YoY_REB_36","YoY_AST_36","YoY_TS","YoY_MPG","YoY_FG3_rate","YoY_RoleLoad_per_min","YoY_FT_rate",
        "DRAFT_NUMBER"
    ]
    X = df.loc[mask, feature_cols].fillna(0)
    y = df.loc[mask, "TARGET_NEXT_BREAKOUT"].astype(int)

    # Simple time-aware split (train <= 2021)
    train_mask = df.loc[mask, "SEASON"] <= 2021
    write_status(phase="train")
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.07, max_iter=500)
    clf.fit(X[train_mask], y[train_mask])

    # --- Inference on latest season: early-career & no prior breakout
    latest_season = int(df["SEASON"].max())
    latest = df[df["SEASON"] == latest_season].copy()
    latest["HAS_PRIOR_BREAKOUT"] = latest.groupby("PLAYER_ID")["LABEL_BREAKOUT"].cummax().shift(1).fillna(0)
    infer_mask = (latest["SEASON_EXP"] <= EARLY_MAX_EXP) & (latest["HAS_PRIOR_BREAKOUT"] == 0)
    latest = latest[infer_mask]

    write_status(phase="score_latest")
    X_latest = latest[feature_cols].fillna(0)
    latest["P_BREAKOUT_NEXT"] = clf.predict_proba(X_latest)[:,1]

    ranked = latest.sort_values("P_BREAKOUT_NEXT", ascending=False)[
        ["PLAYER_ID","DISPLAY_FIRST_LAST","AGE","SEASON_EXP","MPG","PTS_36","TS","P_BREAKOUT_NEXT"]
    ].reset_index(drop=True)
    return ranked

# --- API Schemas ---
class Candidate(BaseModel):
    PLAYER_ID: Optional[int]
    DISPLAY_FIRST_LAST: str
    AGE: Optional[float]
    SEASON_EXP: Optional[float]
    MPG: Optional[float]
    PTS_36: Optional[float]
    TS: Optional[float]
    P_BREAKOUT_NEXT: float

# --- Routes ---
@app.get("/api/health")
def health():
    info = {"ok": True}
    try:
        if os.path.exists(CANDIDATES_PATH):
            st = os.stat(CANDIDATES_PATH)
            info["candidates"] = {
                "path": CANDIDATES_PATH,
                "bytes": int(st.st_size),
                "mtime": dt.datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
            }
    except Exception:
        pass
    try:
        if os.path.exists(STATUS_PATH):
            with open(STATUS_PATH) as f:
                info["status"] = json.load(f)
    except Exception:
        pass
    return info

@app.post("/api/run")
def run(request: Request):
    t0 = time.time()
    use_fast = request.query_params.get("fast_pipeline") == "1" or FAST_PIPELINE
    write_status(phase="start", mode="fast" if use_fast else "full")
    try:
        ranked = build_dataset_fast() if use_fast else build_dataset()
        if ranked.empty:
            # Try to read last status to return a helpful message
            detail = "No data built"
            try:
                with open(STATUS_PATH) as f:
                    s = json.load(f)
                    msg = s.get("message") or s.get("phase")
                    if msg:
                        detail = f"No data built: {msg}"
            except Exception:
                pass
            write_status(phase="error", message=detail, elapsed=int(time.time()-t0))
            raise HTTPException(500, detail)
        os.makedirs(os.path.dirname(CANDIDATES_PATH), exist_ok=True)
        ranked.to_csv(CANDIDATES_PATH, index=False)
        write_status(phase="done", rows=int(len(ranked)), elapsed=int(time.time()-t0))
        return {"rows": len(ranked), "saved_to": CANDIDATES_PATH}
    except Exception as e:
        write_status(phase="error", message=str(e)[:300], elapsed=int(time.time()-t0))
        raise

@app.get("/api/status")
def status():
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except Exception:
        return {"phase":"idle"}

@app.get("/api/candidates", response_model=List[Candidate])
def candidates():
    if not os.path.exists(CANDIDATES_PATH):
        # Not ready yet
        raise HTTPException(status_code=202, detail="candidates file not ready")
    return pd.read_csv(CANDIDATES_PATH).to_dict(orient="records")
