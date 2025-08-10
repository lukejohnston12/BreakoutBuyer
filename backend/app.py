import os, time
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nba_api.stats.static import players as static_players
from nba_api.stats.endpoints import playercareerstats, playergamelog, commonplayerinfo
from sklearn.ensemble import HistGradientBoostingClassifier

# --- App & CORS ---
app = FastAPI(title="BreakoutBuyer API", version="0.2-early")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten to your Railway frontend URL after deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
CANDIDATES_PATH = os.getenv("BREAKOUTBUYER_CANDIDATES", "/data/candidates_latest.csv")
MIN_SEASON = int(os.getenv("MIN_SEASON", 2018))
MAX_SEASON = int(os.getenv("MAX_SEASON", 2025))  # t+1 target
EARLY_MAX_EXP = int(os.getenv("EARLY_MAX_EXP", 3))  # focus rookies/sophs/yr3

# --- Utils ---
def safe_call(fn, max_retries=4, sleep=0.6, **kwargs):
    for i in range(max_retries):
        try:
            return fn(**kwargs)
        except Exception:
            if i == max_retries - 1:
                raise
            time.sleep(sleep * (i+1))

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
def build_dataset(min_season=MIN_SEASON, max_season=MAX_SEASON):
    all_players = static_players.get_players()
    ids=[]
    for pid in pd.DataFrame(all_players)["id"].tolist():
        try:
            cs = safe_call(playercareerstats.PlayerCareerStats, player_id=pid).get_data_frames()[0]
            if not cs.empty and cs["SEASON_ID"].str.contains(str(min_season)).any():
                ids.append(pid)
        except Exception:
            continue

    seasons=list(range(min_season, max_season))
    stacks=[]
    for i,pid in enumerate(ids):
        df=pull_player_season_totals(pid,seasons)
        if not df.empty: stacks.append(df)
        if (i+1)%25==0: print(f"Pulled {i+1}/{len(ids)} players…")
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
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.07, max_iter=500)
    clf.fit(X[train_mask], y[train_mask])

    # --- Inference on latest season: early-career & no prior breakout
    latest_season = int(df["SEASON"].max())
    latest = df[df["SEASON"] == latest_season].copy()
    latest["HAS_PRIOR_BREAKOUT"] = latest.groupby("PLAYER_ID")["LABEL_BREAKOUT"].cummax().shift(1).fillna(0)
    infer_mask = (latest["SEASON_EXP"] <= EARLY_MAX_EXP) & (latest["HAS_PRIOR_BREAKOUT"] == 0)
    latest = latest[infer_mask]

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
def health(): return {"ok": True}

@app.post("/api/run")
def run():
    ranked=build_dataset()
    if ranked.empty: raise HTTPException(500, "No data built")
    os.makedirs(os.path.dirname(CANDIDATES_PATH), exist_ok=True)
    ranked.to_csv(CANDIDATES_PATH, index=False)
    return {"rows": len(ranked), "saved_to": CANDIDATES_PATH}

@app.get("/api/candidates", response_model=List[Candidate])
def candidates():
    if not os.path.exists(CANDIDATES_PATH):
        ranked=build_dataset()
        if ranked.empty: return []
        os.makedirs(os.path.dirname(CANDIDATES_PATH), exist_ok=True)
        ranked.to_csv(CANDIDATES_PATH, index=False)
    return pd.read_csv(CANDIDATES_PATH).to_dict(orient="records")
