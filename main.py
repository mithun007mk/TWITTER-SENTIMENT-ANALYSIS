from transformers import pipeline
import re, uvicorn, os
from datetime import datetime

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nrclex import NRCLex

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

app = FastAPI(title="Twitter Emotion Intensity Analysis API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "Invalid request body"})

@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})


KAGGLE_DF    = None
KAGGLE_STATS = {}

def load_dataset(path: str = "dataset.csv"):
    global KAGGLE_DF, KAGGLE_STATS
    if not os.path.exists(path):
        KAGGLE_STATS = {"loaded": False, "reason": "dataset.csv not found"}
        return
    try:
        df = pd.read_csv(path)

        df.columns = [c.strip().lower() for c in df.columns]

        col_map = {}
        for c in df.columns:
            if c in ("tweet","text","content","sentence"):    col_map[c] = "tweet"
            elif c in ("emotion","label","sentiment","affect"): col_map[c] = "emotion"
            elif c in ("intensity score","intensity_score",
                       "score","intensity"):                   col_map[c] = "intensity"
        df = df.rename(columns=col_map)

        needed = [c for c in ["tweet","emotion"] if c in df.columns]
        df = df[needed].dropna()
        df["emotion"] = df["emotion"].str.strip().str.lower()

        if "intensity" in df.columns:
            df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce").fillna(0.0)
        else:
            df["intensity"] = 0.5

        total       = len(df)
        dist        = df["emotion"].value_counts().to_dict()
        pct         = {k: round(v/total*100, 1) for k,v in dist.items()}
        avg_intensity = {}
        for em in df["emotion"].unique():
            avg_intensity[em] = round(float(df[df["emotion"]==em]["intensity"].mean()), 3)

        KAGGLE_DF = df
        KAGGLE_STATS = {
            "loaded":        True,
            "source":        path,
            "total":         total,
            "emotions":      list(df["emotion"].unique()),
            "distribution":  dist,
            "percentage":    pct,
            "avg_intensity": avg_intensity,
        }
    except Exception as ex:
        KAGGLE_STATS = {"loaded": False, "reason": str(ex)}

load_dataset()

HF_MODEL = "j-hartmann/emotion-english-distilroberta-base"
print(f"Loading HuggingFace model: {HF_MODEL} ...")
hf_classifier = pipeline(
    "text-classification",
    model=HF_MODEL,
    top_k=None,
    device=-1,
)
print("HuggingFace model loaded.")

EMOTIONS = ["joy","anger","fear","sadness","surprise","disgust","anticipation","trust"]
META = {
    "joy":          {"label":"Joy",          "color":"#FFD700"},
    "anger":        {"label":"Anger",        "color":"#FF4136"},
    "fear":         {"label":"Fear",         "color":"#9B59B6"},
    "sadness":      {"label":"Sadness",      "color":"#3498DB"},
    "surprise":     {"label":"Surprise",     "color":"#FF6B35"},
    "disgust":      {"label":"Disgust",      "color":"#2ECC71"},
    "anticipation": {"label":"Anticipation", "color":"#F39C12"},
    "trust":        {"label":"Trust",        "color":"#1ABC9C"},
}

HF_LABEL_MAP = {
    "joy":      "joy",
    "anger":    "anger",
    "fear":     "fear",
    "sadness":  "sadness",
    "surprise": "surprise",
    "disgust":  "disgust",
    "neutral":  None,
}

lemmatizer   = WordNetLemmatizer()
stop_words   = set(stopwords.words("english"))
history_rows = []

class Tweet(BaseModel):
    text: str

def preprocess(text: str) -> dict:
    text       = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text       = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    tokens     = word_tokenize(text.lower())
    no_stop    = [t for t in tokens if t not in stop_words and t.isalpha()]
    lemmatized = [lemmatizer.lemmatize(t) for t in no_stop]
    return {
        "tokens":     tokens[:12],
        "no_stop":    no_stop[:12],
        "lemmatized": lemmatized[:12],
        "clean":      " ".join(lemmatized),
    }

def get_scores(clean_text: str) -> dict:
    e = NRCLex(clean_text)
    raw = {em: 0.0 for em in EMOTIONS}
    try:
        if hasattr(e, "raw_emotion_scores"):
            src = e.raw_emotion_scores
            for k in EMOTIONS:
                raw[k] = float(src.get(k, 0))
        elif hasattr(e, "affect_frequencies"):
            src = e.affect_frequencies
            for k in EMOTIONS:
                raw[k] = float(src.get(k, 0))
        elif hasattr(e, "emotions"):
            src = e.emotions
            for k in EMOTIONS:
                raw[k] = float(src.get(k, 0))
        elif hasattr(e, "affect_list"):
            for emotions in e.affect_list:
                for em in emotions:
                    if em in raw:
                        raw[em] += 1.0
        elif hasattr(e, "top_emotions"):
            for em, score in e.top_emotions:
                if em in raw:
                    raw[em] = float(score)
    except Exception:
        pass
    total = sum(raw.values()) or 1
    return {k: round(raw[k] / total, 3) for k in EMOTIONS}

def get_hf_scores(text: str) -> dict:
    try:
        results = hf_classifier(text[:512])[0]
        raw = {r["label"].lower(): r["score"] for r in results}

        hf_supported = ["joy", "anger", "fear", "sadness", "surprise", "disgust"]
        mapped = {e: 0.0 for e in EMOTIONS}
        for em in hf_supported:
            if em in raw:
                mapped[em] = raw[em]

        total = sum(mapped[e] for e in hf_supported) or 1
        for em in hf_supported:
            mapped[em] = round(mapped[em] / total, 3)

        return mapped
    except Exception:
        return {e: 0.0 for e in EMOTIONS}

def fuse_scores(nrc: dict, hf: dict) -> dict:
    hf_supported = {"joy", "anger", "fear", "sadness", "surprise", "disgust"}
    fused = {}
    for e in EMOTIONS:
        if e in hf_supported:
            fused[e] = round(0.4 * nrc.get(e, 0) + 0.6 * hf.get(e, 0), 4)
        else:
            fused[e] = round(nrc.get(e, 0), 4)

    total = sum(fused.values()) or 1
    return {k: round(v / total, 3) for k, v in fused.items()}

def get_analytics() -> dict:
    if not history_rows:
        return {}
    df          = pd.DataFrame(history_rows)
    most_common = df["dominant"].value_counts().to_dict()
    avg_scores  = {e: round(float(df[e].mean()), 3) for e in EMOTIONS}
    total       = len(df)
    emotion_pct = {e: round(int((df["dominant"] == e).sum()) / total * 100, 1) for e in EMOTIONS}
    return {"total": total, "most_common": most_common,
            "avg_scores": avg_scores, "emotion_pct": emotion_pct}

def find_similar(dominant: str, n: int = 3) -> list:
    if KAGGLE_DF is None:
        return []
    subset = KAGGLE_DF[KAGGLE_DF["emotion"] == dominant]
    if subset.empty:
        subset = KAGGLE_DF[KAGGLE_DF["emotion"].str.contains(dominant, na=False)]
    if subset.empty:
        return []
    rows = subset.nlargest(min(n, len(subset)), "intensity") if "intensity" in subset.columns \
           else subset.sample(min(n, len(subset)))
    result = []
    for _, row in rows.iterrows():
        result.append({
            "tweet":     str(row["tweet"]),
            "emotion":   str(row["emotion"]),
            "intensity": round(float(row.get("intensity", 0.5)), 3),
        })
    return result

@app.post("/analyse")
def analyse(req: Tweet):
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Tweet text cannot be empty"})
    try:
        nlp        = preprocess(req.text)
        nrc_scores = get_scores(nlp["clean"])
        hf_scores  = get_hf_scores(req.text)
        scores     = fuse_scores(nrc_scores, hf_scores)
        dom = max(scores, key=scores.get) if any(scores.values()) else "neutral"
        history_rows.append({"timestamp": datetime.now().isoformat(),
                              "text": req.text, "dominant": dom, **scores})
        return JSONResponse(content={
            "dominant":  dom,
            "label":     META.get(dom, {"label": dom})["label"],
            "color":     META.get(dom, {"color": "#888"})["color"],
            "scores":    scores,
            "nlp":       {"tokens":     nlp["tokens"],
                          "no_stop":    nlp["no_stop"],
                          "lemmatized": nlp["lemmatized"]},
            "similar":   find_similar(dom),
            "analytics": get_analytics(),
        })
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})

@app.get("/dataset-stats")
def dataset_stats():
    return JSONResponse(content=KAGGLE_STATS)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Twitter Emotion Analyser</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--s:#161b22;--c:#1c2128;--b:#30363d;--t:#e6edf3;--m:#8b949e;--bl:#1d9bf0;--gr:#2ea043;--tw:#1d9bf0}
body{background:var(--bg);color:var(--t);font-family:Arial,sans-serif;min-height:100vh}

#login{min-height:100vh;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden}

.login-bg{position:absolute;inset:0;background:linear-gradient(160deg,#0a0e1a 0%,#0d1f3c 40%,#051525 100%)}

.tw-pattern{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.tw-bird{position:absolute;color:rgba(29,155,240,0.07);font-size:24px;animation:drift linear infinite}
@keyframes drift{0%{transform:translateY(0) rotate(0deg);opacity:0}
  10%{opacity:1}90%{opacity:1}100%{transform:translateY(-110vh) rotate(30deg);opacity:0}}

.tweet-cards{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.tc-card{
  position:absolute;background:rgba(29,155,240,0.06);border:1px solid rgba(29,155,240,0.15);
  border-radius:12px;padding:12px 14px;width:220px;backdrop-filter:blur(2px);
  animation:floatcard linear infinite;
}
.tc-card-header{display:flex;align-items:center;gap:6px;margin-bottom:6px}
.tc-avatar{width:20px;height:20px;border-radius:50%;background:rgba(29,155,240,0.3)}
.tc-name{font-size:10px;font-weight:700;color:rgba(29,155,240,0.6);font-family:monospace}
.tc-text{font-size:9px;color:rgba(255,255,255,0.25);line-height:1.5;font-family:monospace}
.tc-footer{display:flex;gap:10px;margin-top:6px}
.tc-icon{font-size:9px;color:rgba(29,155,240,0.3)}
@keyframes floatcard{0%{transform:translateY(110vh) rotate(var(--rot));opacity:0}
  8%{opacity:1}92%{opacity:1}100%{transform:translateY(-30vh) rotate(var(--rot));opacity:0}}

.glow{position:absolute;border-radius:50%;filter:blur(80px);pointer-events:none}

.grid-ov{position:absolute;inset:0;
  background-image:linear-gradient(rgba(29,155,240,0.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(29,155,240,0.03) 1px,transparent 1px);
  background-size:52px 52px}

.hashtag-strip{
  position:absolute;bottom:32px;left:0;right:0;
  display:flex;gap:10px;justify-content:center;flex-wrap:wrap;
  pointer-events:none;
}
.htag{font-size:10px;font-weight:700;color:rgba(29,155,240,0.25);font-family:monospace}

.box{
  background:rgba(13,17,23,0.88);border:1px solid rgba(29,155,240,0.25);border-radius:16px;
  padding:36px 32px;width:340px;text-align:center;position:relative;z-index:10;
  box-shadow:0 0 0 1px rgba(29,155,240,0.08),0 20px 60px rgba(0,0,0,0.7),0 0 100px rgba(29,155,240,0.08);
  backdrop-filter:blur(16px);
}
.twitter-logo{
  width:44px;height:44px;background:var(--tw);border-radius:50%;
  display:flex;align-items:center;justify-content:center;margin:0 auto 12px;
  font-size:22px;font-weight:900;color:#fff;letter-spacing:-1px;
}
.box h2{font-size:19px;font-weight:700;margin-bottom:4px;color:var(--t)}
.box p{font-size:11px;color:var(--m);margin-bottom:24px;font-family:monospace;line-height:1.6}
.box label{display:block;text-align:left;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--m);margin-bottom:5px}
.box input{width:100%;background:rgba(255,255,255,0.04);border:1px solid var(--b);border-radius:8px;color:var(--t);font-size:13px;padding:10px 12px;margin-bottom:14px;outline:none;transition:border-color .2s}
.box input:focus{border-color:var(--tw);background:rgba(29,155,240,0.05)}
.btn{width:100%;background:var(--tw);color:#fff;border:none;border-radius:99px;font-size:14px;font-weight:700;padding:11px;cursor:pointer;letter-spacing:.3px;transition:opacity .2s}
.btn:hover{opacity:.88}.btn:disabled{opacity:.45;cursor:not-allowed}
.err{color:#f85149;font-size:11px;margin-top:8px;min-height:14px;font-family:monospace}
.divider{height:1px;background:var(--b);margin:16px 0}
.emotion-pills{display:flex;gap:5px;justify-content:center;flex-wrap:wrap}
.epill{font-size:9px;font-weight:700;padding:3px 9px;border-radius:99px;font-family:monospace}

#app{display:none}
.nav{background:var(--s);border-bottom:1px solid var(--b);padding:12px 28px;display:flex;align-items:center;justify-content:space-between}
.nav-brand{display:flex;align-items:center;gap:9px}
.nav-logo{width:28px;height:28px;background:var(--tw);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:900;color:#fff}
.nav h1{font-size:16px;font-weight:700;color:var(--t)}
.out-btn{background:none;border:1px solid var(--b);color:var(--m);border-radius:99px;padding:4px 14px;font-size:11px;cursor:pointer;margin-left:10px;transition:all .2s}
.out-btn:hover{border-color:var(--tw);color:var(--tw)}
.main{max-width:1200px;margin:0 auto;padding:20px 18px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
@media(max-width:900px){.main{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.main{grid-template-columns:1fr}}
.card{background:var(--c);border:1px solid var(--b);border-radius:10px;padding:16px}
.span2{grid-column:span 2}.span3{grid-column:1/-1}
.lbl{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--m);margin-bottom:10px}
.sublbl{font-size:8px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--m);margin:8px 0 4px}
textarea{width:100%;background:var(--bg);border:1px solid var(--b);border-radius:6px;color:var(--t);font-size:13px;font-family:monospace;padding:10px;resize:vertical;min-height:72px;outline:none;transition:border-color .2s}
textarea:focus{border-color:var(--tw)}
.row{display:flex;align-items:center;gap:8px;margin-top:9px}
.sbtn{background:none;border:1px solid var(--b);color:var(--m);border-radius:99px;font-size:12px;padding:8px 15px;cursor:pointer}
.cc{margin-left:auto;font-family:monospace;font-size:11px;color:var(--m)}
.dw{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:130px;text-align:center}
.dl{font-size:22px;font-weight:700;letter-spacing:-1px;margin-bottom:4px}
.ds{font-size:10px;color:var(--m);font-family:monospace}
.ph{color:var(--m);font-size:12px;font-family:monospace;line-height:2}
.bars{display:flex;flex-direction:column;gap:6px}
.br{display:grid;grid-template-columns:90px 1fr 38px;align-items:center;gap:6px}
.bn{font-size:11px;font-weight:600}
.bt{height:7px;background:var(--bg);border-radius:99px;overflow:hidden}
.bf{height:100%;border-radius:99px;width:0%;transition:width .8s}
.bv{font-family:monospace;font-size:10px;color:var(--m);text-align:right}
.cw{max-width:220px;margin:0 auto}
.nlp-box{background:var(--bg);border-radius:6px;padding:8px;margin-top:4px;font-family:monospace;font-size:10px;color:var(--m);line-height:1.8;word-break:break-all}
.nlp-box span{color:var(--tw)}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.stat{background:var(--bg);border-radius:6px;padding:10px;text-align:center}
.stat-val{font-size:18px;font-weight:700;color:var(--tw)}
.stat-key{font-size:10px;color:var(--m);margin-top:2px}
.pct-bars{display:flex;flex-direction:column;gap:5px}
.pct-row{display:grid;grid-template-columns:90px 1fr 42px;align-items:center;gap:6px}
.pct-name{font-size:10px;font-weight:600}
.pct-track{height:6px;background:var(--bg);border-radius:99px;overflow:hidden}
.pct-fill{height:100%;border-radius:99px;transition:width 1s}
.pct-val{font-family:monospace;font-size:10px;color:var(--m);text-align:right}
.sim-card{background:var(--bg);border-radius:8px;padding:10px;margin-bottom:7px;border-left:3px solid var(--tw)}
.sim-tweet{font-size:11px;font-family:monospace;color:var(--t);line-height:1.5;margin-bottom:5px}
.sim-meta{display:flex;gap:8px;align-items:center}
.sim-badge{display:inline-flex;padding:2px 7px;border-radius:99px;font-size:9px;font-weight:700}
.sim-intensity{font-size:10px;color:var(--m);font-family:monospace}
.intensity-bar{display:flex;align-items:center;gap:6px;flex:1}
.ibt{height:4px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden;width:60px}
.ibf{height:100%;border-radius:99px}
.he{font-family:monospace;font-size:12px;color:var(--m);text-align:center;padding:12px 0}
table{width:100%;border-collapse:collapse}
th{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--m);padding:5px 8px;text-align:left;border-bottom:1px solid var(--b)}
td{font-size:11px;padding:6px 8px;border-bottom:1px solid rgba(48,54,61,.4);font-family:monospace;vertical-align:middle}
tr:last-child td{border-bottom:none}
.badge{display:inline-flex;align-items:center;gap:3px;padding:2px 8px;border-radius:99px;font-size:10px;font-weight:700}
.tc-cell{max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--t)}
.ds-grid{display:flex;flex-direction:column;gap:5px}
.ds-row{display:flex;justify-content:space-between;align-items:center;font-size:11px;font-family:monospace}
.ds-key{color:var(--m)}.ds-val{color:var(--t);font-weight:700}
.ds-pct-row{display:grid;grid-template-columns:70px 1fr 36px;align-items:center;gap:6px;font-size:10px}
</style>
</head>
<body>

<div id="login">
  <div class="login-bg"></div>
  <div class="grid-ov"></div>

  <div class="glow" style="width:400px;height:400px;background:rgba(29,155,240,0.12);top:-100px;left:-100px"></div>
  <div class="glow" style="width:300px;height:300px;background:rgba(29,155,240,0.08);bottom:-60px;right:-80px"></div>
  <div class="glow" style="width:200px;height:200px;background:rgba(29,155,240,0.06);top:40%;right:15%"></div>

  <div class="tweet-cards" id="tweet-cards"></div>

  <div class="tw-pattern" id="tw-pattern"></div>

  <div class="hashtag-strip">
    <span class="htag">#Joy</span><span class="htag">#Anger</span>
    <span class="htag">#Fear</span><span class="htag">#Sadness</span>
    <span class="htag">#EmotionAnalysis</span><span class="htag">#NLP</span>
    <span class="htag">#Twitter</span><span class="htag">#Kaggle</span>
    <span class="htag">#MachineLearning</span><span class="htag">#FastAPI</span>
    <span class="htag">#NLTK</span><span class="htag">#Python</span>
  </div>

  <div class="box">
    <div class="twitter-logo">X</div>
    <h2>Twitter Emotion Analyser</h2>
    <p>Pandas &middot; NLTK &middot; NRCLex &middot; HuggingFace &middot; FastAPI<br/>Kaggle: Twitter Emotion Intensity Dataset</p>

    <label>Username</label>
    <input type="text" id="un" placeholder="Enter username" autocomplete="off"/>
    <label>Password</label>
    <input type="password" id="pw" placeholder="Enter password"/>
    <button class="btn" onclick="doLogin()">Sign in &rarr;</button>
    <div class="err" id="er"></div>

    <div class="divider"></div>
    <div class="emotion-pills">
      <span class="epill" style="background:#FFD70020;color:#FFD700">Joy</span>
      <span class="epill" style="background:#FF413620;color:#FF4136">Anger</span>
      <span class="epill" style="background:#9B59B620;color:#9B59B6">Fear</span>
      <span class="epill" style="background:#3498DB20;color:#3498DB">Sadness</span>
      <span class="epill" style="background:#FF6B3520;color:#FF6B35">Surprise</span>
      <span class="epill" style="background:#2ECC7120;color:#2ECC71">Disgust</span>
      <span class="epill" style="background:#F39C1220;color:#F39C12">Anticipation</span>
      <span class="epill" style="background:#1ABC9C20;color:#1ABC9C">Trust</span>
    </div>
  </div>
</div>

<div id="app">
  <div class="nav">
    <div class="nav-brand">
      <div class="nav-logo">X</div>
      <h1>Twitter Emotion Analyser</h1>
    </div>
    <div>
      <span id="hu" style="font-size:12px;color:var(--m);font-family:monospace"></span>
      <button class="out-btn" onclick="doLogout()">Sign out</button>
    </div>
  </div>

  <div class="main">

    <div class="card span3">
      <div class="lbl">Analyse Tweet</div>
      <textarea id="tb" placeholder="Type a tweet here... e.g. I am so happy and excited today!" oninput="uc()"></textarea>
      <div class="row">
        <button class="btn" id="ab" style="width:auto;padding:9px 24px;border-radius:99px" onclick="doAnalyse()">Analyse Emotion</button>
        <button class="sbtn" onclick="document.getElementById('tb').value='';uc()">Clear</button>
        <span class="cc" id="cc">0 / 280</span>
      </div>
    </div>

    <div class="card">
      <div class="lbl">Dominant Emotion</div>
      <div class="dw" id="dw"><div class="ph">Enter a tweet<br/>to detect emotions</div></div>
    </div>

    <div class="card">
      <div class="lbl">Emotion Scores</div>
      <div class="bars" id="bars"><div class="ph" style="text-align:center;padding:12px 0">Awaiting analysis...</div></div>
    </div>

    <div class="card">
      <div class="lbl">Radar Chart</div>
      <div class="cw"><canvas id="rc"></canvas></div>
    </div>

    <div class="card span2">
      <div class="lbl">Bar Chart</div>
      <canvas id="bc" height="110"></canvas>
    </div>

    <div class="card">
      <div class="lbl">Pie Chart</div>
      <div class="cw"><canvas id="pc"></canvas></div>
    </div>

    <div class="card span2">
      <div class="lbl">NLTK Text Preprocessing Pipeline</div>
      <div class="sublbl">1. Tokenization — split into words</div>
      <div class="nlp-box" id="nlp-tokens">—</div>
      <div class="sublbl">2. Stopword Removal — remove "is", "the", "am"...</div>
      <div class="nlp-box" id="nlp-nostop">—</div>
      <div class="sublbl">3. Lemmatization — reduce to base form</div>
      <div class="nlp-box" id="nlp-lemma">—</div>
    </div>

    <div class="card">
      <div class="lbl">Pandas Analytics</div>
      <div class="stat-grid" id="stats">
        <div class="stat"><div class="stat-val">0</div><div class="stat-key">Total Analysed</div></div>
        <div class="stat"><div class="stat-val">—</div><div class="stat-key">Most Common</div></div>
        <div class="stat"><div class="stat-val">—</div><div class="stat-key">Highest Avg</div></div>
        <div class="stat"><div class="stat-val">—</div><div class="stat-key">Least Common</div></div>
      </div>
    </div>

    <div class="card span2">
      <div class="lbl">Live Emotion % Dashboard</div>
      <div class="pct-bars" id="pct-bars"><div class="ph" style="text-align:center;padding:10px 0">No data yet...</div></div>
    </div>

    <div class="card">
      <div class="lbl">Emotion Trend (Live)</div>
      <canvas id="tc" height="150"></canvas>
    </div>

    <div class="card span2">
      <div class="lbl">Kaggle Dataset — Similar Tweets by Intensity</div>
      <div class="sublbl" id="sim-label">Run an analysis to see matching tweets from the Kaggle dataset</div>
      <div id="similar"><div class="ph" style="padding:8px 0;font-size:11px">Place dataset.csv in the same folder as app.py</div></div>
    </div>

    <div class="card">
      <div class="lbl">Kaggle Dataset Stats</div>
      <div id="ds-info"><div class="ph" style="padding:4px 0">Loading...</div></div>
    </div>

    <div class="card span3">
      <div class="lbl">Analysis History — Pandas DataFrame View</div>
      <div id="hw"><div class="he">No analyses yet.</div></div>
    </div>

  </div>
</div>

<script>
const CREDS={"Mithunkrishna R":"1833"};
const META={
  joy:{label:"Joy",color:"#FFD700"},anger:{label:"Anger",color:"#FF4136"},
  fear:{label:"Fear",color:"#9B59B6"},sadness:{label:"Sadness",color:"#3498DB"},
  surprise:{label:"Surprise",color:"#FF6B35"},disgust:{label:"Disgust",color:"#2ECC71"},
  anticipation:{label:"Anticipation",color:"#F39C12"},trust:{label:"Trust",color:"#1ABC9C"}
};
const KEYS=Object.keys(META);
const TW="#1d9bf0";
let radarChart=null,barChart=null,pieChart=null,trendChart=null,hist=[];

(function(){
  const tc=document.getElementById("tweet-cards");
  const fakeUsers=["@user_joy","@tweet_mood","@emo_data","@nlp_bot","@kaggle_ai","@sentiment_x"];
  const fakeTweets=[
    "Feeling so happy and grateful today! #Joy",
    "This news makes me so angry! #Anger",
    "I'm really scared about the results #Fear",
    "Missing those good old days... #Sadness",
    "Wow I never expected this! #Surprise",
    "Can't believe what just happened #Emotion",
    "So excited for what comes next #Anticipation",
    "Trust the process, always #Trust",
  ];
  const rotations=["-6deg","4deg","-3deg","7deg","-5deg","3deg"];
  for(let i=0;i<7;i++){
    const card=document.createElement("div");
    card.className="tc-card";
    card.style.cssText=`left:${5+Math.random()*80}%;--rot:${rotations[i%rotations.length]};`+
      `animation-duration:${18+Math.random()*14}s;animation-delay:${Math.random()*16}s`;
    card.innerHTML=
      '<div class="tc-card-header">'+
        '<div class="tc-avatar"></div>'+
        '<span class="tc-name">'+fakeUsers[i%fakeUsers.length]+'</span>'+
      '</div>'+
      '<div class="tc-text">'+fakeTweets[i%fakeTweets.length]+'</div>'+
      '<div class="tc-footer">'+
        '<span class="tc-icon">&#9825; '+Math.floor(Math.random()*200)+'</span>'+
        '<span class="tc-icon">&#8635; '+Math.floor(Math.random()*80)+'</span>'+
      '</div>';
    tc.appendChild(card);
  }

  const tp=document.getElementById("tw-pattern");
  for(let i=0;i<20;i++){
    const b=document.createElement("div");
    b.className="tw-bird";
    b.textContent="✦";
    b.style.cssText=`left:${Math.random()*98}%;font-size:${8+Math.random()*18}px;`+
      `animation-duration:${12+Math.random()*18}s;animation-delay:${Math.random()*18}s`;
    tp.appendChild(b);
  }
})();

document.addEventListener("keydown",e=>{
  if(e.key==="Enter"&&document.getElementById("login").style.display!=="none")doLogin();
});

function doLogin(){
  const u=document.getElementById("un").value.trim(),p=document.getElementById("pw").value;
  if(CREDS[u]&&CREDS[u]===p){
    document.getElementById("login").style.display="none";
    document.getElementById("app").style.display="block";
    document.getElementById("hu").textContent=u;
    initCharts();loadDatasetInfo();
  }else{
    const el=document.getElementById("er");
    el.textContent="Invalid username or password.";
    setTimeout(()=>el.textContent="",3000);
  }
}

function doLogout(){
  document.getElementById("app").style.display="none";
  document.getElementById("login").style.display="flex";
  document.getElementById("un").value="";
  document.getElementById("pw").value="";
}

async function loadDatasetInfo(){
  try{
    const res=await fetch("/dataset-stats");
    const raw=await res.text();
    let d;try{d=JSON.parse(raw);}catch(e){return;}
    const el=document.getElementById("ds-info");
    if(!d.loaded){
      el.innerHTML='<div class="ph" style="font-size:11px;padding:4px 0">'+
        'Place <span style="color:'+TW+'">dataset.csv</span> in the same folder as app.py<br/><br/>'+
        'Download: <span style="color:'+TW+'">Kaggle → Twitter Emotion Intensity Dataset</span></div>';
      return;
    }
    const topDist=Object.entries(d.distribution||{}).sort((a,b)=>b[1]-a[1]);
    let html='<div class="ds-grid">'+
      '<div class="ds-row"><span class="ds-key">Rows</span><span class="ds-val">'+d.total.toLocaleString()+'</span></div>'+
      '<div class="ds-row"><span class="ds-key">Emotions</span><span class="ds-val">'+d.emotions.join(", ")+'</span></div>'+
      '<div class="sublbl" style="margin-top:6px">Distribution</div>';
    topDist.forEach(([k,v])=>{
      const pct=d.percentage[k]||0;
      const col=META[k]?META[k].color:TW;
      const intensity=d.avg_intensity&&d.avg_intensity[k]?d.avg_intensity[k]:0;
      html+='<div class="ds-pct-row">'+
        '<span style="color:'+col+';font-weight:700">'+k+'</span>'+
        '<div style="height:5px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden">'+
          '<div style="height:100%;width:'+pct+'%;background:'+col+';border-radius:99px;transition:width 1s"></div></div>'+
        '<span style="color:var(--m)">'+pct+'%</span></div>'+
        '<div style="font-size:9px;color:var(--m);font-family:monospace;margin-bottom:3px;padding-left:0">'+
          'Avg intensity: <span style="color:'+col+'">'+intensity+'</span> &nbsp;|&nbsp; Count: '+v+'</div>';
    });
    html+='</div>';
    el.innerHTML=html;
  }catch(e){}
}

function initCharts(){
  const base={responsive:true,plugins:{legend:{display:false}},animation:{duration:700}};
  radarChart=new Chart(document.getElementById("rc"),{type:"radar",data:{
    labels:KEYS.map(k=>META[k].label),
    datasets:[{data:new Array(8).fill(0),backgroundColor:"rgba(29,155,240,0.1)",
      borderColor:TW,pointBackgroundColor:KEYS.map(k=>META[k].color),
      pointBorderColor:"#fff",pointRadius:4,borderWidth:2}]},
    options:{...base,scales:{r:{min:0,max:1,ticks:{display:false},
      grid:{color:"rgba(255,255,255,0.06)"},angleLines:{color:"rgba(255,255,255,0.06)"},
      pointLabels:{color:"#8b949e",font:{size:9,weight:"600"}}}}}});

  barChart=new Chart(document.getElementById("bc"),{type:"bar",data:{
    labels:KEYS.map(k=>META[k].label),
    datasets:[{data:new Array(8).fill(0),backgroundColor:KEYS.map(k=>META[k].color+"cc"),
      borderColor:KEYS.map(k=>META[k].color),borderWidth:1,borderRadius:4}]},
    options:{...base,scales:{
      x:{ticks:{color:"#8b949e",font:{size:10}},grid:{color:"rgba(255,255,255,0.04)"}},
      y:{min:0,max:1,ticks:{color:"#8b949e",font:{size:10}},grid:{color:"rgba(255,255,255,0.04)"}}}}});

  pieChart=new Chart(document.getElementById("pc"),{type:"doughnut",data:{
    labels:KEYS.map(k=>META[k].label),
    datasets:[{data:new Array(8).fill(0),backgroundColor:KEYS.map(k=>META[k].color+"cc"),
      borderColor:"#1c2128",borderWidth:2}]},
    options:{...base,plugins:{legend:{display:true,position:"bottom",
      labels:{color:"#8b949e",font:{size:9},boxWidth:10}}}}});

  trendChart=new Chart(document.getElementById("tc"),{type:"line",data:{labels:[],
    datasets:KEYS.map(k=>({label:META[k].label,data:[],borderColor:META[k].color,
      backgroundColor:"transparent",borderWidth:2,pointRadius:2,tension:.4}))},
    options:{...base,scales:{
      x:{ticks:{color:"#8b949e",font:{size:8}},grid:{color:"rgba(255,255,255,0.04)"}},
      y:{min:0,max:1,ticks:{color:"#8b949e",font:{size:8}},grid:{color:"rgba(255,255,255,0.04)"}}},
      plugins:{legend:{display:true,position:"bottom",
        labels:{color:"#8b949e",font:{size:8},boxWidth:8}}}}});
}

function uc(){
  const n=document.getElementById("tb").value.length;
  const el=document.getElementById("cc");
  el.textContent=n+" / 280";
  el.style.color=n>260?"#f85149":"#8b949e";
}

async function doAnalyse(){
  const text=document.getElementById("tb").value.trim();
  if(!text)return;
  const btn=document.getElementById("ab");
  btn.disabled=true;btn.textContent="Analysing...";
  try{
    const res=await fetch("/analyse",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})});
    const raw=await res.text();
    let d;
    try{d=JSON.parse(raw);}
    catch(e){alert("Server error: unexpected response. Please try again.");return;}
    if(!res.ok){alert("Error: "+(d.error||d.detail||"Something went wrong"));return;}
    renderAll(d,text);
  }catch(e){alert("Network error: "+e.message);}
  finally{btn.disabled=false;btn.textContent="Analyse Emotion";}
}

function renderAll(d,text){
  const m=META[d.dominant]||{label:d.dominant,color:"#888"};

  document.getElementById("dw").innerHTML=
    '<div style="width:54px;height:54px;border-radius:50%;background:'+m.color+'1a;border:2px solid '+m.color+'40;display:flex;align-items:center;justify-content:center;margin:0 auto 10px;font-size:15px;font-weight:700;color:'+m.color+'">'+m.label.slice(0,2).toUpperCase()+'</div>'+
    '<div class="dl" style="color:'+m.color+'">'+m.label.toUpperCase()+'</div>'+
    '<div class="ds">dominant emotion detected</div>';

  const sorted=Object.entries(d.scores).sort((a,b)=>b[1]-a[1]);
  let html="";
  for(const[k,v]of sorted){
    const em=META[k];
    html+='<div class="br">'+
      '<div class="bn" style="color:'+em.color+'">'+em.label+'</div>'+
      '<div class="bt"><div class="bf" style="background:'+em.color+';width:0%" data-w="'+Math.round(v*100)+'%"></div></div>'+
      '<div class="bv">'+v.toFixed(3)+'</div></div>';
  }
  document.getElementById("bars").innerHTML=html;
  requestAnimationFrame(()=>document.querySelectorAll(".bf").forEach(el=>el.style.width=el.dataset.w));

  const vals=KEYS.map(k=>d.scores[k]||0);
  radarChart.data.datasets[0].data=vals;radarChart.update();
  barChart.data.datasets[0].data=vals;barChart.update();
  pieChart.data.datasets[0].data=vals;pieChart.update();

  const t=new Date().toLocaleTimeString();
  trendChart.data.labels.push(t);
  KEYS.forEach((k,i)=>trendChart.data.datasets[i].data.push(d.scores[k]||0));
  if(trendChart.data.labels.length>10){
    trendChart.data.labels.shift();
    trendChart.data.datasets.forEach(ds=>ds.data.shift());
  }
  trendChart.update();

  if(d.nlp){
    document.getElementById("nlp-tokens").innerHTML=d.nlp.tokens.map(w=>'<span>'+w+'</span>').join(', ')+'...';
    document.getElementById("nlp-nostop").innerHTML=d.nlp.no_stop.map(w=>'<span>'+w+'</span>').join(', ')+'...';
    document.getElementById("nlp-lemma").innerHTML=d.nlp.lemmatized.map(w=>'<span>'+w+'</span>').join(', ')+'...';
  }

  if(d.similar&&d.similar.length){
    document.getElementById("sim-label").textContent=
      "Top "+d.similar.length+" tweets from Kaggle dataset matching '"+m.label+"' (sorted by intensity):";
    document.getElementById("similar").innerHTML=d.similar.map(s=>{
      const em=META[s.emotion]||{label:s.emotion,color:TW};
      const iw=Math.round(s.intensity*100);
      return '<div class="sim-card">'+
        '<div class="sim-tweet">'+s.tweet+'</div>'+
        '<div class="sim-meta">'+
          '<span class="sim-badge" style="background:'+em.color+'1a;color:'+em.color+'">'+em.label+'</span>'+
          '<div class="intensity-bar">'+
            '<div class="ibt"><div class="ibf" style="width:'+iw+'%;background:'+em.color+'"></div></div>'+
            '<span class="sim-intensity">Intensity: '+s.intensity+'</span>'+
          '</div>'+
        '</div></div>';
    }).join('');
  }else{
    document.getElementById("sim-label").textContent="No matching tweets found in Kaggle dataset";
    document.getElementById("similar").innerHTML='';
  }

  if(d.analytics&&d.analytics.total){
    const a=d.analytics;
    const mc=Object.entries(a.most_common||{}).sort((x,y)=>y[1]-x[1]);
    const topAvg=Object.entries(a.avg_scores||{}).sort((x,y)=>y[1]-x[1]);
    const lc=mc[mc.length-1];
    document.getElementById("stats").innerHTML=
      sv(a.total,"Total Analysed")+sv(mc[0]?mc[0][0]:"—","Most Common")+
      sv(topAvg[0]?topAvg[0][0]:"—","Highest Avg")+sv(lc&&lc[1]>0?lc[0]:"—","Least Common");

    if(a.emotion_pct){
      document.getElementById("pct-bars").innerHTML=
        Object.entries(a.emotion_pct).sort((x,y)=>y[1]-x[1]).map(([k,v])=>{
          const em=META[k];
          return '<div class="pct-row">'+
            '<div class="pct-name" style="color:'+em.color+'">'+em.label+'</div>'+
            '<div class="pct-track"><div class="pct-fill" style="background:'+em.color+';width:'+v+'%"></div></div>'+
            '<div class="pct-val">'+v+'%</div></div>';
        }).join('');
    }
  }

  hist.unshift({text,dominant:d.dominant,meta:m,time:t,scores:d.scores});
  if(hist.length>15)hist.pop();
  document.getElementById("hw").innerHTML=
    '<table><thead><tr><th>Time</th><th>Tweet</th><th>Dominant</th>'+
    KEYS.map(k=>'<th>'+META[k].label.slice(0,3)+'</th>').join('')+
    '</tr></thead><tbody>'+
    hist.map(h=>'<tr><td style="color:var(--m);width:68px">'+h.time+'</td>'+
      '<td class="tc-cell">'+h.text+'</td>'+
      '<td><span class="badge" style="background:'+h.meta.color+'1a;color:'+h.meta.color+'">'+h.meta.label+'</span></td>'+
      KEYS.map(k=>'<td style="color:var(--m)">'+h.scores[k].toFixed(2)+'</td>').join('')+
      '</tr>').join('')+'</tbody></table>';
}

function sv(val,key){
  return '<div class="stat"><div class="stat-val">'+val+'</div><div class="stat-key">'+key+'</div></div>';
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def serve():
    return HTML

if __name__ == "__main__":
    uvicorn.run("__main__:app", host="127.0.0.1", port=8001, reload=False)