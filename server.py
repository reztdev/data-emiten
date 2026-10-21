"""
Jalankan lokal:   uvicorn server:app --port 8000
Start di Render:  uvicorn server:app --host 0.0.0.0 --port $PORT

Endpoint:
  GET /                     cek server hidup
  GET /chart?sym=BBCA.JK    candle harian 5 tahun [{t,o,h,l,c,v}, ...]
  GET /analyze?sym=BBCA.JK  analisis ML: prediksi 5 hari, peluang naik, pola mirip, keputusan
  GET /info?sym=BBCA.JK     info perusahaan dan fundamental
"""
import re
import time

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="MyStock API")

# Data harga bersifat publik. Untuk lebih ketat, ganti ["*"] dengan URL web lo.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

CHART_TTL, INFO_TTL, AN_TTL, STALE_MAX = 300, 3600, 900, 86400
HORIZON = 5  # prediksi return 5 hari ke depan, sama seperti ai-stock.py
_cache: dict = {}
SYM_RE = re.compile(r"^[A-Za-z0-9.^=-]{1,15}$")


def _clean(sym: str) -> str:
    sym = sym.strip().upper()
    if not SYM_RE.match(sym):
        raise HTTPException(400, "Simbol tidak valid")
    return sym


def _cached(key, ttl, loader):
    """Pakai cache kalau masih segar. Kalau Yahoo/loader error, pakai cache lama (maks 24 jam)."""
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        val = loader()
    except HTTPException:
        raise
    except Exception as e:
        if hit and now - hit[0] < STALE_MAX:
            return hit[1]
        raise HTTPException(502, f"Gagal mengambil data dari Yahoo Finance: {e}")
    _cache[key] = (now, val)
    return val


def _history(sym: str) -> pd.DataFrame:
    def load():
        df = yf.Ticker(sym).history(period="5y").dropna(subset=["Close"])  # sama seperti ai-stock.py
        if df.empty:
            raise HTTPException(404, f"Data {sym} tidak ditemukan")
        return df
    return _cached("df:" + sym, CHART_TTL, load)


# ------------------------------------------------------------------ fitur & model
FEATURES = {
    "ret_1": "Return 1 hari", "ret_2": "Return 2 hari lalu", "ret_3": "Return 3 hari lalu",
    "ret_5": "Return 5 hari", "ret_10": "Return 10 hari", "ret_20": "Return 20 hari",
    "sma20": "Harga vs SMA20", "sma50": "Harga vs SMA50",
    "vol10": "Volatilitas 10h", "vol20": "Volatilitas 20h",
    "rsi": "RSI", "macd": "MACD", "atr": "ATR", "volr": "Rasio volume",
    "bb": "Posisi Bollinger", "stoch": "Stochastic", "range": "Rentang harian",
}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur skala-bebas (persentase), jadi pola antar-waktu dan antar-emiten tetap sebanding."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"].astype(float)
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    f["ret_1"], f["ret_2"], f["ret_3"] = r, r.shift(1), r.shift(2)
    f["ret_5"], f["ret_10"], f["ret_20"] = c.pct_change(5), c.pct_change(10), c.pct_change(20)
    f["sma20"] = c / c.rolling(20).mean() - 1
    f["sma50"] = c / c.rolling(50).mean() - 1
    f["vol10"], f["vol20"] = r.rolling(10).std(), r.rolling(20).std()
    d = c.diff()
    rs = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean()
    f["rsi"] = (100 - 100 / (1 + rs)) / 100
    f["macd"] = (c.ewm(span=12).mean() - c.ewm(span=26).mean()) / c
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean() / c
    f["volr"] = v / v.rolling(20).mean()
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    f["bb"] = (c - (m - 2 * s)) / (4 * s)
    f["stoch"] = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min())
    f["range"] = (h - l) / c
    return f.replace([np.inf, -np.inf], np.nan)


def analyze_df(df: pd.DataFrame) -> dict:
    X = make_features(df).dropna()
    y = (df["Close"].shift(-HORIZON) / df["Close"] - 1).reindex(X.index)
    L = X[y.notna()]
    yl = y[y.notna()]
    n = len(L)
    if n < 200:
        raise HTTPException(422, "Data belum cukup untuk analisis (butuh minimal sekitar 1 tahun)")

    reg = lambda: RandomForestRegressor(n_estimators=120, max_depth=7, min_samples_leaf=8,
                                        random_state=42, n_jobs=1)
    clf = lambda: RandomForestClassifier(n_estimators=120, max_depth=7, min_samples_leaf=8,
                                         random_state=42, n_jobs=1)

    # --- uji jujur: urut waktu (tanpa acak), 80% awal latih, 20% akhir uji, jeda 5 hari
    cut = int(n * 0.8)
    Xtr, ytr, Xte, yte = L.iloc[: cut - HORIZON], yl.iloc[: cut - HORIZON], L.iloc[cut:], yl.iloc[cut:]
    r2 = float(r2_score(yte, reg().fit(Xtr, ytr).predict(Xte)))
    acc = float(clf().fit(Xtr, ytr > 0).score(Xte, yte > 0))
    base = float(max((yte > 0).mean(), 1 - (yte > 0).mean()))

    # --- model final: dilatih pada semua data berlabel, prediksi pakai fitur HARI TERAKHIR
    now = X.iloc[[-1]]
    rf = reg().fit(L, yl)
    ret = float(rf.predict(now)[0])
    band = float(np.std([t.predict(now.values)[0] for t in rf.estimators_]))
    prob_up = float(clf().fit(L, yl > 0).predict_proba(now)[0][-1])
    imp = sorted(zip(rf.feature_importances_, L.columns), reverse=True)[:6]

    # --- pola: cari hari-hari bersejarah yang kondisinya paling mirip dengan hari ini
    sc = StandardScaler().fit(L.iloc[:-10])
    Z = sc.transform(L.iloc[:-10])
    k = 25
    dist, idx = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(sc.transform(now))
    an = yl.iloc[:-10].iloc[idx[0]]
    up_share, med = float((an > 0).mean()), float(an.median())
    dates = [L.index[:-10][i].strftime("%Y-%m-%d") for i in idx[0][:3]]

    # --- keputusan: gabungan 3 suara (Random Forest, peluang naik, pola mirip)
    votes = [
        1 if ret > 0.005 else -1 if ret < -0.005 else 0,
        1 if prob_up > 0.55 else -1 if prob_up < 0.45 else 0,
        1 if up_share > 0.6 else -1 if up_share < 0.4 else 0,
    ]
    score = int(sum(votes))
    verdict = "bullish" if score >= 2 else "bearish" if score <= -2 else "netral"
    edge = acc - base
    quality = "tinggi" if (r2 > 0.05 and edge > 0.05) else "sedang" if (r2 > 0 or edge > 0.02) else "rendah"

    price = float(df["Close"].iloc[-1])
    pc = lambda x: f"{x * 100:+.1f}%"
    summary = [
        f"Random Forest memperkirakan return {HORIZON} hari ke depan {pc(ret)} (rentang ketidakpastian ±{band * 100:.1f}%).",
        f"Dari {k} kondisi historis yang paling mirip dengan hari ini, {up_share * 100:.0f}% berakhir naik dalam {HORIZON} hari (median {pc(med)}).",
        "Faktor paling berpengaruh: " + ", ".join(FEATURES[c] for _, c in imp[:3]) + ".",
        (f"Akurasi arah di data uji {acc * 100:.0f}% berbanding {base * 100:.0f}% jika selalu menebak arah yang paling sering. "
         + ("Sinyal lemah, perlakukan sebagai petunjuk saja." if quality == "rendah"
            else "Ada sedikit keunggulan dari tebakan biasa." if quality == "sedang"
            else "Ada keunggulan yang cukup berarti dari tebakan biasa.")),
    ]
    return {
        "sym": None, "asof": X.index[-1].strftime("%Y-%m-%d"), "price": price, "horizon": HORIZON,
        "ret": ret, "band": band, "price_pred": price * (1 + ret), "prob_up": prob_up,
        "holdout": {"r2": r2, "acc": acc, "base": base, "n_test": int(len(yte))},
        "importance": [[FEATURES[c], float(v)] for v, c in imp],
        "analogs": {"k": k, "up_share": up_share, "median": med, "dates": dates},
        "verdict": verdict, "score": score, "quality": quality, "summary": summary,
    }


# ------------------------------------------------------------------ endpoint
@app.get("/")
def root():
    return {"ok": True, "service": "StockLines API"}


@app.get("/chart")
def chart(sym: str):
    sym = _clean(sym)
    df = _history(sym)
    nd = 0 if sym.endswith(".JK") else 2
    return {
        "sym": sym,
        "candles": [
            {"t": ts.strftime("%Y-%m-%d"), "o": round(float(r.Open), nd), "h": round(float(r.High), nd),
             "l": round(float(r.Low), nd), "c": round(float(r.Close), nd),
             "v": int(r.Volume) if r.Volume == r.Volume else 0}
            for ts, r in df.iterrows()
        ],
    }


@app.get("/analyze")
def analyze(sym: str):
    sym = _clean(sym)
    def load():
        out = analyze_df(_history(sym))
        out["sym"] = sym
        return out
    return _cached("an:" + sym, AN_TTL, load)


@app.get("/info")
def info(sym: str):
    sym = _clean(sym)
    def load():
        i = yf.Ticker(sym).info or {}
        keys = ["longName", "sector", "industry", "country", "website", "fullTimeEmployees",
                "marketCap", "trailingPE", "forwardPE", "pegRatio", "priceToBook",
                "dividendYield", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]
        return {"sym": sym, **{k: i.get(k) for k in keys}}
    return _cached("info:" + sym, INFO_TTL, load)
