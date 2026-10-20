"""
Jalankan lokal:   uvicorn server:app --port 8000
Start di Render:  uvicorn server:app --host 0.0.0.0 --port $PORT

Endpoint:
  GET /                  -> cek server hidup
  GET /chart?sym=BBCA.JK -> candle harian 5 tahun [{t,o,h,l,c,v}, ...]
  GET /info?sym=BBCA.JK  -> info perusahaan dan fundamental
"""
import re
import time

import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MyStok API")

# Data harga bersifat publik, jadi semua origin boleh akses.
# Kalau mau lebih ketat, ganti ["*"] dengan URL web lo, mis. ["https://nama-lo.github.io"].

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

CHART_TTL = 300   # detik: cache harga 5 menit
INFO_TTL = 3600   # detik: cache info perusahaan 1 jam
STALE_MAX = 86400 # detik: kalau Yahoo error, boleh pakai cache lama sampai 24 jam

_cache: dict = {}
SYM_RE = re.compile(r"^[A-Za-z0-9.^=-]{1,15}$")


def _clean(sym: str) -> str:
    sym = sym.strip().upper()
    if not SYM_RE.match(sym):
        raise HTTPException(400, "Simbol tidak valid")
    return sym


def _cached(key: str, ttl: int, loader):
    """Ambil dari cache kalau masih segar. Kalau loader gagal, pakai cache lama (stale)."""
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


@app.get("/")
def root():
    return {"ok": True, "service": "StockLines API"}


@app.get("/chart")
def chart(sym: str):
    sym = _clean(sym)

    def load():
        df = yf.Ticker(sym).history(period="5y")  # sama seperti ai-stock.py
        df = df.dropna(subset=["Close"])
        if df.empty:
            raise HTTPException(404, f"Data {sym} tidak ditemukan")
        us = not sym.endswith(".JK")
        nd = 2 if us else 0
        return {
            "sym": sym,
            "candles": [
                {
                    "t": ts.strftime("%Y-%m-%d"),
                    "o": round(float(r.Open), nd),
                    "h": round(float(r.High), nd),
                    "l": round(float(r.Low), nd),
                    "c": round(float(r.Close), nd),
                    "v": int(r.Volume) if r.Volume == r.Volume else 0,
                }
                for ts, r in df.iterrows()
            ],
        }

    return _cached("chart:" + sym, CHART_TTL, load)


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
