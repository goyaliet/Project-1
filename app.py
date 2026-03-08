import json
import csv
import io
import threading
from datetime import datetime
import numpy as np
import yfinance as yf
from flask import Flask, render_template, jsonify, Response, request
from scanner import (
    scan_stream, load_cached_results, save_cache,
    calculate_rsi, DEFAULT_FILTERS,
)

app = Flask(__name__)

# ── Global state ───────────────────────────────────────────────────────────────
_scan_lock   = threading.Lock()
_scan_active = False          # only one scan at a time
_scan_stop   = False          # stop signal

_cache = {                    # last completed scan results
    "results":       [],
    "total_scanned": 0,
    "last_scan":     None,
    "elapsed_sec":   None,
}

# Load previous results on startup
_prev = load_cached_results()
if _prev:
    _cache.update({
        "results":       _prev.get("results",       []),
        "total_scanned": _prev.get("total_scanned", 0),
        "last_scan":     _prev.get("timestamp"),
        "elapsed_sec":   _prev.get("elapsed_sec"),
    })


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/defaults")
def get_defaults():
    return jsonify(DEFAULT_FILTERS)


@app.route("/api/cached")
def get_cached():
    return jsonify(_cache)


# ── SSE stream ─────────────────────────────────────────────────────────────────
@app.route("/api/stream")
def stream():
    global _scan_active

    with _scan_lock:
        if _scan_active:
            def busy():
                yield f"data: {json.dumps({'type':'error','message':'A scan is already running.'})}\n\n"
            return Response(busy(), mimetype="text/event-stream")
        _scan_active = True
        global _scan_stop
        _scan_stop = False

    # Read filters from query params
    def fp(key, cast=float):
        val = request.args.get(key)
        return cast(val) if val is not None else DEFAULT_FILTERS[key]

    def b(key):  # boolean param
        v = request.args.get(key)
        return str(v).lower() in ("1", "true") if v is not None else DEFAULT_FILTERS.get(key, False)

    filters = {
        # Technicals
        "rsi_min":               fp("rsi_min",               int),
        "rsi_max":               fp("rsi_max",               int),
        "rs_min":                fp("rs_min",                float),
        "rs_max":                fp("rs_max",                float),
        "pct_52w_high_min":      fp("pct_52w_high_min",      float),
        "pct_52w_high_max":      fp("pct_52w_high_max",      float),
        "pct_52w_low_enabled":   b("pct_52w_low_enabled"),
        "pct_52w_low_min":       fp("pct_52w_low_min",       float),
        "pct_52w_low_max":       fp("pct_52w_low_max",       float),
        "price_enabled":         b("price_enabled"),
        "price_min":             fp("price_min",             float),
        "price_max":             fp("price_max",             float),
        "price_vs_ma_enabled":   b("price_vs_ma_enabled"),
        "price_vs_ma_dir":       request.args.get("price_vs_ma_dir",    DEFAULT_FILTERS["price_vs_ma_dir"]),
        "price_vs_ma_period":    fp("price_vs_ma_period",    int),
        "ma_order_enabled":      b("ma_order_enabled"),
        "ma_order_fast":         fp("ma_order_fast",         int),
        "ma_order_mid":          fp("ma_order_mid",          int),
        "ma_order_slow":         fp("ma_order_slow",         int),
        "avg_rupee_vol_enabled": b("avg_rupee_vol_enabled"),
        "avg_rupee_vol_min":     fp("avg_rupee_vol_min",     float),
        "ma_adr_enabled":        b("ma_adr_enabled"),
        "ma_adr_days":           fp("ma_adr_days",           int),
        "ma_adr_min":            fp("ma_adr_min",            float),
        # Fundamentals
        "market_cap_min_cr":  fp("market_cap_min_cr",  float),
        "market_cap_max_cr":  fp("market_cap_max_cr",  float),
        "min_avg_volume":     fp("min_avg_volume",      int),
        "pe_enabled":         b("pe_enabled"),
        "pe_min":             fp("pe_min",              float),
        "pe_max":             fp("pe_max",              float),
        "roe_enabled":        b("roe_enabled"),
        "roe_min":            fp("roe_min",             float),
        "roe_max":            fp("roe_max",             float),
        "debt_eq_enabled":    b("debt_eq_enabled"),
        "debt_eq_min":        fp("debt_eq_min",         float),
        "debt_eq_max":        fp("debt_eq_max",         float),
        "eps_growth_enabled": b("eps_growth_enabled"),
        "eps_growth_min":     fp("eps_growth_min",      float),
        "eps_growth_max":     fp("eps_growth_max",      float),
        "rev_growth_enabled": b("rev_growth_enabled"),
        "rev_growth_min":     fp("rev_growth_min",      float),
        "rev_growth_max":     fp("rev_growth_max",      float),
        "free_float_enabled": b("free_float_enabled"),
        "free_float_min":     fp("free_float_min",      float),
        "free_float_max":     fp("free_float_max",      float),
        "promoter_enabled":   b("promoter_enabled"),
        "promoter_min":       fp("promoter_min",   float),
        "promoter_max":       fp("promoter_max",   float),
        # Quarterly growth
        "qoq_eps_enabled":    b("qoq_eps_enabled"),   "qoq_eps_min":   fp("qoq_eps_min",   float),
        "yoy_eps_enabled":    b("yoy_eps_enabled"),   "yoy_eps_min":   fp("yoy_eps_min",   float),
        "qoq_np_enabled":     b("qoq_np_enabled"),    "qoq_np_min":    fp("qoq_np_min",    float),
        "yoy_np_enabled":     b("yoy_np_enabled"),    "yoy_np_min":    fp("yoy_np_min",    float),
        "qoq_sales_enabled":  b("qoq_sales_enabled"), "qoq_sales_min": fp("qoq_sales_min", float),
        "yoy_sales_enabled":  b("yoy_sales_enabled"), "yoy_sales_min": fp("yoy_sales_min", float),
        "qoq_opm_enabled":    b("qoq_opm_enabled"),   "qoq_opm_min":   fp("qoq_opm_min",   float),
        "yoy_opm_enabled":    b("yoy_opm_enabled"),   "yoy_opm_min":   fp("yoy_opm_min",   float),
        "sales_5yr_enabled":  b("sales_5yr_enabled"), "sales_5yr_min": fp("sales_5yr_min", float),
    }

    def generate():
        global _scan_active, _scan_stop
        collected = []
        total_scanned = 0
        elapsed = 0
        try:
            for event in scan_stream(filters, stop_flag=lambda: _scan_stop):
                yield f"data: {json.dumps(event)}\n\n"

                if event["type"] == "result":
                    collected.append(event["data"])
                elif event["type"] == "done":
                    total_scanned = event.get("total_scanned", 0)
                    elapsed       = event.get("elapsed_sec",   0)
                elif event["type"] == "stopped":
                    total_scanned = event.get("processed", 0)
                    elapsed       = event.get("elapsed_sec", 0)
                    break

        except GeneratorExit:
            pass
        except Exception as exc:
            yield f"data: {json.dumps({'type':'error','message':str(exc)})}\n\n"
        finally:
            _scan_active = False
            _scan_stop   = False
            if collected:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _cache["results"]       = collected
                _cache["total_scanned"] = total_scanned
                _cache["last_scan"]     = now
                _cache["elapsed_sec"]   = elapsed
                save_cache(collected, total_scanned, elapsed)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@app.route("/api/stop", methods=["POST"])
def stop_scan():
    global _scan_stop
    _scan_stop = True
    return jsonify({"message": "Stop signal sent"})


# ── Chart data ─────────────────────────────────────────────────────────────────
@app.route("/api/chart/<symbol>")
def get_chart(symbol):
    try:
        df = yf.Ticker(f"{symbol}.NS").history(period="1y", auto_adjust=True)
        if df.empty:
            return jsonify({"error": "No data"}), 404
        df.index = df.index.tz_localize(None)
        rsi_series = calculate_rsi(df["Close"])

        candles = []
        rsi_data = []
        volume_data = []
        for ts, row in df.iterrows():
            d = ts.strftime("%Y-%m-%d")
            candles.append({"time": d, "open": round(float(row["Open"]), 2),
                            "high": round(float(row["High"]), 2),
                            "low":  round(float(row["Low"]),  2),
                            "close":round(float(row["Close"]),2)})
            rv = rsi_series.get(ts)
            rsi_data.append({"time": d,
                             "value": round(float(rv), 2) if rv and not np.isnan(rv) else None})
            volume_data.append({"time": d, "value": int(row["Volume"]),
                                "color": "#10b981" if row["Close"] >= row["Open"] else "#ef4444"})

        return jsonify({
            "symbol":   symbol,
            "candles":  candles,
            "rsi":      rsi_data,
            "volume":   volume_data,
            "high_52w": round(float(df["Close"].max()), 2),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── CSV export ─────────────────────────────────────────────────────────────────
@app.route("/api/export")
def export_csv():
    results = _cache["results"]
    output  = io.StringIO()
    fields  = ["symbol","name","sector","industry","price","change_pct",
               "rsi","high_52w","pct_from_52w_high","market_cap_cr","volume","avg_volume"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in results:
        writer.writerow({k: row.get(k, "") for k in fields})
    fname = f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})


if __name__ == "__main__":
    print("=" * 60)
    print("  Indian Stock Scanner — http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
