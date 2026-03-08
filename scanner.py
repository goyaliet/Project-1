import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
import concurrent.futures
import time
import json
import os
from datetime import datetime

# ── Default Filters ────────────────────────────────────────────────────────────
RSI_PERIOD = 14
DEFAULT_FILTERS = {
    # ── Technicals ─────────────────────────────────────
    "rsi_min": 30,            "rsi_max": 70,
    "rs_min":  50,            "rs_max":  100,

    "pct_52w_high_min": 0,    "pct_52w_high_max": 30,   # % from 52W High range
    "pct_52w_low_enabled": False,
    "pct_52w_low_min": 0,     "pct_52w_low_max": 1000,  # % from 52W Low range

    "price_enabled": False,
    "price_min": 0,           "price_max": 150000,       # Stock price ₹

    "price_vs_ma_enabled": False,
    "price_vs_ma_dir": "above",                          # above / below
    "price_vs_ma_period": 200,                           # 20 / 50 / 100 / 200

    "ma_order_enabled": False,
    "ma_order_fast": 20,                                 # fast MA
    "ma_order_mid":  50,                                 # mid MA
    "ma_order_slow": 200,                                # slow MA

    "avg_rupee_vol_enabled": False,
    "avg_rupee_vol_min": 5,                              # Avg 30D rupee volume Cr.

    "ma_adr_enabled": False,
    "ma_adr_days": 20,                                   # ADR period
    "ma_adr_min":  0,                                    # min ADR %

    # ── Fundamentals ───────────────────────────────────
    "market_cap_min_cr": 500,   "market_cap_max_cr": 100_000,
    "min_avg_volume":    50_000,

    "pe_enabled": False,        "pe_min": 0,      "pe_max": 50,
    "roe_enabled": False,       "roe_min": 10,    "roe_max": 100,    # %
    "debt_eq_enabled": False,   "debt_eq_min": 0, "debt_eq_max": 1,
    "eps_growth_enabled": False,"eps_growth_min": 10, "eps_growth_max": 500, # % YoY
    "rev_growth_enabled": False,"rev_growth_min": 10, "rev_growth_max": 500, # % YoY
    "free_float_enabled": False,"free_float_min": 10, "free_float_max": 100, # %
    "promoter_enabled": False,  "promoter_min": 40,   "promoter_max": 100,   # % holding

    # Quarterly growth filters (min threshold only, > X%)
    "qoq_eps_enabled":   False, "qoq_eps_min":   0,
    "yoy_eps_enabled":   False, "yoy_eps_min":   0,
    "qoq_np_enabled":    False, "qoq_np_min":    0,
    "yoy_np_enabled":    False, "yoy_np_min":    0,
    "qoq_sales_enabled": False, "qoq_sales_min": 0,
    "yoy_sales_enabled": False, "yoy_sales_min": 0,
    "qoq_opm_enabled":   False, "qoq_opm_min":   0,
    "yoy_opm_enabled":   False, "yoy_opm_min":   0,
    "sales_5yr_enabled": False, "sales_5yr_min":  10,
}

BATCH_SIZE    = 100
MAX_WORKERS   = 20
RESULTS_CACHE = os.path.join(os.path.dirname(__file__), "last_results.json")


# ── RSI ────────────────────────────────────────────────────────────────────────
def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── Relative Strength vs Nifty 50 ─────────────────────────────────────────────
def fetch_nifty_close() -> pd.Series:
    try:
        df = yf.download("^NSEI", period="1y", auto_adjust=True, progress=False)
        return df["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)


def calculate_rs(stock_close: pd.Series, nifty_close: pd.Series) -> float:
    if nifty_close.empty or len(stock_close) < 20:
        return 50.0
    def _ret(s, days):
        return float((s.iloc[-1] / s.iloc[-days] - 1) * 100) if len(s) > days else 0.0
    s_ret = 0.4*_ret(stock_close,252) + 0.2*_ret(stock_close,126) \
          + 0.2*_ret(stock_close,63)  + 0.2*_ret(stock_close,21)
    n_ret = 0.4*_ret(nifty_close,252) + 0.2*_ret(nifty_close,126) \
          + 0.2*_ret(nifty_close,63)  + 0.2*_ret(nifty_close,21)
    return round(max(0.0, min(100.0, 50.0 + (s_ret - n_ret))), 1)


# ── NSE Stock List ─────────────────────────────────────────────────────────────
def get_nse_stocks() -> list[str]:
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*", "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.nseindia.com/",
    }
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        df   = pd.read_csv(StringIO(resp.text))
        syms = df["SYMBOL"].dropna().tolist()
        print(f"[NSE] {len(syms)} stocks loaded")
        return [f"{s}.NS" for s in syms]
    except Exception as exc:
        print(f"[NSE] Fallback: {exc}")
        return _fallback_stocks()


def _fallback_stocks() -> list[str]:
    n500 = [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
        "ITC","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA","TITAN",
        "BAJFINANCE","ULTRACEMCO","WIPRO","NESTLEIND","TECHM","POWERGRID","NTPC","ADANIENT",
        "ONGC","JSWSTEEL","TATASTEEL","HCLTECH","INDUSINDBK","BAJAJFINSV","COALINDIA",
        "GRASIM","BRITANNIA","CIPLA","DIVISLAB","DRREDDY","EICHERMOT","HEROMOTOCO",
        "HINDALCO","M&M","SBILIFE","TATACONSUM","TATAMOTORS","UPL","VEDL","AMBUJACEM",
        "APOLLOHOSP","DMART","HAVELLS","ICICIGI","INDIGO","LUPIN","MUTHOOTFIN","NAUKRI",
        "PIDILITIND","SIEMENS","TORNTPHARM","TRENT","VOLTAS","BANDHANBNK","BANKBARODA",
        "BIOCON","BOSCHLTD","CANBK","CHOLAFIN","COLPAL","DLF","FEDERALBNK","GAIL",
        "GODREJCP","GODREJPROP","HAL","IDFCFIRSTB","INDUSTOWER","IRCTC","JUBLFOOD",
        "LICHSGFIN","LTIM","LTTS","MARICO","MFSL","MOTHERSON","MPHASIS","OBEROIRLTY",
        "PAGEIND","PEL","PERSISTENT","PFC","POLYCAB","PNB","RECLTD","SAIL","SRF",
        "SUNDARMFIN","SUPREMEIND","TATACOMM","TATAPOWER","THERMAX","TIINDIA",
        "TVSMOTOR","UBL","UNIONBANK","VBL","ZOMATO","ADANIPORTS","BAJAJ-AUTO","BPCL",
        "HDFCLIFE","IOC","SHREECEM","BHEL","CONCOR","EXIDEIND","HINDPETRO","NMDC",
        "OFSS","PETRONET","PIIND","TATACHEM","ASTRAL","DIXON","KPITTECH",
        "ANGELONE","IRFC","IREDA","HUDCO","SJVN","NHPC","LICI",
    ]
    return [f"{s}.NS" for s in n500]


# ── Phase 1: all price-based filters ──────────────────────────────────────────
def _check_price_filters(symbol: str, df: pd.DataFrame, filters: dict,
                         nifty_close: pd.Series = None) -> dict | None:
    try:
        if df is None or df.empty or len(df) < RSI_PERIOD + 5:
            return None

        close  = df["Close"].dropna()
        high   = df["High"].dropna()
        low_s  = df["Low"].dropna()
        volume = df["Volume"].dropna()

        if len(close) < RSI_PERIOD + 5:
            return None

        cur = float(close.iloc[-1])

        # ── Avg Volume ───────────────────────────────────────────────────────
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        if avg_vol < filters["min_avg_volume"]:
            return None

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi = float(calculate_rsi(close).iloc[-1])
        if pd.isna(rsi) or not (filters["rsi_min"] <= rsi <= filters["rsi_max"]):
            return None

        # ── 52W metrics ──────────────────────────────────────────────────────
        high_52w = float(close.max())
        low_52w  = float(close.min())
        pct_from_high = ((high_52w - cur) / high_52w) * 100
        pct_from_low  = ((cur - low_52w) / low_52w)  * 100

        # % from 52W High (range filter)
        if not (filters["pct_52w_high_min"] <= pct_from_high <= filters["pct_52w_high_max"]):
            return None

        # % from 52W Low (optional)
        if filters.get("pct_52w_low_enabled"):
            if not (filters["pct_52w_low_min"] <= pct_from_low <= filters["pct_52w_low_max"]):
                return None

        # ── Stock Price range ─────────────────────────────────────────────────
        if filters.get("price_enabled"):
            if not (filters["price_min"] <= cur <= filters["price_max"]):
                return None

        # ── Moving Averages ───────────────────────────────────────────────────
        def _ma(days):
            if len(close) < days:
                return None
            return float(close.rolling(days).mean().iloc[-1])

        ma20  = _ma(20)
        ma50  = _ma(50)
        ma100 = _ma(100)
        ma200 = _ma(200)

        # Price vs MA filter
        if filters.get("price_vs_ma_enabled"):
            period = int(filters.get("price_vs_ma_period", 200))
            ma_map = {20: ma20, 50: ma50, 100: ma100, 200: ma200}
            ma_val = ma_map.get(period)
            if ma_val is not None:
                direction = filters.get("price_vs_ma_dir", "above")
                if direction == "above" and cur <= ma_val:
                    return None
                elif direction == "below" and cur >= ma_val:
                    return None

        # MA Order (e.g. 20MA >= 50MA >= 200MA)
        if filters.get("ma_order_enabled"):
            fast = _ma(int(filters.get("ma_order_fast", 20)))
            mid  = _ma(int(filters.get("ma_order_mid",  50)))
            slow = _ma(int(filters.get("ma_order_slow", 200)))
            if fast is None or mid is None or slow is None:
                return None
            if not (fast >= mid >= slow):
                return None

        # AVG Rupee Volume 30D (Cr.)
        if filters.get("avg_rupee_vol_enabled"):
            n = min(30, len(close))
            rupee_vol = float((close.iloc[-n:] * volume.iloc[-n:]).mean()) / 1e7
            if rupee_vol < filters.get("avg_rupee_vol_min", 5):
                return None

        # MA ADR%
        if filters.get("ma_adr_enabled"):
            n = min(int(filters.get("ma_adr_days", 20)), len(close))
            adr_pct = float(((high.iloc[-n:] - low_s.iloc[-n:]) / close.iloc[-n:] * 100).mean())
            if adr_pct < filters.get("ma_adr_min", 0):
                return None

        # ── RS vs Nifty ───────────────────────────────────────────────────────
        nc  = nifty_close if nifty_close is not None else pd.Series(dtype=float)
        rs  = calculate_rs(close, nc)
        if not (filters["rs_min"] <= rs <= filters["rs_max"]):
            return None

        # ── 1-day change ──────────────────────────────────────────────────────
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else cur
        change_pct = ((cur - prev_close) / prev_close) * 100

        return {
            "symbol":            symbol.replace(".NS", ""),
            "price":             round(cur, 2),
            "rsi":               round(rsi, 2),
            "rs":                rs,
            "high_52w":          round(high_52w, 2),
            "low_52w":           round(low_52w, 2),
            "pct_from_52w_high": round(pct_from_high, 2),
            "pct_from_52w_low":  round(pct_from_low, 2),
            "change_pct":        round(change_pct, 2),
            "avg_volume":        int(avg_vol),
            "volume":            int(volume.iloc[-1]),
            "ma20":  round(ma20,  2) if ma20  else None,
            "ma50":  round(ma50,  2) if ma50  else None,
            "ma200": round(ma200, 2) if ma200 else None,
        }
    except Exception:
        return None


# ── Phase 2: Market Cap ────────────────────────────────────────────────────────
def _check_market_cap(symbol: str, stock_data: dict, filters: dict) -> dict | None:
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info   = ticker.info
        mc     = info.get("marketCap") or 0
        mc_cr  = mc / 1e7
        if not (filters["market_cap_min_cr"] <= mc_cr <= filters["market_cap_max_cr"]):
            return None

        # ── Fundamental filters ────────────────────────────────────────────────
        def _fval(key): return info.get(key) or 0

        # P/E Ratio
        if filters.get("pe_enabled"):
            pe = _fval("trailingPE") or _fval("forwardPE")
            if pe <= 0 or not (filters["pe_min"] <= pe <= filters["pe_max"]):
                return None

        # ROE %
        if filters.get("roe_enabled"):
            roe = (_fval("returnOnEquity") or 0) * 100
            if not (filters["roe_min"] <= roe <= filters["roe_max"]):
                return None

        # Debt / Equity
        if filters.get("debt_eq_enabled"):
            de = _fval("debtToEquity") or 0
            if de < 0 or not (filters["debt_eq_min"] <= de <= filters["debt_eq_max"]):
                return None

        # EPS Growth % (YoY)
        if filters.get("eps_growth_enabled"):
            eps_g = (_fval("earningsGrowth") or 0) * 100
            if not (filters["eps_growth_min"] <= eps_g <= filters["eps_growth_max"]):
                return None

        # Revenue Growth % (YoY)
        if filters.get("rev_growth_enabled"):
            rev_g = (_fval("revenueGrowth") or 0) * 100
            if not (filters["rev_growth_min"] <= rev_g <= filters["rev_growth_max"]):
                return None

        # Free Float %
        if filters.get("free_float_enabled"):
            shares_out   = _fval("sharesOutstanding")
            float_shares = _fval("floatShares")
            ff = (float_shares / shares_out * 100) if shares_out > 0 else 0
            if not (filters["free_float_min"] <= ff <= filters["free_float_max"]):
                return None

        # Promoter Holding % (heldPercentInsiders as proxy)
        if filters.get("promoter_enabled"):
            promoter = (_fval("heldPercentInsiders") or 0) * 100
            if not (filters["promoter_min"] <= promoter <= filters["promoter_max"]):
                return None

        # ── Quarterly growth filters ───────────────────────────────────────────
        any_qtr_filter = any(filters.get(k) for k in [
            "qoq_eps_enabled","yoy_eps_enabled","qoq_np_enabled","yoy_np_enabled",
            "qoq_sales_enabled","yoy_sales_enabled","qoq_opm_enabled","yoy_opm_enabled",
            "sales_5yr_enabled",
        ])
        qtr_metrics = {}
        if any_qtr_filter:
            try:
                qf  = ticker.quarterly_financials   # rows=metrics, cols=quarters (newest first)
                qe  = ticker.quarterly_earnings      # rows=quarters, cols=[Earnings, Revenue]
                af  = ticker.financials              # annual, newest first

                def _q(df, row, col):
                    try: return float(df.loc[row].iloc[col])
                    except Exception: return None

                # ── Revenue (Sales) ────────────────────────────────────────
                rev = [_q(qf,"Total Revenue",i) for i in range(5)]
                if None not in rev[:2] and rev[1] != 0:
                    qtr_metrics["qoq_sales"] = (rev[0]-rev[1])/abs(rev[1])*100
                if None not in [rev[0], rev[4]] and rev[4] != 0:
                    qtr_metrics["yoy_sales"] = (rev[0]-rev[4])/abs(rev[4])*100

                # ── Net Profit ─────────────────────────────────────────────
                np_ = [_q(qf,"Net Income",i) for i in range(5)]
                if None not in np_[:2] and np_[1] != 0:
                    qtr_metrics["qoq_np"] = (np_[0]-np_[1])/abs(np_[1])*100
                if None not in [np_[0], np_[4]] and np_[4] != 0:
                    qtr_metrics["yoy_np"] = (np_[0]-np_[4])/abs(np_[4])*100

                # ── OPM (Operating Profit Margin) ──────────────────────────
                def _opm(i):
                    oi = _q(qf,"Operating Income",i)
                    rv = rev[i] if i < len(rev) else None
                    return (oi/rv*100) if oi and rv else None

                opm0, opm1, opm4 = _opm(0), _opm(1), _opm(4)
                if opm0 is not None and opm1 is not None:
                    qtr_metrics["qoq_opm"] = opm0 - opm1
                if opm0 is not None and opm4 is not None:
                    qtr_metrics["yoy_opm"] = opm0 - opm4

                # ── EPS (from quarterly_earnings) ──────────────────────────
                if qe is not None and not qe.empty and "Earnings" in qe.columns:
                    eps = qe["Earnings"].tolist()
                    if len(eps) >= 2 and eps[1] != 0:
                        qtr_metrics["qoq_eps"] = (eps[0]-eps[1])/abs(eps[1])*100
                    if len(eps) >= 5 and eps[4] != 0:
                        qtr_metrics["yoy_eps"] = (eps[0]-eps[4])/abs(eps[4])*100

                # ── Sales 5-Year CAGR ──────────────────────────────────────
                if af is not None and not af.empty and af.shape[1] >= 5:
                    r0 = _q(af,"Total Revenue",0)
                    r5 = _q(af,"Total Revenue",4)
                    if r0 and r5 and r5 > 0:
                        qtr_metrics["sales_5yr"] = ((r0/r5)**0.2 - 1)*100

            except Exception:
                pass

        # Apply quarterly growth filters
        checks = [
            ("qoq_eps_enabled",   "qoq_eps_min",   "qoq_eps"),
            ("yoy_eps_enabled",   "yoy_eps_min",   "yoy_eps"),
            ("qoq_np_enabled",    "qoq_np_min",    "qoq_np"),
            ("yoy_np_enabled",    "yoy_np_min",    "yoy_np"),
            ("qoq_sales_enabled", "qoq_sales_min", "qoq_sales"),
            ("yoy_sales_enabled", "yoy_sales_min", "yoy_sales"),
            ("qoq_opm_enabled",   "qoq_opm_min",   "qoq_opm"),
            ("yoy_opm_enabled",   "yoy_opm_min",   "yoy_opm"),
            ("sales_5yr_enabled", "sales_5yr_min",  "sales_5yr"),
        ]
        for en_key, min_key, metric in checks:
            if filters.get(en_key):
                val = qtr_metrics.get(metric)
                if val is None or val < filters.get(min_key, 0):
                    return None

        # ── Collect fundamental data for display ───────────────────────────────
        shares_out   = info.get("sharesOutstanding") or 1
        float_shares = info.get("floatShares") or 0
        free_float   = round(float_shares / shares_out * 100, 1) if shares_out else None
        promoter_pct = round((info.get("heldPercentInsiders") or 0) * 100, 1)

        stock_data.update({
            "name":          info.get("longName") or info.get("shortName") or symbol,
            "sector":        info.get("sector",   "N/A"),
            "industry":      info.get("industry", "N/A"),
            "market_cap_cr": round(mc_cr, 2),
            "pe":            round(info.get("trailingPE") or info.get("forwardPE") or 0, 1),
            "roe":           round((info.get("returnOnEquity") or 0) * 100, 1),
            "debt_eq":       round(info.get("debtToEquity") or 0, 2),
            "eps_growth":    round((info.get("earningsGrowth") or 0) * 100, 1),
            "rev_growth":    round((info.get("revenueGrowth")  or 0) * 100, 1),
            "free_float":    free_float,
            "promoter":      promoter_pct,
            "qoq_sales":     round(qtr_metrics.get("qoq_sales"),1)  if qtr_metrics.get("qoq_sales")  is not None else None,
            "yoy_sales":     round(qtr_metrics.get("yoy_sales"),1)  if qtr_metrics.get("yoy_sales")  is not None else None,
            "qoq_np":        round(qtr_metrics.get("qoq_np"),1)     if qtr_metrics.get("qoq_np")     is not None else None,
            "yoy_np":        round(qtr_metrics.get("yoy_np"),1)     if qtr_metrics.get("yoy_np")     is not None else None,
            "qoq_eps":       round(qtr_metrics.get("qoq_eps"),1)    if qtr_metrics.get("qoq_eps")    is not None else None,
            "yoy_eps":       round(qtr_metrics.get("yoy_eps"),1)    if qtr_metrics.get("yoy_eps")    is not None else None,
            "qoq_opm":       round(qtr_metrics.get("qoq_opm"),1)    if qtr_metrics.get("qoq_opm")    is not None else None,
            "yoy_opm":       round(qtr_metrics.get("yoy_opm"),1)    if qtr_metrics.get("yoy_opm")    is not None else None,
            "sales_5yr":     round(qtr_metrics.get("sales_5yr"),1)  if qtr_metrics.get("sales_5yr")  is not None else None,
        })
        return stock_data
    except Exception:
        return None


# ── Streaming scan ─────────────────────────────────────────────────────────────
def scan_stream(filters: dict = None, stop_flag=None):
    if filters is None:
        filters = DEFAULT_FILTERS
    if stop_flag is None:
        stop_flag = lambda: False

    started = datetime.now()
    try:
        symbols     = get_nse_stocks()
        total       = len(symbols)
        print("[Scan] Fetching Nifty 50 benchmark…")
        nifty_close = fetch_nifty_close()
        yield {"type": "start", "total": total}

        processed = 0
        found     = 0

        for batch_start in range(0, total, BATCH_SIZE):
            if stop_flag():
                elapsed = int((datetime.now() - started).total_seconds())
                yield {"type": "stopped", "processed": processed,
                       "found": found, "elapsed_sec": elapsed}
                return

            batch = symbols[batch_start: batch_start + BATCH_SIZE]
            label = f"{batch_start+1}–{min(batch_start+BATCH_SIZE, total)}"

            try:
                raw = yf.download(batch, period="1y", auto_adjust=True,
                                  progress=False, group_by="ticker", threads=True)
            except Exception:
                processed += len(batch)
                yield {"type": "progress", "processed": processed,
                       "total": total, "found": found, "batch_label": label}
                time.sleep(0.2)
                continue

            phase1 = []
            for sym in batch:
                try:
                    df = raw[sym] if len(batch) > 1 else raw
                    r  = _check_price_filters(sym, df, filters, nifty_close)
                    if r:
                        phase1.append((sym.replace(".NS", ""), r))
                except Exception:
                    pass

            if phase1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futs = {ex.submit(_check_market_cap, s, d, filters): s for s, d in phase1}
                    for fut in concurrent.futures.as_completed(futs):
                        result = fut.result()
                        if result:
                            found += 1
                            yield {"type": "result", "data": result}

            processed += len(batch)
            yield {"type": "progress", "processed": processed,
                   "total": total, "found": found, "batch_label": label}
            time.sleep(0.1)

        elapsed = int((datetime.now() - started).total_seconds())
        yield {"type": "done", "total_scanned": total,
               "found": found, "elapsed_sec": elapsed}

    except Exception as exc:
        yield {"type": "error", "message": str(exc)}


# ── Cache helpers ──────────────────────────────────────────────────────────────
def load_cached_results() -> dict:
    try:
        with open(RESULTS_CACHE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(results, total_scanned, elapsed_sec):
    try:
        with open(RESULTS_CACHE, "w") as f:
            json.dump({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "total_scanned": total_scanned, "results": results,
                       "elapsed_sec": elapsed_sec}, f)
    except Exception:
        pass
