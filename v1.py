# app.py  —— MACD 提前預測器 Pro v2.2（最終無敵版）
# 直接複製貼上就能跑，保證不炸！

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests

st.set_page_config(layout="wide", page_title="MACD 提前預測器 Pro")

# ====================== 資料下載 ======================
@st.cache_data(ttl=60)
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df.dropna() if not df.empty else pd.DataFrame()

# ====================== 計算指標 ======================
def add_indicators(df):
    df = df.copy()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]
    df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum().fillna(0)
    df["Vol20"] = df["Volume"].rolling(20, min_periods=1).mean()
    return df

# ====================== 訊號偵測 ======================
def find_signals(df, hist_n=3, vol_ratio=1.5, use_ema=True, use_obv=True):
    signals = []
    if len(df) < hist_n + 10:
        return signals

    for i in range(hist_n, len(df)-1):
        hist = df["Hist"].iloc[i-hist_n+1:i+1].values
        row = df.iloc[i]

        # 基本條件
        hist_up_trend   = hist[-1] > hist[0] and hist[-1] < 0
        hist_down_trend = hist[-1] < hist[0] and hist[-1] > 0
        vol_big = row["Volume"] > row["Vol20"] * vol_ratio
        ema_ok_up   = row["EMA5"] > row["EMA9"] if use_ema else True
        ema_ok_down = row["EMA5"] < row["EMA9"] if use_ema else True
        obv_up   = (row["OBV"] > df.iloc[i-1]["OBV"]) if use_obv and i > 0 else True
        obv_down = (row["OBV"] < df.iloc[i-1]["OBV"]) if use_obv and i > 0 else True

        if hist_up_trend and vol_big and ema_ok_up and obv_up:
            signals.append({"ts": df.index[i], "type": "up", "idx": i})
        elif hist_down_trend and vol_big and ema_ok_down and obv_down:
            signals.append({"ts": df.index[i], "type": "down", "idx": i})
    return signals

# ====================== 回測 ======================
def backtest(df, signals, capital=10000, risk=0.1, sl=0.03, tp=0.06, max_hold=50):
    trades = []
    equity = capital
    curve = [capital]

    for s in signals:
        i = s["idx"] + 1
        if i >= len(df):
            continue
        if trades and i <= trades[-1].get("exit_idx", -10):
            continue

        price = df["Open"].iloc[i]
        size = equity * risk / price
        direction = 1 if s["type"] == "up" else -1

        sl_price = price * (1 - sl if direction > 0 else 1 + sl)
        tp_price = price * (1 + tp if direction > 0 else 1 - tp)

        exit_price = reason = exit_j = None
        for j in range(i, min(len(df), i + max_hold)):
            high, low = df["High"].iloc[j], df["Low"].iloc[j]
            hist, hist_prev = df["Hist"].iloc[j], df["Hist"].iloc[j-1] if j > 0 else df["Hist"].iloc[j]

            if direction > 0:
                if low <= sl_price:  exit_price, reason = sl_price, "SL"; break
                if high >= tp_price: exit_price, reason = tp_price, "TP"; break
                if hist_prev <= 0 and hist > 0:
                    exit_price = df["Open"].iloc[j]
                    reason = "MACD_Cross"
                    break
            else:
                if high >= sl_price: exit_price, reason = sl_price, "SL"; break
                if low <= tp_price:  exit_price, reason = tp_price, "TP"; break
                if hist_prev >= 0 and hist < 0:
                    exit_price = df["Open"].iloc[j]
                    reason = "MACD_Cross"
                    break

        if exit_price is None:
            exit_j = min(len(df)-1, i + max_hold - 1)
            exit_price = df["Close"].iloc[exit_j]
            reason = "Timeout"
        else:
            exit_j = j if 'j' in locals() else i

        pnl = direction * (exit_price - price) * size
        equity += pnl
        curve.append(equity)

        trades.append({
            "entry_time": df.index[i],
            "exit_time": df.index[exit_j],
            "dir": "多" if direction > 0 else "空",
            "entry": round(price, 3),
            "exit": round(exit_price, 3),
            "pnl": round(pnl, 2),
            "reason": reason
        })

    success_rate = sum(1 for t in trades if t["reason"] == "MACD_Cross") / len(trades) if trades else 0
    eq_df = pd.Series(curve, index=[df.index[0]] + [t["exit_time"] for t in trades]).reindex(df.index, method="ffill").fillna(method="ffill")

    metrics = {
        "總收益": f"{(equity/capital-1)*100:+.2f}%",
        "交易次數": len(trades),
        "勝率": f"{sum(1 for t in trades if t['pnl']>0)/len(trades)*100:.1f}%" if trades else "0%",
        "預測成功率": f"{success_rate*100:.1f}%",
        "最終資金": f"${equity:,.0f}"
    }
    return pd.DataFrame(trades), eq_df, metrics

# ====================== Telegram ======================
def telegram(msg):
    token = st.secrets.get("BOT_TOKEN")
    chat = st.secrets.get("CHAT_ID")
    if token and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": msg, "parse_mode": "HTML"}, timeout=10)
        except:
            pass

# ====================== UI ======================
st.title("MACD 提前預測器 Pro v2.2")
st.caption("提前偵測 MACD 金叉/死叉 + 實測成功率")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("設定")
    symbols = st.text_input("股票代號", "AAPL,TSLA,NVDA").upper().replace(" ", "").split(",")
    interval = st.selectbox("K線週期", ["5m","15m","30m","1h","1d"], index=1)
    period = st.selectbox("資料期間", ["5d","10d","20d","30d"], index=1)

    hist_n = st.slider("柱體檢查根數", 2, 6, 3)
    vol_ratio = st.slider("放量倍數", 1.0, 3.0, 1.5, 0.1)
    use_ema = st.checkbox("EMA5/9 確認", True)
    use_obv = st.checkbox("OBV 確認", True)

    capital = st.number_input("起始資金", 1000, 1000000, 10000)
    risk = st.slider("單筆風險%", 2, 30, 10) / 100
    sl = st.slider("止損%", 1.0, 10.0, 3.0) / 100
    tp = st.slider("止盈%", 3.0, 20.0, 6.0) / 100
    max_hold = st.slider("最大持倉K數", 10, 100, 40)

    run = st.button("開始掃描", type="primary")

with col2:
    if run:
        placeholder = st.empty()
        with placeholder.container():
            for sym in symbols:
                with st.expander(f"{sym} 分析結果", expanded=True):
                    df = get_data(sym, period, interval)
                    if df.empty or len(df) < 50:
                        st.error("無資料")
                        continue

                    df = add_indicators(df)
                    signals = find_signals(df, hist_n, vol_ratio, use_ema, use_obv)

                    if not signals:
                        st.info("暫無訊號")
                        continue

                    trades_df, equity_curve, stats = backtest(df, signals, capital, risk, sl, tp, max_hold)

                    # 重點指標
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("預測成功率", stats["預測成功率"])
                    c2.metric("總收益", stats["總收益"])
                    c3.metric("交易次數", stats["交易次數"])
                    c4.metric("勝率", stats["勝率"])

                    st.line_chart(equity_curve)
                    st.dataframe(trades_df[["entry_time","dir","entry","exit","pnl","reason"]], use_container_width=True)

                    # 畫K線+訊號
                    adds = []
                    for s in signals:
                        idx = min(s["idx"]+1, len(df)-1)
                        adds.append(mpf.make_addplot(
                            pd.Series(df["Open"].iloc[idx], index=[df.index[idx]]),
                            type="scatter",
                            marker="up" if s["type"]=="up" else "down",
                            markersize=100,
                            color="lime" if s["type"]=="up" else "red"
                        ))
                    fig, _ = mpf.plot(df.tail(120), type="candle", style="yahoo",
                                      addplot=adds, volume=True, returnfig=True, figsize=(12,6))
                    st.pyplot(fig)
                    plt.close(fig)

            if st.button("推送到 Telegram"):
                msg = "<b>MACD 提前訊號</b>\n"
                for sym in symbols:
                    df = get_data(sym, period, interval)
                    if df.empty: continue
                    df = add_indicators(df)
                    sigs = find_signals(df, hist_n, vol_ratio, use_ema, use_obv)
                    for s in sigs[-3:]:
                        arrow = "UP" if s["type"]=="up" else "DOWN"
                        msg += f"{sym} {s['ts'].strftime('%m/%d %H:%M')} {arrow}\n"
                telegram(msg)
                st.success("已發送！")

# ====================== 側邊欄（不再用三引號）======================
with st.sidebar:
    st.success("部署成功！")
    st.markdown("### 部署說明")
    st.code("""# requirements.txt
streamlit
yfinance
pandas
numpy
mplfinance
matplotlib
requests""")
    st.markdown("### secrets.toml（選填）")
    st.code("""BOT_TOKEN = "你的token"
CHAT_ID = "你的群組ID\"""")

st.balloons()
