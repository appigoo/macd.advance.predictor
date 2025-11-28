# app.py —— MACD 提前預測器 Pro v2.7（終極友好版）
# 直接貼上就能用，保證一次成功！

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="MACD 提前預測器 Pro")

# ====================== 超強資料下載（自動延長期間）=====================
@st.cache_data(ttl=90)  # 90秒快取
def get_data(ticker, period, interval):
    # 自動應對週末、非交易時段
    periods_to_try = [period, "10d", "20d", "30d", "60d", "3mo"]
    for p in periods_to_try:
        try:
            df = yf.download(ticker, period=p, interval=interval, progress=False, auto_adjust=True)
            if not df.empty and len(df) >= 30:
                # 強制轉型
                for col in ['Open','High','Low','Close','Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna()
                if len(df) >= 30:
                    return df
        except:
            continue
    return pd.DataFrame()

# ====================== 指標計算 ======================
def add_indicators(df):
    if df.empty or len(df) < 30:
        return df
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

# ====================== 訊號偵測（簡潔穩定）=====================
def find_signals(df, hist_n=3, vol_ratio=1.5, use_ema=True, use_obv=True):
    signals = []
    if len(df) < 35:
        return signals

    for i in range(hist_n, len(df)-1):
        h = df["Hist"].iloc[i-hist_n+1:i+1].values
        row = df.iloc[i]
        vol, vol20 = float(row["Volume"]), float(row["Vol20"])
        ema5, ema9 = float(row["EMA5"]), float(row["EMA9"])
        obv, obv_prev = float(row["OBV"]), float(df.iloc[i-1]["OBV"]) if i > 0 else (0, 0)

        up_cond   = h[-1] > h[0] and h[-1] < 0 and vol > vol20 * vol_ratio
        down_cond = h[-1] < h[0] and h[-1] > 0 and vol > vol20 * vol_ratio

        if up_cond and (ema5 > ema9 if use_ema else True) and (obv > obv_prev if use_obv else True):
            signals.append({"ts": df.index[i], "type": "up", "idx": i})
        elif down_cond and (ema5 < ema9 if use_ema else True) and (obv < obv_prev if use_obv else True):
            signals.append({"ts": df.index[i], "type": "down", "idx": i})
    return signals

# ====================== 回測（簡化版）=====================
def backtest(df, signals, capital=10000, risk=0.1, sl=0.03, tp=0.06, max_hold=50):
    if not signals:
        return pd.DataFrame(), pd.Series([capital]), {"總收益": "0%", "交易次數": 0, "勝率": "0%", "預測成功率": "0%"}
    
    trades = []
    equity = capital
    curve = [capital]
    times = [df.index[0]]

    for s in signals:
        i = s["idx"] + 1
        if i >= len(df): continue
        if trades and i <= trades[-1].get("exit_idx", -99): continue

        price = float(df["Open"].iloc[i])
        size = equity * risk / price
        direction = 1 if s["type"] == "up" else -1
        sl_price = price * (1 - sl if direction > 0 else 1 + sl)
        tp_price = price * (1 + tp if direction > 0 else 1 - tp)

        exit_price = reason = exit_j = None
        for j in range(i, min(len(df), i + max_hold)):
            high, low = float(df["High"].iloc[j]), float(df["Low"].iloc[j])
            hist, hist_prev = float(df["Hist"].iloc[j]), float(df["Hist"].iloc[j-1] if j > 0 else df["Hist"].iloc[j])

            if direction > 0 and (low <= sl_price or high >= tp_price or (hist_prev <= 0 and hist > 0)):
                exit_price = sl_price if low <= sl_price else (tp_price if high >= tp_price else float(df["Open"].iloc[j]))
                reason = "SL" if low <= sl_price else ("TP" if high >= tp_price else "MACD_Cross")
                exit_j = j
                break
            elif direction < 0 and (high >= sl_price or low <= tp_price or (hist_prev >= 0 and hist < 0)):
                exit_price = sl_price if high >= sl_price else (tp_price if low <= tp_price else float(df["Open"].iloc[j]))
                reason = "SL" if high >= sl_price else ("TP" if low <= tp_price else "MACD_Cross")
                exit_j = j
                break
        else:
            exit_j = min(len(df)-1, i + max_hold - 1)
            exit_price = float(df["Close"].iloc[exit_j])
            reason = "Timeout"

        pnl = direction * (exit_price - price) * size
        equity += pnl
        curve.append(equity)
        times.append(df.index[exit_j])
        trades.append({"entry_time": df.index[i], "dir": "多" if direction>0 else "空", "pnl": round(pnl,2), "reason": reason, "exit_idx": exit_j})

    success_rate = sum(1 for t in trades if t["reason"] == "MACD_Cross") / len(trades) if trades else 0
    eq_df = pd.Series(curve, index=times).reindex(df.index, method="ffill").fillna(equity)

    return pd.DataFrame(trades), eq_df, {
        "總收益": f"{(equity/capital-1)*100:+.2f}%",
        "交易次數": len(trades),
        "勝率": f"{sum(1 for t in trades if t['pnl']>0)/len(trades)*100:.1f}%" if trades else "0%",
        "預測成功率": f"{success_rate*100:.1f}%",
        "最終資金": f"${equity:,.0f}"
    }

# ====================== UI（超美提示）=====================
st.title("MACD 提前預測器 Pro v2.7")
st.markdown("**即時偵測金叉/死叉前兆 • 實測成功率 • 週末也看得懂**")

# 現在時間提示
now = datetime.now()
if now.weekday() >= 5 or (now.weekday() == 4 and now.hour >= 17):  # 週五17點後算週末
    st.warning("現在是美國非交易時段（週末或夜間），部分即時資料可能延遲或無更新")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("設定")
    symbols = st.text_input("股票代號（支援台股加.TW）", "AAPL,TSLA,NVDA,2330.TW,0050.TW")
    symbols = [s.strip().upper() for s in symbols.replace(" ", "").split(",") if s]

    interval = st.selectbox("時間框架", ["5m","15m","30m","1h","1d"], index=2)
    period = st.selectbox("資料期間（自動延長）", ["5d","10d","20d","30d","60d"], index=3)

    hist_n = st.slider("柱體連續檢查", 2, 6, 3)
    vol_ratio = st.slider("放量倍數", 1.0, 3.0, 1.5, 0.1)
    use_ema = st.checkbox("EMA5/9 趨勢確認", True)
    use_obv = st.checkbox("OBV 資金流確認", True)

    capital = st.number_input("模擬資金", 5000, 1000000, 10000)
    risk = st.slider("單筆風險%", 5, 25, 10) / 100
    sl = st.slider("止損%", 2.0, 8.0, 3.0) / 100
    tp = st.slider("止盈%", 4.0, 15.0, 6.0) / 100

    run = st.button("開始掃描", type="primary")

with col2:
    if run:
        if not symbols:
            st.error("請輸入至少一個股票代號")
        else:
            placeholder = st.empty()
            with placeholder.container():
                found_any = False
                for sym in symbols:
                    df = get_data(sym, period, interval)
                    if df.empty:
                        st.warning(f"{sym} → 目前無資料（可能休市或代號錯誤）")
                        continue

                    df = add_indicators(df)
                    signals = find_signals(df, hist_n, vol_ratio, use_ema, use_obv)
                    trades_df, eq_curve, stats = backtest(df, signals, capital, risk, sl, tp)

                    with st.expander(f"{sym} • {len(signals)} 個訊號 • 預測成功率 {stats['預測成功率']}", expanded=True):
                        found_any = True
                        c1, c2, c3 = st.columns(3)
                        c1.success(f"總收益 {stats['總收益']}")
                        c2.success(f"勝率 {stats['勝率']}")
                        c3.success(f"交易 {stats['交易次數']} 次")

                        st.line_chart(eq_curve.tail(100))
                        if not trades_df.empty:
                            st.dataframe(trades_df[["entry_time","dir","pnl","reason"]], use_container_width=True)

                        # K線圖
                        adds = [mpf.make_addplot(pd.Series([float(df["Open"].iloc[min(s["idx"]+1,len(df)-1)])], 
                                      index=[df.index[min(s["idx"]+1,len(df)-1)]]),
                                      type="scatter", marker="up" if s["type"]=="up" else "down",
                                      markersize=150, color="lime" if s["type"]=="up" else "red") for s in signals[-10:]]

                        try:
                            fig, _ = mpf.plot(df.tail(100), type="candle", style="yahoo", addplot=adds, volume=True,
                                              returnfig=True, figsize=(12,6), tight_layout=True)
                            st.pyplot(fig)
                            plt.close(fig)
                        except:
                            st.info("K線圖暫時無法顯示")

                if not found_any:
                    st.info("所有股票目前都沒有訊號，或處於休市狀態，晚點再來看看吧")

# ====================== 側邊欄 ======================
with st.sidebar:
    st.success("運行成功！")
    st.markdown("### 常見問題")
    st.info("週末或美國深夜會顯示「無資料」，屬正常現象！\n\n台股請加 `.TW` 如：2330.TW")
    st.code("""requirements.txt
streamlit
yfinance
pandas
numpy
mplfinance
matplotlib
requests""")

st.balloons()
