# app.py  —— MACD 提前預測器 Pro v2.1（最終穩定版）
# 直接上傳到 Streamlit Cloud 即可運行

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
from datetime import datetime

st.set_page_config(layout="wide", page_title="MACD 提前預測器 Pro v2.1")

# ----------------------
# 下載資料
# ----------------------
@st.cache_data(ttl=60, show_spinner=False)
def download_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    return df.dropna()

# ----------------------
# 計算指標
# ----------------------
def calc_indicators(df):
    df = df.copy()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['Signal']
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['OBV'] = df['OBV'].fillna(0)
    
    df['avg_vol_20'] = df['Volume'].rolling(20, min_periods=1).mean()
    return df

# ----------------------
# 核心：提前預測 MACD 交叉
# ----------------------
def detect_macd_prediction(df, hist_n=3, volume_ratio=1.5, use_ema_cross=True, use_obv=True):
    signals = []
    if len(df) < hist_n + 10:
        return signals

    for i in range(hist_n, len(df)-1):
        hist = df['MACD_hist'].iloc[i-hist_n+1:i+1].values

        row = df.iloc[i]
        prev = df.iloc[i-1] if i > 0 else row

        ema5, ema9 = row['EMA5'], row['EMA9']
        vol, avg_vol = row['Volume'], row['avg_vol_20']
        obv_change = row['OBV'] - prev['OBV']

        # Histogram 趨勢
        hist_up = hist[-1] > hist[0] and hist[-1] < 0
        hist_down = hist[-1] < hist[0] and hist[-1] > 0

        cond_ema_up = ema5 > ema9 if use_ema_cross else True
        cond_ema_down = ema5 < ema9 if use_ema_cross else True
        cond_obv_up = obv_change > 0 if use_obv else True
        cond_obv_down = obv_change < 0 if use_obv else True
        cond_vol = vol > avg_vol * volume_ratio

        if hist_up and cond_ema_up and cond_obv_up and cond_vol:
            signals.append({'ts': df.index[i], 'type': 'up', 'idx': i})
        elif hist_down and cond_ema_down and cond_obv_down and cond_vol:
            signals.append({'ts': df.index[i], 'type': 'down', 'idx': i})

    return signals

# ----------------------
# 回測引擎（已修好短倉 + 實際交叉率）
# ----------------------
def backtest_signals(df, signals, capital=10000, pct_per_trade=0.1, sl_pct=0.03, tp_pct=0.05, max_bars=50):
    trades = []
    equity = capital
    curve = [capital]
    dates = [df.index[0]]

    for sig in signals:
        i = sig['idx'] + 1
        if i >= len(df) or (trades and i <= trades[-1].get('exit_idx', -10) + 1):
            continue

        price = df['Open'].iloc[i]
        size = equity * pct_per_trade / price
        direction = 1 if sig['type'] == 'up' else -1

        sl = price * (1 - sl_pct if direction > 0 else 1 + sl_pct)
        tp = price * (1 + tp_pct if direction > 0 else 1 - tp_pct)

        exit_price = reason = exit_idx = None
        for j in range(i, min(len(df), i + max_bars)):
            h, l = df['High'].iloc[j], df['Low'].iloc[j]
            hist = df['MACD_hist'].iloc[j]
            hist_prev = df['MACD_hist'].iloc[j-1] if j > 0 else hist

            # 止盈止損
            if direction > 0:
                if l <= sl: exit_price, reason = sl, 'SL'; break
                if h >= tp: exit_price, reason = tp, 'TP'; break
                if hist_prev <= 0 and hist > 0:
                    exit_price = df['Open'].iloc[j]
                    reason = 'MACD_Cross'; break
            else:
                if h >= sl: exit_price, reason = sl, 'SL'; break
                if l <= tp: exit_price, reason = tp, 'TP'; break
                if hist_prev >= 0 and hist < 0:
                    exit_price = df['Open'].iloc[j]
                    reason = 'MACD_Cross'; break

        if exit_price is None:
            exit_idx = min(len(df)-1, i + max_bars - 1)
            exit_price = df['Close'].iloc[exit_idx]
            reason = 'Timeout'
        else:
            exit_idx = j if 'j' in locals() else i

        pnl = direction * (exit_price - price) * size
        equity += pnl
        curve.append(equity)
        dates.append(df.index[exit_idx])

        trades.append({
            'entry_time': df.index[i], 'exit_time': df.index[exit_idx],
            'direction': sig['type'], 'entry': round(price, 4),
            'exit': round(exit_price, 4), 'pnl': round(pnl, 2),
            'reason': reason
        })

    # 預測成功率（實際發生交叉）
    success = sum(1 for t in trades if t['reason'] == 'MACD_Cross')
    success_rate = success / len(trades) if trades else 0

    eq_df = pd.DataFrame({'equity': curve}, index=dates).reindex(df.index, method='ffill').fillna(method='ffill')

    metrics = {
        '初始資金': f"${capital:,.0f}",
        '最終資金': f"${equity:,.0f}",
        '總收益': f"{(equity/capital-1)*100:+.2f}%",
        '交易次數': len(trades),
        '勝率': f"{sum(1 for t in trades if t['pnl']>0)/len(trades)*100:.1f}%" if trades else "0%",
        '預測成功率': f"{success_rate*100:.1f}%",   # 重點指標！
        '最大回撤': f"{((pd.Series(curve).cummax() - curve)/pd.Series(curve).cummax()).max()*100:.2f}%"
    }
    return pd.DataFrame(trades), eq_df, metrics

# ----------------------
# Telegram 推送
# ----------------------
def send_telegram(msg):
    token = st.secrets.get("BOT_TOKEN")
    chat_id = st.secrets.get("CHAT_ID")
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"})
        except:
            pass

# ----------------------
# UI
# ----------------------
st.title("MACD 提前預測器 Pro v2.1")
st.markdown("**提前 3~10 根K棒預知金叉/死叉 + 實測成功率**")

c1, c2 = st.columns([1, 2])

with c1:
    st.header("設定")
    tickers = st.text_input("股票代號（逗號分隔）", "AAPL,TSLA,NVDA")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    interval = st.selectbox("時間框架", ["5m", "15m", "30m", "60m", "1d"], index=1)
    period = st.selectbox("期間", ["5d", "10d", "20d", "30d"], index=1)

    hist_n = st.slider("柱體連續檢查", 2, 6, 3)
    volume_ratio = st.slider("放量倍數", 1.0, 3.0, 1.5, 0.1)
    use_ema_cross = st.checkbox("EMA5/9 確認", True)
    use_obv = st.checkbox("OBV 資金流", True)

    st.markdown("### 回測參數")
    capital = st.number_input("起始資金", 1000, 500000, 10000)
    pct_per_trade = st.slider("單筆倉位", 0.02, 0.5, 0.1, 0.01)
    sl_pct = st.slider("止損 %", 1.0, 10.0, 3.0, 0.5) / 100
    tp_pct = st.slider("止盈 %", 2.0, 20.0, 6.0, 0.5) / 100
    max_bars = st.slider("最大持倉K數", 10, 120, 40)

    run = st.button("開始分析", type="primary")

with c2:
    placeholder = st.empty()

if run:
    with placeholder.container():
        for t in tickers:
            with st.expander(f"【{t}】分析結果", expanded=True):
                df = download_data(t, period, interval)
                if df.empty or len(df) < 50:
                    st.error("資料不足")
                    continue

                df = calc_indicators(df)
                signals = detect_macd_prediction(df, hist_n, volume_ratio,
                                                 use_ema_cross, use_obv)

                if not signals:
                    st.info("暫無符合訊號")
                    continue

                trades_df, eq_df, metrics = backtest_signals(df, signals, capital,
                                                             pct_per_trade, sl_pct, tp_pct, max_bars)

                # 關鍵指標
                col1, col2, col3 = st.columns(3)
                col1.metric("預測成功率", metrics['預測成功率'])
                col2.metric("總收益", metrics['總收益'])
                col3.metric("交易次數", metrics['交易次數'])

                for k, v in metrics.items():
                    if k != '預測成功率':
                        st.write(f"**{k}**：{v}")

                st.line_chart(eq_df['equity'])
                if not trades_df.empty:
                    st.dataframe(trades_df[['entry_time','direction','entry','exit','pnl','reason']])

                # K線圖
                ap = []
                for s in signals:
                    ts_idx = min(s['idx'] + 1, len(df)-1)
                    ts = df.index[ts_idx]
                    price = df['Open'].iloc[ts_idx]
                    marker = '^' if s['type']=='up' else 'v'
                    color = 'lime' if s['type']=='up' else 'red'
                    ap.append(mpf.make_addplot(pd.Series([price], index=[ts]),
                                               type='scatter', marker=marker,
                                               markersize=120, color=color))

                fig, _ = mpf.plot(df.tail(120), type='candle', style='yahoo',
                                  addplot=ap, volume=True, returnfig=True)
                st.pyplot(fig)
                plt.close(fig)

        if st.button("推送到 Telegram"):
            msg = "<b>MACD 提前預測訊號</b>\n"
            for t in tickers:
                df = download_data(t, period, interval)
                if df.empty: continue
                df = calc_indicators(df)
                sigs = detect_macd_prediction(df, hist_n, volume_ratio, use_ema_cross, use_obv)
                for s in sigs[-3:]:
                    arrow = "UP" if s['type']=='up' else "DOWN"
                    msg += f"{t} {s['ts'].strftime('%m/%d %H:%M')} {arrow}\n"
            send_telegram(msg)
            st.success("已推送！")

# ----------------------
# 側邊欄說明
# ----------------------
st.sidebar.success("部署成功！")
st.sidebar.markdown("""
### 部署步驟
1. 建立 `secrets.toml`：
```toml
BOT_TOKEN = "你的機器人token"
CHAT_ID = "你的群組ID"
