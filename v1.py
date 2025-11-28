# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
import time

st.set_page_config(layout="wide", page_title="MACD Advance Predictor (MVP)")

# ----------------------
# Helper functions
# ----------------------
@st.cache_data(ttl=60)
def download_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    if df.empty:
        return df
    df = df.dropna()
    return df

def calc_indicators(df):
    df = df.copy()
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['Signal']
    # OBV
    df['OBV'] = ((np.sign(df['Close'].diff()).fillna(0)) * df['Volume']).cumsum()
    df['avg_vol_20'] = df['Volume'].rolling(20, min_periods=1).mean()
    return df

def detect_macd_prediction(df, hist_n=3, volume_ratio=1.5, use_ema_cross=True, use_obv=True):
    """
    Return a list of signals: dict with ts(index), type ('up'/'down'), price (open next), idx
    This is detection of *prediction* (before MACD actually crosses).
    """
    signals = []
    for i in range(hist_n, len(df)-1):
        hist = df['MACD_hist'].iloc[i-hist_n+1:i+1]  # last hist_n values ending at i
        # Check monotonic behavior: negative -> increasing towards zero (i.e. monotonic increasing)
        # positive -> decreasing towards zero (i.e. monotonic decreasing)
        ema5 = df['EMA5'].iat[i]
        ema9 = df['EMA9'].iat[i]
        obv_change = df['OBV'].iat[i] - df['OBV'].iat[i-1] if i>=1 else 0
        vol = df['Volume'].iat[i]
        avg_vol = df['avg_vol_20'].iat[i] if not np.isnan(df['avg_vol_20'].iat[i]) else vol

        # cond up: histogram increasing (getting closer to 0) when negative
        hist_vals = hist.values
        cond_hist_up = (hist_vals[-1] > hist_vals[0]) and (hist_vals[-1] < 0 or hist_vals.mean() < 0)
        cond_hist_down = (hist_vals[-1] < hist_vals[0]) and (hist_vals[-1] > 0 or hist_vals.mean() > 0)

        cond_ema_up = (ema5 > ema9) if use_ema_cross else True
        cond_ema_down = (ema5 < ema9) if use_ema_cross else True
        cond_obv_up = (obv_change > 0) if use_obv else True
        cond_obv_down = (obv_change < 0) if use_obv else True
        cond_vol_up = vol > avg_vol * volume_ratio
        cond_vol_down = vol < avg_vol / volume_ratio

        # Predict upward
        if cond_hist_up and cond_ema_up and cond_obv_up and cond_vol_up:
            signals.append({'ts': df.index[i], 'type': 'up', 'idx': i})
        # Predict downward
        elif cond_hist_down and cond_ema_down and cond_obv_down and cond_vol_down:
            signals.append({'ts': df.index[i], 'type': 'down', 'idx': i})
    return signals

# Simple backtest
def backtest_signals(df, signals, capital=10000, pct_per_trade=0.1, sl_pct=0.03, tp_pct=0.05, max_bars_hold=50):
    """
    Simple backtest:
    - Enter next bar open after signal
    - Long for 'up' signals, Short for 'down' signals
    - Exit when MACD actual crosses Signal (i.e., MACD_hist crosses 0), or hit TP/SL, or after max_bars_hold
    - Position sizing by pct_per_trade of capital
    """
    trades = []
    equity = capital
    equity_curve = []
    positions = []  # current open trades
    last_entry_idx = -999

    for sig in signals:
        i = sig['idx']
        # avoid overlapping too-close signals (simple throttle)
        if i <= last_entry_idx + 1:
            continue
        entry_idx = i + 1
        if entry_idx >= len(df):
            continue
        entry_price = df['Open'].iat[entry_idx]
        size = (capital * pct_per_trade) / entry_price  # shares
        direction = sig['type']  # up -> long, down -> short
        stop = entry_price * (1 - sl_pct) if direction == 'up' else entry_price * (1 + sl_pct)
        take = entry_price * (1 + tp_pct) if direction == 'up' else entry_price * (1 - tp_pct)

        # simulate bar by bar
        exit_price = None
        exit_idx = None
        for j in range(entry_idx, min(len(df), entry_idx + max_bars_hold)):
            low = df['Low'].iat[j]
            high = df['High'].iat[j]
            # stop or take hit intrabar? assume if high >= take (for long) it's TP, if low <= stop it's SL
            if direction == 'up':
                if low <= stop:
                    exit_price = stop
                    exit_idx = j
                    reason = 'SL'
                    break
                if high >= take:
                    exit_price = take
                    exit_idx = j
                    reason = 'TP'
                    break
            else:
                # short
                if high >= stop:
                    exit_price = stop
                    exit_idx = j
                    reason = 'SL'
                    break
                if low <= take:
                    exit_price = take
                    exit_idx = j
                    reason = 'TP'
                    break
            # MACD_hist cross check - exit when MACD_hist crosses below 0 (long) or above 0 (short)
            macd_hist = df['MACD_hist'].iat[j]
            macd_hist_prev = df['MACD_hist'].iat[j-1] if j-1 >= 0 else macd_hist
            if direction == 'up' and macd_hist_prev > 0 and macd_hist < 0:
                exit_price = df['Open'].iat[j]  # conservative: exit at open
                exit_idx = j
                reason = 'MACD_cross'
                break
            if direction == 'down' and macd_hist_prev < 0 and macd_hist > 0:
                exit_price = df['Open'].iat[j]
                exit_idx = j
                reason = 'MACD_cross'
                break
        if exit_price is None:
            # force exit at last available close
            exit_idx = min(len(df)-1, entry_idx + max_bars_hold - 1)
            exit_price = df['Close'].iat[exit_idx]
            reason = 'Timeout'

        # P/L
        pnl = (exit_price - entry_price) * size if direction == 'up' else (entry_price - exit_price) * size
        ret = pnl / (capital * pct_per_trade)
        equity += pnl
        trades.append({
            'entry_idx': entry_idx, 'entry_time': df.index[entry_idx], 'entry_price': entry_price,
            'exit_idx': exit_idx, 'exit_time': df.index[exit_idx], 'exit_price': exit_price,
            'direction': direction, 'pnl': pnl, 'return': ret, 'reason': reason
        })
        equity_curve.append({'time': df.index[exit_idx], 'equity': equity})
        last_entry_idx = exit_idx

    # Build equity series as dataframe
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve).set_index('time').resample(df.index.freq or '1min').ffill().fillna(method='ffill')
        eq_df = eq_df.reindex(df.index, method='ffill').fillna(method='ffill')
        eq_df['equity'] = eq_df['equity'].fillna(capital)
    else:
        eq_df = pd.DataFrame({'equity': capital}, index=df.index)

    # Metrics
    trade_df = pd.DataFrame(trades)
    total_trades = len(trade_df)
    wins = trade_df[trade_df['pnl'] > 0]
    win_rate = len(wins) / total_trades if total_trades>0 else 0
    gross_pnl = trade_df['pnl'].sum() if total_trades>0 else 0
    max_drawdown = compute_max_drawdown(eq_df['equity'].values)

    metrics = {
        'initial_capital': capital,
        'ending_capital': equity,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'gross_pnl': gross_pnl,
        'max_drawdown': max_drawdown
    }
    return trade_df, eq_df, metrics

def compute_max_drawdown(equity_array):
    peak = -np.inf
    max_dd = 0.0
    for x in equity_array:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak>0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd

def df_to_csv_bytes(df):
    return df.to_csv(index=True).encode('utf-8')

def send_telegram(text):
    # Use st.secrets to store BOT_TOKEN and CHAT_ID
    if "BOT_TOKEN" in st.secrets and "CHAT_ID" in st.secrets:
        token = st.secrets["BOT_TOKEN"]
        chat = st.secrets["CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": chat, "text": text})
        except Exception as e:
            st.error("Telegram 發送失敗: " + str(e))
    else:
        st.info("未設定 Telegram secrets (st.secrets['BOT_TOKEN'] & st.secrets['CHAT_ID'])")

# ----------------------
# UI
# ----------------------
st.title("MACD Advance Predictor — MVP (完整版)")
left_col, right_col = st.columns([1,2])

with left_col:
    st.header("設定")
    tickers = st.text_input("股票 (逗號分隔, 支援美股代號)", value="TSLA,NVDA,AAPL")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()!='']
    interval = st.selectbox("時間框架", ["5m", "15m", "60m", "1d"])
    period = st.selectbox("抓取期間", ["5d","14d","30d","90d"], index=1)
    hist_n = st.slider("MACD 柱子敏感度 (連續檢查根數)", 2, 6, 3)
    volume_ratio = st.slider("量能放大倍數 (預測上升)", 1.0, 3.0, 1.5)
    use_ema_cross = st.checkbox("使用 EMA5/EMA9 交叉確認", value=True)
    use_obv = st.checkbox("使用 OBV 確認", value=True)
    # Backtest params
    st.markdown("**回測設定**")
    capital = st.number_input("起始資金 (USD)", value=10000.0)
    pct_per_trade = st.slider("每筆交易資金比率", 0.01, 0.5, 0.1, 0.01)
    sl_pct = st.slider("Stop Loss (%)", 0.005, 0.1, 0.03, 0.005)  # 0.5% ~ 10%
    tp_pct = st.slider("Take Profit (%)", 0.005, 0.2, 0.05, 0.005)
    max_bars_hold = st.number_input("最大持倉 K 根數 (bar)", min_value=1, max_value=500, value=50)
    run_btn = st.button("開始分析 (Fetch → Detect → Backtest)")

with right_col:
    st.header("結果")
    status = st.empty()
    output_area = st.empty()

if run_btn:
    status.info("抓取資料中 ...")
    all_signals = {}
    all_trades = {}
    all_metrics = {}
    all_equity = {}

    for t in tickers:
        status.info(f"抓 {t} 資料 ({period} / {interval}) ...")
        df = download_data(t, period=period, interval=interval)
        if df.empty:
            st.warning(f"{t} 無法抓到資料或資料為空，請確認代號與時間範圍")
            continue
        df = calc_indicators(df)
        signals = detect_macd_prediction(df, hist_n=hist_n, volume_ratio=volume_ratio,
                                         use_ema_cross=use_ema_cross, use_obv=use_obv)
        status.info(f"{t} 偵測到 {len(signals)} 筆預測信號")
        trade_df, eq_df, metrics = backtest_signals(df, signals,
                                                   capital=capital, pct_per_trade=pct_per_trade,
                                                   sl_pct=sl_pct, tp_pct=tp_pct, max_bars_hold=int(max_bars_hold))
        all_signals[t] = signals
        all_trades[t] = trade_df
        all_metrics[t] = metrics
        all_equity[t] = eq_df
        # UI display per ticker
        st.subheader(f"{t} — 指標圖 / 訊號")
        # mplfinance plot
        mpf_df = df.copy()
        # mark predicted signals on plot
        ap = []
        for s in signals:
            idx = s['idx'] + 1 if (s['idx'] + 1) < len(df) else s['idx']
            ts = df.index[idx]
            price = df['Open'].iat[idx]
            color = 'g' if s['type']=='up' else 'r'
            ap.append(mpf.make_addplot(pd.Series(price,index=[ts]), type='scatter', markersize=70, marker='^' if s['type']=='up' else 'v'))
        try:
            fig, ax = mpf.plot(mpf_df, type='candle', volume=True, addplot=ap, returnfig=True, style='yahoo', mav=(5,9))
            st.pyplot(fig)
        except Exception as e:
            st.write("繪圖失敗：", e)

        st.write("偵測到的訊號（時間 / 型態）")
        sig_table = pd.DataFrame([{'time':s['ts'], 'type':s['type']} for s in signals])
        st.dataframe(sig_table)

        st.write("回測績效")
        st.markdown(metrics_to_md(metrics := metrics))
        if not trade_df.empty:
            st.write("交易紀錄 (回測結果)")
            st.dataframe(trade_df[['entry_time','entry_price','exit_time','exit_price','direction','pnl','reason']])
            csv_btn = st.download_button(f"下載 {t} 交易紀錄 CSV", data=df_to_csv_bytes(trade_df), file_name=f"{t}_trades.csv", mime="text/csv")
        # equity curve
        st.line_chart(eq_df['equity'])

    status.success("分析完成 ✅")
    # Global download options
    st.header("匯出與推播")
    # Export all signal lists as CSV
    combined_signals = []
    for t, sgns in all_signals.items():
        for s in sgns:
            combined_signals.append({'ticker': t, 'time': s['ts'], 'type': s['type']})
    if combined_signals:
        cs_df = pd.DataFrame(combined_signals)
        st.download_button("下載所有預測訊號 CSV", data=df_to_csv_bytes(cs_df), file_name="predicted_signals.csv", mime="text/csv")

    # Telegram push option
    if st.button("推播所有預測訊號 到 Telegram"):
        msg = "MACD Advance Predictor - 預測訊號\n"
        for t, sgns in all_signals.items():
            for s in sgns:
                msg += f"{t} | {s['ts']} | {s['type']}\n"
        send_telegram(msg)
        st.success("已發送（或嘗試發送）")

# ----------------------
# Helper for metrics display (small)
# ----------------------
def metrics_to_md(metrics):
    md = f"""
- 初始資金: ${metrics.get('initial_capital',0):,.2f}  
- 最終資金: ${metrics.get('ending_capital',0):,.2f}  
- 交易次數: {metrics.get('total_trades',0)}  
- 勝率: {metrics.get('win_rate',0)*100:,.2f}%  
- 毛損益: ${metrics.get('gross_pnl',0):,.2f}  
- 最大回撤: {metrics.get('max_drawdown',0)*100:,.2f}%  
"""
    return md

# ----------------------
# End of app
# ----------------------

# pip install streamlit yfinance pandas numpy mplfinance matplotlib requests
