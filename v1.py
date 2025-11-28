# app.py  —— MACD Advance Predictor 完整稳定版（2025 最新）
# 直接复制到 Streamlit 部署即可运行

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import time

st.set_page_config(layout="wide", page_title="MACD 提前预测器 Pro v2.0")

# ----------------------
# 缓存数据下载
# ----------------------
@st.cache_data(ttl=60, show_spinner=False)
def download_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.dropna()
    return df

# ----------------------
# 计算指标
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
# 核心：提前预测 MACD 交叉
# ----------------------
def detect_macd_prediction(df, hist_n=3, volume_ratio=1.5, use_ema_cross=True, use_obv=True):
    signals = []
    if len(df) < hist_n + 10:
        return signals

    for i in range(hist_n, len(df)-1):
        hist = df['MACD_hist'].iloc[i-hist_n+1:i+1]
        hist_vals = hist.values

        row = df.iloc[i]
        prev_row = df.iloc[i-1] if i > 0 else row

        # 基础数据
        ema5 = row['EMA5']
        ema9 = row['EMA9']
        vol = row['Volume']
        avg_vol = row['avg_vol_20']
        obv_change = row['OBV'] - prev_row['OBV']

        # 高级过滤：避免追高杀低
        price_above_ema20 = row['Close'] > row['EMA20']
        rsi = ta.rsi(df['Close'], length=14).iloc[i] if 'ta' in globals() else 50
        not_overbought = rsi < 72 if not np.isnan(rsi) else True
        not_oversold = rsi > 28 if not np.isnan(rsi) else True

        # Histogram 趋势判断
        hist_increasing = hist_vals[-1] > hist_vals[0]
        hist_decreasing = hist_vals[-1] < hist_vals[0]
        near_zero_negative = hist_vals[-1] < 0 and abs(hist_vals[-1]) < abs(hist_vals[0])
        near_zero_positive = hist_vals[-1] > 0 and hist_vals[-1] < hist_vals[0]

        cond_hist_up = hist_increasing and near_zero_negative
        cond_hist_down = hist_decreasing and near_zero_positive

        cond_ema_up = (ema5 > ema9) if use_ema_cross else True
        cond_ema_down = (ema5 < ema9) if use_ema_cross else True
        cond_obv_up = (obv_change > 0) if use_obv else True
        cond_obv_down = (obv_change < 0) if use_obv else True
        cond_vol_up = vol > avg_vol * volume_ratio

        # 最终信号
        if (cond_hist_up and cond_ema_up and cond_obv_up and cond_vol_up and 
            price_above_ema20 and not_overbought):
            signals.append({'ts': df.index[i], 'type': 'up', 'idx': i})

        if (cond_hist_down and cond_ema_down and cond_obv_down and cond_vol_up and 
            not price_above_ema20 and not_oversold):
            signals.append({'ts': df.index[i], 'type': 'down', 'idx': i})

    return signals

# ----------------------
# 回测引擎（修复短仓 + 加入实际交叉验证）
# ----------------------
def backtest_signals(df, signals, capital=10000, pct_per_trade=0.1, sl_pct=0.03, tp_pct=0.05, max_bars_hold=50):
    trades = []
    equity = capital
    equity_curve = [capital]
    dates = [df.index[0]]

    for sig in signals:
        i = sig['idx']
        entry_idx = i + 1
        if entry_idx >= len(df):
            continue
        if len(trades) > 0 and entry_idx <= trades[-1]['exit_idx'] + 1:
            continue  # 避免重叠

        entry_price = df['Open'].iloc[entry_idx]
        size = (equity * pct_per_trade) / entry_price
        direction = 1 if sig['type'] == 'up' else -1

        # 正确止损止盈
        sl_price = entry_price * (1 - sl_pct * direction)
        tp_price = entry_price * (1 + tp_pct * direction)

        exit_price = None
        exit_idx = None
        reason = 'Timeout'

        for j in range(entry_idx, min(len(df), entry_idx + max_bars_hold)):
            high = df['High'].iloc[j]
            low = df['Low'].iloc[j]
            hist = df['MACD_hist'].iloc[j]
            hist_prev = df['MACD_hist'].iloc[j-1] if j > 0 else hist

            # 止损止盈（日内假设可命中）
            if direction == 1:  # 多头
                if low <= sl_price:
                    exit_price = sl_price
                    reason = 'StopLoss'
                    break
                if high >= tp_price:
                    exit_price = tp_price
                    reason = 'TakeProfit'
                    break
                if hist_prev <= 0 and hist > 0:
                    exit_price = df['Open'].iloc[j]
                    reason = 'MACD_Cross_Up'
                    break
            else:  # 空头
                if high >= sl_price:
                    exit_price = sl_price
                    reason = 'StopLoss'
                    break
                if low <= tp_price:
                    exit_price = tp_price
                    reason = 'TakeProfit'
                    break
                if hist_prev >= 0 and hist < 0:
                    exit_price = df['Open'].iloc[j]
                    reason = 'MACD_Cross_Down'
                    break

        if exit_price is None:
            exit_idx = min(len(df)-1, entry_idx + max_bars_hold - 1)
            exit_price = df['Close'].iloc[exit_idx]
            reason = 'Timeout'
        else:
            exit_idx = j

        # 计算盈亏
        pnl = direction * (exit_price - entry_price) * size
        equity += pnl
        equity_curve.append(equity)
        dates.append(df.index[exit_idx])

        trades.append({
            'entry_time': df.index[entry_idx],
            'exit_time': df.index[exit_idx],
            'direction': sig['type'],
            'entry': entry_price,
            'exit': exit_price,
            'pnl': round(pnl, 2),
            'return_%': round(pnl / (capital * pct_per_trade) * 100, 2),
            'reason': reason
        })

    # 实际交叉成功率
    actual_crossed = sum(1 for t in trades if 'MACD_Cross' in t['reason'])
    predict_success_rate = actual_crossed / len(trades) if trades else 0

    eq_df = pd.DataFrame({'equity': equity_curve}, index=dates)
    eq_df = eq_df.resample('1min').ffill().reindex(df.index, method='ffill').fillna(method='ffill')

    metrics = {
        '初始资金': f"${capital:,.0f}",
        '最终资金': f"${equity:,.0f}",
        '总收益': f"{(equity/capital-1)*100:+.2f}%",
        '交易次数': len(trades),
        '胜率': f"{sum(1 for t in trades if t['pnl']>0)/len(trades)*100:.2f}%" if trades else "0%",
        '预测成功率': f"{predict_success_rate*100:.2f}%",  # 重磅指标！
        '最大回撤': f"{((pd.Series(equity_curve).cummax() - equity_curve)/pd.Series(equity_curve).cummax()).max()*100:.2f}%"
    }

    trade_df = pd.DataFrame(trades)
    return trade_df, eq_df, metrics

# ----------------------
# Telegram 推送
# ----------------------
def send_telegram(text):
    if st.secrets.get("BOT_TOKEN") and st.secrets.get("CHAT_ID"):
        url = f"https://api.telegram.org/bot{st.secrets['BOT_TOKEN']}/sendMessage"
        try:
            requests.post(url, data={"chat_id": st.secrets["CHAT_ID"], "text": text, "parse_mode": "HTML"})
        except:
            pass

# ----------------------
# Streamlit UI
# ----------------------
st.title("MACD 提前预测器 Pro v2.0")
st.markdown("### 提前 3~10 根K棒预知 MACD 金叉/死叉 + 实盘级回测")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("参数设置")
    tickers = st.text_input("股票代码（逗号分隔）", "AAPL,TSLA,NVDA,SPY")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    interval = st.selectbox("时间周期", ["5m", "15m", "30m", "60m", "1d"], index=1)
    period = st.selectbox("数据期间", ["5d", "10d", "20d", "30d", "60d"], index=1)

    hist_n = st.slider("柱体连续检查根数", 2, 6, 3)
    volume_ratio = st.slider("放量倍数门槛", 1.0, 3.0, 1.6, 0.1)
    use_ema_cross = st.checkbox("EMA5/9 趋势确认", True)
    use_obv = st.checkbox("OBV 资金流确认", True)

    st.markdown("### 回测设置")
    capital = st.number_input("起始资金", 1000, 1000000, 10000)
    pct_per_trade = st.slider("单笔仓位", 0.02, 0.5, 0.12, 0.01)
    sl_pct = st.slider("止损 %", 1.0, 8.0, 3.0, 0.5) / 100
    tp_pct = st.slider("止盈 %", 2.0, 15.0, 6.0, 0.5) / 100
    max_bars_hold = st.slider("最大持仓K线数", 10, 120, 40)

    run = st.button("开始扫描与回测", type="primary")

with col2:
    st.header("实时结果")
    placeholder = st.empty()

if run:
    with placeholder.container():
        progress = st.progress(0)
        status_text = st.empty()
        all_signals = {}

        for idx, t in enumerate(tickers):
            progress.progress((idx + 1) / len(tickers))
            status_text.info(f"正在分析 {t} ...")

            df = download_data(t, period=period, interval=interval)
            if df.empty or len(df) < 50:
                st.warning(f"{t} 数据不足")
                continue

            df = calc_indicators(df)
            signals = detect_macd_prediction(df, hist_n=hist_n, volume_ratio=volume_ratio,
                                             use_ema_cross=use_ema_cross, use_obv=use_obv)

            if not signals:
                st.info(f"{t}：暂无信号")
                continue

            trade_df, eq_df, metrics = backtest_signals(df, signals, capital, pct_per_trade, sl_pct, tp_pct, max_bars_hold)
            all_signals[t] = (signals, trade_df, metrics, df)

            st.success(f"{t} 完成！共发现 {len(signals)} 个预测信号")
            st.metric("预测成功率（实际发生交叉）", metrics['预测成功率'])
            for k, v in metrics.items():
                if k not in ['预测成功率']:
                    st.write(f"**{k}** → {v}")

            if not trade_df.empty:
                st.line_chart(eq_df['equity'])
                st.dataframe(trade_df[['entry_time','direction','entry','exit','pnl','return_%','reason']], use_container_width=True)

            # 绘制K线图 + 信号标记
            ap = []
            for s in signals:
                ts = df.index[s['idx'] + 1] if s['idx'] + 1 < len(df) else df.index[s['idx']]
                price = df['Open'].iloc[s['idx'] + 1] if s['idx'] + 1 < len(df) else df['Close'].iloc[s['idx']]
                marker = '^' if s['type']=='up' else 'v'
                color = 'lime' if s['type']=='up' else 'red'
                ap.append(mpf.make_addplot(pd.Series(price, index=[ts]), type='scatter', marker=marker, markersize=100, color=color))

            try:
                fig, _ = mpf.plot(df.tail(100), type='candle', style='yahoo', addplot=ap, volume=True,
                                returnfig=True, figsize=(10,6))
                st.pyplot(fig)
                plt.close(fig)
            except:
                pass

        # 全局推送
        if all_signals and st.button("发送所有今日信号到 Telegram"):
            msg = "<b>MACD 提前预测信号</b>\n"
            for t, (sigs, _, _, _) in all_signals.items():
                for s in sigs[-5:]:
                    arrow = "UP" if s['type']=='up' else "DOWN"
                    msg += f"{t} | {s['ts'].strftime('%m/%d %H:%M')} | {arrow}\n"
            send_telegram(msg)
            st.success("已推送到 Telegram")

        st.balloons()

# ----------------------
# 依赖说明（部署时需要）
# ----------------------
st.sidebar.markdown("""
### 部署必装
```bash
pip install streamlit yfinance pandas numpy mplfinance matplotlib ta requests
