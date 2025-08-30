
import numpy as np
import matplotlib.pyplot as plt

def calculate_ema(data, period):
    alpha = 2 / (period + 1)
    ema = [np.mean(data[:period])]
    for price in data[period:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    return np.array(ema)

def calculate_macd(data, short_period, long_period, signal_period):
    ema_short = calculate_ema(data, short_period)
    ema_long = calculate_ema(data, long_period)

    min_len = min(len(ema_short), len(ema_long))
    macd_line = ema_short[-min_len:] - ema_long[-min_len:]
    signal_line = calculate_ema(macd_line, signal_period)

    macd_line = macd_line[-len(signal_line):]
    return macd_line, signal_line

def calculate_sma(data, period):
    return np.convolve(data, np.ones(period) / period, mode='valid')

def calculate_rsi(data, period):
    deltas = np.diff(data)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(data)
    rsi[:period] = 100 - 100 / (1 + rs)
    for i in range(period, len(data)):
        delta = deltas[i - 1]
        upval = max(delta, 0)
        downval = -min(delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100 - 100 / (1 + rs)
    return rsi

def backtest_strategy(data,
                      sma_period, ema_period,
                      macd_short, macd_long, macd_signal,
                      rsi_period, rsi_oversold, rsi_overbought):

    sma = calculate_sma(data, sma_period)
    ema = calculate_ema(data, ema_period)
    rsi = calculate_rsi(data, rsi_period)

    macd_line, signal_line = calculate_macd(data, macd_short, macd_long, macd_signal)

    min_len = min(len(sma), len(ema), len(signal_line), len(rsi))

    sma = sma[-min_len:]
    ema = ema[-min_len:]
    rsi = rsi[-min_len:]
    macd_line = macd_line[-min_len:]
    signal_line = signal_line[-min_len:]
    price_aligned = data[-min_len:]

    position = 0
    cash = 0
    entry_price = 0

    for i in range(min_len):
        price = price_aligned[i]

        if position == 0:
            if (price > sma[i] and price > ema[i] and
                macd_line[i] > signal_line[i] and
                macd_line[i-1] <= signal_line[i-1] and
                rsi[i] < rsi_oversold):
                position = 1
                entry_price = price

        elif position == 1:
            if (price < sma[i] or price < ema[i] or
                (macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]) or
                rsi[i] > rsi_overbought):
                cash += price - entry_price
                position = 0

    if position == 1:
        cash += data[-1] - entry_price

    return cash

def optimize_params(data):
    best_profit = -np.inf
    best_params = None

    for sma_p in range(5, 31, 5):
        for ema_p in range(5, 31, 5):
            for macd_s in range(8, 16, 2):
                for macd_l in range(20, 31, 2):
                    if macd_l <= macd_s:
                        continue
                    for macd_sig in range(5, 11):
                        for rsi_p in range(10, 21, 5):
                            for rsi_os in range(20, 41, 5):
                                for rsi_ob in range(60, 81, 5):
                                    profit = backtest_strategy(data,
                                                              sma_p, ema_p,
                                                              macd_s, macd_l, macd_sig,
                                                              rsi_p, rsi_os, rsi_ob)
                                    if profit > best_profit:
                                        best_profit = profit
                                        best_params = (sma_p, ema_p, macd_s, macd_l, macd_sig, rsi_p, rsi_os, rsi_ob)

    return best_params, best_profit


# Dati sintetici per test
np.random.seed(42)
days = 500
prices = np.cumsum(np.random.normal(0, 1, days)) + 100

best_params, best_profit = optimize_params(prices)
print("Migliori parametri (SMA, EMA, MACD_short, MACD_long, MACD_signal, RSI_period, RSI_oversold, RSI_overbought):")
print(best_params)
print("Profitto simulato:", best_profit)

# Plot prezzi
plt.plot(prices)
plt.title("Prezzi simulati con ottimizzazione parametri")
plt.show()
