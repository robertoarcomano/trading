
    # def calculate_rsi(data, period):
    #     deltas = np.diff(data)
    #     seed = deltas[:period]
    #     up = seed[seed >= 0].sum() / period
    #     down = -seed[seed < 0].sum() / period
    #     rs = up / down if down != 0 else 0
    #     rsi = np.zeros_like(data)
    #     rsi[:period] = 100 - 100 / (1 + rs)
    #     for i in range(period, len(data)):
    #         delta = deltas[i - 1]
    #         upval = max(delta, 0)
    #         downval = -min(delta, 0)
    #         up = (up * (period - 1) + upval) / period
    #         down = (down * (period - 1) + downval) / period
    #         rs = up / down if down != 0 else 0
    #         rsi[i] = 100 - 100 / (1 + rs)
    #     return rsi
    
    # def calculate_ema(data, period):
    #     alpha = 2 / (period + 1)
    #     ema = [np.mean(data[:period])]
    #     for price in data[period:]:
    #         ema.append(alpha * price + (1 - alpha) * ema[-1])
    #     return np.array(ema)

    # def calculate_macd(data, short_period, long_period, signal_period):
    #     ema_short = calculate_ema(data, short_period)
    #     ema_long = calculate_ema(data, long_period)

    #     min_len = min(len(ema_short), len(ema_long))
    #     macd_line = ema_short[-min_len:] - ema_long[-min_len:]
    #     signal_line = calculate_ema(macd_line, signal_period)

    #     macd_line = macd_line[-len(signal_line):]
    #     return macd_line, signal_line
