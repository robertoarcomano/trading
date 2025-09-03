import ast
import pandas as pd

# Apri e carica i dati dal file txt
with open('btceur.txt', 'r') as f:
    data = ast.literal_eval(f.read())

# Estraggo il prezzo di chiusura per ogni candela (close è il secondo elemento)
closes = [candle[1] for candle in data]

# Creo una Series pandas dei prezzi di chiusura
prices = pd.Series(closes)

# Calcolo EMA20 e EMA50
ema20 = prices.ewm(span=20, adjust=False).mean()
ema50 = prices.ewm(span=50, adjust=False).mean()

# Stato iniziale
in_position = False
entry_price = 0
trade_log = []
commission_rate = 0.001  # 0.1%

for i in range(1, len(prices)):
    # Condizione di acquisto: EMA20 incrocia EMA50 verso l'alto e close > EMA20
    if not in_position and ema20.iloc[i] > ema50.iloc[i] and ema20.iloc[i - 1] <= ema50.iloc[i - 1] and prices.iloc[i] > ema20.iloc[i]:
        entry_price = prices.iloc[i]
        in_position = True
        trade_log.append({'type': 'BUY', 'price': entry_price, 'index': i})
    
    # Condizione di vendita: EMA20 incrocia EMA50 verso il basso o close < EMA20
    elif in_position and (ema20.iloc[i] < ema50.iloc[i] or prices.iloc[i] < ema20.iloc[i]):
        exit_price = prices.iloc[i]
        in_position = False
        
        # Calcolo guadagno e commissioni
        gross_profit = (exit_price - entry_price)
        commission = (entry_price + exit_price) * commission_rate
        net_profit = gross_profit - commission
        
        trade_log.append({
            'type': 'SELL',
            'price': exit_price,
            'index': i,
            'gross_profit': gross_profit,
            'commission': commission,
            'net_profit': net_profit
        })

# Output risultati
total_net_profit = 0
for trade in trade_log:
    if trade['type'] == 'BUY':
        print(f"[{trade['index']}] BUY @ {trade['price']:.2f}")
    else:
        print(f"[{trade['index']}] SELL @ {trade['price']:.2f} | Gross profit: {trade['gross_profit']:.2f} | Commission: {trade['commission']:.2f} | Net profit: {trade['net_profit']:.2f}")
        total_net_profit += trade['net_profit']

print("\nTotal net profit: {:.2f}".format(total_net_profit))
