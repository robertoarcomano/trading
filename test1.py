import numpy as np
import matplotlib.pyplot as plt


# Crea 100 punti equidistanti tra 0 e 10 sull'asse x
x = np.linspace(0, 10, 100)

# Genera funzione sinusoidale con rumore casuale
y = np.sin(x) + np.random.normal(0, 0.1, size=x.size)


window_size = 10

# Calcolo della SMA su una finestra di window_size punti
sma = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

# Calcolo deviazione standard mobile sulla stessa finestra
std = np.array([np.std(y[i:i+window_size]) for i in range(len(y) - window_size + 1)])

# Calcola le bande di Bollinger (moltiplicatore di 2)
upper_band = sma + 2 * std
lower_band = sma - 2 * std


# Funzione per calcolare la EMA con window_size
def calculate_ema(data, window_size):
    alpha = 2 / (window_size + 1)
    ema = [np.mean(data[:window_size])]  # Inizializza con la media semplice dei primi window_size valori
    for price in data[window_size:]:
        new_ema = (price * alpha) + (ema[-1] * (1 - alpha))
        ema.append(new_ema)
    return np.array(ema)


# Calcolo della EMA
ema = calculate_ema(y, window_size)


# Calcolo MACD e linea di segnale
# MACD = EMA12 - EMA26; useremo 12 e 26 come periodi
ema_12 = calculate_ema(y, 12)
ema_26 = calculate_ema(y, 26)

# Per allineare le due EMA, tagliamo la più lunga
min_len = min(len(ema_12), len(ema_26))
macd = ema_12[-min_len:] - ema_26[-min_len:]

# Linea di segnale: EMA a 9 periodi della MACD
signal = calculate_ema(macd, 9)

# Allineiamo l'asse x per tutti i grafici
x_sma = x[window_size - 1:]
x_macd = x[-min_len + 9 - 1:]  # allineiamo considerando l'EMA della MACD

# Grafico
plt.plot(x, y, label='Funzione con rumore', color='blue')
plt.plot(x_sma, sma, color='red', label=f'SMA (window={window_size})')
plt.plot(x_sma, upper_band, color='gray', label='Banda superiore')
plt.plot(x_sma, lower_band, color='darkgray', label='Banda inferiore')
plt.plot(x_sma, ema, color='yellow', label=f'EMA (window={window_size})')

plt.plot(x_macd, macd[-len(signal):], color='green', label='MACD')
plt.plot(x_macd, signal, color='orange', label='Linea di segnale')

plt.title('Funzione con rumore, SMA, EMA, Bande di Bollinger, MACD e Segnale')
plt.xlabel('Asse X')
plt.ylabel('Asse Y')
plt.legend()
plt.show()
