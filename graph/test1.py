import matplotlib.pyplot as plt

# Dati dei punti x e y
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Creazione del grafico a dispersione (scatter plot)
plt.scatter(x, y, color='blue', marker='o')

# Aggiunta di titolo e etichette degli assi
plt.title('Grafico a punti')
plt.xlabel('Asse X')
plt.ylabel('Asse Y')

# Mostra il grafico
plt.show()
