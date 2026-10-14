import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from tabulate import tabulate

os.makedirs("graphs", exist_ok=True)

# Задача: кластеризація HTTP-запитів до вебсервера
# x1 - час відповіді сервера, мс
# x2 - розмір відповіді, КБ
rng = np.random.default_rng(42)
groups = [  # (кількість, сер. час, СКВ, сер. розмір, СКВ)
    (170, 50, 15, 6, 3),          # легкі запити: статика, кеш
    (140, 300, 45, 80, 15),       # звичайні сторінки та API
    (70, 500, 70, 320, 40),       # важкі запити: файли, звіти
    (70, 1300, 150, 30, 10),      # аномально повільні: проблеми з БД
]
data = np.vstack([
    np.column_stack((rng.normal(t, s_t, n), rng.normal(sz, s_sz, n)))
    for n, t, s_t, sz, s_sz in groups
])
data[:, 0] = np.clip(data[:, 0], 5, None)     # час відповіді >= 5 мс
data[:, 1] = np.clip(data[:, 1], 0.5, None)   # розмір >= 0.5 КБ
print(f"Кількість запитів: {len(data)}")
print(f"Час відповіді: від {data[:, 0].min():.1f} до {data[:, 0].max():.1f} мс")
print(f"Розмір відповіді: від {data[:, 1].min():.1f} до {data[:, 1].max():.1f} КБ")

plt.scatter(data[:, 0], data[:, 1], s=12)
plt.xlabel("Час відповіді, мс")
plt.ylabel("Розмір відповіді, КБ")
plt.title("HTTP-запити до вебсервера")
plt.grid(True)
plt.savefig("graphs/01_data.png")
plt.show()

# Мін-макс нормалізація: ознаки мають різні шкали
data_min = data.min(axis=0)
data_max = data.max(axis=0)
data_norm = (data - data_min) / (data_max - data_min)

# Вибір кількості кластерів за коефіцієнтом розбиття FPC
fpcs = []
for c in range(2, 9):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_norm.T, c, m=2, error=0.005, maxiter=1000, seed=42)
    fpcs.append(fpc)
    print(f"c = {c}: FPC = {fpc:.4f}, ітерацій = {p}")

plt.plot(range(2, 9), fpcs, marker="o")
plt.xlabel("Кількість кластерів")
plt.ylabel("FPC")
plt.title("Коефіцієнт нечіткого розбиття")
plt.grid(True)
plt.savefig("graphs/02_fpc.png")
plt.show()

# Нечітка кластеризація FCM з 4 кластерами
n_clusters = 4
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_norm.T, n_clusters, m=2, error=0.005, maxiter=1000, seed=42)
labels = np.argmax(u, axis=0)
centers = cntr * (data_max - data_min) + data_min   # у вихідних одиницях

print(f"Кількість ітерацій: {p}, FPC = {fpc:.4f}")
table = [[i + 1, round(centers[i, 0], 2), round(centers[i, 1], 2),
          np.sum(labels == i)] for i in range(n_clusters)]
print(tabulate(table, headers=["Кластер", "Час, мс", "Розмір, КБ",
                               "Запитів"], tablefmt="grid"))

for i in range(n_clusters):
    pts = data[labels == i]
    plt.scatter(pts[:, 0], pts[:, 1], s=12, label=f"Кластер {i + 1}")
plt.scatter(centers[:, 0], centers[:, 1], marker="*", s=250,
            color="black", label="Центри")
plt.xlabel("Час відповіді, мс")
plt.ylabel("Розмір відповіді, КБ")
plt.title("Кластери з центрами")
plt.legend()
plt.grid(True)
plt.savefig("graphs/03_clusters.png")
plt.show()

# Зміна значень цільової функції
print(tabulate([[i + 1, round(v, 4)] for i, v in enumerate(jm)],
               headers=["Ітерація", "J"], tablefmt="grid"))

plt.plot(range(1, len(jm) + 1), jm, marker="o")
plt.xlabel("Кількість ітерацій")
plt.ylabel("Значення цільової функції")
plt.title("Зміна значень цільової функції")
plt.grid(True)
plt.savefig("graphs/04_objective.png")
plt.show()
