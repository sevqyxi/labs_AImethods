import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.makedirs("graphs", exist_ok=True)

# Варіант 2:  y = sin(x)/x,  z = (x - y) * sin(x + y)
x_values = np.linspace(0, 20, 100)
y_values = np.sinc(x_values / np.pi)  # sin(x)/x, при x = 0 -> 1
z_values = (x_values - y_values) * np.sin(x_values + y_values)

plt.plot(x_values, y_values)
plt.title("Y-function")
plt.savefig("graphs/01_y_function.png")
plt.show()

plt.plot(x_values, z_values)
plt.title("Z-function")
plt.savefig("graphs/02_z_function.png")
plt.show()

# ---------- Трикутні функції приналежності ----------
x_means = np.linspace(min(x_values), max(x_values), 6)
y_means = np.linspace(min(y_values), max(y_values), 6)
z_means = np.linspace(min(z_values), max(z_values), 9)

x_diff, y_diff, z_diff = 3, 0.2, 4

y_range = np.linspace(min(y_values), max(y_values), 100)
z_range = np.linspace(min(z_values), max(z_values), 100)

mx = [fuzz.trimf(x_values, [x_means[i] - x_diff, x_means[i], x_means[i] + x_diff])
      for i in range(6)]
my = [fuzz.trimf(y_range, [y_means[i] - y_diff, y_means[i], y_means[i] + y_diff])
      for i in range(6)]
mf = [fuzz.trimf(z_range, [z_means[i] - z_diff, z_means[i], z_means[i] + z_diff])
      for i in range(9)]

for i in range(6):
    plt.plot(x_values, mx[i])
plt.title("X Trimf")
plt.savefig("graphs/03_x_trimf.png")
plt.show()

for i in range(6):
    plt.plot(y_range, my[i])
plt.title("Y Trimf")
plt.savefig("graphs/04_y_trimf.png")
plt.show()

for i in range(9):
    plt.plot(z_range, mf[i])
plt.title("Z Trimf")
plt.savefig("graphs/05_z_trimf.png")
plt.show()


def calculate_trimf(x, a, b, c):
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0


def calculate_gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def calculate_trapmf(x, a, b, c, d):
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    elif c < x <= d:
        return (d - x) / (d - c)
    else:
        return 0


def membership(value, mean, diff, kind):
    if kind == "trimf":
        return calculate_trimf(value, mean - diff, mean, mean + diff)
    if kind == "gaussmf":
        return calculate_gaussmf(value, mean, diff)
    return calculate_trapmf(value, mean - diff, mean - diff / 3,
                            mean + diff / 3, mean + diff)


def functionCompare(value, means, diff, kind="trimf"):
    best_func_value = -float("inf")
    best_index = -1
    for index, mean in enumerate(means):
        ff = membership(value, mean, diff, kind)
        if ff > best_func_value:
            best_func_value = ff
            best_index = index
    return best_index


print("Таблиця значень")
table = [["y\\x"] + [str(round(x, 2)) for x in x_means]]
for y_value in y_means:
    row = [round(y_value, 2)]
    for x in x_means:
        z = (x - y_value) * np.sin(x + y_value)
        row.append(round(z, 2))
    table.append(row)
print(tabulate(table, tablefmt="grid"))

rules = {}
print("Таблиця з назвами функцій")
table = [["y\\x"] + ["mx" + str(i) for i in range(1, 7)]]
for i in range(6):
    row = ["my" + str(i + 1)]
    for j in range(6):
        z = (x_means[j] - y_means[i]) * np.sin(x_means[j] + y_means[i])
        best_func = functionCompare(z, z_means, z_diff)
        row.append("mf" + str(best_func + 1))
        rules[(j, i)] = best_func
    table.append(row)
print(tabulate(table, tablefmt="grid"))

print("\nRules:")
for rule in rules.keys():
    print(f"if (x is mx{rule[0] + 1}) and (y is my{rule[1] + 1}) then (z is mf{rules[rule] + 1})")


# Вихід моделі: ступінь виконання правила = min(mu_x, mu_y),
# результат - зважене середнє центрів вихідних функцій
def fuzzy_output(x, y, rules, x_p, y_p, kind):
    mu_x = [membership(x, m, x_p, kind) for m in x_means]
    mu_y = [membership(y, m, y_p, kind) for m in y_means]
    num, den = 0, 0
    for (i, j), k in rules.items():
        w = min(mu_x[i], mu_y[j])
        num += w * z_means[k]
        den += w
    return num / den if den > 0 else 0


def relative_error(z_true, z_model):
    shift = 1 - min(z_true)   # "піднімаємо" обидві величини, щоб еталон >= 1
    return np.mean(np.abs((z_true + shift) - (z_model + shift)) / (z_true + shift)) * 100


z_output = []
for x in x_values:
    y = np.sinc(x / np.pi)
    z_output.append(fuzzy_output(x, y, rules, x_diff, y_diff, "trimf"))

plt.plot(x_values, z_output, label="Model")
plt.plot(x_values, z_values, label="True")
plt.title("Справжня і змодельована функції")
plt.legend()
plt.savefig("graphs/06_model_trimf.png")
plt.show()

mse = mean_squared_error(z_values, z_output)
mae = mean_absolute_error(z_values, z_output)
print(f"\nMean Squared Error (MSE) = {mse}")
print(f"Mean Absolute Error (MAE) = {mae}")
print(f"Relative Error = {relative_error(z_values, np.array(z_output))} %")

# ---------- Гауссові функції приналежності ----------
x_sigma = (max(x_values) - min(x_values)) / 6 / 2
y_sigma = (max(y_values) - min(y_values)) / 6 / 2
z_sigma = (max(z_values) - min(z_values)) / 9 / 2

x_mf_gaussian = [fuzz.gaussmf(x_values, x_means[i], x_sigma) for i in range(6)]
y_mf_gaussian = [fuzz.gaussmf(y_range, y_means[i], y_sigma) for i in range(6)]
z_mf_gaussian = [fuzz.gaussmf(z_range, z_means[i], z_sigma) for i in range(9)]

for i in range(6):
    plt.plot(x_values, x_mf_gaussian[i])
plt.title("X Gaussmf")
plt.savefig("graphs/07_x_gaussmf.png")
plt.show()

for i in range(6):
    plt.plot(y_range, y_mf_gaussian[i])
plt.title("Y Gaussmf")
plt.savefig("graphs/08_y_gaussmf.png")
plt.show()

for i in range(9):
    plt.plot(z_range, z_mf_gaussian[i])
plt.title("Z Gaussmf")
plt.savefig("graphs/09_z_gaussmf.png")
plt.show()

rules_gauss = {}
print("Таблиця з назвами функцій")
table = [["y/x"] + ["mx" + str(i) for i in range(1, 7)]]
for i in range(6):
    row = ["my" + str(i + 1)]
    for j in range(6):
        z = (x_means[j] - y_means[i]) * np.sin(x_means[j] + y_means[i])
        best_func = functionCompare(z, z_means, z_sigma, "gaussmf")
        row.append("mf" + str(best_func + 1))
        rules_gauss[(j, i)] = best_func
    table.append(row)
print(tabulate(table, tablefmt="grid"))

print("\nRules:")
for rule in rules_gauss.keys():
    print(f"if (x is mx{rule[0] + 1}) and (y is my{rule[1] + 1}) then (z is mf{rules_gauss[rule] + 1})")

z_output = []
for x in x_values:
    y = np.sinc(x / np.pi)
    z_output.append(fuzzy_output(x, y, rules_gauss, x_sigma, y_sigma, "gaussmf"))

plt.plot(x_values, z_output, label="Model")
plt.plot(x_values, z_values, label="True")
plt.legend()
plt.savefig("graphs/10_model_gaussmf.png")
plt.show()

mse = mean_squared_error(z_values, z_output)
mae = mean_absolute_error(z_values, z_output)
print(f"\nMean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Relative Error: {relative_error(z_values, np.array(z_output))} %")

# ---------- Трапецієподібні функції приналежності ----------
x_mf_trap = [fuzz.trapmf(x_values, [m - x_diff, m - x_diff / 3, m + x_diff / 3, m + x_diff])
             for m in x_means]
y_mf_trap = [fuzz.trapmf(y_range, [m - y_diff, m - y_diff / 3, m + y_diff / 3, m + y_diff])
             for m in y_means]
z_mf_trap = [fuzz.trapmf(z_range, [m - z_diff, m - z_diff / 3, m + z_diff / 3, m + z_diff])
             for m in z_means]

for name, universe, mfs in [("X", x_values, x_mf_trap), ("Y", y_range, y_mf_trap),
                            ("Z", z_range, z_mf_trap)]:
    for m in mfs:
        plt.plot(universe, m)
    plt.title(f"{name} Trapmf")
    plt.savefig(f"graphs/11_{name.lower()}_trapmf.png")
    plt.show()

z_output = []
for x in x_values:
    y = np.sinc(x / np.pi)
    z_output.append(fuzzy_output(x, y, rules, x_diff, y_diff, "trapmf"))

plt.plot(x_values, z_output, label="Model")
plt.plot(x_values, z_values, label="True")
plt.legend()
plt.savefig("graphs/12_model_trapmf.png")
plt.show()

mse = mean_squared_error(z_values, z_output)
mae = mean_absolute_error(z_values, z_output)
print(f"\nTrapmf Mean Squared Error (MSE): {mse}")
print(f"Trapmf Mean Absolute Error (MAE): {mae}")
print(f"Trapmf Relative Error: {relative_error(z_values, np.array(z_output))} %")

# ---------- Зменшення кількості правил: лише діагональ таблиці ----------
diag_rules = {(i, i): rules[(i, i)] for i in range(6)}
print("\nDiagonal rules:")
for rule in diag_rules.keys():
    print(f"if (x is mx{rule[0] + 1}) and (y is my{rule[1] + 1}) then (z is mf{diag_rules[rule] + 1})")

z_output = []
for x in x_values:
    y = np.sinc(x / np.pi)
    z_output.append(fuzzy_output(x, y, diag_rules, x_diff, y_diff, "trimf"))

plt.plot(x_values, z_output, label="Model")
plt.plot(x_values, z_values, label="True")
plt.title("Модель лише з діагональними правилами")
plt.legend()
plt.savefig("graphs/13_model_diag.png")
plt.show()

mse = mean_squared_error(z_values, z_output)
mae = mean_absolute_error(z_values, z_output)
print(f"\nDiagonal Mean Squared Error (MSE): {mse}")
print(f"Diagonal Mean Absolute Error (MAE): {mae}")
print(f"Diagonal Relative Error: {relative_error(z_values, np.array(z_output))} %")
