"""
Завдання:
    Прогноз заряду батареї через 1 годину за поточним зарядом,
    яскравістю екрана та навантаженням процесора (мережа TSK / ANFIS).
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
N_SAMPLES = 500
TEST_PART = 0.2
N_MF = 2               # ФП на кожен вхід
EPOCHS = 40
LEARNING_RATE = 0.01

INPUT_NAMES = ["Заряд, %", "Яскравість, %", "Навантаження CPU, %"]
MF_NAMES = ["Низький", "Високий"]
OUTPUT_NAME = "Заряд через 1 год, %"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def generate_data(n, seed):
    """Генерує дані розряду батареї за 1 годину."""
    rng = np.random.default_rng(seed)
    charge = rng.uniform(5, 100, n)
    brightness = rng.uniform(0, 100, n)
    cpu = rng.uniform(0, 100, n)

    # витрата за годину: система + екран + процесор (нелінійно)
    consumption = 4 + 0.10 * brightness + 0.08 * cpu + 0.0012 * cpu ** 2
    # режим енергозбереження при заряді < 20 %
    consumption = np.where(charge < 20, consumption * 0.7, consumption)
    consumption += rng.normal(0, 1.0, n)

    charge_next = np.clip(charge - consumption, 0, 100)
    X = np.column_stack([charge, brightness, cpu])
    return np.round(X, 1), np.round(charge_next, 1)


def save_dat(path, X, y):
    np.savetxt(path, np.column_stack([X, y]), fmt="%.1f", delimiter="\t")


def load_dat(path):
    data = np.loadtxt(path)
    return data[:, :3], data[:, 3]


def split_data(X, y, test_part, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_part)
    test, train = idx[:n_test], idx[n_test:]
    return X[train], y[train], X[test], y[test]


def gbellmf(x, a, b, c):
    """Узагальнена дзвоноподібна ФП."""
    return 1.0 / (1.0 + np.abs((x - c) / a) ** (2 * b))


class ANFIS:
    """Нейро-нечітка мережа TSK."""

    def __init__(self, n_inputs, n_mf):
        self.n_inputs = n_inputs
        self.n_mf = n_mf
        # правила - усі комбінації ФП входів
        self.rules = np.array(list(itertools.product(range(n_mf), repeat=n_inputs)))
        self.n_rules = len(self.rules)

        # параметри ФП (як у genfis: a = 0.5/(m-1), b = 2)
        self.c = np.tile(np.linspace(0, 1, n_mf), (n_inputs, 1))
        self.a = np.full((n_inputs, n_mf), 0.5 / (n_mf - 1))
        self.b = np.full((n_inputs, n_mf), 2.0)

        # лінійні параметри TSK: p0, p1, ..., pN для кожного правила
        self.p = np.zeros((self.n_rules, n_inputs + 1))

    def layer1(self, X):
        """Шар 1: фазифікація."""
        return gbellmf(X[:, :, None], self.a[None], self.b[None], self.c[None])

    def layer2(self, mu):
        """Шар 2: активізація правил (добуток)."""
        w = np.ones((mu.shape[0], self.n_rules))
        for i in range(self.n_inputs):
            w *= mu[:, i, self.rules[:, i]]
        return w

    def forward(self, X):
        mu = self.layer1(X)
        w = self.layer2(mu)
        w_sum = w.sum(axis=1, keepdims=True) + 1e-12
        w_norm = w / w_sum
        X1 = np.column_stack([np.ones(len(X)), X])
        f = X1 @ self.p.T                        # шар 3: функції TSK
        y = (w_norm * f).sum(axis=1)             # шари 4-5
        return y, mu, w, w_sum, w_norm, f, X1

    def predict(self, X):
        return self.forward(X)[0]

    def fit_linear(self, X, y):
        """Лінійні параметри - методом найменших квадратів."""
        _, _, _, _, w_norm, _, X1 = self.forward(X)
        A = (w_norm[:, :, None] * X1[:, None, :]).reshape(len(X), -1)
        ridge = 1e-6
        p = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y)
        self.p = p.reshape(self.n_rules, self.n_inputs + 1)

    def fit_nonlinear(self, X, y, lr):
        """Параметри ФП (a, b, c) - градієнтним спуском."""
        y_pred, mu, w, w_sum, _, f, _ = self.forward(X)
        err = y_pred - y
        dy_dw = (f - y_pred[:, None]) / w_sum

        grad_a = np.zeros_like(self.a)
        grad_b = np.zeros_like(self.b)
        grad_c = np.zeros_like(self.c)

        for i in range(self.n_inputs):
            others = np.ones_like(w)
            for k in range(self.n_inputs):
                if k != i:
                    others *= mu[:, k, self.rules[:, k]]
            for j in range(self.n_mf):
                mask = self.rules[:, i] == j
                dE_dmu = err * (dy_dw[:, mask] * others[:, mask]).sum(axis=1)

                # похідні ФП за параметрами
                m = mu[:, i, j]
                t = (X[:, i] - self.c[i, j]) / self.a[i, j]
                t = np.where(np.abs(t) < 1e-9, 1e-9, t)
                k_ = m * (1 - m)
                dmu_da = 2 * self.b[i, j] / self.a[i, j] * k_
                dmu_db = -2 * np.log(np.abs(t)) * k_
                dmu_dc = 2 * self.b[i, j] / (self.a[i, j] * t) * k_

                grad_a[i, j] = np.mean(dE_dmu * dmu_da)
                grad_b[i, j] = np.mean(dE_dmu * dmu_db)
                grad_c[i, j] = np.mean(dE_dmu * dmu_dc)

        # крок по нормованому градієнту
        norm = np.sqrt((grad_a ** 2).sum() + (grad_b ** 2).sum() + (grad_c ** 2).sum()) + 1e-12
        self.a -= lr * grad_a / norm
        self.b -= lr * grad_b / norm
        self.c -= lr * grad_c / norm
        self.a = np.maximum(self.a, 0.05)
        self.b = np.maximum(self.b, 0.5)

    def train(self, X, y, X_test, y_test, epochs, lr):
        """Гібридне навчання."""
        history = {"train": [], "test": []}
        for epoch in range(1, epochs + 1):
            self.fit_linear(X, y)
            self.fit_nonlinear(X, y, lr)
            self.fit_linear(X, y)
            history["train"].append(rmse(y, self.predict(X)))
            history["test"].append(rmse(y_test, self.predict(X_test)))
            if epoch == 1 or epoch % 10 == 0:
                print(f"  Епоха {epoch:3d}: RMSE train = {history['train'][-1] * 100:.3f} %, "
                      f"RMSE test = {history['test'][-1] * 100:.3f} %")
        return history


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


def metrics(y, y_pred):
    mae = np.mean(np.abs(y - y_pred))
    mask = y > 1  # без майже нульового заряду
    mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"MAE, %": mae, "RMSE, %": rmse(y, y_pred), "MAPE, %": mape, "R^2": r2}


def rules_text(model):
    """Правила ЯКЩО-ТО з висновками у відсотках."""
    lines = []
    short = ["Заряд", "Яскравість", "CPU"]
    for k, rule in enumerate(model.rules):
        cond = " І ".join(f"{short[i]} = {MF_NAMES[j]}" for i, j in enumerate(rule))
        p = model.p[k]
        concl = (f"y = {100 * p[0]:.2f} {p[1]:+.3f}*Заряд "
                 f"{p[2]:+.3f}*Яскравість {p[3]:+.3f}*CPU")
        lines.append(f"Правило {k + 1:2d}: ЯКЩО {cond}, ТО {concl}")
    return "\n".join(lines)


def plot_structure(model, path):
    fig, ax = plt.subplots(figsize=(11, 8))
    n_in, n_mf, n_r = model.n_inputs, model.n_mf, model.n_rules
    y_in = np.linspace(0.8, 0.2, n_in)
    y_mf = np.linspace(0.95, 0.05, n_in * n_mf)
    y_r = np.linspace(0.98, 0.02, n_r)
    xs = [0.05, 0.25, 0.5, 0.72, 0.95]

    for k in range(n_r):
        for i, j in enumerate(model.rules[k]):
            ax.plot([xs[1], xs[2]], [y_mf[i * n_mf + j], y_r[k]], color="lightgray", lw=0.4)
        ax.plot([xs[2], xs[3]], [y_r[k], y_r[k]], color="lightgray", lw=0.4)
        ax.plot([xs[3], xs[4]], [y_r[k], 0.5], color="lightgray", lw=0.4)
    for i in range(n_in):
        for j in range(n_mf):
            ax.plot([xs[0], xs[1]], [y_in[i], y_mf[i * n_mf + j]], color="gray", lw=0.8)

    ax.scatter([xs[0]] * n_in, y_in, s=500, c="tab:blue", zorder=3)
    ax.scatter([xs[1]] * (n_in * n_mf), y_mf, s=200, c="tab:orange", zorder=3)
    ax.scatter([xs[2]] * n_r, y_r, s=60, c="tab:green", zorder=3)
    ax.scatter([xs[3]] * n_r, y_r, s=60, c="tab:purple", zorder=3)
    ax.scatter([xs[4]], [0.5], s=500, c="tab:red", zorder=3)

    for i in range(n_in):
        ax.text(xs[0], y_in[i] + 0.05, INPUT_NAMES[i], ha="center", fontsize=9)
        for j in range(n_mf):
            ax.text(xs[1] - 0.02, y_mf[i * n_mf + j], MF_NAMES[j], ha="right",
                    va="center", fontsize=7)
    ax.text(xs[4], 0.56, "Вихід y", ha="center", fontsize=9)

    titles = ["Входи", "Шар 1\nфазифікація (ФП)", f"Шар 2\nправила ({n_r})",
              "Шари 3-4\nфункції TSK,\nнормалізація", "Шар 5\nвихід"]
    for x, t in zip(xs, titles):
        ax.text(x, 1.04, t, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.15)
    ax.axis("off")
    ax.set_title("Структура гібридної нейро-нечіткої мережі TSK (ANFIS)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_mfs(params_before, model, path):
    x = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(2, model.n_inputs, figsize=(13, 6), sharey=True)
    for row, (title, (a, b, c)) in enumerate(
            [("до навчання", params_before), ("після навчання", (model.a, model.b, model.c))]):
        for i in range(model.n_inputs):
            ax = axes[row, i]
            for j in range(model.n_mf):
                ax.plot(x * 100, gbellmf(x, a[i, j], b[i, j], c[i, j]), label=MF_NAMES[j])
            ax.set_title(f"{INPUT_NAMES[i]} ({title})", fontsize=10)
            ax.set_xlabel(INPUT_NAMES[i])
            ax.grid(alpha=0.3)
        axes[row, 0].set_ylabel("Ступінь належності")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_learning(history, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(history["train"]) + 1)
    ax.plot(epochs, np.array(history["train"]) * 100, label="Навчальна вибірка")
    ax.plot(epochs, np.array(history["test"]) * 100, label="Тестова вибірка")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("RMSE, %")
    ax.set_title("Крива навчання гібридної мережі")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_check(y_test, y_pred, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    n = min(50, len(y_test))
    axes[0].plot(y_test[:n], "o-", label="Фактичне", ms=4)
    axes[0].plot(y_pred[:n], "x--", label="Прогноз мережі", ms=5)
    axes[0].set_xlabel("Номер тестового прикладу")
    axes[0].set_ylabel(OUTPUT_NAME)
    axes[0].set_title(f"Порівняння на перших {n} тестових прикладах")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_test, y_pred, s=12, alpha=0.7)
    axes[1].plot([0, 100], [0, 100], "r--", label="Ідеальний прогноз")
    axes[1].set_xlabel("Фактичне, %")
    axes[1].set_ylabel("Прогноз, %")
    axes[1].set_title("Фактичні vs прогнозовані значення")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].hist(y_pred - y_test, bins=20, edgecolor="black")
    axes[2].set_xlabel("Похибка прогнозу, %")
    axes[2].set_ylabel("Кількість")
    axes[2].set_title("Розподіл похибок (тестова вибірка)")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_surface(model, path, charge=80):
    br, cpu = np.meshgrid(np.linspace(0, 100, 40), np.linspace(0, 100, 40))
    X = np.column_stack([np.full(br.size, charge), br.ravel(), cpu.ravel()]) / 100
    Z = model.predict(X).reshape(br.shape) * 100
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(br, cpu, Z, cmap="viridis")
    ax.set_xlabel(INPUT_NAMES[1])
    ax.set_ylabel(INPUT_NAMES[2])
    ax.set_zlabel(OUTPUT_NAME)
    ax.set_title(f"Поверхня нечіткого виводу (поточний заряд = {charge} %)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def predict_one(model, charge, brightness, cpu):
    x = np.array([[charge, brightness, cpu]]) / 100
    return float(np.clip(model.predict(x)[0] * 100, 0, 100))


def interactive(model):
    """Ручне введення значень (Enter - вихід)."""
    print("\nВведіть: заряд яскравість CPU (через пробіл), або Enter для виходу.")
    while True:
        s = input(">>> ").strip()
        if not s:
            break
        try:
            c, b, p = map(float, s.replace(",", ".").split())
            print(f"    Прогноз заряду через 1 год: {predict_one(model, c, b, p):.1f} %")
        except ValueError:
            print("    Потрібно ввести три числа, наприклад: 75 60 40")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Навчальна вибірка
    print("1. Генерування навчальної вибірки...")
    X, y = generate_data(N_SAMPLES, SEED)
    data_path = os.path.join(DATA_DIR, "battery_data.dat")
    save_dat(data_path, X, y)
    X, y = load_dat(data_path)
    X_train, y_train, X_test, y_test = split_data(X, y, TEST_PART, SEED)
    save_dat(os.path.join(DATA_DIR, "battery_train.dat"), X_train, y_train)
    save_dat(os.path.join(DATA_DIR, "battery_test.dat"), X_test, y_test)
    print(f"   Всього: {len(X)}, навчальна: {len(X_train)}, тестова: {len(X_test)}")
    print("   Перші 5 рядків (заряд, яскравість, CPU -> заряд через 1 год):")
    for row_x, row_y in zip(X[:5], y[:5]):
        print(f"   {row_x[0]:6.1f} {row_x[1]:6.1f} {row_x[2]:6.1f} -> {row_y:6.1f}")

    # нормування до [0, 1]
    Xtr, ytr, Xte, yte = X_train / 100, y_train / 100, X_test / 100, y_test / 100

    # 2. Структура мережі
    print("\n2. Генерування структури мережі TSK...")
    model = ANFIS(n_inputs=3, n_mf=N_MF)
    n_lin = model.p.size
    n_nonlin = model.a.size * 3
    print(f"   Входів: 3, ФП на вхід: {N_MF} (gbellmf), правил: {model.n_rules}")
    print(f"   Лінійних параметрів: {n_lin}, нелінійних параметрів: {n_nonlin}")
    plot_structure(model, os.path.join(RESULTS_DIR, "1_structure.png"))
    params_before = (model.a.copy(), model.b.copy(), model.c.copy())

    # 3. Навчання
    print(f"\n3. Навчання (гібридний метод, епох: {EPOCHS}, крок: {LEARNING_RATE})...")
    history = model.train(Xtr, ytr, Xte, yte, EPOCHS, LEARNING_RATE)
    plot_mfs(params_before, model, os.path.join(RESULTS_DIR, "2_membership_functions.png"))
    plot_learning(history, os.path.join(RESULTS_DIR, "3_learning_curve.png"))

    # 4. Система нечіткого виводу
    print("\n4. Система нечіткого виводу (база правил збережена у results/rules.txt):")
    text = rules_text(model)
    with open(os.path.join(RESULTS_DIR, "rules.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n".join(text.splitlines()[:5]) + "\n   ...")
    plot_surface(model, os.path.join(RESULTS_DIR, "5_surface.png"))

    # 5. Перевірка адекватності
    print("\n5. Перевірка адекватності моделі:")
    pred_train = np.clip(model.predict(Xtr) * 100, 0, 100)
    pred_test = np.clip(model.predict(Xte) * 100, 0, 100)
    for name, yt, yp in [("Навчальна", y_train, pred_train), ("Тестова", y_test, pred_test)]:
        m = metrics(yt, yp)
        print(f"   {name:9s}: " + ", ".join(f"{k} = {v:.3f}" for k, v in m.items()))
    plot_check(y_test, pred_test, os.path.join(RESULTS_DIR, "4_check.png"))

    print("\n6. Приклади прогнозу:")
    examples = [(100, 30, 10), (80, 70, 50), (60, 100, 95), (50, 20, 5), (15, 50, 60)]
    for c, b, p in examples:
        print(f"   Заряд {c:3d} %, яскравість {b:3d} %, CPU {p:3d} % -> "
              f"через 1 год: {predict_one(model, c, b, p):5.1f} %")

    print(f"\nГрафіки збережено у папці: {RESULTS_DIR}")
    interactive(model)
    plt.show()


if __name__ == "__main__":
    main()