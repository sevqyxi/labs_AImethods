
import os
import sys
import gzip
import ssl
import subprocess
import time
import urllib.request

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
DATA_DIR = os.path.join(BASE_DIR, "mnist_data")
os.makedirs(GRAPHS_DIR, exist_ok=True)


def save_and_open(fig, fname):
    """Зберігає графік у PDF в папку graphs і одразу відкриває його."""
    path = os.path.join(GRAPHS_DIR, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [графік] {path}")
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("win"):
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"  не вдалося відкрити {fname}: {e}")


# 1. Завантаження датасету MNIST
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

def ssl_context():
    """SSL-контекст із сертифікатами certifi (якщо пакет встановлено)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

def download_mnist():
    """Завантажує файли MNIST, якщо їх ще немає у папці mnist_data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    ctx = ssl_context()
    for fname in MNIST_FILES.values():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            continue
        for mirror in MNIST_MIRRORS:
            try:
                print(f"Завантаження {fname} ...")
                with urllib.request.urlopen(mirror + fname, context=ctx,
                                            timeout=60) as r:
                    data = r.read()
                with open(path, "wb") as f:
                    f.write(data)
                break
            except Exception as e:
                print(f"  не вдалося з {mirror}: {e}")
        else:
            raise RuntimeError(
                f"Не вдалося завантажити {fname}. "
                "Встановіть сертифікати: python3 -m pip install certifi")


def read_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0


def read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)


def load_mnist():
    download_mnist()
    p = lambda k: os.path.join(DATA_DIR, MNIST_FILES[k])
    x_train = read_images(p("train_images"))
    y_train = read_labels(p("train_labels"))
    x_test = read_images(p("test_images"))
    y_test = read_labels(p("test_labels"))
    return x_train, y_train, x_test, y_test

# 2. Шари згорткової нейронної мережі
class Conv2D:
    """
    Згортковий шар. Кожне ядро (матриця ваг k x k) ковзає по вхідному
    шару з кроком stride = 1, поелементно множиться на фрагмент
    зображення, результат підсумовується і записується у відповідну
    позицію карти ознак. Zero padding додає нульові пікселі по краях,
    щоб розмір виходу дорівнював розміру входу.
    """

    def __init__(self, in_ch, out_ch, k, padding):
        self.k, self.pad = k, padding
        fan_in = in_ch * k * k
        self.W = (np.random.randn(out_ch, in_ch, k, k) *
                  np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        N, C, H, W = x.shape
        k, p = self.k, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        Ho, Wo = H + 2 * p - k + 1, W + 2 * p - k + 1
        # усі фрагменти зображення під ядром (N, C, Ho, Wo, k, k)
        win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * k * k)
        out = cols @ self.W.reshape(self.W.shape[0], -1).T + self.b
        self.cache = (x.shape, cols, Ho, Wo)
        return out.reshape(N, Ho, Wo, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        (N, C, H, W), cols, Ho, Wo = self.cache
        k, p, F = self.k, self.pad, self.W.shape[0]
        d = dout.transpose(0, 2, 3, 1).reshape(-1, F)
        self.dW = (d.T @ cols).reshape(self.W.shape)
        self.db = d.sum(axis=0)
        dcols = (d @ self.W.reshape(F, -1)).reshape(N, Ho, Wo, C, k, k)
        dxp = np.zeros((N, C, H + 2 * p, W + 2 * p), dtype=np.float32)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + Ho, j:j + Wo] += \
                    dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p:p + H, p:p + W]

    def params(self):
        return [(self.W, self.dW, self.vW), (self.b, self.db, self.vb)]


class ReLU:
    """Функція активації ReLU: f(x) = max(0, x)."""

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

    def params(self):
        return []


class MaxPool2D:
    """
    Шар підвибірки (max pooling 2x2): з чотирьох сусідніх нейронів
    карти ознак вибирається максимальний -> розмір карти зменшується вдвічі.
    """

    def forward(self, x):
        N, C, H, W = x.shape
        r = x.reshape(N, C, H // 2, 2, W // 2, 2)
        out = r.max(axis=(3, 5))
        self.mask = (r == out[:, :, :, None, :, None])
        self.shape = x.shape
        return out

    def backward(self, dout):
        d = self.mask * dout[:, :, :, None, :, None]
        return d.reshape(self.shape)

    def params(self):
        return []


class Flatten:
    """Перетворення кінцевих карт ознак у вектор."""

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

    def params(self):
        return []


class Dense:
    """Повнозв'язний шар."""

    def __init__(self, n_in, n_out):
        self.W = (np.random.randn(n_in, n_out) *
                  np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return [(self.W, self.dW, self.vW), (self.b, self.db, self.vb)]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# 3. Структура згорткової нейронної мережі

class CNN:
    """
    Вхід 1x28x28
      -> Conv 8 ядер 5x5, padding 2  -> 8x28x28 -> ReLU
      -> MaxPool 2x2                 -> 8x14x14
      -> Conv 16 ядер 3x3, padding 1 -> 16x14x14 -> ReLU
      -> MaxPool 2x2                 -> 16x7x7
      -> Flatten                     -> 784
      -> Dense 784->64 -> ReLU
      -> Dense 64->10  -> Softmax (10 класів: цифри 0..9)
    """

    def __init__(self):
        self.layers = [
            Conv2D(1, 8, 5, padding=2), ReLU(), MaxPool2D(),
            Conv2D(8, 16, 3, padding=1), ReLU(), MaxPool2D(),
            Flatten(),
            Dense(16 * 7 * 7, 64), ReLU(),
            Dense(64, 10),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def step(self, lr, momentum=0.9):
        """Оновлення ваг: SGD з моментом."""
        for layer in self.layers:
            for w, dw, v in layer.params():
                v *= momentum
                v -= lr * dw
                w += v

    def predict_proba(self, x, batch=500):
        res = [softmax(self.forward(x[i:i + batch]))
               for i in range(0, len(x), batch)]
        return np.concatenate(res)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def feature_maps(self, x):
        """Карти ознак після першого та другого згорткових шарів."""
        maps = []
        for layer in self.layers:
            x = layer.forward(x)
            if isinstance(layer, ReLU) and len(maps) < 2 and x.ndim == 4:
                maps.append(x.copy())
        return maps

    def summary(self):
        print("Структура згорткової нейронної мережі:")
        shape = (1, 28, 28)
        total = 0
        x = np.zeros((1,) + shape, dtype=np.float32)
        for layer in self.layers:
            x = layer.forward(x)
            n = sum(p[0].size for p in layer.params())
            total += n
            name = layer.__class__.__name__
            if isinstance(layer, Conv2D):
                name += f" ({layer.W.shape[0]} ядер {layer.k}x{layer.k}, padding={layer.pad})"
            print(f"  {name:<38} вихід: {str(x.shape[1:]):<14} параметрів: {n}")
        print(f"  Усього параметрів: {total}\n")


def cross_entropy(logits, y):
    p = softmax(logits)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    grad = p.copy()
    grad[np.arange(n), y] -= 1
    return loss, grad / n

# 4. Навчання мережі методом зворотного поширення помилки
def train(model, x_train, y_train, x_test, y_test,
          epochs=3, batch_size=64, lr=0.01):
    history = {"loss": [], "train_acc": [], "test_acc": []}
    n = len(x_train)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        idx = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            logits = model.forward(x_train[b])
            loss, grad = cross_entropy(logits, y_train[b])
            model.backward(grad)
            model.step(lr)
            losses.append(loss)
            if (i // batch_size) % 200 == 0:
                print(f"  епоха {epoch}, крок {i // batch_size:4d}/{n // batch_size}, "
                      f"помилка = {loss:.4f}")
        sub = np.random.choice(n, 5000, replace=False)
        tr_acc = (model.predict(x_train[sub]) == y_train[sub]).mean()
        te_acc = (model.predict(x_test) == y_test).mean()
        history["loss"].append(float(np.mean(losses)))
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        print(f"Епоха {epoch}: середня помилка = {np.mean(losses):.4f}, "
              f"точність (навч.) = {tr_acc * 100:.2f}%, "
              f"точність (тест) = {te_acc * 100:.2f}%, "
              f"час = {time.time() - t0:.1f} с\n")
    return history

# 5. Зображення, що містять помилки
def add_gaussian_noise(x, sigma):
    return np.clip(x + np.random.randn(*x.shape).astype(np.float32) * sigma, 0, 1)


def add_salt_pepper(x, p):
    x = x.copy()
    r = np.random.rand(*x.shape)
    x[r < p / 2] = 0.0
    x[(r >= p / 2) & (r < p)] = 1.0
    return x


def erase_block(x, size):
    """Затирає випадковий квадратний фрагмент зображення."""
    x = x.copy()
    for img in x:
        i, j = np.random.randint(4, 24 - size + 1, size=2)
        img[:, i:i + size, j:j + size] = 0.0
    return x


def shift_images(x, d):
    """Зсув зображення на d пікселів вправо та вниз."""
    return np.roll(np.roll(x, d, axis=2), d, axis=3)


# 6. Графіки
def plot_training(history):
    ep = np.arange(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["loss"], "o-")
    ax[0].set_title("Помилка навчання (перехресна ентропія)")
    ax[0].set_xlabel("Епоха")
    ax[0].set_ylabel("Помилка")
    ax[0].grid(True)
    ax[1].plot(ep, np.array(history["train_acc"]) * 100, "o-", label="Навчальна вибірка")
    ax[1].plot(ep, np.array(history["test_acc"]) * 100, "s-", label="Тестова вибірка")
    ax[1].set_title("Точність розпізнавання")
    ax[1].set_xlabel("Епоха")
    ax[1].set_ylabel("Точність, %")
    ax[1].legend()
    ax[1].grid(True)
    for a in ax:
        a.set_xticks(ep)
    fig.tight_layout()
    save_and_open(fig, "training.pdf")


def plot_predictions(model, x, y, title, fname, n=20):
    pred = model.predict(x[:n])
    proba = model.predict_proba(x[:n]).max(axis=1)
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 0.9, 5.2))
    for i, a in enumerate(axes.flat):
        a.imshow(x[i, 0], cmap="gray")
        color = "green" if pred[i] == y[i] else "red"
        a.set_title(f"{pred[i]} ({proba[i] * 100:.0f}%)\nсправж.: {y[i]}",
                    fontsize=8, color=color)
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_and_open(fig, fname)


def plot_distortions(model, x, y, distortions, fname, n=8):
    rows = len(distortions)
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.3, rows * 1.6))
    for r, (name, xd) in enumerate(distortions):
        pred = model.predict(xd[:n])
        for c in range(n):
            a = axes[r, c]
            a.imshow(xd[c, 0], cmap="gray")
            color = "green" if pred[c] == y[c] else "red"
            a.set_title(f"{pred[c]}", fontsize=9, color=color)
            a.set_xticks([])
            a.set_yticks([])
            if c == 0:
                a.set_ylabel(name, fontsize=8)
    fig.suptitle("Розпізнавання зображень, що містять помилки "
                 "(над зображенням — відповідь мережі)")
    fig.tight_layout()
    save_and_open(fig, fname)


def plot_noise_curves(curves):
    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.6))
    for a, (title, xlabel, levels, accs) in zip(axes, curves):
        a.plot(levels, np.array(accs) * 100, "o-")
        a.set_title(title)
        a.set_xlabel(xlabel)
        a.set_ylabel("Точність, %")
        a.set_ylim(0, 100)
        a.grid(True)
    fig.tight_layout()
    save_and_open(fig, "accuracy_vs_noise.pdf")


def plot_feature_maps(model, img, label):
    maps = model.feature_maps(img[None])
    W1 = model.layers[0].W
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(4, 9, 1)
    ax.imshow(img[0], cmap="gray")
    ax.set_title(f"Вхід ({label})", fontsize=8)
    ax.axis("off")
    for i in range(8):
        ax = fig.add_subplot(4, 9, 2 + i)
        ax.imshow(W1[i, 0], cmap="bwr")
        ax.set_title(f"Ядро {i + 1}", fontsize=8)
        ax.axis("off")
        ax = fig.add_subplot(4, 9, 11 + i)
        ax.imshow(maps[0][0, i], cmap="viridis")
        ax.set_title(f"Карта 1.{i + 1}", fontsize=8)
        ax.axis("off")
    for i in range(16):
        ax = fig.add_subplot(4, 9, 19 + i + (1 if i >= 8 else 0) + 1)
        ax.imshow(maps[1][0, i], cmap="viridis")
        ax.set_title(f"Карта 2.{i + 1}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Ядра першого згорткового шару та карти ознак (feature maps)")
    fig.tight_layout()
    save_and_open(fig, "feature_maps.pdf")

# 7. Головна програма
def main():
    print("Завантаження MNIST ...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Навчальна вибірка: {x_train.shape}, тестова: {x_test.shape}\n")

    model = CNN()
    model.summary()

    print("Навчання мережі ...")
    history = train(model, x_train, y_train, x_test, y_test,
                    epochs=3, batch_size=64, lr=0.01)
    plot_training(history)

    #Навчальні зображення
    tr_acc = (model.predict(x_train[:10000]) == y_train[:10000]).mean()
    te_acc = (model.predict(x_test) == y_test).mean()
    print(f"Точність на навчальних зображеннях (10000 шт.): {tr_acc * 100:.2f}%")
    print(f"Точність на тестових зображеннях (10000 шт.):  {te_acc * 100:.2f}%\n")
    plot_predictions(model, x_train, y_train,
                     "Розпізнавання навчальних зображень", "train_predictions.pdf")
    plot_predictions(model, x_test, y_test,
                     "Розпізнавання тестових зображень", "test_predictions.pdf")

    #Зображення з помилками
    xs, ys = x_test[:2000], y_test[:2000]
    distortions = [
        ("Оригінал", xs),
        ("Гаусів шум σ=0.3", add_gaussian_noise(xs, 0.3)),
        ("Шум «сіль-перець» 15%", add_salt_pepper(xs, 0.15)),
        ("Затертий блок 8x8", erase_block(xs, 8)),
        ("Зсув на 3 пікселі", shift_images(xs, 3)),
    ]
    print("Точність на зображеннях з помилками:")
    for name, xd in distortions:
        acc = (model.predict(xd) == ys).mean()
        print(f"  {name:<24}: {acc * 100:.2f}%")
    plot_distortions(model, xs, ys, distortions, "distorted_predictions.pdf")

    sigmas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    sp = [0, 0.05, 0.1, 0.2, 0.3, 0.4]
    blocks = [0, 4, 6, 8, 10, 12]
    curves = [
        ("Гаусів шум", "σ", sigmas,
         [(model.predict(add_gaussian_noise(xs, s)) == ys).mean() for s in sigmas]),
        ("Шум «сіль-перець»", "частка пікселів", sp,
         [(model.predict(add_salt_pepper(xs, p)) == ys).mean() for p in sp]),
        ("Затертий блок", "розмір блоку, пікс.", blocks,
         [(model.predict(erase_block(xs, b) if b else xs) == ys).mean() for b in blocks]),
    ]
    print("\nЗалежність точності від рівня спотворень:")
    for title, xlabel, levels, accs in curves:
        print(f"  {title}: " + ", ".join(
            f"{l}: {a * 100:.1f}%" for l, a in zip(levels, accs)))
    plot_noise_curves(curves)

    plot_feature_maps(model, x_test[0], y_test[0])
    print(f"\nГрафіки збережено у папку '{GRAPHS_DIR}'.")


if __name__ == "__main__":
    main()