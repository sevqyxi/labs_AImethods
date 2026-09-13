from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz


# Базова множина
x = np.linspace(0, 10, 1001)
output_dir = Path(__file__).resolve().parent / "graphs_pdf"
output_dir.mkdir(exist_ok=True)


def save_plot(filename, title, curves, xlabel="x", ylabel="Ступінь приналежності"):
    """Допоміжна функція для збереження графіків у форматі PDF."""
    plt.figure(figsize=(9, 5))

    for y, label, style in curves:
        plt.plot(x, y, style, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    file_path = output_dir / filename
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Збережено: {file_path}")


# 1. Трикутна і трапецієподібна функції приналежності

triangle = fuzz.trimf(x, [2, 5, 8])
trapezoid = fuzz.trapmf(x, [1, 3, 7, 9])

save_plot(
    "01_triangular_and_trapezoidal.pdf",
    "Трикутна і трапецієподібна функції приналежності",
    [
        (triangle, "trimf [2, 5, 8]", "-"),
        (trapezoid, "trapmf [1, 3, 7, 9]", "--"),
    ],
)


# 2. Проста і двостороння функції приналежності Гауса

gauss = fuzz.gaussmf(x, 5, 1.2)

gauss2_1 = fuzz.gauss2mf(x, 3, 0.7, 7, 0.7)
gauss2_2 = fuzz.gauss2mf(x, 2.5, 1.0, 6.5, 1.3)
gauss2_3 = fuzz.gauss2mf(x, 4, 0.6, 8, 1.0)

save_plot(
    "02_gaussian_functions.pdf",
    "Проста і двосторонні функції приналежності Гауса",
    [
        (gauss, "gaussmf", "-"),
        (gauss2_1, "gauss2mf №1", "--"),
        (gauss2_2, "gauss2mf №2", "-."),
        (gauss2_3, "gauss2mf №3", ":"),
    ],
)


# 3. Функція приналежності "узагальнений дзвін"

bell = fuzz.gbellmf(x, 2, 4, 5)

save_plot(
    "03_generalized_bell.pdf",
    'Функція приналежності "узагальнений дзвін"',
    [
        (bell, "gbellmf [2, 4, 5]", "-"),
    ],
)

# 4. Набір сигмоїдальних функцій
# sigmf: основна одностороння сигмоїдна функція
# dsigmf: різниця двох сигмоїд
# psigmf: добуток двох сигмоїд

sigmoid = fuzz.sigmf(x, 5, 2)
dsigmoid = fuzz.dsigmf(x, 3, 3, 7, 3)
psigmoid = fuzz.psigmf(x, 3, 3, 7, -3)

save_plot(
    "04_sigmoid_functions.pdf",
    "Сигмоїдальні функції приналежності",
    [
        (sigmoid, "sigmf", "-"),
        (dsigmoid, "dsigmf", "--"),
        (psigmoid, "psigmf", "-."),
    ],
)


# 5. Поліноміальні функції приналежності: Z, PI і S функції

z_function = fuzz.zmf(x, 2, 5)
pi_function = fuzz.pimf(x, 1, 3, 7, 9)
s_function = fuzz.smf(x, 5, 8)

save_plot(
    "05_polynomial_functions.pdf",
    "Поліноміальні функції приналежності",
    [
        (z_function, "zmf", "-"),
        (pi_function, "pimf", "--"),
        (s_function, "smf", "-."),
    ],
)

# 6. Мінімаксна інтерпретація логічних операторів
# 
# кон'юнкція (AND) = minimum
# диз'юнкція (OR) = maximum
#
# Використаємо дві функції приналежності Гауса.

gauss_a = fuzz.gaussmf(x, 4, 1.3)
gauss_b = fuzz.gaussmf(x, 6, 1.3)

intersection_min = np.fmin(gauss_a, gauss_b)
union_max = np.fmax(gauss_a, gauss_b)

save_plot(
    "06_minimax_intersection.pdf",
    "Мінімаксна інтерпретація: перетин (AND)",
    [
        (gauss_a, "A", "--"),
        (gauss_b, "B", "--"),
        (intersection_min, "min(A, B)", "-"),
    ],
)

save_plot(
    "07_minimax_union.pdf",
    "Мінімаксна інтерпретація: об'єднання (OR)",
    [
        (gauss_a, "A", "--"),
        (gauss_b, "B", "--"),
        (union_max, "max(A, B)", "-"),
    ],
)

# 7. Імовірнісна інтерпретація кон'юнкції і диз'юнкції
# Кон'юнкція:
#     A AND B = A * B
#
# Диз'юнкція (алгебраїчна сума):
#     A OR B = A + B - A * B

intersection_prob = gauss_a * gauss_b
union_prob = gauss_a + gauss_b - gauss_a * gauss_b

save_plot(
    "08_probability_intersection.pdf",
    "Імовірнісна інтерпретація: кон'юнкція",
    [
        (gauss_a, "A", "--"),
        (gauss_b, "B", "--"),
        (intersection_prob, "A · B", "-"),
    ],
)

save_plot(
    "09_probability_union.pdf",
    "Імовірнісна інтерпретація: диз'юнкція",
    [
        (gauss_a, "A", "--"),
        (gauss_b, "B", "--"),
        (union_prob, "A + B - A·B", "-"),
    ],
)


# 8. Доповнення нечіткої множини
#     NOT A = 1 - A

complement = 1 - gauss_a

save_plot(
    "10_fuzzy_complement.pdf",
    "Доповнення нечіткої множини",
    [
        (gauss_a, "A", "--"),
        (complement, "NOT A = 1 - A", "-"),
    ],
)

