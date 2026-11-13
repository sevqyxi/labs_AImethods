import os
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

os.makedirs("graphs", exist_ok=True)


# Варіант 2: y = sin(x)/x, z = (x - y) * sin(x + y)
def func_y(x):
    return np.sinc(x[0] / np.pi)  # sin(x)/x, при x = 0 -> 1


# ГА шукає мінімум, тому для максимуму z мінімізуємо -z
def func_z(x):
    return -((x[0] - x[1]) * np.sin(x[0] + x[1]))


# Параметри ГА
POP_SIZE = 10
GENERATIONS = 100
ELITE_COUNT = 1
CROSSOVER_FRACTION = 0.8
RUNS = 5


def create_population(n_vars, low, high):
    return np.random.uniform(low, high, (POP_SIZE, n_vars))


def evaluate(population, func):
    return np.array([func(ind) for ind in population])


def tournament_selection(population, fitness, k=2):
    # з k випадкових особин обираємо ту, що має менше значення функції
    idx = np.random.randint(0, POP_SIZE, k)
    return population[idx[np.argmin(fitness[idx])]]


def crossover(parent1, parent2):
    # арифметичне схрещення: нащадок між двома батьками
    alpha = np.random.rand(len(parent1))
    return alpha * parent1 + (1 - alpha) * parent2


def mutation(individual, generation, low, high):
    # гаусова мутація, крок зменшується з номером покоління
    sigma = (high - low) * 0.5 * (1 - generation / GENERATIONS)
    child = individual + np.random.normal(0, sigma, len(individual))
    return np.clip(child, low, high)


def average_distance(population):
    dist = []
    for i in range(POP_SIZE):
        for j in range(i + 1, POP_SIZE):
            dist.append(np.linalg.norm(population[i] - population[j]))
    return np.mean(dist)


def genetic_algorithm(func, n_vars, low, high):
    population = create_population(n_vars, low, high)
    fitness = evaluate(population, func)
    best_history, mean_history, dist_history = [], [], []

    for gen in range(GENERATIONS):
        order = np.argsort(fitness)
        new_population = [population[i] for i in order[:ELITE_COUNT]]  # еліта

        n_crossover = round((POP_SIZE - ELITE_COUNT) * CROSSOVER_FRACTION)
        for _ in range(n_crossover):
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            new_population.append(crossover(p1, p2))

        while len(new_population) < POP_SIZE:
            parent = tournament_selection(population, fitness)
            new_population.append(mutation(parent, gen, low, high))

        population = np.array(new_population)
        fitness = evaluate(population, func)

        best_history.append(fitness.min())
        mean_history.append(fitness.mean())
        dist_history.append(average_distance(population))

    best = population[np.argmin(fitness)]
    return best, fitness.min(), best_history, mean_history, dist_history


def plot_ga(result, title, filename):
    best, best_fit, best_history, mean_history, dist_history = result
    generations = range(1, GENERATIONS + 1)

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 2, 1)
    plt.plot(generations, best_history, "k.", label="Best fitness")
    plt.plot(generations, mean_history, "b.", label="Mean fitness")
    plt.title(f"Best: {best_fit:.4f} Mean: {mean_history[-1]:.4f}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness value")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.bar(range(1, len(best) + 1), best)
    plt.title("Current Best Individual")
    plt.xlabel(f"Number of variables ({len(best)})")
    plt.ylabel("Current best individual")
    plt.xticks(range(1, len(best) + 1))

    plt.subplot(2, 2, 3)
    plt.plot(generations, dist_history, "b.")
    plt.title("Average Distance Between Individuals")
    plt.xlabel("Generation")
    plt.ylabel("Average Distance")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"graphs/{filename}")
    plt.show()


def run_several_times(func, n_vars, low, high):
    # ГА запускається кілька разів, обирається найкраще рішення
    results = [genetic_algorithm(func, n_vars, low, high) for _ in range(RUNS)]
    table = [[i + 1, *np.round(r[0], 4), round(r[1], 4)] for i, r in enumerate(results)]
    best_run = min(range(RUNS), key=lambda i: results[i][1])
    return results, table, best_run


# 1) Мінімум функції однієї змінної
low1, high1 = -4, 1
results1, table1, best_run1 = run_several_times(func_y, 1, low1, high1)
print("Мінімізація y = sin(x)/x на відрізку [-4; 1]")
print(tabulate(table1, headers=["Запуск", "x", "y(x)"], tablefmt="grid"))

x_min, y_min = results1[best_run1][0][0], results1[best_run1][1]
print(f"Найкращий запуск №{best_run1 + 1}: x = {x_min:.4f}, y_min = {y_min:.4f}\n")

plot_ga(results1[best_run1], "Minimization of y = sin(x)/x", "01_ga_min_y.png")

x_values = np.linspace(low1, high1, 500)
plt.plot(x_values, np.sinc(x_values / np.pi))
plt.plot(x_min, y_min, "ro", label=f"min ({x_min:.3f}; {y_min:.4f})")
plt.title("Y-function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.savefig("graphs/02_y_function.png")
plt.show()

# 2) Максимум функції двох змінних
low2, high2 = -1, 3
results2, table2, best_run2 = run_several_times(func_z, 2, low2, high2)
for row in table2:
    row[-1] = -row[-1]  # повертаємо знак, щоб вивести саме z
print("Максимізація z = (x - y)·sin(x + y) на відрізку [-1; 3]")
print(tabulate(table2, headers=["Запуск", "x", "y", "z(x, y)"], tablefmt="grid"))

(x_max, y_max), z_max = results2[best_run2][0], -results2[best_run2][1]
print(f"Найкращий запуск №{best_run2 + 1}: x = {x_max:.4f}, y = {y_max:.4f}, "
      f"z_max = {z_max:.4f}")

plot_ga(results2[best_run2], "Maximization of z = (x - y)·sin(x + y)", "03_ga_max_z.png")

grid = np.linspace(low2, high2, 100)
X, Y = np.meshgrid(grid, grid)
Z = (X - Y) * np.sin(X + Y)

ax = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
ax.scatter(x_max, y_max, z_max, color="red", s=50, label=f"max = {z_max:.4f}")
ax.view_init(elev=35, azim=-140)
ax.set_title("Z-function")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.savefig("graphs/04_z_surface.png")
plt.show()
