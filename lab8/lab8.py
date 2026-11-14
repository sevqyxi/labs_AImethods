import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tabulate import tabulate

os.makedirs("graphs", exist_ok=True)

# Габарити монтажного поля
A, B = 100, 80

# Габарити елементів розміщення (a_i, b_i)
sizes = np.array([
    [20, 15], [15, 10], [25, 20], [10, 10], [30, 15],
    [15, 15], [20, 10], [10, 20], [25, 10], [15, 20],
])
N = len(sizes)

# Матриця зв'язків C: c_ij - кількість з'єднань між елементами i та j
C = np.array([
    [0, 3, 0, 1, 0, 0, 2, 0, 0, 0],
    [3, 0, 2, 0, 0, 1, 0, 0, 0, 0],
    [0, 2, 0, 0, 3, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 2, 0, 0, 0, 1],
    [0, 0, 3, 0, 0, 0, 0, 2, 1, 0],
    [0, 1, 0, 2, 0, 0, 3, 0, 0, 0],
    [2, 0, 0, 0, 0, 3, 0, 0, 0, 2],
    [0, 0, 1, 0, 2, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 1, 0, 0, 3, 0, 2],
    [0, 0, 0, 1, 0, 0, 2, 0, 2, 0],
])

# Допустимі межі центрів, щоб елемент не виходив за поле
low = sizes / 2
high = np.array([A, B]) - sizes / 2

L_MAX = N ** 2 * np.sqrt(A ** 2 + B ** 2)

# Параметри ГА
POP_SIZE = 50
MAX_GENERATIONS = 1000
ELITE_COUNT = 2
K = 0.01
STAGNATION = 100  # поколінь без покращення після досягнення нульового перекриття


def total_length(z):
    # L = сума d_ij * c_ij по всіх парах i, j
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
    return (d * C).sum()


def total_overlap(z):
    # S_ij = [0.5(a_i + a_j) - |x_i - x_j|] * [0.5(b_i + b_j) - |y_i - y_j|]
    half = (sizes[:, None, :] + sizes[None, :, :]) / 2
    dist = np.abs(z[:, None, :] - z[None, :, :])
    over = np.clip(half - dist, 0, None)
    s = over[:, :, 0] * over[:, :, 1]
    np.fill_diagonal(s, 0)
    return s.sum()


def fitness(z, k=K):
    # F = k * O(L) + P(S_заг)
    return k * total_length(z) / L_MAX + total_overlap(z) / (N * A * B)


def random_chromosome():
    return np.random.uniform(low, high)


def selection_probs(f_values):
    # для мінімізації: менше F -> більша ефективність
    eff = 1 / f_values
    ps = eff / eff.sum()
    p_min, p_max = ps.min(), ps.max()
    if p_max == p_min:
        return np.ones_like(ps) / len(ps)
    psl = (ps - p_min) / (p_max - p_min)  # масштабування в [0; 1]
    return psl / psl.sum()


def crossover(parent1, parent2):
    # одноточковий кросовер, розріз по межі генів
    point = np.random.randint(1, N)
    child1 = np.vstack([parent1[:point], parent2[point:]])
    child2 = np.vstack([parent2[:point], parent1[point:]])
    return child1, child2


def mutation(z):
    # нові випадкові координати для випадкового гена
    z = z.copy()
    i = np.random.randint(N)
    z[i] = np.random.uniform(low[i], high[i])
    return z


def genetic_algorithm(mutation_prob, k=K):
    population = [random_chromosome() for _ in range(POP_SIZE)]
    f_values = np.array([fitness(z, k) for z in population])
    history = []
    best_f, last_improve = np.inf, 0

    for gen in range(1, MAX_GENERATIONS + 1):
        order = np.argsort(f_values)  # ранжування
        population = [population[i] for i in order]
        f_values = f_values[order]
        probs = selection_probs(f_values)

        new_population = population[:ELITE_COUNT]  # елітний відбір
        while len(new_population) < POP_SIZE:
            i, j = np.random.choice(POP_SIZE, 2, replace=False, p=probs)
            for child in crossover(population[i], population[j]):
                if np.random.rand() < mutation_prob:
                    child = mutation(child)
                new_population.append(child)

        population = new_population[:POP_SIZE]
        f_values = np.array([fitness(z, k) for z in population])

        best = population[np.argmin(f_values)]
        history.append(f_values.min())
        if f_values.min() < best_f - 1e-12:
            best_f, last_improve = f_values.min(), gen

        # зупинка: нульове перекриття і немає покращення STAGNATION поколінь
        if total_overlap(best) == 0 and gen - last_improve >= STAGNATION:
            break

    return best, history, gen


def draw_placement(z, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.add_patch(Rectangle((0, 0), A, B, fill=False, linewidth=2))
    for i in range(N):
        for j in range(i + 1, N):
            if C[i, j]:
                ax.plot([z[i, 0], z[j, 0]], [z[i, 1], z[j, 1]], "r-",
                        linewidth=C[i, j], alpha=0.5)
    for i in range(N):
        w, h = sizes[i]
        ax.add_patch(Rectangle((z[i, 0] - w / 2, z[i, 1] - h / 2), w, h,
                               facecolor="lightblue", edgecolor="blue", alpha=0.7))
        ax.text(z[i, 0], z[i, 1], str(i + 1), ha="center", va="center", fontsize=11)
    ax.set_xlim(-5, A + 5)
    ax.set_ylim(-5, B + 5)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nL = {total_length(z):.1f}, "
                 f"S = {total_overlap(z):.1f}, F = {fitness(z):.5f}")
    plt.savefig(f"graphs/{filename}")
    plt.show()


# 1) Дослідження ймовірності мутації
RUNS = 5
table = []
mean_gens, mean_f, zero_counts = [], [], []
probs_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for pm in probs_list:
    results = [genetic_algorithm(mutation_prob=pm) for _ in range(RUNS)]
    gens = [r[2] for r in results]
    f_best = [r[1][-1] for r in results]
    zero = sum(total_overlap(r[0]) == 0 for r in results)
    mean_gens.append(np.mean(gens))
    mean_f.append(np.mean(f_best))
    zero_counts.append(zero)
    table.append([pm, round(np.mean(gens), 1), f"{np.mean(f_best):.5f}",
                  f"{min(f_best):.5f}", f"{zero}/{RUNS}"])

print("Вплив імовірності мутації (середнє за 5 запусків)")
print(tabulate(table, headers=["P мутації", "Поколінь", "F середнє", "F найкраще",
                               "S = 0"], tablefmt="grid"))

# спершу найбільше запусків без перекриття, потім найменше середнє F
best_idx = min(range(len(probs_list)), key=lambda i: (-zero_counts[i], mean_f[i]))
MUTATION_PROB = probs_list[best_idx]
print(f"Обрана ймовірність мутації: {MUTATION_PROB}")

plt.figure()
plt.bar([str(p) for p in probs_list], mean_gens, color="steelblue", edgecolor="black")
plt.title("Duration of evolution vs mutation probability")
plt.xlabel("Mutation probability")
plt.ylabel("Generations")
plt.savefig("graphs/01_mutation_prob.png")
plt.show()

# 2) Дослідження вагового коефіцієнта k
table_k = []
for k in [0.01, 0.05, 0.1, 0.5, 1]:
    results = [genetic_algorithm(MUTATION_PROB, k) for _ in range(RUNS)]
    best = min(results, key=lambda r: r[1][-1])[0]
    table_k.append([k, round(total_length(best), 1), round(total_overlap(best), 1)])

print("\nВплив вагового коефіцієнта k (найкращий з 5 запусків)")
print(tabulate(table_k, headers=["k", "L", "S перекриття"], tablefmt="grid"))

# 3) Розв'язання задачі з обраними параметрами
initial = random_chromosome()
draw_placement(initial, "Initial random placement", "02_initial.png")

results = [genetic_algorithm(MUTATION_PROB) for _ in range(RUNS)]
best, history, gens = min(results, key=lambda r: (total_overlap(r[0]) > 0, r[1][-1]))

print(f"\nНайкраще розміщення (зупинка на поколінні {gens}):")
print(f"L = {total_length(best):.1f}, S перекриття = {total_overlap(best):.1f}, "
      f"F = {history[-1]:.5f}")
print(tabulate([[i + 1, *sizes[i], *np.round(best[i], 2)] for i in range(N)],
               headers=["Елемент", "a", "b", "x", "y"], tablefmt="grid"))

plt.figure()
plt.plot(range(1, len(history) + 1), history)
plt.title("Optimality function F")
plt.xlabel("Generation")
plt.ylabel("F")
plt.grid()
plt.savefig("graphs/03_fitness.png")
plt.show()

draw_placement(best, "GA placement", "04_result.png")
