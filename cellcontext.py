import json, random

def run(seed=7):
    rng = random.Random(seed)
    contexts, genes, perturbations = 5, 24, 8
    base = [[rng.uniform(1, 5) for _ in range(genes)] for _ in range(contexts)]
    shared = [[rng.uniform(-.7, .7) for _ in range(genes)] for _ in range(perturbations)]
    slopes = [[rng.uniform(-.32, .32) for _ in range(genes)] for _ in range(perturbations)]
    interaction = [[[slopes[p][g] * (c-2) + rng.gauss(0, .03) for g in range(genes)]
                    for p in range(perturbations)] for c in range(contexts)]
    train, test = [], []
    for c in range(contexts):
        for p in range(perturbations):
            target = [base[c][g] + shared[p][g] + interaction[c][p][g] +
                      rng.gauss(0, .08) for g in range(genes)]
            (test if (c * 3 + p) % 5 == 0 else train).append((c, p, target))
    global_effect = [[0.0] * genes for _ in range(perturbations)]
    counts = [0] * perturbations
    context_effect = {}
    for c, p, y in train:
        effect = [y[g] - base[c][g] for g in range(genes)]
        counts[p] += 1
        global_effect[p] = [a + b for a, b in zip(global_effect[p], effect)]
        context_effect[c, p] = effect
    global_effect = [[v / counts[p] for v in row] for p, row in enumerate(global_effect)]
    global_err = context_err = n = 0
    for c, p, y in test:
        nearest = min((cc for cc, pp in context_effect if pp == p), key=lambda cc: abs(cc-c))
        for g in range(genes):
            global_err += abs(base[c][g] + global_effect[p][g] - y[g])
            context_err += abs(base[c][g] + context_effect[nearest, p][g] - y[g])
            n += 1
    result = {"global_mae": round(global_err/n, 4), "context_mae": round(context_err/n, 4)}
    result["mae_reduction_pct"] = round(100 * (1-result["context_mae"]/result["global_mae"]), 1)
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
