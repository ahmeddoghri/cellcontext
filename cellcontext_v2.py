"""Data-driven context similarity, as a parallel non-destructive fix.

cellcontext.py's "context-aware" method picks the nearest context by literal
array index (min(abs(cc - c))). That only works because the synthetic
generator makes the interaction term a linear function of context index
itself (slopes[p][g] * (c - 2)). Real biological contexts have no such
built-in ordering. This module replaces index proximity with a similarity
measured from training data: two contexts are "near" if their observed
effects agree on perturbations both have training data for.
"""
import json
import random


def _make_dataset(seed, shuffle_identity):
    rng = random.Random(seed)
    contexts, genes, perturbations = 5, 24, 8
    base = [[rng.uniform(1, 5) for _ in range(genes)] for _ in range(contexts)]
    shared = [[rng.uniform(-.7, .7) for _ in range(genes)] for _ in range(perturbations)]
    slopes = [[rng.uniform(-.32, .32) for _ in range(genes)] for _ in range(perturbations)]

    identity = list(range(contexts))
    if shuffle_identity:
        rng.shuffle(identity)

    interaction = [[[slopes[p][g] * (identity[c] - 2) + rng.gauss(0, .03) for g in range(genes)]
                    for p in range(perturbations)] for c in range(contexts)]
    train, test = [], []
    for c in range(contexts):
        for p in range(perturbations):
            target = [base[c][g] + shared[p][g] + interaction[c][p][g] +
                      rng.gauss(0, .08) for g in range(genes)]
            (test if (c * 3 + p) % 5 == 0 else train).append((c, p, target))
    return contexts, genes, perturbations, base, train, test


def _fit_effects(train, base, genes, perturbations):
    global_effect = [[0.0] * genes for _ in range(perturbations)]
    counts = [0] * perturbations
    context_effect = {}
    for c, p, y in train:
        effect = [y[g] - base[c][g] for g in range(genes)]
        counts[p] += 1
        global_effect[p] = [a + b for a, b in zip(global_effect[p], effect)]
        context_effect[c, p] = effect
    global_effect = [[v / counts[p] for v in row] for p, row in enumerate(global_effect)]
    return global_effect, context_effect


def run_index_baseline(seed=7, shuffle_identity=False):
    """Same index-proximity method as cellcontext.py's run(), parameterized
    so it can be evaluated against the shuffle_identity adversarial variant."""
    contexts, genes, perturbations, base, train, test = _make_dataset(seed, shuffle_identity)
    global_effect, context_effect = _fit_effects(train, base, genes, perturbations)
    global_err = context_err = n = 0
    for c, p, y in test:
        nearest = min((cc for cc, pp in context_effect if pp == p), key=lambda cc: abs(cc - c))
        for g in range(genes):
            global_err += abs(base[c][g] + global_effect[p][g] - y[g])
            context_err += abs(base[c][g] + context_effect[nearest, p][g] - y[g])
            n += 1
    result = {"global_mae": round(global_err / n, 4), "context_mae": round(context_err / n, 4)}
    result["mae_reduction_pct"] = round(100 * (1 - result["context_mae"] / result["global_mae"]), 1)
    return result


def run(seed=7, shuffle_identity=False):
    rng = random.Random(seed)
    contexts, genes, perturbations = 5, 24, 8
    base = [[rng.uniform(1, 5) for _ in range(genes)] for _ in range(contexts)]
    shared = [[rng.uniform(-.7, .7) for _ in range(genes)] for _ in range(perturbations)]
    slopes = [[rng.uniform(-.32, .32) for _ in range(genes)] for _ in range(perturbations)]

    identity = list(range(contexts))
    if shuffle_identity:
        rng.shuffle(identity)

    interaction = [[[slopes[p][g] * (identity[c] - 2) + rng.gauss(0, .03) for g in range(genes)]
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

    def similarity(c1, c2):
        shared_p = [p for p in range(perturbations) if (c1, p) in context_effect and (c2, p) in context_effect]
        if not shared_p:
            return None
        dist = 0.0
        for p in shared_p:
            e1, e2 = context_effect[c1, p], context_effect[c2, p]
            dist += sum(abs(a - b) for a, b in zip(e1, e2)) / len(e1)
        return dist / len(shared_p)

    global_err = context_err = n = 0
    for c, p, y in test:
        candidates = [cc for cc, pp in context_effect if pp == p]
        sims = [(cc, similarity(c, cc)) for cc in candidates]
        sims = [(cc, s) for cc, s in sims if s is not None]
        nearest = min(sims, key=lambda t: t[1])[0] if sims else candidates[0]
        for g in range(genes):
            global_err += abs(base[c][g] + global_effect[p][g] - y[g])
            context_err += abs(base[c][g] + context_effect[nearest, p][g] - y[g])
            n += 1
    result = {"global_mae": round(global_err / n, 4), "context_mae": round(context_err / n, 4)}
    result["mae_reduction_pct"] = round(100 * (1 - result["context_mae"] / result["global_mae"]), 1)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
