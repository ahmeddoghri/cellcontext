# cellcontext

**Context-aware single-cell perturbation prediction, with a baseline honest enough to beat.**

Tell the same joke to eight different people and you get eight different reactions — a couple of laughs, a wince, one person who reports you to HR. A cell reacts to a perturbation the same way. Knock out the same gene at the same dose in two different cellular contexts and you get two different transcriptional responses, not one "true" effect with noise sprinkled on top. Most perturbation-response benchmarks average across the room and call the average ground truth. cellcontext keeps track of who's actually in the room.

It's a compact, inspectable implementation inspired by [Nature Methods' 2025 benchmark of 27 perturbation-response methods across 29 datasets](https://www.nature.com/articles/s41592-025-02980-0), rebuilt small enough to read in one sitting and run without a GPU, a checkpoint, or an API key.

## The result

```bash
python cellcontext.py
```
```json
{
  "global_mae": 0.2699,
  "context_mae": 0.1813,
  "mae_reduction_pct": 32.8
}
```

Average every training response per perturbation into one number and you get `global_mae`. Look up the nearest *observed* context and transfer its residual response instead and you get `context_mae` — a 32.8% drop in error on held-out context/perturbation pairs the model never trained on. Same data, same perturbations, the only difference is refusing to pretend every cell is the same cell.

**Update:** "nearest observed context" turned out to mean nearest array
*index*, `min(abs(cc - c))`, not nearest by anything biological. It only
worked because the synthetic generator happens to make the interaction term
a linear function of context index. Decouple index from the latent value
that actually drives the effect and the method collapses to worse than the
naive baseline (mean -27.7% to -35.5% "reduction," i.e. it makes things
worse). `cellcontext_v2.py` replaces index proximity with a similarity
measured from training data itself and holds up. Details below.

## How it works

Training data is synthetic expression profiles built from a shared per-perturbation effect plus a genuine context-specific interaction term, so there's a real, known reason context should matter — this isn't noise dressed up as signal. The baseline averages every training response per perturbation into one global effect. The context-aware method indexes responses by context and, at test time, borrows the residual from the nearest context that's actually been observed for that perturbation. It's nearest-neighbor transfer, not a foundation model, and that's on purpose: the mechanism stays visible instead of hiding behind 200M parameters.

## "Nearest context" was nearest array index

`cellcontext.py`'s context-aware method picks
`min((cc for cc, pp in context_effect if pp == p), key=lambda cc: abs(cc-c))`
— literally the context whose integer index is numerically closest to the
test context's index. That's only a meaningful notion of "similar" because
the data generator builds the interaction term as
`slopes[p][g] * (identity[c] - 2)`, where `identity[c] == c` by default: the
effect is a straight line in context index. Real single-cell contexts (cell
types, donors, tissues) have no such built-in ordering.

```bash
python eval_v2.py
```
```
tuning (30 seeds):  index_mean_reduction_pct=-27.7   data_mean_reduction_pct=31.5
holdout (20 seeds): index_mean_reduction_pct=-35.5   data_mean_reduction_pct=30.1
```

`adversarial.py` shuffles that identity so index order no longer tracks the
true latent driver of the interaction term — the situation any real dataset
would actually be in. Run through the unmodified index-proximity method,
mean MAE reduction goes solidly negative (worse than just averaging) across
30 tuning seeds and a disjoint 20-seed holdout evaluated once, with worst
cases past -76%. `cellcontext_v2.py`'s `run()` replaces index distance with
a similarity computed from training data: two contexts are "near" if their
observed effects agree on the perturbations both have training data for.
That holds a positive ~30% mean reduction on both sweeps, never dropping
below +13.8% even in the worst holdout seed, and matches
(actually slightly beats, 33.9% vs 32.8%) the original index-based number on
the original, unshuffled data. `cellcontext.py` is untouched and the
published 32.8% still reproduces exactly.

## Run it

```bash
python cellcontext.py
python -m unittest discover -s tests -v
```

## What is tested

The test compares the context-aware method against the deliberately legible global-average baseline and requires `mae_reduction_pct >= 30`. The data generator is seeded, so the number in this README, in CI, and in the portfolio case study are the same number, not three different ones that happen to rhyme.

## Scope

This is an educational research reproduction on controlled synthetic data. It is not a clinical, diagnostic, production genomics, or safety-critical system, and it makes no claim about real single-cell datasets. The point is to make one mechanism — don't average away the context — measurable without hiding it behind a checkpoint.

## Research basis

- [Nature Methods' 2025 benchmark of 27 perturbation-response methods across 29 datasets](https://www.nature.com/articles/s41592-025-02980-0)
- Original implementation and benchmark in this repository are MIT licensed.

## License

MIT
