"""Adversarial seeds for the context-similarity fix.

TUNING_SEEDS: used to characterize and tune cellcontext_v2's data-driven
similarity against cellcontext.py's index-proximity method.
HOLDOUT_SEEDS: disjoint, evaluated exactly once after the fix was finalized.

Both are run with shuffle_identity=True, which decouples context array index
from the latent value that actually drives the interaction term -- the
scenario a real biological dataset would look like, where nothing guarantees
context order in the data matches context similarity.
"""

TUNING_SEEDS = list(range(100, 130))
HOLDOUT_SEEDS = list(range(200, 220))
