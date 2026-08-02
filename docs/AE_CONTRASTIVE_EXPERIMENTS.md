# AE Contrastive Loss Experiments

This experiment tests whether adding the AE latent triplet loss improves
landmark usefulness.

The baseline must disable the new auxiliary loss explicitly:

```bash
python scripts/train_pointmaze.py --steps 500000 --seed 0 \
  --ae-contrastive-lambda 0.0 \
  --log-file logs/pm_ae_base_s0.log \
  --save logs/pm_ae_base_s0.pt
```

The random-negative AE contrastive variant uses the proposed default:

```bash
python scripts/train_pointmaze.py --steps 500000 --seed 0 \
  --ae-contrastive-lambda 0.1 \
  --ae-contrastive-margin 1.0 \
  --ae-negatives-per-anchor 1 \
  --ae-negative-mode random \
  --log-file logs/pm_ae_contrast_s0.log \
  --save logs/pm_ae_contrast_s0.pt
```

For a smoke test, add `--short` to both commands. Do not use smoke results as
evidence; they only verify that the pipeline runs.

## Matched-Seed Ablation

Run both variants for the same seeds:

```bash
for seed in 0 1 2; do
  python scripts/train_pointmaze.py --steps 500000 --seed "$seed" \
    --ae-contrastive-lambda 0.0 \
    --log-file "logs/pm_ae_base_s${seed}.log" \
    --save "logs/pm_ae_base_s${seed}.pt"

  python scripts/train_pointmaze.py --steps 500000 --seed "$seed" \
    --ae-contrastive-lambda 0.1 \
    --ae-contrastive-margin 1.0 \
    --ae-negatives-per-anchor 1 \
    --ae-negative-mode random \
    --log-file "logs/pm_ae_contrast_s${seed}.log" \
    --save "logs/pm_ae_contrast_s${seed}.pt"
done
```

Keep all non-AE-contrastive settings fixed: env steps, seed, eval episodes,
planner settings, landmarks, hindsight range, and network sizes.

## What to Track

- Long-horizon eval success rate.
- Learning curve shape, not just final score.
- `ae_rec`, `ae_latent`, and `ae_contrastive`.
- `ae_rank_acc`: fraction of sampled triples where
  `dist(anchor, positive) < dist(anchor, negative)` in AE latent space.
- Landmark spread/coverage via existing eval/plot utilities.

Plot a single log:

```bash
python scripts/plot_log.py --log logs/pm_ae_contrast_s0.log \
  --out logs/pm_ae_contrast_s0_curve.png
```

For Kaggle, use:

```text
notebooks/kaggle_ae_contrastive_experiment.ipynb
```

Start with random negatives. Hard negatives are intentionally not implemented
yet, so the first ablation is clean.
