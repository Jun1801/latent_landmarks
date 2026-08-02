# PN-LMCGS Design

## Purpose

Extend this repository's L3P implementation with Positive-Negative Landmark Monte Carlo Graph Search (PN-LMCGS) as specified by `PN_LMCGS_IMPLEMENTATION_GUIDE.md`. The existing actor, distance critic `D`, goal-to-goal value model `V`, autoencoder, positive landmark optimizer, HER replay, soft-Floyd search, and temporally extended subgoal execution remain the baseline.

PN-LMCGS is opt-in. With `pn_lmcgs_enabled=False`, training, evaluation, environment behavior, replay sampling, and checkpoint loading must retain the existing L3P behavior.

## Chosen Approach

Use an additive PN-LMCGS package and thin adapters around the existing L3P trainer and environments. This keeps the baseline code path explicit and makes the new safety and macro-planning components independently testable.

Two alternatives were rejected:

- Treating all wall contacts or ordinary collisions as violations would create noisy negative labels because those events do not have consistent safety meaning across the existing environments.
- Requiring every environment to expose exactly one new API would make Safety-Gymnasium and existing custom GoalEnv wrappers unnecessarily difficult to use.

Instead, all supported environment return formats are normalized at the wrapper boundary into one internal transition record. A collision-to-cost rule is permitted only through an explicit, disabled-by-default adapter.

## Compatibility Contract

- `pn_lmcgs_enabled=False` selects the original `LatentPlanner` and original training schedule.
- Missing safety information means `safety_cost=0.0`; it is not an error and does not change baseline behavior.
- Existing PointMaze geometry and dynamics remain the default. The hazardous two-route variant is separately configurable.
- Existing L3P checkpoints load without PN-LMCGS state. New modules initialize normally and remain inactive until their data gates pass.
- PN-LMCGS checkpoints include all new network, optimizer, landmark, replay metadata, calibration, activation, and scheduler state needed to resume training.
- The original distance critic remains a task-distance model. Safety targets never alter its rewards or TD targets.

## Configuration

Add flat dataclass fields grouped by a `pn_` prefix because this repository uses a single `Config` dataclass rather than nested YAML groups. The fields cover:

- feature activation and environment adapters;
- hazardous PointMaze geometry and hazard cost;
- positive and negative landmark counts and activation gates;
- cost critic weight and optimizer settings;
- macro replay, horizon, model, and calibration settings;
- MCGS candidate, PUCT, risk, leaf, penalty, and fallback settings;
- phased-training gates and collection-mode probabilities.

Environment-specific L3P settings such as `d_max`, horizon, actor capacity, and replay budget remain inherited unless explicitly overridden.

## Safety-Cost Interface

### Normalized step result

Create an environment helper that accepts the step results used in this repository and returns:

```python
NormalizedStep(
    observation,
    reward,
    terminated,
    truncated,
    info,
    safety_cost,
)
```

Supported input forms are:

```text
(obs, reward, done, info)                         legacy Gym/repository
(obs, reward, terminated, truncated, info)       Gymnasium
(obs, reward, cost, terminated, truncated, info) Safety-Gymnasium
```

Cost precedence is deterministic:

1. the separate Safety-Gymnasium cost return;
2. `info["safety_cost"]`;
3. `info["cost"]`;
4. an explicitly configured collision-to-cost adapter;
5. zero.

The chosen value is converted to a scalar and binarized with `float(value > 0)`. Conflicting lower-precedence labels are ignored. The normalized `info` always contains `safety_cost`, `goal_reached`, and `termination_reason` without mutating the environment-owned dictionary.

Timeout/truncation is not a violation. Goal failure is not a violation. A collision is not a violation unless the optional adapter names and enables a specific collision signal.

### Optional collision adapter

The adapter is configured with an information key and, where needed, a matching value. It is disabled by default. It exists for environments whose owners explicitly define a particular contact event as unsafe; it is not enabled for current L3P environments.

### Two-route hazardous PointMaze

Add an optional PointMaze layout with two routes from start to goal:

- a short central route crossing a designated rectangular hazard region;
- a longer route around that region with no hazard exposure.

Entering or remaining inside the hazard region emits `info["safety_cost"]=1.0`; all other transitions emit zero. Hazard contact need not terminate the episode, allowing the safety return to represent any violation before goal/termination. The info dictionary also identifies the hazard and uses the standard normalized termination fields.

The mode is disabled by default so existing PointMaze runs and checkpoints retain their original task distribution.

## Primitive Replay and Episode Semantics

Extend `HERReplayBuffer` in place with episodic arrays for:

```text
cost, violation, goal_reached, terminated, truncated, termination_reason
```

Legacy calls to `store_episode` may omit these keys; zero/false/default values are inserted. Sampling returns the new fields in addition to the existing L3P batch.

HER keeps the observed transition cost unchanged. After goal relabeling, `goal_reached` is recomputed against the relabeled goal, and the cost-critic bootstrap stops at that relabeled success. Negative violation endpoints are never introduced as a separate positive HER-goal pool.

Variable early terminations are padded to the fixed episodic storage horizon with a valid-step mask. Optimization and auxiliary memories sample only real transitions. This avoids fabricating repeated terminal transitions while retaining the buffer's efficient array layout.

## Landmark Memories

Create a `LandmarkMemoryManager` that ingests each completed real episode.

Positive candidates are achieved goals from segments whose commanded goal is eventually reached without an intervening cost. Safe timeouts and ordinary failures are neutral. The existing GLS operation sparsifies positive samples before the existing `LatentLandmarks` ELBO update.

Hard-negative memory stores exactly one endpoint for each transition with binary cost one:

- use `next_achieved_goal` when available;
- otherwise use `achieved_goal` and mark the sample as approximate/pre-impact.

Earlier warning-window states never enter hard-negative memory. Each sample retains its episode ID so activation can require both `min_negative_samples` and `min_negative_episodes`.

Instantiate a second `LatentLandmarks` module for negative centroids. Its slots remain persistent and are never sorted after updates. Negative landmarks are outcome prototypes only and never selectable graph nodes.

## Goal-Conditioned Cost Critic

Add `ViolationCritic(state, action, goal) -> probability` with a sigmoid output and a target copy. It consumes the same normalized observations and goals as the actor and distance critic.

For each sampled transition:

```text
reached   = relabeled goal reached at next achieved goal
stop      = terminated or truncated or reached
target    = cost + (1 - cost) * (1 - stop) * target_next_probability
loss      = binary cross entropy(prediction, detached target)
```

The target is bounded in `[0, 1]`. Target-network updates use the existing Polyak cadence.

When PN-LMCGS is active, the actor loss adds `pn_cost_critic_weight * violation_probability` to the existing `-Q + action_l2` objective. Gradients update the actor but not the cost critic during the actor step. When PN-LMCGS is disabled, actor loss is byte-for-byte equivalent in behavior to the baseline expression.

## Macro Attempts and Execution

Create a separate ring buffer of raw `MacroAttempt` records. Each record contains start, command, end, optional violation goal, duration, context, outcome booleans, episode ID, and timestep range. Commands are stored as raw goal vectors, never centroid IDs.

A real macro attempt begins whenever a high-level command is issued and ends at the earliest of:

- commanded goal reached;
- safety violation;
- primitive episode termination/truncation;
- adaptive macro horizon exhaustion.

The horizon is `ceil(D(current_state, actor(current_state, command), command))`, clipped to `[pn_k_min, pn_k_max]`. After each completed attempt, the high-level planner replans from the new real state.

Labels are derived at sample time using the current encoder and current centroids, in mandatory priority order: `VIOLATION`, `TARGET`, `DRIFT`, `STUCK`. A safe failure becomes `DRIFT` only when the end encoding is within the configured positive assignment radius; otherwise it is `STUCK`.

## Macro Transition Model

Implement a factorized model with detached encoder inputs:

```text
u = [z_start, z_command, z_command - z_start, context]
```

A shared MLP produces:

- four outcome logits ordered `TARGET`, `DRIFT`, `STUCK`, `VIOLATION`;
- a positive-landmark query scored against current detached positive centroids;
- a negative-landmark query scored against current detached negative centroids;
- four bounded duration predictions, one per outcome.

This query-based output remains valid as centroid positions move and does not require rebuilding the network. Macro losses must not backpropagate into the encoder or either centroid set.

Training uses weighted or stratified outcome batches, conditional positive/negative identity loss, and normalized smooth-L1 duration loss. A held-out natural-frequency validation partition drives scalar temperature calibration, expected calibration error, reliability bins, and activation gating.

Empty inactive negative centroid sets produce no negative-ID distribution or loss while retaining the generic `VIOLATION` outcome probability.

## MCGS Planner

Create `RiskConstrainedMCGS` and a trainer-facing adapter. Every real planning call constructs an independent search with:

```text
ROOT, positive landmarks, GOAL
```

Negative landmarks are terminal chance outcomes, not selectable nodes. Transpositions share state by `(node_id, remaining_macro_depth)` only within the call.

### Edge prediction

Build a per-call cache for all local directed candidates. Root distances use the original current-state `D`; internal distances use `V`. Each cached edge also stores calibrated macro outcome probabilities, positive/negative outcome distributions, duration predictions, and prior.

Candidates must pass `d_max`, immediate violation probability, non-self, path-cycle, and `top_k` filters. The final goal is considered whenever it passes those filters. Priors combine target probability, local plus remaining distance, and violation probability exactly as specified in the implementation guide.

### Search and risk

Each search edge owns visit count, reward sum, and Beta risk parameters initialized from the model's immediate violation probability. Selection uses PUCT only among edges whose conservative risk estimate is below `pn_search_risk_limit`.

Chance simulation handles:

- `TARGET`: move to the commanded node, or finish safely at `GOAL`;
- `DRIFT`: sample a positive node and terminate with a loop penalty if it repeats;
- `STUCK`: terminate with a safe task penalty;
- `VIOLATION`: terminate with safety return one and a violation penalty.

The existing soft-Floyd routine is reused through a PN-specific edge-cost adapter. Task and minimum-risk leaf tables are precomputed once per real plan. No-path leaves have task value zero and risk one.

After the simulation budget, the root is filtered by the stricter root risk limit and ranked by visits, mean reward, then lower risk. Evaluation returns `NO_SAFE_PLAN` when no action is feasible. An unsafe training fallback is possible only behind the explicit disabled-by-default flag.

The planner adapter returns the actual command vector and diagnostics. The trainer executes one macro attempt, records its real outcome, and replans.

## Phased Training

Implement explicit phases derived from data gates rather than only elapsed steps:

1. L3P warm-up: direct collection, baseline learning, safety-cost collection, cost-critic learning; no MCGS.
2. Landmark activation: positive landmarks activate after safe-success coverage; negative landmarks activate after both negative gates.
3. Macro bootstrap: collect attempts using original planning, risk-masked soft-Floyd, local positive exploration, and direct goals; train and calibrate the macro model.
4. Joint PN-LMCGS: enable search only after macro-attempt and validation gates, then mix MCGS, original-planner, and direct-goal collection according to normalized configured probabilities.

All phases continue updating the original L3P modules. If the PN feature is disabled, phase selection and collection reduce to the existing trainer logic.

## Checkpoints

Version trainer checkpoints. New checkpoints include:

- baseline L3P model state;
- cost critic and target plus optimizer state;
- negative landmarks and optimizer state;
- landmark memories and activation counters;
- macro replay metadata, model, optimizer, and calibration temperature;
- phase/scheduler counters and RNG state.

Loading an unversioned baseline checkpoint supplies absent PN state with fresh defaults and logs that PN modules were not restored. Landmark count mismatches continue to rebuild modules before loading. Malformed partial PN state fails with a clear migration error rather than silently mixing incompatible components.

## Logging

Preserve existing scalar logs and add the guide's minimum safety, memory, macro-model, and planner diagnostics. Evaluation reports goal success, safe success, episode violation, path length, no-safe-plan rate, and planning latency. Natural-frequency outcome metrics are kept separate from stratified training-batch metrics.

## Tests

Testing follows isolated red-green cycles for each milestone.

### Safety and replay

- normalize all three step return formats and cost-key precedence;
- default missing cost to zero;
- verify collision adapter is opt-in;
- verify standard and hazardous PointMaze modes;
- verify safe positive segments, timeout neutrality, endpoint-only negatives, HER cost preservation, relabeled stopping, and legacy episode insertion.

### Learning modules

- verify critic target boundaries and stopping behavior;
- verify actor safety gradient without cost-critic parameter gradients;
- verify negative activation gates and persistent slots;
- verify macro shapes, probability sums, duration bounds, moving-centroid compatibility, detached encoder/centroids, label priority, conditional losses, and calibration.

### Planner

- verify shortest zero-risk route and longer safe-route selection;
- verify chance outcomes, loop termination, root `D` versus internal `V`, transposition sharing, risk filtering, leaf behavior, no-safe-plan, and evaluation fallback rules;
- use deterministic model stubs and seeded sampling so failures are reproducible.

### Integration and regression

- verify disabled PN-LMCGS follows the existing planner and actor paths;
- verify old checkpoints load with zero-cost-compatible defaults;
- run the existing six tests unchanged;
- run a small hazardous PointMaze collection/update/search smoke test without requiring learning convergence.

## Non-Goals

The v1 implementation excludes moving-context dynamics, belief-state planning, progressive widening, continuous landmark proposals, latent dynamics, ensembles, CVaR/distributional backup, negative-proximity masks, and pre-violation hard negatives.
