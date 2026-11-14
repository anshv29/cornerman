# Audit

Two parts. The early audit Ansh asked for on 2026-11-14, before moving past Step 7 (below). The full Fable upgrade pass (end of SPEC.md) comes later and will add to this file.

## 2026-11-14: early audit (leakage, the math, optimistic numbers)

Scope, as asked: leakage in every evaluation so far; the math behind Dawid-Skene, the prior correction, and the identity and classifier evaluations; and any number likely to be too optimistic. Numbers below were recomputed today on the current predictions (symmetric classifier) unless a date says otherwise. "Agent-judged" means I looked at the frames myself, blind to the set but not to the footage.

### Summary
- No train/test leakage across fights in the state classifier: folds are by fight, Dawid-Skene and the family rates are refit inside each fold, the +-1 s windows stay inside a fight.
- But there is no untouched test set for the states. Several design choices (balanced training, the family-first decision rule, the red-top fix) were made while looking at silver, and silver is also the score. Silver numbers are development numbers.
- The family Dawid-Skene model can't test its own independence assumption: with two labellers it fits the R-by-VLM table exactly. Everything rests on the fixed VLM accuracy (0.95, from 32 agent-judged clips), and the clinch rate swings from 0.1% to 10% as that number goes from 0.80 to 0.99. The corrected probabilities `q_*` inherit this.
- Silver direction scores can't see identity errors: the VLM was shown the same red and blue boxes the classifier uses.
- Replays aren't detected, so they may be counted twice. They stay inside one fight, so they don't leak across folds, but they inflate counts.
- Most headline numbers are moment-level and treat moments as independent. Resampling by fight widens them a lot (silver family accuracy 0.84, 95% interval 0.75 to 0.92 over 29 fights).

### 1. Leakage, evaluation by evaluation

| Evaluation | Same fight in train and test? | Same fighter? | Tuned on the test? | Verdict |
|---|---|---|---|---|
| State classifier vs silver (`eval/states.py`) | No. 5 folds by fight; Dawid-Skene, class balance and the family rates are fit on training fights only; windows stay in the fight. | Possible: 15 fighters have 2+ fights and can fall in different folds. The features are geometry (boxes, pose, masks, fence distance), not appearance, so the risk is low. | Yes. Balanced training was chosen by comparing silver scores (DECISIONS 2026-10-27: family accuracy 0.74 to 0.85 on the same data); the family-first decision rule came from looking at argmax on silver; the red-top fix was checked on silver. | No leak of labels, but silver is a development set, not a test. |
| Lower tier (prompts agree, rules abstain) | Same as above. | Same. | Not used for choices, only reported. | Cleaner than silver, but still the VLM's own answers. |
| Constants learned from the silver clips | The VLM family anchor (0.95), B-top trust (0.85) and B-controller trust (0.75) come from spot checks of silver and lower-tier clips, from all fights, including each fold's test fights. | n/a | One number each, so mild. | Small leak; worth removing once there is a fresh check set. |
| Silver direction (top and bottom) | n/a | n/a | n/a | Blind to identity: the VLM saw the red and blue boxes from our identity stage, and the classifier's red and blue features come from the same boxes. A swapped corner looks right to both. |
| Identity, whole fight (`identity_v3`) | Dev 30 fights, test 24 new fights, method frozen and committed first. The fighter classifier was trained on the dev fights only. | Fighters with 2+ fights can be in both dev and test; the method is per fight (no cross-fight appearance), so no leak. | The 0.5 flag threshold was chosen after the test (already in DECISIONS 2026-10-19: "89% kept right" is optimistic, needs re-validating on fresh fights). | Clean apart from the threshold. Junior removal left 13 of 18 right (72%, Wilson 95% interval 49 to 88%). |
| Identity per shot (`identity_segments`) | Round 1 dev, rule v2 committed, round 2 on 40 fresh shots. | n/a | v2 was set on round 1, tested on round 2. | Clean. 26 of 28 judgeable right (77 to 98%). |
| Fence rule G | Tuned on a dev set, then a fresh blind check (14 of 20 calls right, 18 of 18 hard negatives). | n/a | No. | Clean, agent-judged, small. |
| Strike rule, round 1 | Thresholds set by reasoning before any result. | n/a | Not retuned. | Clean, agent-judged, small (9 of 17). |
| Takedown rule, rounds 1 and 2 | Round 2's rule was designed after round 1 (Ansh's spec); round 2 sampled fresh moments, at least 10 s from anything in round 1. | n/a | One change before round 2 was based on pass counts, not on correctness: the stability test moved from `p_*` to `q_*` when `p_*` let almost nothing through. | Clean as a sanity check; 4 of 8 (Wilson 22 to 78%). |
| VLM pilots and swap tests | Prompts were revised after earlier pilots on overlapping clips. | n/a | Prompt wording yes, on a few dozen clips. | Mild; the silver set came after the prompts were frozen. |
| Replays | A replay repeats a moment of the same fight, so it can't cross folds. | n/a | n/a | Not a leak, but double counting: replays aren't detected. 18% of live moments have no readable running clock (the score bar is often hidden during replays), 82% have one. Not measured further. |

### 2. The math

**Family Dawid-Skene (`weak_labels/labellers.fit_family`).**
- Independence can't be tested here. On the 2,322 moments where both R and the VLM vote, the fitted model reproduces the 3 x 3 table of their votes exactly (largest difference 0.0000). Two labellers give 8 free cells; the model has 8 free numbers (2 for the prior, 6 for R's confusion) once the VLM's confusion is fixed. So any dependence between R's and the VLM's mistakes is absorbed into R's estimated confusion, unseen.
- Everything hangs on the anchor. Refitting with other values for the VLM's family accuracy:

  | VLM accuracy | standing rate | clinch rate | ground rate | R right on standing / clinch / ground |
  |---|---|---|---|---|
  | 0.80 | 0.801 | 0.001 | 0.197 | 0.49 / 0.66 / 0.72 |
  | 0.85 | 0.733 | 0.028 | 0.239 | 0.49 / 0.62 / 0.61 |
  | 0.90 | 0.658 | 0.056 | 0.285 | 0.48 / 0.59 / 0.51 |
  | 0.95 (used) | 0.598 | 0.081 | 0.321 | 0.48 / 0.56 / 0.44 |
  | 0.99 | 0.558 | 0.099 | 0.342 | 0.48 / 0.55 / 0.40 |

  The 0.95 rests on 32 of 32 agent-judged clips (16 silver, 16 lower tier). The 95% interval for that runs down to about 0.89, where the clinch rate would be about 5%, not 8%; and the check leaned to easy clips, so the VLM's accuracy on hard moments may be lower still. So the clinch rate and the clinch probabilities per fighter are uncertain by a factor of about 2, and the soft labels with them.
- R looks weak at standing (0.48): of moments the VLM calls standing, R says clinch about as often as standing. That is either R's contact test firing on fighters who are close but not clinched, or the VLM under-calling clinch. Two labellers can't tell which. This needs a third, independent view (Ansh's labels would settle it; he doesn't label, so the next best is a by-eye check of R-clinch / VLM-standing disagreements).
- Missing votes are assumed to say nothing about the class. They do: the VLM only votes on the 780 silver-set clips, sampled by strata built from the rules' own proxies (ground from torso angle, merged boxes), so the clips over-represent clinch and ground and the moments where R is likely wrong. The prior is refit over all moments with the confusions fixed, which handles the over-sampling of classes, but not R's errors being different inside the sampled strata.

**Direction (`weak_labels/labellers.direction`).** Votes are multiplied as if independent given the truth. R-top and the VLM's top call look at the same frames, so their errors are probably correlated and the direction probabilities are too sure where both vote. The trusts are rough: R-top 0.95 is from ViCoS (jiu-jitsu, a different domain), B-top 0.85 from 6 of 6 silver clips plus 4 of 6 lower tier, B-controller 0.75 is an assumption. Clinch direction is already never reported (Ansh, 2026-11-14).

**Prior correction (`events/states.prior_correct`).**
- The formula is the standard one: q is p times (target rate / training rate) per family, renormalised. The training rate is the family share of the actual training weights (after balancing and the colour-swap copies), which is the right one.
- It assumes only the label mix changes between training and use (each family's features look the same). The training rows are the moments where R or the VLM voted (92% of moments, since R almost always votes), so that holds well enough.
- That q's family shares match the target rates (0.583 / 0.085 / 0.331 against 0.595 / 0.083 / 0.322) is close to true by construction; it is a sanity check, not a validation.
- Calibration against silver, as far as it can be measured: expected calibration error of q's family probability 0.06 on silver, 0.20 on the lower tier (too sure of standing on hard clips). Silver's own family mix (0.71 / 0.19 / 0.11) differs from the target rates, so neither number is a clean calibration test.
- The target rates themselves come from Dawid-Skene, so all of the anchor uncertainty above passes straight into q.

**Classifier evaluation (`eval/states.py`).**
- Truth is the clip's state, given to every moment of a 3 s clip. A clip that changes state mid-way gives some moments a wrong truth (noise, not bias, but it caps the score).
- Moments within a clip and a fight are strongly correlated. Fight-bootstrap 95% intervals (2,000 resamples of fights):

  | | moments | clips | fights | family accuracy | balanced |
  |---|---|---|---|---|---|
  | silver | 772 | 127 | 29 | 0.84 (0.75 to 0.92) | 0.82 (0.71 to 0.92) |
  | lower tier | 860 | 141 | 29 | 0.63 (0.54 to 0.72) | 0.62 (0.52 to 0.71) |

- The "rules" baseline agrees with silver at family level by construction (silver needs the rules to agree), so beating it on silver says little. The lower tier is the fair comparison there (0.63 against the rules' 0.34).
- The majority baseline uses the most common predicted state, not the training labels; it is trivially beatable either way.

**Identity evaluation.** Verdicts are agent-judged from crop montages before the flags or keys were read. The per-fight accuracy counts a fight right only if both rows are clean, which is strict. The numbers are small (18 and 28 judged units), and their intervals are already in DECISIONS.

### 3. Numbers likely to be too optimistic

1. **Silver family accuracy 0.84** (above the 80% target). Easy clips (rules and both prompts must agree), choices made on it, no fresh test. Interval 0.75 to 0.92. The lower tier's 0.63 is the honest number for hard moments; the truth for all fighting time is somewhere between.
2. **Silver direction where known: pair 0.97, ground top recall 1.0.** Picked moments (direction marginal at least 0.8), on easy clips, and blind to identity mistakes. Per-fighter top time is right only if identity is right too (72% of whole test fights, 93% of kept shots).
3. **Clinch rate 8% and the clinch probabilities per fighter.** Set mostly by the 0.95 anchor; about 5 to 10% if the 32-clip check holds, lower if the VLM is worse on hard moments.
4. **Identity "about 9 in 10 kept fights right" at threshold 0.5.** Threshold chosen after the test (already flagged in DECISIONS).
5. **VLM family accuracy 32 of 32.** Agent-judged, 16 of them on the easy silver clips. Used as a fixed truth by Dawid-Skene.
6. **Takedown 4 of 8, strike 9 of 17, fence 14 of 20.** Judged by the same agent who wrote the rules. Fine as sanity checks, not as accuracies.
7. **Counts of fighting time and events.** Replays aren't removed; walk-outs and celebrations sometimes pass the fighting filter (seen in the per-shot check).
8. **Action counts.** Stand-ups (356) and clinch entries (134) come from the same state changes that made rule takedowns 1 in 4 right; they are likely just as noisy. They are out of the event tables already.

### 4. Smaller things found on the way
- 184 red and 183 blue fence distances are infinite (no real mat edge found in that frame). The tree model puts them in the top bin, which works, but they should be missing values. Not fixed yet.
- The state classifier wasn't colour-symmetric before today (fixed 2026-11-14, `tests/property/test_colour_swap.py`). Identity can't be put under an exact swap test, since red and blue kits are different colours; its colour features are known to be lopsided (PROGRESS, Shaky).
- The VLM actions pilot gives both fighters the same actions in 21 of 34 live answers; when it's compared to the rules, agreement from copying must be told apart from real agreement.

### 5. What I'd do about it (for Ansh to decide)
1. **A fresh state test set.** Freeze the state classifier as it is now, and score it once on silver clips from fights it has never been developed on (the next batch of fights, or a held-back part of the current VLM run). Report that number as the test number; silver stays a dev number.
2. **Pin down the anchor.** A blind, agent-judged check of 40 moments where R says clinch and the VLM says standing (and the reverse) would show which labeller is wrong there, and give the VLM's accuracy on hard moments, not just easy ones. Then refit with the measured value, and carry its uncertainty (fit at the low and high ends) into the per-fighter numbers.
3. **Report intervals by fight** for every accuracy from now on (`eval/states.py` can do the bootstrap).
4. **Folds by fighter group** (fights sharing a fighter in the same fold), to remove the small same-fighter risk; cheap.
5. **Replay detection** before any per-fighter rate goes public: matching pose sequences against earlier footage, as planned in Phase 2.
6. Fence distances: infinite to missing (small model change; rerun states).
