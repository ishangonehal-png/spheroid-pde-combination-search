# Sensitivity Analysis of the Savitar Kernel

*Mechanistic-interpretability expansion, Savitar 2.0. This report explains why the Savitar kernel wins on some benchmark contexts and loses on others, using sensitivity-analysis paradigms drawn from Bayesian optimization, the wider ML interpretability literature, and systems biology. It grounds the explanation in the drug-combination pillar and turns the diagnosis into a prioritized plan for augmenting the kernel.*

*All numbers below are computed from the precomputed per-seed results in `results/` and from re-fitting the actual kernel on the real benchmark observed sets; nothing is re-simulated beyond what is stated in Methods. Companion files: `paradigm_survey.md` (literature), `references.bib` (31 verified sources), `augmentation_roadmap.md` (the plan), and the figures and CSVs named inline.*

## Executive summary

Savitar wins on 60 of 93 benchmark contexts (win rate 0.645). The win/loss pattern is not random and it is not primarily about problem difficulty: it is about whether the kernel's central structural prior, the CP interaction tensor, is *identifiable* from the tiny observation budget, and whether the activity gate operates in its informative regime.

Three findings, in increasing depth:

1. **Where the variance lives.** Domain identity explains 41% of the variance in Savitar's regret advantage; benchmark-within-domain another 14%; the rest is context-idiosyncratic. Within a domain, harder problems tend to favour Savitar (finance Spearman rho = -0.905, drug -0.418), but this reverses across domains, a Simpson's paradox. The two structural priors dissociate cleanly by domain: on the drug pillar the activity gate is load-bearing (removing it raises regret on 82% of contexts, median +40 regret units), while on finance neither prior matters and the base kernel carries the win.

2. **What the drug pillar reveals.** Savitar's drug wins are *not* explained by multi-target synergy: the best drug combinations are not enriched for cross-mechanism pairs (Fisher OR 1.09, p = 0.40). The win is carried by the interaction tensor fitting pair-specific structure that a per-drug dose-response model misses, and by the gate operating in its active band. On the Tan/HIV loss the gate collapses "off" (65% of doses below the no-effect point) and the full kernel loses to a plain dose-response baseline.

3. **The mechanism, from parameter geometry.** Re-fitting the real kernel and computing the sloppy-model eigenspectrum of its marginal likelihood shows the fit is extremely sloppy everywhere (spectrum spans ~11 decades; effective rank 2.4 to 6.9 of up to 1099 parameters). The interaction tensor A is *unidentified in every context* (dimension-normalized stiff-subspace enrichment 0.01 to 0.18, always below 1). Losses occur where interaction identifiability collapses toward zero and all modeling capacity flows into the per-drug curve machinery, so the kernel degenerates into a worse-parameterized dose-response model. This overturns the natural hypothesis (that A is well-constrained on wins) and replaces it with a sharper one: **A is a useful low-rank inductive bias that helps precisely when the fit has enough identified directions to pay for it.**

The augmentation roadmap follows directly: the highest-priority fixes target *identifiability*, not expressiveness. An identifiability-gated fallback to the curve model would have rescued the Tan/HIV loss; a shrinkage prior on A and a budget-aware interaction rank address the universal sloppiness. All three are implementable inside the marginal-likelihood objective without redesigning the kernel.

## The question, and why sensitivity analysis

Savitar is a structured Gaussian-process kernel for low-budget Bayesian optimization over finite combinatorial pools with rare high-value winners. It carries two structural priors: an **activity gate** that encodes Bliss-style combination independence (a combination contributes no interaction signal when any constituent is at its no-effect point), and a **symmetric CP interaction tensor** that shares interaction parameters across all constituent orders, giving O(Dq) parameters regardless of arity. The paper reports strong aggregate results but a mixed per-context picture, and gives a qualitative regime theory for the exceptions.

The goal of this analysis is to make that regime theory *quantitative* and *mechanistic*: to attribute Savitar's per-context performance to measurable properties of the problem and of the kernel's own fitted parameters. That is exactly the remit of sensitivity analysis (SA). The companion `paradigm_survey.md` surveys SA across three tiers, BO/GP-native (Sobol decomposition, fANOVA hyperparameter importance, ARD relevance), ML-wide interpretability-as-SA (permutation importance, SHAP, influence functions), and cross-field imports from systems biology (sloppy-model analysis, practical identifiability, active subspaces), and maps each onto Savitar. This report executes the three highest-value lenses that survey identifies.

## Methods

**SA dataset (`sa_dataset.parquet`, `feature_table.csv`).** One row per (domain, benchmark, context), 93 contexts, built from 11,492 per-seed regret records across 12 canonical result files spanning the three domains (drug-combination, Ni-Sn-Zn alloy chemistry, ETF options finance). For each context we record Savitar's final regret, the best baseline's regret, and derived advantage metrics (advantage = best-baseline regret minus Savitar regret; log-regret ratio; per-prior ablation costs). Problem features are joined per context where available: constituent-universe size D, pool size P, rare-winner fraction, and, for the three anti-infective drug contexts, Hill-fit quality. Model-free difficulty (random-search mean regret) covers all 93 contexts; the two ablation costs cover the 33 chemistry, finance, and anti-infective-drug contexts that carry both ablation variants. A known coverage limit: the ALMANAC and NCI-60 raw pools are not in the repository, so their 64 contexts have regret and difficulty but not full pool features.

**Ablation semantics (from `src/combo_kernel.py`).** The `no_gate` variant fixes the activity gate to 1 (keeping M and A); the `no_m` variant freezes the Hill-to-embedding projection M to the identity (keeping the gate and A). The CP interaction tensor A is never directly ablated, so its contribution is inferred: when Savitar wins even as the gate and M are removed, A is doing the work.

**Parameter-geometry diagnostic.** We re-fit the actual `MonoFingerprintKernel` (q = 8, learnable M) in a dedicated environment on the *real* benchmark observed sets, reconstructing each observed set by re-running the true offline BO loop (n_init = 4, n_iter = 10, pool subsampled to 64 at the repo's fixed seed), 6 seeds per context. At each fitted optimum we compute the Hessian of the exact negative log marginal likelihood over all learnable parameters. We verified our marginal-likelihood matches gpytorch's to machine precision, and computed the Hessian by double-backward on the leaf parameters (the functorch and functional-call paths silently drop gpytorch's cached-kernel gradients and must not be used here). We report converged-seed means only.

## Findings

### Finding 1: where the advantage variance lives

Savitar's regret advantage is structured by domain. A nested variance decomposition of the log-regret ratio (`sa_variance_results.csv`) attributes **41.2%** to domain, **13.9%** to benchmark-within-domain, and **44.8%** to context-idiosyncratic residual. A one-way ANOVA on domain is strongly significant (F = 31.6, p = 4e-11, eta-squared = 0.41).

Difficulty acts through a Simpson's paradox. Overall, difficulty is uncorrelated with the advantage (Spearman rho = -0.002). But *within* domains, harder problems favour Savitar: finance rho = -0.905 (p = 1e-6, n = 16), drug rho = -0.418 (p = 3e-4, n = 71); chemistry is the exception (rho = +0.37, n.s., n = 6). The partial correlation controlling for domain is rho = -0.423 (p = 2e-5). Harder-within-domain problems are where structure pays off; the cross-domain average hides it.

The two structural priors dissociate by domain, which is the mechanistic core of this finding. On the **drug** pillar the activity gate is load-bearing: removing it raises regret on 82% of contexts, with a median cost of +40 regret units. On **finance**, removing either prior changes nothing (both mean costs are negative or zero), yet Savitar still wins, so on finance the base RBF-on-curve kernel, not the drug-specific priors, is what wins. On **chemistry** the effects are tiny in absolute terms. This is why the roadmap treats the drug priors as context-earned rather than always-on.

![Variance decomposition of Savitar's regret advantage]({{artifact:art_edac9a1f-17e4-4b1d-b2b2-221b7fd7ad94}})

*Figure 1 (`sa_variance_decomposition.png`). (a) Nested variance decomposition. (b) Difficulty-vs-advantage by domain, showing the within-domain trend and the cross-domain Simpson's paradox. (c) Fraction of contexts where each prior is load-bearing, by domain. (d) Per-benchmark median advantage.*

### Finding 2: what the drug pillar reveals

The bio pillar is where the mechanism is most legible, because the priors are pharmacologically motivated. We tested the natural hypothesis that Savitar wins when synergy is genuinely multi-target, on three anti-infective contexts spanning the outcome range: Brochado (E. coli, a clear win), Cokol (S. cerevisiae, a tie), and Tan/HIV (a loss).

**The win is not multi-target synergy.** In Brochado, the best drug combinations are *not* enriched for cross-mechanism (cross-MoA) drug pairs. The overall cross-MoA rate among viable combinations is 0.840; among the top 5% of combinations it is 0.851, statistically indistinguishable (Fisher OR = 1.09, p = 0.40). Mechanistic diversity does not predict which combinations win. So Savitar's advantage comes from the interaction tensor fitting *pair-specific* structure that a per-drug dose-response model cannot represent, not from a bias toward mechanistically diverse pairs.

**The gate is load-bearing but regime-dependent.** The activity gate (1 minus viability) is informative on the Brochado win: gate median 0.34, with 100% of queried doses in the active band, even though the underlying Hill fits are degenerate (median R-squared = -0.59). It is partially active on the Cokol tie (median 0.13, 65% active, clean Hill fits R-squared = 1.00). It collapses on the Tan/HIV loss: gate median 0.016, with 65% of doses at viability near 1, i.e. below the no-effect point, where the gate can inject no interaction signal.

**The ablation ladder locates the failure.** Removing the gate raises regret on all three contexts. Removing M raises regret on Brochado (0.040 to 1.146) and Tan/HIV, but is roughly neutral on Cokol (no_m 0.014 versus 0.018), so "both priors always help" is false: the gate helps universally, M does not help on Cokol. The decisive contrast is Tan/HIV: even with both priors, the full structured kernel (regret 5.43) loses to a plain per-drug dose-response baseline (2.65). The problem is not a prior pointing the wrong way; it is that the structured kernel spends capacity on interaction structure the data do not support.

![Bio-grounded case analysis of the drug pillar]({{artifact:art_65ab2972-578e-4c6e-94fd-e71770e89ecb}})

*Figure 2 (`bio_case_analysis.png`). (a) Gate operating regime by context. (b) Ablation ladder: on Tan/HIV the full kernel loses to a simple dose-response model. (c) Cross-MoA rate by combination-quality decile (flat: no multi-target enrichment). (d) Gate-value distributions. Numbers in `bio_case_results.csv`.*

A methodological caveat established during this analysis: the assay concentration floor is not zero (control-at-floor viability is 52.7% in Brochado, not 100%), so there is no true no-drug arm. Absolute Bliss-synergy levels are therefore confounded, and we report only interaction *spread*, not signed synergy fractions. This confound itself motivates a roadmap item (floor-aware normalization).

### Finding 3: the mechanism, from parameter geometry

The deepest result comes from the flagship cross-field import: sloppy-model analysis (Gutenkunst et al. 2007; Transtrum et al. 2015) and practical-identifiability analysis (Raue et al. 2009), applied to the curvature of Savitar's own marginal likelihood.

**The fit is sloppy everywhere.** On all three contexts the eigenspectrum of the marginal-likelihood Hessian spans about 11 decades, the textbook sloppy signature: a few stiff, data-constrained directions and a long tail of unconstrained ones. The effective rank (participation ratio of the spectrum) is only 2.4 to 6.9, out of 299 to 1099 nominal parameters. This is a direct, quantitative confirmation of the paper's O(Dq) design claim: the model genuinely operates in a low-dimensional identified subspace, not in its full parameter count.

**The interaction tensor is unidentified in every context.** Normalizing each parameter block's share of the stiff subspace by its share of the parameters (so that A's dominant dimensionality cannot create a spurious signal), the CP interaction tensor A has an enrichment of 0.01 to 0.18, *always below 1*: it is under-represented in the stiff directions in every context. At a 14-observation budget, A's hundreds of parameters are never pinned by data. What the data do identify is the per-drug curve machinery, the M projection (enrichment 7 to 33) and the lengthscale (enrichment 4 to 34).

**Losses are identifiability collapse, not a bad prior.** The discriminator between win and loss is the effective rank together with A-identifiability, not A-stiffness. Brochado (win) has effective rank 6.9, about three times the others, so it has the most identified structure overall; its A-enrichment (0.11) is comparable to Cokol's (0.18), both far above Tan/HIV's, and the identified structure is real and helps. Cokol (tie) has effective rank 2.4 but the problem is trivially easy (regret goes to zero for everyone), so low identifiability costs nothing. Tan/HIV (loss) has effective rank 2.4 *and* A-enrichment collapsed to 0.01, with all capacity flowing into the curve terms (M enrichment 32.9, lengthscale 33.8). There, Savitar degenerates into a worse-parameterized dose-response model, and the plain baseline beats it. This is the mechanistic signature of "surplus structure": the interaction prior is not backfiring, it is unidentifiable and unpaid-for.

![Parameter-space geometry diagnostic]({{artifact:art_77070308-5ad5-4149-b57a-69ca63dab2df}})

*Figure 3 (`param_geometry_eigenspectrum.png`). (a) Marginal-likelihood eigenspectra (sloppy, ~11 decades). (b) Effective rank per context. (c) Stiff-subspace enrichment by parameter block: A is sloppy everywhere, the curve machinery is what data identify. (d) The loss regime: low effective rank and collapsed A-identifiability together. Numbers in `param_geometry_results.csv`.*

## Synthesis: a revised theory of when Savitar wins

Combining the three findings, the paper's qualitative regime theory sharpens into a mechanistic one. Savitar's advantage requires two things to hold together:

1. **The interaction structure must be identifiable enough to pay for itself.** A is a low-rank inductive bias, not a precisely fitted object, at low budget it is never precisely fitted. It helps when the fit has enough identified directions (higher effective rank) that the interaction contribution rises above noise. When identifiability collapses (Tan/HIV), the structured kernel reduces to a worse dose-response model and loses to the plain one.

2. **The activity gate must be in its informative regime.** The gate carries the drug pillar, but only when queried doses sit in the active band. When doses fall below the no-effect point (Tan/HIV), the gate reads near-constant and injects no signal.

Ties, by contrast, are cheap: on easy problems (Cokol, high rare-winner density) everyone reaches zero regret, so poor identifiability has no cost. And the whole drug-specific story is domain-local: on finance, neither prior matters and the base kernel wins on its own.

This theory is both more predictive and more actionable than "finite pool, small budget, rare winners, informative curve." It says *which measurable quantity* (interaction identifiability, gate regime) decides each case, and it points directly at fixes.

## Augmentation roadmap (summary)

The full plan with evidence, mechanisms, citable paradigms, and implementation sketches is in `augmentation_roadmap.md`. The priority ordering follows from the findings above:

- **P0, identifiability-gated interaction fallback.** Estimate A's identifiability online from the fitted kernel and scale the interaction contribution down when it is unconstrained, relaxing toward the curve-only model that already beats Savitar on Tan/HIV. Imports Raue-style practical identifiability as a control signal. Highest expected regret reduction per unit effort; implementable as a post-fit hook.
- **P1, shrinkage/sparsity prior on the CP factors.** A is universally sloppy and currently unregularized; a horseshoe or ARD prior pulls unsupported interaction loadings toward zero. Imports the sparse axis-aligned prior (SAASBO) that the paper reports beating Savitar on the small-universe alloy context.
- **P1, budget-aware interaction rank.** Grow q with the observation count so the interaction parameter count tracks the number of identified directions. An own addition motivated directly by the effective-rank measurement.
- **P2, Hill-quality-weighted gate** and **floor-aware activity normalization.** Make the gate trust its no-effect point only when the Hill fit supports it, and anchor viability to the assay's real floor.
- **P3, hierarchical context-adaptive priors.** Partially pool hyperparameters across contexts within a benchmark, addressing the 14% benchmark-within-domain variance.

The top three all target one root cause, the interaction tensor is expressive but unidentifiable at low budget, from different angles, and none require redesigning the kernel.

## Limitations

- **Pool-feature coverage.** Full problem features (D, P, rarity, Hill quality) exist for the 3 anti-infective drug contexts; finance has D/P/rarity but no Hill; chemistry has only D; the 64 ALMANAC and NCI-60 contexts have regret and difficulty but no pool features, because their raw pools are not in the repository. Difficulty (all 93 contexts) and the two ablation costs (33 contexts) are the cross-domain-complete signals, and the domain-level conclusions rest on them.
- **Assay floor confound.** The non-zero concentration floor means absolute synergy is not cleanly measurable; we report interaction spread and gate regime, which are robust to the floor, not signed synergy.
- **Parameter-geometry scope.** The eigenspectrum diagnostic was run on the three anti-infective drug contexts (the win/tie/loss triad), 6 seeds each, at the core20 budget. Extending it to chemistry and finance contexts is future work; the drug pillar was prioritized because the priors are pharmacologically interpretable there.
- **Correlational, not causal.** The variance decomposition and geometry analyses explain existing results; the roadmap items are hypotheses to be tested by actually implementing them and re-benchmarking, which is the next expansion phase.

## Reproducibility

Every figure and table is regenerated from `results/` and, for Finding 3, from re-fitting the kernel. Key artifacts under `analysis/`:

| File | Contents |
|---|---|
| `sa_dataset.parquet` | 93-context tidy SA dataset (per-context regret + features) |
| `feature_table.csv` | Human-readable feature table |
| `sa_dataset_datadict.json`, `sa_dataset_manifest.csv` | Data dictionary and source provenance |
| `paradigm_survey.md`, `references.bib` | Literature survey and 31 verified citations |
| `sa_variance_decomposition.png`, `sa_variance_results.csv` | Finding 1 |
| `bio_case_analysis.png`, `bio_case_results.csv` | Finding 2 |
| `param_geometry_eigenspectrum.png`, `param_geometry_results.csv` | Finding 3 |
| `augmentation_roadmap.md` | The prioritized plan |

The parameter-geometry re-fit uses the repository's own kernel and BO loop with the fixed core20 configuration (n_init 4, n_iter 10, pool 64, embedding_dim 8, interaction_order 2, 30 Adam steps at lr 0.05), extended only by a longer optimizer run to reach a true optimum before computing curvature.
