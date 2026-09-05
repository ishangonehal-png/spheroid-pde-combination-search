# PDE-extractable readouts and what they actually predict

Exactly one family of quantities in this literature has been validated against overall survival in patients
*and* is computable from an avascular spheroid reaction-diffusion model: the **tumour-dynamics (TGI) metrics**,
namely the growth rate constant KG, the time at which net growth resumes (TTG), and the depth and earliness of
shrinkage. Everything else divides into metrics that are outcome-validated but need vasculature, imaging
calibration or tissue assays we cannot simulate (hypoxic fraction from pO2 histograms, the glioma invasiveness
ratio, gene-expression hypoxia classifiers), and metrics our model computes cleanly but which no one has tied
to a patient endpoint (necrotic core radius, penetration depth, surviving-rim cellularity). The honest short
list is six readouts with a defensible clinical anchor, plus three spatial diagnostics that earn their place as
mechanism rather than prognosis.

The single most useful result for our purposes is also the most uncomfortable one. In the founding analysis of
tumour growth-rate constants, survival correlated strongly with the *growth* constant (log g, Pearson r = -0.72)
and only weakly with the *regression* constant (log d, r = -0.218) across 112 men with metastatic
castration-resistant prostate cancer [Stein 2008](https://doi.org/10.1634/theoncologist.2008-0075). The regression constant is precisely what an in vitro
Hill curve gives us most directly. The growth constant is what emerges from the interaction of kill, regrowth,
oxygen limitation and drug penetration, which is the part a spatial model can actually change.

## Tumour-dynamics metrics (Tier A)

### Growth rate constant KG

KG is the best-evidenced quantity in this entire survey. Beyond the original prostate cohort, it was
re-estimated across eight randomised metastatic castration-resistant prostate cancer trials pooled through
Project Data Sphere, with rates obtainable in 2353 of 2678 evaluable patients; g both correlated with overall
survival and separated docetaxel from prednisone and mitoxantrone [Wilkerson 2017](https://doi.org/10.1016/S1470-2045%2816%2930633-7). In five successive
intramural NCI trials, g fell roughly tenfold across a decade of increasingly effective regimens and
outperformed PSA doubling time as a survival predictor [Stein 2011](https://doi.org/10.1158/1078-0432.CCR-10-1762).

The property that matters most for us is that KG can separate treatments when conventional response endpoints
cannot. In the atezolizumab NSCLC programme, progression-free survival was similar between arms while the TGI
profiles crossed at 25 weeks (more shrinkage with docetaxel, slower growth with atezolizumab) and a survival
model built on KG from the phase II POPLAR trial predicted the phase III OAK hazard ratio at 0.73 (95% prediction
interval 0.63–0.85) against 0.73 observed [Claret 2018](https://doi.org/10.1158/1078-0432.CCR-17-3662). That framework then transported to a different drug
class: in the ALEX trial of alectinib versus crizotinib, 286 of 303 patients yielded a predicted hazard ratio of
0.612 (95% PI 0.480–0.770) against 0.625 observed [Kassir 2023](https://doi.org/10.1007/s00280-023-04558-z).

One caveat travels with KG and belongs in our paper. In the vaccine arm of the NCI prostate series, survival
benefit appeared without a corresponding change in g, which the authors attribute to immunity developing after
vaccination [Stein 2011](https://doi.org/10.1158/1078-0432.CCR-10-1762). KG is a surrogate for cytotoxic and cytostatic effect, not for every mechanism of
benefit. Since our model contains no immune compartment, this is a limit we inherit rather than one we can test.

### Time to tumour growth

TTG was the winner of a direct head-to-head comparison of tumour-size metrics: across 923 Western and 203
Chinese first-line metastatic colorectal cancer patients in two phase III bevacizumab trials, TTG best predicted
overall survival and fully captured the bevacizumab effect, with no impact of ethnicity on the TTG–OS
relationship [Claret 2013](https://doi.org/10.1200/JCO.2012.45.0973). An independent NSCLC study reached the same conclusion, identifying
time-to-growth as the most significant TGI metric in a study-specific model from GEMSTONE-302 [Sheng 2023](https://doi.org/10.1002/psp4.13094).
TTG is a clean simulation output, the time at which the simulated burden trajectory turns from net shrinkage to
net growth, and it is the metric a spatial model should move most, because a hypoxic core that shelters viable
cells brings the turning point forward without necessarily changing the nadir depth.

### Depth of response and early tumour shrinkage

These two are the same signal read as a continuous nadir depth and as a fixed-timepoint threshold. Both were
associated with progression-free, post-progression and overall survival in the phase III TRIBE trial, and both
predicted survival as accurately as RECIST response while avoiding its categorical coarseness
[Cremolini 2015](https://doi.org/10.1093/annonc/mdv112). The largest effect sizes in this survey come from the ETS analysis of the CRYSTAL and OPUS
trials in KRAS wild-type disease: ETS ≥ 20% versus < 20% gave progression-free survival of 14.1 versus 7.3 months
(HR 0.32) and overall survival of 30.0 versus 18.6 months (HR 0.53) in CRYSTAL, with HRs of 0.22 and 0.43 in OPUS
[Piessevaux 2013](https://doi.org/10.1200/JCO.2012.42.8532). A systematic review across ten trials reaches the same direction [Heinemann 2015](https://doi.org/10.1016/j.ejca.2015.06.116).
The founding TGI–OS paper had already shown that week-7 tumour-size change plus baseline burden was enough to
predict an independent phase III survival distribution, at 431 days predicted (90% PI 362–514) against 401 observed
[Claret 2009](https://doi.org/10.1200/JCO.2008.21.0807); a comparable NSCLC analysis across four registration trials found week-8 change, ECOG score
and baseline size to be the survival predictors [Wang 2009](https://doi.org/10.1038/clpt.2009.64).

Note what the co-predictors are in both cases: performance status and baseline tumour burden. Neither is
derivable from a spheroid. Our simulation contributes the tumour-dynamics term of these models and nothing else,
and a fair statement of what we predict is "the tumour-dynamics component of a validated survival model," not
"survival."

### Diagnostics, not objectives

Time to nadir is analytically determined by the two rate constants in a biexponential model, and the original
authors treat it and the nadir value as surrogates for d and g [Stein 2008](https://doi.org/10.1634/theoncologist.2008-0075), so it is worth logging but
carries no information beyond KS and KG. Baseline tumour burden is genuinely prognostic (in KEYNOTE-001, below-median
baseline size gave an objective response rate of 44% versus 23% and an OS hazard ratio of 0.38, holding up in
multivariate analysis across 583 patients) [Joseph 2018](https://doi.org/10.1158/1078-0432.CCR-17-2386), but in a simulation it is an input: it must be
*held constant* across combinations or it will confound every other readout. Untreated volume doubling time is
likewise set by our growth calibration; the natural-history evidence for it comes from 39 patients with 59
untreated small hepatocellular carcinomas, where doubling time ranged from 27 to 606 days [Barbara 1992](https://doi.org/10.1002/hep.1840160122).

## Hypoxia and necrosis

Hypoxia has the strongest and largest patient evidence base of any spatial quantity here. The definitive cohort
is 397 head and neck tumours across seven centres, where the fraction of pO2 readings at or below 2.5 mmHg
(HP2.5) above the population median predicted poor overall survival (P = 0.006) and was the most significant
factor in a stratified multivariate Cox model; five-year survival was roughly flat for HP2.5 between 0 and 20%
and approached zero in the most hypoxic tumours [Nordsmark 2005](https://doi.org/10.1016/j.radonc.2005.06.038). Smaller cohorts agree: in 28 stage IV head
and neck patients, median pO2 above versus below 10 mmHg gave 12-month disease-free survival of 78% versus 22%
(P = 0.009) [Brizel 1997](https://doi.org/10.1016/S0360-3016%2897%2900101-6), and in 103 patients with locally advanced cervical carcinoma, hypoxic tumours had
significantly worse disease-free and overall survival, with oxygenation and FIGO stage the leading independent
prognostic factors (Hockel 1996, PMID-only). In glioblastoma, FMISO-PET hypoxic volume and maximum tissue-to-blood ratio
before radiotherapy both predicted shorter time to progression and survival across 22 patients (P ≤ 0.001)
[Spence 2008](https://doi.org/10.1158/1078-0432.CCR-07-4995). The generality of the association is reviewed by [Vaupel 2007](https://doi.org/10.1007/s10555-007-9055-1).

The computability verdict is **partial**, and the reason is mechanistic rather than technical. Clinical hypoxia
is predominantly *perfusion*-limited, arising from structurally and functionally disturbed microcirculation
[Hockel 2001](https://doi.org/10.1093/jnci/93.4.266); our hypoxia is purely *diffusion*-limited in an avascular geometry. A hypoxic fraction from
our model is an analogue of HP2.5, not a measurement of it, and the numeric values are not on a common scale. Two
further hypoxia metrics are not computable at all: the 15-gene expression classifier, validated in 323
randomised head and neck patients where classifier-hypoxic tumours did worse and were restored to the
non-hypoxic outcome level by hypoxic modification [Toustrup 2011](https://doi.org/10.1158/0008-5472.CAN-11-1182), and immunohistochemical pimonidazole
scores with vascular density, where low control was seen in hypoxic or poorly vascularised tumours across 43
biopsies (Kaanders 2002, PMID-only). Vascular density has no counterpart in an avascular model.

Necrosis has one very large cohort behind it: across 3009 surgically treated renal cell carcinomas, coagulative
tumour necrosis carried a death-from-RCC risk ratio of 5.27 (95% CI 4.56–6.09) in clear cell and 4.20 (95% CI
1.65–10.68) in chromophobe disease, but was not significant in papillary (1.49, 95% CI 0.81–2.74)
[Sengupta 2005](https://doi.org/10.1002/cncr.21206). The tumour-type dependence is the caution: this is not a universal prognosticator, and RCC
lines are a minor part of the NCI-60 panel [Holbeck 2017](https://doi.org/10.1158/0008-5472.CAN-17-0489). Necrotic core radius is trivially computable from
our model; the histologic entity it resembles has partly different causes.

The most important negative result in this survey is also about hypoxia. Adding the hypoxia-activated cytotoxin
tirapazamine to chemoradiotherapy in 861 unselected head and neck patients across 89 sites produced two-year
overall survival of 66.2% versus 65.7% for cisplatin alone, with no difference in failure-free survival, time to
locoregional failure or quality of life [Rischin 2010](https://doi.org/10.1200/JCO.2009.27.4449). Hypoxia is prognostic; a hypoxia-directed
intervention in unselected patients was not beneficial. A model that predicts hypoxia-modulated kill is
therefore not thereby predicting patient benefit, and we should say so explicitly.

## Invasion metrics

The Fisher-KPP metrics, invasion velocity v = 2·√(D·ρ) and the invasiveness ratio ρ/D, have real patient
validation, and it is validation of a kind we cannot inherit. Across 32 newly diagnosed glioblastomas, net
proliferation and invasion rates estimated by fitting the proliferation-invasion model to each patient's own
serial pretreatment MRI were significantly associated with prognosis after controlling for age and Karnofsky
performance status [Wang 2009](https://doi.org/10.1158/0008-5472.CAN-08-3863). The largest such study, 243 contrast-enhancing gliomas, showed that ρ/D
modifies surgical benefit: nodular (high ρ/D) tumours gained over eight months of median survival from gross
total resection (P = 0.00142) while diffuse tumours gained nothing (P = 0.532) [Baldock 2014](https://doi.org/10.1371/journal.pone.0099057). Velocity of
radial expansion was demonstrated on paired pretreatment scans as an explicit proof of principle in a small
series [Swanson 2008](https://doi.org/10.1016/j.clon.2008.01.006), and the modelling lineage runs from [Swanson 2000](https://doi.org/10.1046/j.1365-2184.2000.00177.x) through [Harpold 2007](https://doi.org/10.1097/nen.0b013e31802d9000).

Three things disqualify these as readouts for us. The validation is *patient-imaging-calibrated*: D and ρ were
fitted per patient, not assumed. The endpoint in the strongest study is benefit from *surgery*, not from
chemotherapy. And in our model D is a fixed physical property of the medium and ρ a fixed growth parameter, so
v and ρ/D cannot vary with drug combination at all; reporting them as predictive readouts would be reporting a
constant. Related work links model kinetics to FMISO hypoxic burden across 11 patients [Szeto 2009](https://doi.org/10.1158/0008-5472.CAN-08-3884), useful
as a face-validity check that our coupled model reproduces the right direction but not as a survival claim. The
per-patient radiotherapy-efficacy fitting in [Rockne 2010](https://doi.org/10.1088/0031-9155/55/12/001) rests on nine patients and again needs imaging.
The scope limits are stated plainly by [Jackson 2015](https://doi.org/10.1007/s11538-015-0067-7) and [Rockne 2019](https://doi.org/10.1088/1478-3975/ab1a09).

There is one genuinely transferable idea in this literature: **Days Gained**, the time shift between the
model-predicted untreated trajectory and the observed treated one, which was prognostic for time to recurrence
and overall survival across 63 newly diagnosed glioblastoma patients and separated true progression from
pseudoprogression [Neal 2013](https://doi.org/10.1158/0008-5472.CAN-12-3588). The *construction* (simulate untreated, simulate treated, take the horizontal
displacement at matched burden) transfers cleanly to a spheroid and is arguably our best-motivated spatial
readout. The *validation* does not transfer, because it was obtained under radiotherapy with imaging
calibration in glioma.

## Drug penetration

This is where the spatial model earns its existence, and where the evidence stops at xenografts and 3D culture.
Doxorubicin concentration falls exponentially with distance from tumour blood vessels, halving over roughly
40–50 µm, while hypoxic regions sit 90–140 µm away, so many viable cells see no detectable drug after a single
injection [Primeau 2005](https://doi.org/10.1158/1078-0432.CCR-05-1664). Taxanes reach barely 100 µm into tissue, with paclitaxel penetrating up to
twofold better than docetaxel, and the consequence is measurable as depleted S-phase fractions near vessels but
not far from them [Kyle 2007](https://doi.org/10.1158/1078-0432.CCR-06-1941). In multicell layers about 200 µm thick, cisplatin, etoposide, gemcitabine,
paclitaxel and vinblastine all penetrated slowly relative to the bare support membrane (Tannock 2002, PMID-only), and
penetration of paclitaxel, doxorubicin, methotrexate and 5-fluorouracil was significantly better through loosely
packed than tightly packed layers [Grantab 2006](https://doi.org/10.1158/0008-5472.CAN-05-3077); methotrexate transport determinants are characterised in
[Cowan 2001](https://doi.org/10.1002/1097-0215%2820010101%2991:1<120::AID-IJC1021>3.0.CO;2-Y), and the general synthesis is [Minchinton 2006](https://doi.org/10.1038/nrc1893). The premise that 3D context creates
resistance invisible in monolayer goes back to alkylating-agent resistance conferred by mechanisms operative
only in vivo [Teicher 1990](https://doi.org/10.1126/science.247.4949.1457).

Note how many of those drugs are in the ALMANAC panel. Drug-specific penetration differences are the concrete
mechanism by which our spatial model can reorder combinations relative to their in vitro potency, which is the
whole point of the extension. But not one of these papers reports a patient endpoint. Penetration depth is
**Tier B**, and the claim it supports is "this mechanism plausibly explains why in vitro potency mis-ranks
combinations," not "penetration depth predicts survival."

The closest thing to an outcome validation for this family is the spheroid-size series in which colorectal
spheroids containing both hypoxic and necrotic zones most closely resembled in vivo expression profiles and were
the most 5-fluorouracil-resistant [Daster 2017](https://doi.org/10.18632/oncotarget.13857). That is our core mechanism, demonstrated experimentally, with an
ALMANAC drug. The oxygen length scale itself traces to the original observation that necrotic centres appear
once tumour cords exceed about twice the oxygen diffusion distance [Thomlinson 1955](https://doi.org/10.1038/bjc.1955.55); the parameters come
from direct spheroid measurements [Mueller-Klieser 1984](https://doi.org/10.1016/S0006-3495%2884%2984030-8) and from growth curves derived mechanistically from
doubling time and oxygen consumption rate [Grimes 2016](https://doi.org/10.1371/journal.pone.0153692), and the zone-partition readouts we propose have a
methodological precedent in [Bull 2020](https://doi.org/10.1371/journal.pcbi.1007961). The oxygen enhancement ratio [Gray 1953](https://doi.org/10.1259/0007-1285-26-312-638) explains why hypoxia
was historically measured at all but is irrelevant to a chemotherapy-only model.

## Regrowth and residual disease

Residual burden after a full course of therapy has strong patient validation in breast cancer: residual cancer
burden was independently prognostic for distant relapse-free survival across 382 patients (hazard ratio 2.50,
95% CI 1.70–3.69), with minimal residual disease behaving like a pathologic complete response and extensive
residual disease carrying poor prognosis regardless of receptor status [Symmans 2007](https://doi.org/10.1200/JCO.2007.10.6823). But clinical RCB is a
pathologist's composite over tumour bed size, cellularity, and nodal number and size. Nodes are not simulable.
Our surviving-rim cellularity maps onto the primary-tumour cellularity term by analogy only, so it is **Tier C
as we compute it** even though the index it resembles is Tier A. Time to regrowth is in the same position:
mechanistically sound, closely related to the validated TTG and Days Gained metrics, but not itself an
outcome-validated endpoint [Bruno 2020](https://doi.org/10.1158/1078-0432.CCR-19-0287).

Two preclinical-translation results calibrate how much weight the whole enterprise can bear. Simulated xenograft
tumour growth inhibition correlated with clinical response at r = 0.91 (P = 0.0008) across eight agents, but
only after substituting human pharmacokinetics for mouse; the correlation did not hold using inhibition observed
at the mouse maximum tolerated dose [Wong 2012](https://doi.org/10.1158/1078-0432.CCR-12-0738). Since our model has no pharmacokinetics, this supports our
approach only under an explicit statement that we compare combinations at matched *exposure*, not matched dose;
the PK-PD lineage is [Simeoni 2004](https://doi.org/10.1158/0008-5472.CAN-03-2524) and [Rocchetti 2007](https://doi.org/10.1016/j.ejca.2007.05.011). And the best 3D ex vivo predictiveness data are
drug-specific in a way that should temper any general claim: patient-derived organoids identified non-responders
to irinotecan-based therapy in over 80% of patients without misclassifying benefiters, yet failed entirely for
5-fluorouracil plus oxaliplatin [Ooft 2019](https://doi.org/10.1126/scitranslmed.aay2574), even though organoid responses broadly recapitulated patient
responses in an earlier trial-embedded biobank [Vlachogiannis 2018](https://doi.org/10.1126/science.aao2774). Growth-law choice is not innocuous
either [Benzekry 2014](https://doi.org/10.1371/journal.pcbi.1003800), and the standard model structures and their identifiability problems are catalogued
in [Ribba 2014](https://doi.org/10.1038/psp.2014.12).

## Recommended readouts

Nine quantities, in descending order of how much clinical weight they can carry. The first six have a Tier A
anchor; the last three are spatial diagnostics that explain *why* the first six moved and should be labelled
Tier C wherever they appear.

1. **KG_sim**: growth rate constant from a biexponential fit to simulated total viable burden over the on-treatment window. Tier A anchor ([Stein 2008](https://doi.org/10.1634/theoncologist.2008-0075), [Wilkerson 2017](https://doi.org/10.1016/S1470-2045%2816%2930633-7), [Claret 2018](https://doi.org/10.1158/1078-0432.CCR-17-3662), [Kassir 2023](https://doi.org/10.1007/s00280-023-04558-z)). Direction: higher is worse. This is the primary objective candidate.
2. **TTG_sim**: time at which net growth resumes (zero-crossing of dN/dt, i.e. nadir time of the fitted trajectory). Tier A anchor ([Claret 2013](https://doi.org/10.1200/JCO.2012.45.0973), [Sheng 2023](https://doi.org/10.1002/psp4.13094)). Higher is better. Best-performing metric in a head-to-head comparison, and the one the spatial mechanism should move most.
3. **DpR_sim**: depth of response, 1 − min_t N(t)/N(0). Tier A anchor ([Cremolini 2015](https://doi.org/10.1093/annonc/mdv112), [Heinemann 2015](https://doi.org/10.1016/j.ejca.2015.06.116)). Higher is better. Continuous and cutoff-free, which suits a Bayesian-optimisation objective.
4. **ETS_sim**: relative burden reduction at a fixed early simulated timepoint. Tier A anchor ([Piessevaux 2013](https://doi.org/10.1200/JCO.2012.42.8532), [Claret 2009](https://doi.org/10.1200/JCO.2008.21.0807), [Wang 2009](https://doi.org/10.1038/clpt.2009.64)). Higher is better. Largest published effect sizes of any metric here.
5. **Residual burden fraction**: N(t_end)/N(0) at end of the simulated course. Tier A anchor for the concept, Tier C as we compute it ([Symmans 2007](https://doi.org/10.1200/JCO.2007.10.6823)). Lower is better.
6. **Hypoxic fraction**: viable-cell-weighted fraction below an oxygen threshold, reported both at nadir and time-averaged, and as an absolute hypoxic volume. Tier A anchor with a partial-computability caveat ([Nordsmark 2005](https://doi.org/10.1016/j.radonc.2005.06.038), [Brizel 1997](https://doi.org/10.1016/S0360-3016%2897%2900101-6), (Hockel 1996, PMID-only), [Spence 2008](https://doi.org/10.1158/1078-0432.CCR-07-4995)). Higher is worse. Report as a *diffusion-limited analogue* of clinical hypoxia, never as HP2.5.
7. **Necrotic core radius and viable rim thickness**: from the density and oxygen fields. Tier C for outcome; Tier A only in the tumour-type-specific histologic-necrosis sense ([Sengupta 2005](https://doi.org/10.1002/cncr.21206), [Thomlinson 1955](https://doi.org/10.1038/bjc.1955.55)). Mechanism readout.
8. **Per-drug penetration depth**: radial distance over which each drug's concentration halves, time-averaged over dosing. Tier B ([Primeau 2005](https://doi.org/10.1158/1078-0432.CCR-05-1664), [Kyle 2007](https://doi.org/10.1158/1078-0432.CCR-06-1941), (Tannock 2002, PMID-only), [Grantab 2006](https://doi.org/10.1158/0008-5472.CAN-05-3077)). This is the diagnostic that explains rank changes relative to in vitro potency.
9. **Growth delay (Days-Gained analogue)**: horizontal time shift between treated and untreated simulated trajectories at matched burden. Construction borrowed from a Tier A metric, but Tier C as we compute it ([Neal 2013](https://doi.org/10.1158/0008-5472.CAN-12-3588)).

Log **KS_sim** (the kill rate constant) and **time to nadir** as well, but as diagnostics rather than
objectives: KS is the metric most directly inherited from the in vitro Hill curve and it is the one that did
*not* predict survival [Stein 2008](https://doi.org/10.1634/theoncologist.2008-0075), and time to nadir is algebraically determined by KS and KG. Keep
baseline burden and untreated doubling time fixed across combinations so they cannot confound the comparison
[Joseph 2018](https://doi.org/10.1158/1078-0432.CCR-17-2386), [Barbara 1992](https://doi.org/10.1002/hep.1840160122).

## Do not claim this

- **Do not claim we predict overall survival, PFS, or any patient endpoint.** We compute the tumour-dynamics *component* of models that were fitted and validated in patients. Baseline burden, performance status, albumin, metastatic-site count and neutrophil-lymphocyte ratio are co-predictors in every published TGI–OS model ([Claret 2018](https://doi.org/10.1158/1078-0432.CCR-17-3662), [Wang 2009](https://doi.org/10.1038/clpt.2009.64), [Sheng 2023](https://doi.org/10.1002/psp4.13094)) and none is derivable from a spheroid.
- **Do not report invasion velocity v = 2·√(D·ρ) or the invasiveness ratio ρ/D as predictive readouts.** Their patient validation is imaging-calibrated glioma, the strongest endpoint is benefit from surgery rather than chemotherapy, and in our model D and ρ are fixed inputs, so these quantities are constants across combinations [Baldock 2014](https://doi.org/10.1371/journal.pone.0099057), [Wang 2009](https://doi.org/10.1158/0008-5472.CAN-08-3863).
- **Do not equate our hypoxic fraction with a clinical hypoxia measurement.** Clinical hypoxia is largely perfusion-limited [Hockel 2001](https://doi.org/10.1093/jnci/93.4.266); ours is purely diffusion-limited and avascular. Same direction, different quantity, no shared scale.
- **Do not claim that hypoxia-modulated kill implies patient benefit.** Tirapazamine added to chemoradiotherapy in 861 unselected patients gave 66.2% versus 65.7% two-year survival [Rischin 2010](https://doi.org/10.1200/JCO.2009.27.4449).
- **Do not present penetration depth, necrotic core size, surviving-rim cellularity, or time to regrowth as outcome-validated.** They are Tier B and Tier C. No paper in this survey ties any of them to a patient endpoint.
- **Do not headline the kill rate constant KS or in vitro potency as a survival surrogate.** log(d) correlated with survival at r = -0.218 against r = -0.72 for log(g) [Stein 2008](https://doi.org/10.1634/theoncologist.2008-0075).
- **Do not claim anything requiring vasculature, immune infiltration, drug pharmacokinetics, or patient imaging calibration.** That excludes perfusion-limited hypoxia, EPR effects, nodal residual disease [Symmans 2007](https://doi.org/10.1200/JCO.2007.10.6823), gene-expression hypoxia classifiers [Toustrup 2011](https://doi.org/10.1158/0008-5472.CAN-11-1182), pimonidazole-plus-vascularity scores (Kaanders 2002, PMID-only), immunotherapy benefit [Stein 2011](https://doi.org/10.1158/1078-0432.CCR-10-1762), and any dose-to-exposure translation [Wong 2012](https://doi.org/10.1158/1078-0432.CCR-12-0738), [Rocchetti 2007](https://doi.org/10.1016/j.ejca.2007.05.011).
- **Do not generalise 3D-model predictiveness across agents.** Organoid prediction worked for irinotecan-based therapy and failed for 5-fluorouracil plus oxaliplatin in the same cohort [Ooft 2019](https://doi.org/10.1126/scitranslmed.aay2574).
- **Do not describe our readouts as breaking monotonicity until it is demonstrated.** That in vitro potency and spatial outcome can dissociate is mechanistically supported ([Primeau 2005](https://doi.org/10.1158/1078-0432.CCR-05-1664), [Kyle 2007](https://doi.org/10.1158/1078-0432.CCR-06-1941), [Daster 2017](https://doi.org/10.18632/oncotarget.13857)) but it is a hypothesis about our model, to be shown by Spearman correlation between in vitro AUC and each simulated readout across combinations. It is not a literature result.

## Raising a metric a tier

Nothing in this literature validates a spheroid-derived spatial metric against a patient endpoint. The nearest
achievable step is the middle rung: correlate simulated readouts against xenograft tumour growth inhibition for
ALMANAC pairs with published in vivo data, at matched exposure rather than matched dose [Wong 2012](https://doi.org/10.1158/1078-0432.CCR-12-0738). The rung
above that requires paired ex vivo and clinical response data of the kind organoid studies now generate
[Ooft 2019](https://doi.org/10.1126/scitranslmed.aay2574), [Vlachogiannis 2018](https://doi.org/10.1126/science.aao2774), which is a different experiment from ours.
