# Where does the selectivity signal appear?
### Interpretable QSAR for D₂ / 5-HT₂A receptor selectivity

Code and data for a study on model interpretability in drug discovery. The
question we set out to answer: when you train a model to predict receptor
selectivity, does the way you set up the prediction task change how good the
resulting explanations are?

We work with selectivity between the dopamine D₂ and serotonin 5-HT₂A
receptors. The 5-HT₂A/D₂ affinity ratio matters clinically. It is what
separates atypical antipsychotics (clozapine, risperidone) from typical ones
(haloperidol), and it tracks with a lower rate of extrapyramidal motor side
effects (Meltzer, 1989). On top of predicting that ratio, we check whether
the models' explanations point at the parts of each molecule a chemist would
say are responsible for selectivity, and whether that depends on how the
model was trained.

## 1. The question and the three approaches

Selectivity is the difference in binding affinity:

```
ΔpKi(m) = pKi_5HT2A(m) − pKi_D2(m)
```

There are three sensible ways to model this. We run all three on the same
data split, the same feature space, and the same tuning budget, so that the
training objective is the only thing that changes between them.

| | Approach | How selectivity is obtained |
|---|---|---|
| **A** | Direct | One model trained directly on ΔpKi |
| **B** | Post-hoc | Two independent single-receptor models, predictions subtracted |
| **C** | Multi-task | One shared-encoder model with two receptor heads, outputs subtracted |

We test six hypotheses about how the choice of approach affects predictive
accuracy and, mainly, explanation quality. We treat explanation quality as
two separate things: whether the explanations are stable (reproducible across
reruns) and whether their content is right (whether they recover known
pharmacophores).

## 2. Repository layout

```
qsar_selectivity_repo/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── utils.py               shared functions, imported by every notebook
│   ├── 01_data.ipynb          ChEMBL fetch, curation, EDA, scaffold split
│   ├── 02_models.ipynb        RF / XGB / LightGBM / MLP + Optuna tuning
│   ├── 03_explanations.ipynb  SHAP attribution + EH1 / EH2 / EH4
│   ├── 04_benchmark.ipynb     EH3 pharmacophore-recovery benchmark
│   └── 05_mmp.ipynb           matched-molecular-pair enrichment (supporting)
├── data/
│   └── explanation_benchmark_dataset.csv    18-compound curated benchmark
├── presentation/
│   ├── qsar_selectivity.pptx
│   └── qsar_selectivity.pdf
└── docs/
    └── QSAR_Interpretability_Literature_Review.pdf
```

The notebooks are a linear pipeline. Each one writes files that the next one
reads (section 5), so run them in order the first time.

## 3. Data and sources

Training data comes straight from ChEMBL (release 34) through the official
REST API. Nothing bulk is committed to the repo; notebook 01 fetches it
reproducibly.

| Target | ChEMBL ID | Approx. Ki records | Role |
|---|---|---|---|
| Dopamine D₂      | `CHEMBL217` | ~9,800 | receptor 1 |
| Serotonin 5-HT₂A | `CHEMBL224` | ~5,200 | receptor 2 |

Curation (in `utils.py`):

- Ki only. IC₅₀ and other assay types are excluded on purpose. Mixing assay
  formats puts a systematic offset into ΔpKi that you cannot separate from
  real selectivity signal (Kalliokoski et al., 2013).
- Salt stripping, units harmonised to nM, conversion to pKi.
- Replicate measurements per compound aggregated by median.
- Overlap set: the ~2,700 compounds with a measured Ki at both receptors.
  These are the only compounds for which selectivity is actually defined.
- Bemis–Murcko scaffold split (70/15/15), shared across all three approaches
  so no compound class leaks from train into test.

Features are ECFP4 Morgan fingerprints (radius 2, 2048 bits) plus 9
physicochemical descriptors (MW, logP, TPSA, HBD, HBA, rotatable bonds,
aromatic rings, fraction Csp³, formal charge).

### The pharmacophore benchmark

No public dataset tells you which atoms drive selectivity, so we built one.
`data/explanation_benchmark_dataset.csv` has 18 well-characterised ligands in
four groups, each with literature Ki values, a computed ΔpKi, and the
scaffold expected to drive its selectivity.

| Group | n | Examples | Purpose |
|---|---|---|---|
| G1a, 5-HT₂A selective | 4 | Risperidone, Ketanserin, Clozapine, Volinanserin | pharmacophore recovery |
| G1b, D₂ selective | 3 | Raclopride, Sulpiride, Amisulpride | pharmacophore recovery |
| G2, inactive controls | 4 | Atropine, Naloxone, Caffeine, Scopolamine | out-of-distribution check |
| G3, active non-selective | 6 | Haloperidol, Olanzapine, Spiperone, … | direction accuracy only |

One row (asparagine) is a placeholder and gets dropped at load time, leaving
17 usable compounds.

## 4. How to run

### Environment

Python 3.11 is recommended. The PyTorch code auto-detects the device, so
Apple Silicon, CUDA, and CPU all work.

```bash
# from the repository root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name qsar-selectivity
```

If `pip install rdkit` fails on your platform, use conda instead:
`conda install -c conda-forge rdkit`.

### Running the pipeline

```bash
jupyter lab notebooks/
```

Run them in order:

1. `01_data.ipynb` fetches D₂ and 5-HT₂A activity from ChEMBL, curates it,
   runs EDA, and builds the scaffold split. Needs internet. Writes `data.pkl`.
2. `02_models.ipynb` trains and tunes every model (RF, XGB, LightGBM, MLP)
   for all three approaches. Writes `models.pkl`.
3. `03_explanations.ipynb` computes SHAP attributions and tests EH1, EH2,
   EH4. Writes `shap_*.npy`.
4. `04_benchmark.ipynb` runs the EH3 pharmacophore-recovery benchmark. Reads
   `data.pkl`, `models.pkl`, and the benchmark CSV.
5. `05_mmp.ipynb` runs the matched-molecular-pair enrichment analysis. Reads
   `data.pkl` and `shap_*.npy`.

Timing: notebook 01 is network-bound, about 10–20 min. Notebook 02 is the
expensive one; a full Optuna run is an overnight job, but each search has a
trial budget at the top of its cell that you can drop for a quick pass.
Notebooks 03–05 take a few minutes each. Everything is seeded (`SEED = 42`;
`seed_everything()` covers Python, NumPy, PyTorch, CUDA, and MPS).

A note on the benchmark: EH3 scores the explanations of the fitted models,
not held-out prediction, so the benchmark compounds are run through the
models as trained. Some of them may overlap the training distribution. That
is fine for judging explanation quality, and we flag it in the limitations
rather than work around it.

## 5. Pipeline data flow

```
01_data        →  data.pkl
02_models      →  models.pkl                     (reads data.pkl)
03_explanations →  shap_*.npy                     (reads data.pkl, models.pkl)
04_benchmark                                      (reads data.pkl, models.pkl, benchmark CSV)
05_mmp                                            (reads data.pkl, shap_*.npy)
```

`data.pkl`, `models.pkl`, and `shap_*.npy` are generated, not committed.
Running the notebooks in order recreates them.

## 6. Results

### Hypotheses and verdicts

| | Hypothesis | Verdict |
|---|---|---|
| H0 | Selectivity is not learnable beyond noise | Rejected, best val R² ≈ 0.64 |
| Main | Direct learning gives better explanation quality | Partially supported (below) |
| EH1 | Direct explanations are the most stable across seeds | Supported |
| EH2 | Shared representations give more receptor-specific explanations | Not supported, the reverse holds |
| EH3 | Direct explanations recover known pharmacophores better | Not supported, all approaches tie |
| EH4 | Direct explanations are more localised | Partial, entropy order C < A < B |

### Two axes of explanation quality

The main result is that explanation quality is not a single quantity. It
splits into two things that the training objective affects differently.

Stability (EH1). Cross-seed agreement of feature rankings, as Kendall τ over
5 seeds:

| Approach | Mean τ | Std |
|---|---|---|
| A, Direct | 0.801 | 0.035 |
| C, Multi-task | 0.709 | 0.043 |
| B, Post-hoc | 0.623 | 0.081 |

Direct learning gives the most reproducible explanations. Post-hoc
subtraction is the least stable, which makes sense: subtracting two
independently noisy SHAP vectors amplifies the noise. The direct approach is
also the most consistent across explanation methods. GradientSHAP, Integrated
Gradients, DeepLIFT, and TreeSHAP agree at ρ ≥ 0.95.

Content alignment (EH3). Pharmacophore recovery on the benchmark is the same
across all three approaches within noise; the recall curves overlap from
k = 1 to 30. But all three sit well above random. About 60% of the top-10
attributed bits are pharmacophore bits, against roughly 15% expected from
random selection over active bits, so around 4× enrichment, checked against a
2,000-draw random baseline.

So the main hypothesis holds in one direction only. The training objective
controls how reproducible an explanation is, but not what the explanation
says chemically. For a model interpretation you need to be stable across
reruns, train directly on selectivity. If you only care which fragments get
flagged, the three approaches are interchangeable.

### Other findings

Multi-task conflates rather than separates (EH2). We expected the shared
encoder to pull the two receptors' signals apart. It does the opposite. The
per-receptor attribution vectors are more similar under multi-task
(cosine 0.51) than under independent models (0.29; Mann–Whitney p < 0.001).
The shared encoder learns features that predict both receptors at once, so
"what matters for D₂?" and "what matters for 5-HT₂A?" come back nearly
identical. If you want receptor-specific explanations, independent models are
the better choice.

A model trained on a bioactivity database has no notion of "inactive" (G2).
On the inactive controls (caffeine, atropine, naloxone) the model assigns
attributions just as large as it does for genuinely selective drugs. Naloxone
has 13 bits above the 0.1-ΔpKi threshold, and caffeine's single largest
attribution (0.51) is the biggest anywhere in the benchmark. This is a data
limitation, not a model bug. ChEMBL contains almost only active compounds;
the training pKi values are essentially all above 2.2, so the model never
sees inactivity and cannot represent it. For an out-of-distribution molecule
it just splits attribution between "D₂-like" and "5-HT₂A-like" patterns,
which roughly cancel into a small net prediction built from large, opposed
contributions.

### Predictive performance

For context, not as the main result. Scaffold-split test R² for the best
model per family on Approach A: trees 0.54, MLP 0.53. Validation R² across
all approaches and families runs 0.61–0.66, so the approaches are
predictively comparable, which is what makes the explanation comparison fair.
Full per-approach numbers are in notebook 02 and on slide 5 of the
presentation.

## 7. Other material in this repo

- `presentation/qsar_selectivity.pdf` (and `.pptx`): the final presentation,
  with the full results tables, the EH1 stability chart, the EH2 result, the
  benchmark design, and annotated fragment maps for the best-explained
  compounds (Ketanserin, Amisulpride).
- `docs/QSAR_Interpretability_Literature_Review.pdf`: background reading on
  interpretable QSAR and the pharmacology of D₂/5-HT₂A selectivity.
- `05_mmp.ipynb`: matched-molecular-pair analysis. The ~168k MMP pairs
  (17k of them selectivity cliffs) show how hard the problem is and confirm
  that top-SHAP bits are enriched in transformations that shift selectivity a
  lot (χ², all p ≈ 0). The mismatch report flags fluorine as over-weighted by
  SHAP and tertiary-amine substitution as under-weighted, both model-free SAR
  observations.

## 8. Limitations

- Benchmark size. Only 7 compounds carry a defined pharmacophore, so the EH3
  comparison is underpowered (Wilcoxon n = 7). Read the "no best approach"
  result as no detectable difference, not as proof they are exactly equal.
- One receptor pair. Everything here is D₂/5-HT₂A. Whether the EH1 stability
  result carries over to other GPCR pairs is the obvious next step.
- Two model families. Trees and MLPs only; no graph neural network in the
  main comparison.
- The pharmacophore SMARTS come from the medicinal-chemistry literature. A
  structure-based annotation from co-crystal contacts would be firmer, and is
  the better route for any extension.

## 9. Where this could go

The pipeline is receptor-agnostic. The fetch, curation, split, modelling, and
explanation code all take target IDs as parameters. The natural extensions
are to add a second receptor pair with plenty of ChEMBL data and a
well-characterised pharmacophore (μ/κ opioid or β₁/β₂ adrenergic) to see if
EH1 generalises, to build a structure-based benchmark from PDB co-crystal
contacts instead of literature SMARTS, and to add a message-passing GNN as a
third model family.

## 10. References

- Meltzer, H. Y. (1989). Clinical studies on the mechanism of action of
  clozapine. Psychopharmacology.
- Kalliokoski, T., Kramer, C., Vulpetti, A., Gedeck, P. (2013). Comparability
  of mixed IC₅₀ data, a statistical analysis. PLoS ONE 8(4):e61007.
- Wang, S. et al. (2018). D₂ dopamine receptor structure. Nature
  (PMID 29466326). Structural basis of the conserved Asp3.32 anchor.
- Rowley, M. et al. (1996). Indole-piperidine D₂ antagonists. J. Med. Chem.
  (PMID 8648587).
- Lundberg, S. M., Lee, S.-I. (2017). A unified approach to interpreting model
  predictions (SHAP). NeurIPS.
- Data: ChEMBL database, release 34. https://www.ebi.ac.uk/chembl/
