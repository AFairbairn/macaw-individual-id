# Individual identification from parrot calls

This repository produces every result table in the paper on acoustic individual identification
in two macaw species. It covers 15 pre-trained embedding models, an MFCC baseline, and a
metric-learning head.

Every output of the pipeline is a CSV file. Figures are made separately from those files, so
a change to a figure never changes a result.

The species are:

- `aa`, *Ara ambiguus* (great green macaw), 8 individuals, one call type.
- `ag`, *Ara glaucogularis* (blue-throated macaw), 16 individuals, 8 call types.

---

## 1. Requirements

### 1.1 Software

Create the environment. Use conda where it is available.

```bash
conda env create -f environment.yml
conda activate parrot-id
```

On a machine with no conda, use pip and a virtual environment. `requirements.txt` holds the
same set.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

To record the environment that produced a set of results, run this once, in the activated
environment on the machine that produced them.

```bash
python src/08_freeze_environment.py freeze
```

That writes `environment.lock/`, which holds every installed package version, an md5 checksum
for every model weight file, and the Python, OS and CUDA versions. On every later run,
`run_all.sh` compares the environment against that record and reports a difference. To stop
the run on a difference, set `STRICT_ENV=1`.

### 1.2 Data

The audio and the embeddings are archived separately from the code. Download the archive,
then point `PARROT_DATA` at it.

```bash
export PARROT_DATA=/path/to/the/dataset
```

If `PARROT_DATA` is not set, the pipeline reads `./data`. That directory holds the master
metadata tables only.

The archive holds three things.

| What | Where it goes | Which path needs it |
|---|---|---|
| The audio | `$PARROT_DATA` | `./run_all.sh all` |
| The 15 model embeddings | `bacpipe_results/` in the repository | `./run_all.sh analyse` |
| The MFCC features | `mfcc_results/` in the repository | `./run_all.sh analyse` |

The dataset follows the metadata framework of Knight et al. (2024).

- Recording method: targeted.
- Spatial extent: single location.
- Temporal extent: multiple recordings within one season.
- Annotation: call level.

### 1.3 Hardware

| Stage | Hardware |
|---|---|
| 0, 2, 3, 4, 5 | 8 CPU cores and 32 GB of memory. |
| 1 | One GPU. |

Stages 2 to 5 run scikit-learn on fewer than 1,100 vectors, so a GPU gives no useful speed
increase.

Stage 1 extracts the embeddings. A small GPU is sufficient. On the LRZ AI Systems cluster,
select the P100. Do not queue for the A100, because the extra memory gives no benefit here.

Note: Three groups of models always run on CPU, even on a GPU node.

- `perch_bird`, `perch_v2`, and `surfperch` use JAX.
- `birdnet` uses TensorFlow.
- `avesecho_passt` is pinned to CPU. See section 8.3.

---

## 2. How to run

To reproduce every result table from the published embeddings, run the analysis path. It
runs on CPU.

```bash
./run_all.sh analyse
```

To reproduce the pipeline from the raw audio instead, run the full path. It extracts the
embeddings first, so it needs a GPU and the audio.

```bash
./run_all.sh all
```

To extract the embeddings and stop there, run `./run_all.sh embed`.

The outputs go to `results/`. Each stage prints its duration when it finishes, and the run
ends with a table of all of them.

Every stage skips work it has already done, so a rerun after a failure continues from the
point it stopped.

The pipeline detects the device. To override the detected device, set `FORCE_DEVICE`.

```bash
FORCE_DEVICE=cpu ./run_all.sh analyse
```

---

## 3. Terms

Every term used in this repository has one meaning, and that meaning is in
`docs/GLOSSARY.md`. The code, the documentation and the manuscript use the same word for the
same thing. The most important ones are below.

| Term | Meaning |
|---|---|
| **call** | One vocalisation. The unit of the audio files. |
| **clip** | One audio file. One clip holds exactly one call from one bird. |
| **recording** | One continuous source recording. Calls are cut from a recording. A recording can hold calls from more than one bird. |
| **recording_id** | The identifier of a recording. This is the unit that the train and test split groups on. |
| **encounter** | The set of calls from one bird within one recording. |
| **subset** | A group of birds that is scored together. `aa` has `all`. `ag` has `all` and `lab`. |
| **probe** | A logistic regression classifier that is fitted to frozen embeddings. |
| **embedding** | The output vector of a model for one clip. |
| **head** | The small metric-learning network that is trained on top of frozen embeddings. |
| **EER** | Equal error rate. The verification metric. Lower is better. Chance is 0.5. |
| **leakage delta** | The random split accuracy minus the by-recording accuracy. |

The rules that govern how this repository is written are in `docs/WRITING_STANDARD.md`.
`tests/test_writing.py` enforces them.

---

## 4. What the pipeline does

The pipeline has six stages. Stage 1 is the only stage that uses a GPU.

| Stage | Script | Output | Hardware |
|---|---|---|---|
| 0 | `00_build_master_metadata.py` | `data/<sp>/metadata/<sp>_master.csv` | CPU |
| 1 | `01_extract_embeddings.py`, `01b_verify_embeddings.py` | `bacpipe_results/<sp>/embeddings/` | GPU |
| 2 | `02_extract_mfcc.py` | `mfcc_results/<sp>/embeddings/` | CPU |
| 3 | `03_classify.py` | `results/<sp>/rows.csv` | CPU |
| 4 | `04_metric_learning.py` | `results/<sp>/<subset>/metric_learning/` | CPU |
| 5 | `05_diagnostics.py`, `06_leakage_experiment.py`, `09_supplementary_bouts.py`, `07_manifest.py` | `results/diagnostics/` | CPU |

Stage 0 builds the master metadata table. This table is the single source of truth for the
bird identity and the `recording_id` of every clip. Every later stage reads it.

---

## 5. Repository layout

```
macaw-individual-id/
├── README.md                     This file.
├── LICENSE                       MIT.
├── environment.yml               Pinned dependencies, for conda.
├── requirements.txt              The same set, for pip.
├── config.yaml                   Every analysis choice, in one file.
├── run_all.sh                    The single entry point.
├── docs/
│   ├── GLOSSARY.md               One meaning for every term.
│   └── WRITING_STANDARD.md       How this repository is written.
├── src/
│   ├── 00_build_master_metadata.py
│   ├── 01_extract_embeddings.py
│   ├── 01b_verify_embeddings.py
│   ├── 02_extract_mfcc.py
│   ├── 03_classify.py
│   ├── 04_metric_learning.py
│   ├── 05_diagnostics.py
│   ├── 06_leakage_experiment.py
│   ├── 07_manifest.py
│   ├── 08_freeze_environment.py
│   └── 09_supplementary_bouts.py
├── tests/
│   ├── test_splits.py            Asserts that the split does not leak.
│   └── test_writing.py           Asserts the writing standard.
├── data/<sp>/metadata/           The master tables. Tracked here.
└── results/                      All outputs. Regenerated, never edited by hand.
```

---

## 6. The evaluation protocol

### 6.1 The split groups on the recording

The train and test split uses `GroupKFold` with `recording_id` as the group. Calls from one
recording never appear in both the train set and the test set.

This is the most important choice in the analysis. Recordings carry an acoustic signature.
Within one bird, a model can identify which recording a call came from at 2.7 to 3.7 times
chance. If calls from one recording appear in both the train set and the test set, the model
can use that signature instead of the voice, and the reported accuracy is too high.

The pipeline reports both splits:

- **By recording.** The primary result. This split does not leak.
- **Random.** A stratified split that ignores the recording. This result is too high.

The difference between the two is the leakage delta. The delta is 0.08 to 0.10 for `aa` and
0.22 to 0.29 for `ag`.

### 6.2 The call set

The analysis uses single calls only, for both species. One clip holds one call.
This is the matched acoustic unit across the two species.

The *Ara ambiguus* collection also holds 255 repeated call bouts. A bout gives a
model more acoustic material, and it raises *Ara ambiguus* accuracy by 0.03 to
0.07. *Ara glaucogularis* has no bouts.

Warning: Do not add the bouts to the main analysis. The main result of the paper
is that *Ara ambiguus* is easier to identify than *Ara glaucogularis*. If *Ara
ambiguus* used bouts and *Ara glaucogularis* did not, part of that difference
would come from the acoustic unit rather than from the biology.

`src/09_supplementary_bouts.py` scores the bouts separately, for the supplement.

### 6.3 The subsets

`ag` is reported for two subsets, because the birds are not all housed together.

- `all` holds all 16 birds. Some birds live in separate rooms, so the room can stand in for
  the identity. This result is an optimistic upper bound.
- `lab` holds the 12 birds that share one room. The room cannot stand in for the identity.
  This result is the honest one. Report it as the primary `ag` result.

### 6.4 The metrics

The pipeline reports five metrics for every model.

1. **Linear probe accuracy.** A logistic regression classifier on the frozen embeddings.
2. **Cosine nearest-centroid accuracy.** One mean template for each bird. This metric matches
   the enrolment case.
3. **Verification EER and AUROC.** Positive pairs are the same bird in different recordings.
   This metric generalises to the open-set case.
4. **Clustering.** KMeans, affinity propagation, and HDBSCAN. See Section 6.6.
5. **Encounter accuracy.** See Section 6.5.

Caution: `ag` is not balanced. The number of calls for each bird runs from 41 to 108. Report
the majority-class baseline (0.108 for `all`, 0.111 for `lab`) next to `1/n_birds`. If you
report `1/n_birds` alone, the result looks better than it is.

`aa` is balanced at 60 calls for each bird, so `1/n_birds` (0.125) is correct there.

### 6.5 Encounter accuracy and its assumption

Encounter accuracy pools the calls of one bird within one recording into one mean query. The
query is then matched to the nearest enrolled centroid.

This metric answers one question: given a set of calls that are known to come from one bird
in one recording, which bird is it?

The metric assumes that the calls have already been grouped by individual. That assumption
holds when the encounter has one bird, or when source localisation has separated the birds.
The assumption is necessary here, because 14 of 74 `ag` recordings and 2 of 211 `aa`
recordings hold calls from two birds. To pool all calls in those recordings would average two
birds into one query.

`03_classify.py` records the number of multi-bird recordings in every result row, so the
assumption stays visible in the output.

### 6.6 Clustering metrics

The pipeline reports NMI, AMI, ARI, and the inferred number of clusters.

Caution: Do not report NMI on its own. NMI increases as the number of clusters increases.
Affinity propagation infers 37 to 85 clusters for 8 to 16 birds, so its NMI is higher than
the NMI of KMeans while its ARI is lower. Always report NMI next to AMI or ARI, and next to
the inferred number of clusters.

---

## 7. How to verify a run

Every run writes `results/MANIFEST.csv`. This file holds the md5 checksum of every output.

To confirm that a rerun gives the same results, compare the manifests.

```bash
diff <(sort results/MANIFEST.csv) <(sort results_previous/MANIFEST.csv)
```

If the two manifests match, the results are identical. If a line differs, that file changed.

To confirm that the split does not leak, run the tests.

```bash
pytest tests/
```

`tests/test_splits.py` asserts that no `recording_id` appears in both the train set and the
test set, for every species and every subset.

`tests/test_writing.py` asserts that every comment, docstring and documentation file follows
`docs/WRITING_STANDARD.md`.

---

## 8. Known issues

### 8.1 bacpipe does not detect CUDA

`bacpipe` reads its device from `settings.yaml` inside the installed package. It does not
detect CUDA. The shipped value is `cpu`.

Caution: If you do not change this value, the torch models run on CPU on a GPU node. The run
is then much slower and gives no error.

`run_all.sh` sets this value automatically and keeps a backup of the original file.

### 8.2 bacpipe prints harmless errors

After `bacpipe` writes the embeddings, it runs a dashboard step that this pipeline does not
use. That step prints many tracebacks. The messages below are harmless.

- `annotations.csv` not found
- `dim_reduced_embeddings`
- `list.remove(x)`
- `Length mismatch between time_bins`
- `Input image size 128*300`

Do not use the absence of tracebacks to confirm success. Run `01b_verify_embeddings.py`
instead. That script counts the `.npy` files for each model.

### 8.3 avesecho_passt fails on CUDA

`avesecho_passt` moves the model to the device but not the input audio. On CUDA, every file
fails and the model writes zero embeddings. `01_extract_embeddings.py` pins this model to CPU
through the `CPU_ONLY_MODELS` list.

### 8.4 Do not upgrade JAX

Warning: Do not run `pip install -U jax[cuda12]`. The `-U` flag upgrades numpy to version 2,
which breaks the pinned torch and bacpipe environment.

To repair a broken environment, delete it and rebuild it from `environment.yml`. Then run
`python src/08_freeze_environment.py verify`.

### 8.5 Stage 4 writes only at the end of a subset

`04_metric_learning.py` writes its results when a subset finishes. If the process stops in the
middle of a subset, the work for that subset is lost.

Run stage 4 inside `tmux` or `screen`, or submit it as a batch job. Stage 3 is different. It
appends its results for each model and skips completed work when you rerun it.
