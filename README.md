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

**Python 3.11 or newer is required.** bacpipe publishes no build for 3.10, and bacpipe
supplies 13 of the 15 pre-trained models.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` names five packages. bacpipe pins the rest of the stack, and it pulls
torch, torchaudio, transformers, librosa, numpy, pandas, scipy, scikit-learn and PyYAML with
it.

If the machine has no Python 3.11 and you cannot install one, use `uv`. It puts a standalone
interpreter in your home directory and needs no administrator rights.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Stage 1 downloads model weights from `huggingface.co` the first time it runs, so that
machine needs network access. Later runs read the cache. Stages 2 to 5 need no network.

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

The archive holds the audio at these two paths, and the pipeline resolves every clip from
them. Point `PARROT_DATA` either at the archive root or at the `data` directory inside it.
Both work.

```
data/aa/all_calls/single/*.wav
data/ag/audio/single/<bird>/<call_type>/*.wav
```

The master metadata tables are not part of the archive. They are curated, they are versioned
with this code, and every stage reads them from `data/<sp>/metadata/` in the repository. Only
the audio is read from `$PARROT_DATA`. If the archive holds its own copy of a master table,
the repository copy is the one that counts.

The archive holds three things.

| What | Where it goes | Which path needs it |
|---|---|---|
| The audio | `$PARROT_DATA` | every path. Stage 0 pads it and stage 2 reads it. |
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
- `avesecho_passt` is pinned to CPU. See section 8.5.

---

## 2. How to run

To reproduce every result table from the published embeddings, run the analysis path. It
needs no GPU. It still needs the audio, because stage 0 pads every clip and stage 2 computes
the MFCC baseline from it.

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

Stage 1 and stage 3 skip work they have already done, so a rerun after a failure continues
from the point it stopped. Stage 1 skips a model that wrote a file for every clip, and stage 3
skips a model that already has a row. Stages 0, 2, 4 and 5 recompute in full every time.

To recompute one model of stage 1 on its own:

```bash
python src/01_extract_embeddings.py --species aa --models birdnet --device cuda
```

That deletes the output of the named model first, then rebuilds it. Every other model is left
alone.

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

The pipeline has six stages, numbered 0 to 5. Stage 1 is the only stage that uses a GPU.

| Stage | Script | Output | Hardware |
|---|---|---|---|
| 0 | `00_build_master_metadata.py`, `00a_pad_audio.py` | `audio_padded/` | CPU |
| 1 | `01a_fetch_checkpoints.py`, `01_extract_embeddings.py`, `01b_verify_embeddings.py` | `bacpipe_results/<sp>/embeddings/` | GPU |
| 2 | `02_extract_mfcc.py` | `mfcc_results/<sp>/embeddings/` | CPU |
| 3 | `03_classify.py` | `results/<sp>/rows.csv` | CPU |
| 4 | `04_metric_learning.py` | `results/<sp>/<subset>/metric_learning/` | CPU |
| 5 | `05_diagnostics.py`, `06_leakage_experiment.py`, `09_supplementary_bouts.py`, `07_manifest.py` | `results/diagnostics/`, `results/supplementary/`, `results/MANIFEST.csv` | CPU |

Stage 0 builds the master metadata table. This table is the single source of truth for the
bird identity and the `recording_id` of every clip. Every later stage reads it.

---

## 5. Repository layout

```
macaw-individual-id/
├── README.md                     This file.
├── LICENSE                       MIT.
├── requirements.txt              The five direct dependencies.
├── config.yaml                   Every analysis choice, in one file.
├── run_all.sh                    The single entry point.
├── docs/
│   ├── GLOSSARY.md               One meaning for every term.
│   └── WRITING_STANDARD.md       How this repository is written.
├── src/
│   ├── 00_build_master_metadata.py
│   ├── 00a_pad_audio.py
│   ├── 01a_fetch_checkpoints.py
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
│   ├── test_output_paths.py      Asserts that every model writes where the stages read.
│   ├── test_padding.py           Asserts that padding changes the length and nothing else.
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

The pipeline reports six metrics for every model.

1. **Macro F1.** The primary metric. `ag` is not balanced, and macro F1 gives every bird the
   same weight whatever its number of calls. Column `macro_f1_byrec`.
2. **Linear probe accuracy.** A logistic regression classifier on the frozen embeddings.
3. **Cosine nearest-centroid accuracy.** One mean template for each bird. This metric matches
   the enrolment case.
4. **Verification EER and AUROC.** Positive pairs are the same bird in different recordings.
   This metric generalises to the open-set case.
5. **Clustering.** KMeans, given the true number of birds. See Section 6.6.
6. **Encounter accuracy.** See Section 6.5.

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

`00_build_master_metadata.py` reports the number of multi-bird recordings for each species
when it validates the tables, and `tests/test_splits.py` asserts the two counts above.

### 6.6 Clustering metrics

KMeans runs with the true number of birds. The pipeline reports four columns:
`cluster_ari`, `cluster_ami` and `cluster_nmi` against the bird, and `cluster_ari_recording`
against the recording.

Caution: Do not report NMI on its own. NMI rises as the number of clusters rises, so it
rewards a method that splits the data more finely. Report it next to AMI and ARI.

`cluster_ari_recording` is the control. A model that clusters by recording rather than by
bird scores highly on it, and that is the failure this study is about.

---

## 7. How to verify a run

Every run writes `results/MANIFEST.csv`. This file holds the md5 checksum of every output.

To confirm that a rerun gives the same results, compare the manifests.

```bash
diff <(sort results/MANIFEST.csv) <(sort results_previous/MANIFEST.csv)
```

If the two manifests match, the results are identical. If a line differs, that file changed.

The manifest holds the path, the size and the md5 of every output, so two runs of the same
code on the same data produce the same file. Save `results/MANIFEST.csv` under another name
before a rerun, and compare against that.

To confirm that the split does not leak, run the tests.

```bash
pytest tests/
```

`tests/test_splits.py` asserts that no `recording_id` appears in both the train set and the
test set, for every species and every subset.

`tests/test_writing.py` asserts that every comment, docstring and documentation file follows
`docs/WRITING_STANDARD.md`.

`tests/test_output_paths.py` asserts that stage 1 writes to the directory that stage 1b and
stage 3 read. `tests/test_padding.py` asserts that a padded clip keeps the sample rate,
channel count and sample format of its source.

---

## 8. Known issues

### 8.1 bacpipe does not detect CUDA

`bacpipe` reads its device from `settings.yaml` inside the installed package. It does not
detect CUDA. The shipped value is `cpu`.

Caution: If you do not change this value, the torch models run on CPU on a GPU node. The run
is then much slower and gives no error.

`run_all.sh` sets this value automatically and keeps a backup of the original file.

### 8.2 Two models fail on the shortest clips

BirdNET raises `index 1 is out of bounds for axis 0 with size 1` on the shortest clips. It
skips the clip and continues, so the model writes no embedding for it and the log gives no
total. WavLMForXVector fails the same way, because its convolution and TDNN stack needs a
minimum number of frames.

`src/00a_pad_audio.py` adds trailing silence to any clip below `padding.min_seconds` in
`config.yaml` and copies the rest unchanged, into `audio_padded/`. Stage 1 and stage 2 both
read that copy, so every representation sees the same audio. The rule is applied to both
species, because padding one and not the other would put a preprocessing difference inside
the species comparison.

Report `min_seconds` in the methods. It is a preprocessing choice, not a property of the
recordings.

### 8.3 bacpipe does not fetch the BirdNET or the BEATs checkpoint

bacpipe downloads the weights of most of its models on first use. BirdNET and BEATs are the
exceptions. Without the checkpoint the model writes zero embeddings, and the log looks
normal.

`src/01a_fetch_checkpoints.py` downloads it into `bacpipe/model_checkpoints`, which is where
bacpipe looks. `run_all.sh` runs it before stage 1 and stops the run if the download fails.
The dataset repository and the file patterns are in the `checkpoints` block of `config.yaml`.

### 8.4 bacpipe prints harmless errors

After `bacpipe` writes the embeddings, it runs a dashboard step that this pipeline does not
use. That step prints many tracebacks. The messages below are harmless.

- `annotations.csv` not found
- `dim_reduced_embeddings`
- `list.remove(x)`
- `Length mismatch between time_bins`
- `Input image size 128*300`

Do not use the absence of tracebacks to confirm success. Run `01b_verify_embeddings.py`
instead. That script counts the `.npy` files for each model.

### 8.5 avesecho_passt fails on CUDA

`avesecho_passt` moves the model to the device but not the input audio. On CUDA, every file
fails and the model writes zero embeddings. `01_extract_embeddings.py` pins this model to CPU
through the `CPU_ONLY_MODELS` list.

### 8.6 Do not upgrade JAX

Warning: Do not run `pip install -U jax[cuda12]`. The `-U` flag upgrades numpy to version 2,
which breaks the pinned torch and bacpipe environment.

To repair a broken environment, delete it and rebuild it from `requirements.txt`. Then run
`python src/08_freeze_environment.py verify`.

### 8.7 Stage 4 writes only at the end of a subset

`04_metric_learning.py` writes its results when a subset finishes. If the process stops in the
middle of a subset, the work for that subset is lost.

Run stage 4 inside `tmux` or `screen`, or submit it as a batch job. Stage 3 is different. It
appends its results for each model and skips completed work when you rerun it.
