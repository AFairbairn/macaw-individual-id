# Glossary

Each term below has one meaning in this project. The code, the documentation and the
manuscript use the same word for the same thing. If you need a new term, add it here first.

The rules that govern how the terms are used are in `docs/WRITING_STANDARD.md`.

---

## 1. The data

| Term | Meaning |
|---|---|
| **call** | One vocalisation. The unit of analysis. |
| **clip** | One audio file. A clip holds exactly one call from one bird. |
| **single call** | A clip that holds one call. This is the call set of the main analysis, for both species. |
| **bout** | A run of several calls from one bird, held in one clip. *Ara ambiguus* has bouts. *Ara glaucogularis* has none. Bouts are scored only in the supplement. |
| **recording** | One continuous source recording. Calls are cut from a recording. A recording can hold calls from more than one bird. |
| **recording_id** | The identifier of a recording. This is the unit that the train and test split groups on. |
| **encounter** | The set of calls from one bird within one recording. |
| **subset** | A group of birds that is scored together. `aa` has `all`. `ag` has `all` and `lab`. |
| **`aa`** | *Ara ambiguus*, the great green macaw. 8 birds, one call type. |
| **`ag`** | *Ara glaucogularis*, the blue-throated macaw. 16 birds, 8 call types. |
| **`all`** | The subset that holds every bird of a species. For `ag`, some of those birds live in separate rooms, so the room can stand in for the identity. The `ag` `all` result is an optimistic upper bound. |
| **`lab`** | The `ag` subset that holds the 12 birds which share one room. The room cannot stand in for the identity. This is the honest `ag` result. |
| **master table** | `data/<species>/metadata/<species>_master.csv`. The single source of truth for the bird identity and the recording of every clip. Every stage reads it. |

## 2. The models

| Term | Meaning |
|---|---|
| **representation** | One way of turning a clip into a vector. The panel holds 18: 15 pre-trained models and 3 MFCC variants. |
| **pre-trained model** | A model that was trained on other audio and is used here without further training. |
| **frozen** | The weights of the model are not changed. Only the small classifier or head on top is fitted. |
| **embedding** | The output vector of a model for one clip. |
| **family** | The reporting group of a model. Six families: bird-supervised, animal self-supervised, audio self-supervised, audio-text, speech speaker verification, and handcrafted. |
| **MFCC** | Mel-frequency cepstral coefficients. A handcrafted description of the shape of the spectrum. The classical baseline of this study. |
| **CMVN** | Cepstral mean and variance normalisation. Each MFCC coefficient is centred and scaled within one recording. |
| **probe** | A logistic regression classifier that is fitted to frozen embeddings. It measures whether the identity is present in the embedding. |
| **head** | The small metric-learning network that is trained on top of frozen embeddings. It changes the geometry of the space. |
| **prototype** | The mean embedding of the calls of one bird. Also called a centroid or a template. |

## 3. The evaluation

| Term | Meaning |
|---|---|
| **split by recording** | The train and test split that uses `GroupKFold` with `recording_id` as the group. No recording appears on both sides. This is the primary result. |
| **random split** | A stratified split that ignores the recording. It puts calls from one recording on both sides. This result is too high. It is reported only for the comparison. |
| **leakage** | Information that reaches the test set but that would not be available in real use. Here, the shared acoustic signature of a recording. |
| **leakage delta** | The random split accuracy minus the by-recording accuracy. The size of the overstatement. |
| **calls per recording** | The number of calls cut from one recording. This is the variable that sets the leakage delta. |
| **domain shift** | The acoustic difference between recordings, from the microphone, the position of the bird, the state of the room, or the time of day. Measured by predicting the recording within one bird. |
| **chance level** | The accuracy of a model that has no information. Report the majority-class baseline next to `1 / n_birds`, because `ag` is not balanced. |
| **verification** | The task of deciding whether two calls come from the same bird. Scored with EER and AUROC. |
| **EER** | Equal error rate. The verification metric. Lower is better. Chance is 0.5. |
| **AUROC** | The area under the receiver operating characteristic curve. Higher is better. Chance is 0.5. |
| **encounter accuracy** | The accuracy when the calls of one bird within one recording are pooled into one query. It assumes the calls are already grouped by bird. |
| **open set** | The case in which a test call can come from a bird that was never enrolled. |
| **enrolment** | The step that stores one prototype for each known bird, before any test call arrives. |
| **NMI, AMI, ARI** | Three scores for a clustering against a known grouping. Never report NMI alone, because NMI increases with the number of clusters. |

## 4. Machine-learning terms in plain words

Use the plain wording in the right column. The jargon in the left column is listed so that a
reader who meets it elsewhere can map it onto this project.

| Do not write | Write instead |
|---|---|
| ablation | changing one thing while holding the rest the same |
| overfits | learns the training data instead of the general pattern |
| early stopping | stops when the score stops improving |
| logit | the raw score |
| x-vector | the fixed-length output |
| backbone | the pre-trained model |
| downstream | later |
| hyperparameter | setting |
| latent space | the space the model maps into |
| fine-tune | continue training the model itself, not only the head |
| zero-shot | used with no training on our data |

`tests/test_writing.py` fails if a term in the left column appears in any comment,
docstring, or documentation file.
