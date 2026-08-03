# Writing standard

This standard applies to every comment, every docstring, every log message and every
documentation file in this repository. It is not a style preference. It exists so that the
analysis can be read, checked and defended by a reader who is a bird behaviour specialist
and not a deep-learning specialist.

The standard follows ASD-STE100 Simplified Technical English, with the additions below.

This is guidance for anyone who edits this repository. Read it before you write a comment, a
docstring or a documentation file. It is not enforced by a test, because a test cannot judge
whether a sentence is clear, and a repository that passes a style test is not the same thing
as a repository a reader can follow.

---

## Rule 1. State facts. Never estimate.

Do not write a number unless it was measured, read from the data, or taken from a cited
paper.

This applies to run times, file sizes, memory, speed and counts.

An earlier version of this repository carried an invented run time in every script. One
script was annotated "about 5 minutes". That script takes 20 seconds. `run_all.sh` now
measures each stage and prints the duration, so the figure comes from the run itself.

If you do not have a measurement, do not write a number. Write what the reader must do to
get one.

Wrong, in a script docstring:

```
RUNTIME
    About 4 to 8 hours for each species on a GPU.
```

Right, in `run_all.sh`, where the figure comes from the clock:

```
STAGE 1 of 5   Embedding extraction   [GPU, resumable]
         STAGE 1 of 5   Embedding extraction   [GPU, resumable] took 05:12:44
```

Do not write a hedge word in front of a number (`about 5`, `roughly 200`, `~3`), and do not
write a `RUNTIME` heading in a docstring.

## Rule 2. Use the correct term. Define it once.

Every technical term lives in `docs/GLOSSARY.md` with one meaning. The code and the
manuscript use the same word for the same thing.

Prefer the plain word to the jargon. Section 4 of the glossary lists the terms to avoid and
the wording to use instead.

A term is not banned because it is technical. It is banned because a plain word carries the
same meaning with less cost to the reader. `EER`, `GroupKFold` and `MFCC` are correct
technical terms and stay.

## Rule 3. No em dashes. No semicolons. No contractions.

- An em dash is not used in this project. Use parentheses for an aside, or write two
  sentences.
- ASD-STE100 does not allow the semicolon. Write two sentences.
- Write "do not", not "don't". Write "it is", not "it's".

## Rule 4. Condition before action.

Write the condition first, then the action. The reader then knows whether the sentence
applies before reading the instruction.

Wrong: "Delete the cache if the build fails."

Right: "If the build fails, delete the cache."

## Rule 5. Every heading a script needs.

The module docstring of every script in `src/` starts with these four headings, in this
order:

| Heading | Content |
|---|---|
| `PURPOSE` | What the script produces, and why that output is needed. |
| `USAGE` | The exact command. Every argument that the pipeline uses. |
| `INPUT` | Every file the script reads. |
| `OUTPUT` | Every file the script writes. |

Any further heading is free text in capitals, for example `CAUTION`, `RESUMING` or
`THE DESIGN`. `RUNTIME` is not allowed. See rule 1.

## Rule 6. Explain the choice, not the code.

A comment that repeats the code is noise. Record why the choice was made.

Wrong:

```python
# Set the seed to 42.
seed = 42
```

Right:

```python
# One seed for the whole pipeline, so a rerun reproduces the published numbers.
seed = 42
```

When a method was tested and rejected, record the number that rejected it. A reader who
does not know that a method was already tried will ask for it in review.

## Rule 7. No placeholders. No dangling references.

The repository holds the final code set only.

- Do not commit a script that reads a file which does not exist.
- Do not reference a document that has not been written.
- If a value is not yet known, leave it out. Add it when you have it.

## Rule 8. Write for the reviewer as well as the user.

Where a result depends on an assumption, state the assumption next to the result, not in a
separate document. `03_classify.py` records the number of multi-bird recordings in every
result row for this reason. The assumption then travels with the number.
