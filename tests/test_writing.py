#!/usr/bin/env python
"""
Tests for the writing standard.

PURPOSE
    Assert that every comment, docstring and documentation file follows
    docs/WRITING_STANDARD.md. A checklist is followed once. A test is followed
    on every commit.

USAGE
    pytest tests/

WHAT EACH TEST CHECKS
    test_no_estimated_number
        Rule 1. No hedge word in front of a number, and no RUNTIME heading.
    test_no_banned_term
        Rule 2. No term from section 4 of docs/GLOSSARY.md.
    test_no_em_dash, test_no_semicolon, test_no_contraction
        Rule 3.
    test_script_has_the_four_headings
        Rule 5. PURPOSE, USAGE, INPUT and OUTPUT, in that order.
    test_referenced_file_exists
        Rule 7. Every repository file named in prose is present on disk.

WHAT COUNTS AS PROSE
    Python    The docstrings and the comment lines. Not the code.
    Shell     The comment lines, and the text the script prints. A stage banner
              is read by the person who runs the pipeline, so it follows the
              same rules as a comment.
    YAML      The comment lines. Not the values.
    Markdown  Everything outside a fenced code block and outside backticks.

    The code itself is exempt, because a variable name is read by the
    interpreter and not by a person looking for an explanation.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PYTHON_FILES = sorted(ROOT.glob("src/*.py")) + sorted(ROOT.glob("tests/*.py"))
SHELL_FILES = sorted(ROOT.glob("*.sh"))
YAML_FILES = sorted(ROOT.glob("*.yaml"))
MARKDOWN_FILES = sorted(ROOT.glob("*.md")) + sorted(ROOT.glob("docs/*.md"))
ALL_FILES = PYTHON_FILES + SHELL_FILES + YAML_FILES + MARKDOWN_FILES

# Section 4 of docs/GLOSSARY.md. The replacement wording is in that file.
BANNED_TERMS = [
    "ablation",
    "overfit",
    "early stopping",
    "logit",
    "x-vector",
    "backbone",
    "downstream",
    "hyperparameter",
    "latent space",
    "zero-shot",
]

# Rule 1. A hedge word in front of a quantity means it was not measured. The
# quantity can be a digit or a number word ('about one day' is an estimate).
HEDGE_BEFORE_NUMBER = re.compile(
    r"\b(about|around|roughly|approximately|circa|nearly|some|maybe|perhaps)\s+"
    r"(?:[0-9]|one|two|three|four|five|six|seven|eight|nine|ten|half|dozen)\b",
    re.IGNORECASE,
)
TILDE_NUMBER = re.compile(r"~\s*[0-9]")

# The start of a here-document, and the word that ends it.
HEREDOC_START = re.compile(r"<<-?\s*'?\"?([A-Za-z_][A-Za-z0-9_]*)'?\"?\s*$")

# The shell commands that print a message to the person running the pipeline.
PRINTED_TEXT = re.compile(r'\b(?:say|begin_stage|end_stage|die|echo)\s+"([^"\n]*)"')

CONTRACTION = re.compile(r"\b\w+('t|'re|'ll|'ve|'m)\b|\b(it's|let's|that's|there's|here's)\b",
                         re.IGNORECASE)

# Rule 5.
REQUIRED_HEADINGS = ["PURPOSE", "USAGE", "INPUT", "OUTPUT"]

# Rule 7. A path in prose must exist, unless the pipeline generates it or the
# path holds a placeholder such as <species>.
GENERATED_PREFIXES = ("results/", "logs/", "bacpipe_results/", "mfcc_results/",
                      "supplementary/", "data/")
PATH_IN_PROSE = re.compile(r"\b((?:src|tests|docs|environment\.lock)/[\w./-]+|"
                           r"(?:README\.md|config\.yaml|run_all\.sh|environment\.yml))")


def strip_markdown_code(text):
    """Return the markdown text with code blocks and backtick spans removed."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", " ", text)
    return text


def prose_of(path):
    """Return the prose of one file, as a list of (line number, line) pairs.

    Code is excluded. See WHAT COUNTS AS PROSE in the module docstring.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    if path.suffix == ".md":
        kept = strip_markdown_code(text).split("\n")
        return [(i + 1, line) for i, line in enumerate(kept) if line.strip()]

    if path.suffix in (".sh", ".yaml"):
        out = []
        for i, line in enumerate(lines):
            code, _, comment = line.partition("#")
            if comment.strip():
                out.append((i + 1, comment))
            if path.suffix == ".sh":
                # The text the script prints. A banner that names a run time is
                # the same violation as a comment that names one. Only the
                # message commands are read, so a quoted command stays exempt.
                for quoted in PRINTED_TEXT.findall(code):
                    if quoted.strip() and not quoted.startswith("$"):
                        out.append((i + 1, quoted))
        out += heredoc_lines(lines)
        return out

    # Python. The docstrings plus the comment lines.
    out = []
    tree = ast.parse(text)
    docstring_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                continue
            first = node.body[0]
            for number in range(first.lineno, first.end_lineno + 1):
                docstring_lines.add(number)
    for i, line in enumerate(lines, start=1):
        if i in docstring_lines:
            out.append((i, line))
        elif "#" in line and not line.strip().startswith("#!"):
            comment = line.split("#", 1)[1]
            if comment.strip():
                out.append((i, comment))
    return [(i, line) for i, line in out if line.strip()]


def heredoc_lines(lines):
    """Return the (line number, line) pairs inside every here-document.

    A here-document holds text that the script prints, so it follows the same
    rules as a comment. The quoted-string reader above does not see it.
    """
    out, terminator = [], None
    for i, line in enumerate(lines, start=1):
        if terminator is None:
            match = HEREDOC_START.search(line)
            if match:
                terminator = match.group(1)
            continue
        if line.strip() == terminator:
            terminator = None
            continue
        if line.strip():
            out.append((i, line))
    return out


def report(path, hits):
    """Return one readable message for a list of (line number, line) hits."""
    name = path.relative_to(ROOT)
    return "\n".join(f"{name}:{number}: {line.strip()}" for number, line in hits)


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_estimated_number(path):
    """Rule 1. Every number is measured, from the data, or from a cited paper."""
    if path.name in ("WRITING_STANDARD.md", "test_writing.py"):
        pytest.skip("This file quotes the wrong form as the example.")
    hits = [(n, l) for n, l in prose_of(path)
            if HEDGE_BEFORE_NUMBER.search(l) or TILDE_NUMBER.search(l)]
    assert not hits, (
        "An unmeasured number. See rule 1 of docs/WRITING_STANDARD.md.\n" + report(path, hits)
    )


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_runtime_heading(path):
    """Rule 1. run_all.sh measures each stage, so no script states its own time."""
    hits = [(n, l) for n, l in prose_of(path) if l.strip() == "RUNTIME"]
    assert not hits, (
        "A RUNTIME heading. run_all.sh prints the measured duration instead.\n"
        + report(path, hits)
    )


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
@pytest.mark.parametrize("term", BANNED_TERMS)
def test_no_banned_term(path, term):
    """Rule 2. Section 4 of docs/GLOSSARY.md holds the wording to use instead."""
    if path.name in ("GLOSSARY.md", "WRITING_STANDARD.md", "test_writing.py"):
        pytest.skip("This file defines the banned list.")
    pattern = re.compile(r"\b" + re.escape(term), re.IGNORECASE)
    hits = [(n, l) for n, l in prose_of(path) if pattern.search(l)]
    assert not hits, (
        f"The term '{term}' is banned. See section 4 of docs/GLOSSARY.md.\n" + report(path, hits)
    )


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_dash(path):
    """Rule 3. Use parentheses for an aside, or write two sentences."""
    hits = [(n, l) for n, l in prose_of(path) if "—" in l or " -- " in l]
    assert not hits, "An em dash. See rule 3 of docs/WRITING_STANDARD.md.\n" + report(path, hits)


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_semicolon(path):
    """Rule 3. ASD-STE100 does not allow the semicolon. Write two sentences."""
    if path.name in ("WRITING_STANDARD.md", "test_writing.py"):
        pytest.skip("This file names the rule.")
    hits = [(n, l) for n, l in prose_of(path) if ";" in l]
    assert not hits, "A semicolon. See rule 3 of docs/WRITING_STANDARD.md.\n" + report(path, hits)


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_contraction(path):
    """Rule 3. Write 'do not', not the short form."""
    if path.name in ("WRITING_STANDARD.md", "test_writing.py"):
        pytest.skip("This file names the rule.")
    hits = [(n, l) for n, l in prose_of(path) if CONTRACTION.search(l)]
    assert not hits, "A contraction. See rule 3 of docs/WRITING_STANDARD.md.\n" + report(path, hits)


@pytest.mark.parametrize("path", sorted(ROOT.glob("src/*.py")),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_script_has_the_four_headings(path):
    """Rule 5. PURPOSE, USAGE, INPUT and OUTPUT, in that order."""
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=False)
    assert doc is not None, f"{path.name} has no module docstring."
    found = [line.strip() for line in doc.split("\n")
             if line.strip() in REQUIRED_HEADINGS]
    missing = [h for h in REQUIRED_HEADINGS if h not in found]
    assert not missing, (
        f"{path.name} is missing {missing}. See rule 5 of docs/WRITING_STANDARD.md."
    )
    assert found == REQUIRED_HEADINGS, (
        f"{path.name} has the four headings in the order {found}. "
        f"The required order is {REQUIRED_HEADINGS}."
    )


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_referenced_file_exists(path):
    """Rule 7. No dangling reference to a file that was never written."""
    missing = []
    for number, line in prose_of(path):
        for match in PATH_IN_PROSE.findall(line):
            # A path at the end of a sentence carries the full stop.
            match = match.rstrip(".")
            if "<" in match or match.startswith(GENERATED_PREFIXES):
                continue
            if not (ROOT / match).exists():
                missing.append((number, f"{match}  (in: {line.strip()})"))
    assert not missing, (
        "A reference to a file that does not exist. See rule 7 of "
        "docs/WRITING_STANDARD.md.\n" + report(path, missing)
    )
