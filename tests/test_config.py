#!/usr/bin/env python
"""
Tests for config.yaml.

PURPOSE
    Assert that config.yaml holds every key the code reads. An edit to this
    file can remove a block that a script depends on, and a script that reads a
    missing block with .get() carries on and does nothing.

    That happened. An edit to the checkpoints block removed the padding block
    beside it. 00_prepare_data.py printed "config.yaml has no padding block.
    Nothing to do", returned 0, and the run continued with no padded audio.

USAGE
    pytest tests/

WHAT EACH TEST CHECKS
    test_every_key_the_code_reads_exists
        Every CONFIG["key"] and CONFIG.get("key") in src/ resolves.
    test_the_required_blocks_are_present
        The blocks a run cannot proceed without are named here as well, so a
        rename in the code cannot quietly take one with it.
    test_every_species_has_subsets
        Each species has at least one subset, and each subset rule is either
        empty or one column and one value.
"""
import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

# A run cannot proceed without these. They are listed here as well as being
# found in the code, so that a key renamed in one place fails here.
REQUIRED_BLOCKS = [
    "seed",
    "species",
    "subsets",
    "call_set",
    "split",
    "padding",
    "mfcc",
    "metric_learning",
    "cpu_only_models",
]

SOURCES = sorted(ROOT.glob("src/*.py"))


def keys_read_by(path):
    """Return every top-level config key that one script reads.

    The function reads the syntax tree, so a key inside a comment or a string
    is not counted.
    """
    found = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        # CONFIG["key"]
        if isinstance(node, ast.Subscript):
            value, index = node.value, node.slice
            if (isinstance(value, ast.Name) and value.id == "CONFIG"
                    and isinstance(index, ast.Constant)
                    and isinstance(index.value, str)):
                found.add(index.value)

        # CONFIG.get("key")
        if isinstance(node, ast.Call):
            function = node.func
            if (isinstance(function, ast.Attribute) and function.attr == "get"
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "CONFIG"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found.add(node.args[0].value)

    return found


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_key_the_code_reads_exists(path):
    """Every top-level config key a script reads is present in config.yaml."""
    missing = sorted(key for key in keys_read_by(path) if key not in CONFIG)
    assert not missing, (
        f"{path.name} reads {missing} from config.yaml, and config.yaml does "
        f"not hold {'it' if len(missing) == 1 else 'them'}. "
        "A script that reads a missing block with .get() does nothing and "
        "reports success."
    )


@pytest.mark.parametrize("block", REQUIRED_BLOCKS)
def test_the_required_blocks_are_present(block):
    """A run cannot proceed without these blocks."""
    assert block in CONFIG, f"config.yaml has no {block} block."


def test_padding_holds_a_positive_minimum():
    """The padding minimum is a number above zero."""
    minimum = CONFIG["padding"]["min_seconds"]
    assert isinstance(minimum, (int, float))
    assert minimum > 0


@pytest.mark.parametrize("species", CONFIG["species"])
def test_every_species_has_subsets(species):
    """Each species has at least one subset, and each rule is well formed."""
    subsets = CONFIG["subsets"][species]
    assert subsets, f"{species} has no subset."

    for name, rule in subsets.items():
        if rule is None:
            continue
        assert len(rule) == 1, (
            f"The {species} subset {name} names {len(rule)} columns. "
            "A subset rule is one column and one value."
        )


def test_no_key_is_defined_twice():
    """A duplicated key silently overwrites the one above it."""
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    top_level = re.findall(r"(?m)^([a-z_]+):", text)
    duplicated = sorted({k for k in top_level if top_level.count(k) > 1})
    assert not duplicated, f"Defined more than once: {duplicated}"
