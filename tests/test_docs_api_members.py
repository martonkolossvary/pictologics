"""Tests validating mkdocstrings ``members:`` entries against the real API surface.

``mkdocs build --strict`` does not catch a typo'd member name in a ``members:`` list
under a ``::: dotted.path`` directive — mkdocstrings silently renders an empty section
instead of failing the build. This module parses ``docs/api/*.md`` the same way
mkdocstrings reads it and asserts every listed member actually resolves on its target
(method/attribute, dataclass field, or class annotation).

Formerly a gitignored dev script (``dev/check_api_members.py``); moved here so the
check runs in CI as part of the normal test suite.
"""

from __future__ import annotations

import dataclasses
import importlib
import pathlib
import re
from typing import Any

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
API_DIR = REPO / "docs" / "api"


# =============================================================================
# Helpers (mirror the logic of the former dev/check_api_members.py)
# =============================================================================


def extract_members(text: str) -> set[tuple[str, str]]:
    """Parse ``:::``/``members:`` directives in *text*.

    Returns the set of (target, member) pairs declared, following the same grammar
    mkdocstrings uses when rendering docs/api/*.md: a ``::: dotted.path`` line sets
    the current target, a ``#`` heading or ``---`` divider resets it, a ``members:``
    line opens a member list, and each ``- name`` line under it pairs that name with
    the current target.
    """
    pairs: set[tuple[str, str]] = set()
    target: str | None = None
    in_members = False
    for raw in text.splitlines():
        directive = re.match(r"\s*:::\s+(\S+)", raw)
        if directive:
            target = directive.group(1)
            in_members = False
            continue
        if re.match(r"^#", raw) or raw.strip() == "---":
            target, in_members = None, False
            continue
        if re.match(r"\s*members:\s*$", raw):
            in_members = True
            continue
        if in_members:
            item = re.match(r"\s+-\s+(\S+)", raw)
            if item and target is not None:
                pairs.add((target, item.group(1)))
            elif raw.strip():
                in_members = False
    return pairs


def has_member(obj: Any, member: str) -> bool:
    """True if ``member`` is a method/attr, a dataclass field, or a class annotation."""
    return (
        hasattr(obj, member)
        or member in getattr(obj, "__dataclass_fields__", {})
        or member in getattr(obj, "__annotations__", {})
    )


def resolve(dotted: str) -> Any:
    """Import the longest importable module prefix of *dotted*, then getattr the rest."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"cannot import any module prefix of {dotted!r}")


# =============================================================================
# Unit tests for extract_members (every parser branch)
# =============================================================================


class TestExtractMembers:
    """Tests for the ``:::``/``members:`` parser."""

    def test_parses_all_branches(self):
        text = (
            "::: pictologics.foo.Bar\n"
            "    options:\n"
            "      members:\n"
            "        - method_a\n"
            "    some trailing text\n"
            "# Heading resets target\n"
            "::: pictologics.baz.Qux\n"
            "    members:\n"
            "        - method_b\n"
            "---\n"
        )
        assert extract_members(text) == {
            ("pictologics.foo.Bar", "method_a"),
            ("pictologics.baz.Qux", "method_b"),
        }


# =============================================================================
# Unit tests for has_member (all three resolution strategies)
# =============================================================================


class TestHasMember:
    """Tests for has_member's hasattr / dataclass-field / annotation checks."""

    def test_hasattr_branch(self):
        class Foo:
            def bar(self) -> None: ...

        assert has_member(Foo, "bar") is True

    def test_dataclass_field_branch(self):
        @dataclasses.dataclass
        class Foo:
            field_no_default: int

        assert not hasattr(Foo, "field_no_default")
        assert has_member(Foo, "field_no_default") is True

    def test_annotation_only_branch(self):
        class Foo:
            plain_attr: int

        assert not hasattr(Foo, "plain_attr")
        assert has_member(Foo, "plain_attr") is True

    def test_missing_member(self):
        class Foo:
            pass

        assert has_member(Foo, "nonexistent") is False


# =============================================================================
# Unit tests for resolve (module-prefix search)
# =============================================================================


class TestResolve:
    """Tests for resolve's longest-importable-prefix search."""

    def test_resolves_class_via_module_prefix(self):
        from pictologics.deduplication import DeduplicationRules

        assert resolve("pictologics.deduplication.DeduplicationRules") is DeduplicationRules

    def test_bogus_target_raises_import_error(self):
        with pytest.raises(ImportError):
            resolve("definitely_not_a_real_top_level_package_xyz123.Nope")


# =============================================================================
# Integration test: every docs/api/*.md members entry must resolve
# =============================================================================


def test_docs_api_members_resolve():
    """Every ``members:`` entry under a ``:::`` directive in docs/api/*.md must resolve."""
    errors: list[str] = []
    for md in sorted(API_DIR.rglob("*.md")):
        for target, member in extract_members(md.read_text()):
            obj = resolve(target)
            if not has_member(obj, member):  # pragma: no cover  # docs are valid; see sanity check
                errors.append(f"{md.relative_to(REPO)}: `{target}` has no member `{member}`")
    assert not errors, "\n".join(errors)
