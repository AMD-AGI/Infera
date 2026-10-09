"""Criterion 11 — a validator inside the producing task's zone is rejected.

Two properties, and the second is the trap.

**Resolution is mandatory, not prudent.** Spec §9.1 *sanctions* cross-package
symlinks, so the dangerous case is a link in a neutral package pointing **into**
the producer's zone: lexically innocent, and it executes the producer's bytes.

**The fail-closed direction is inverted** relative to `handoff` and `env_mgr`.
There *contained* means allow, so unresolvable means deny. Here *contained* means
**reject**, so unresolvable must be treated as **inside** — and importing
`handoff.check_contained` and negating at the call site would negate the
fail-closed behaviour too, accepting a dangling validator symlink.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.validator.conftest import validator_record
from validator.protocols import SeparationViolation
from validator.separation import check_separation, producer_zones, reaches
from validator.spec import admit


@pytest.fixture
def layout(tmp_path: Path) -> Path:
    """A producer zone, a neutral shared package, and somewhere to hang links."""
    for rel in ("pkg_producer/checks", "pkg_shared/checks", "pkg_neutral"):
        (tmp_path / rel).mkdir(parents=True)
    (tmp_path / "pkg_producer/checks/readme.md").write_text("# the producer's own")
    (tmp_path / "pkg_shared/checks/readme.md").write_text("# neutral")
    return tmp_path


def spec(readme: str, **kw):
    return admit(validator_record("shape", readme=readme, entry=None, **kw), origin="o")


def producer(*zones: Path) -> dict:
    return {"name": "collect_trace", "permissions": [str(z) for z in zones]}


def test_logic_inside_producer_permissions_rejected(layout: Path) -> None:
    """Criterion 11. The candidate writes the exam, the answer key, and grades it."""
    with pytest.raises(SeparationViolation) as exc:
        check_separation(
            spec("pkg_producer/checks/readme.md"),
            producer(layout / "pkg_producer"),
            package_root=layout,
        )
    assert str(layout / "pkg_producer/checks/readme.md") in str(exc.value)


def test_shared_package_symlink_admitted(layout: Path) -> None:
    """The benign case, and it is the one spec §9.1 sanctions: a relative symlink
    crossing packages is how two packages share a handoff kind."""
    link = layout / "pkg_neutral/readme.md"
    link.symlink_to(layout / "pkg_shared/checks/readme.md")
    check_separation(
        spec("pkg_neutral/readme.md"), producer(layout / "pkg_producer"), package_root=layout
    )


def test_symlink_into_zone_rejected(layout: Path) -> None:
    """The dangerous inverse: a link in a *neutral* package pointing into the
    producer's zone. Lexically innocent, and it executes the producer's bytes.
    Measured over five checks × six layouts, only `realpath` plus a trailing
    separator got all six right."""
    link = layout / "pkg_neutral/readme.md"
    link.symlink_to(layout / "pkg_producer/checks/readme.md")
    with pytest.raises(SeparationViolation):
        check_separation(
            spec("pkg_neutral/readme.md"), producer(layout / "pkg_producer"), package_root=layout
        )


def test_unresolvable_logic_path_is_rejected(layout: Path) -> None:
    """**The inverted fail-closed.** A dangling validator symlink must be
    *rejected*, where the same idea in `handoff` denies. Two uses of one
    primitive, two failure directions, both loud."""
    link = layout / "pkg_neutral/readme.md"
    link.symlink_to(layout / "pkg_shared/checks/gone.md")
    assert reaches(layout / "pkg_producer", link) is True
    with pytest.raises(SeparationViolation):
        check_separation(
            spec("pkg_neutral/readme.md"), producer(layout / "pkg_producer"), package_root=layout
        )


def test_prefix_sibling_is_not_contained(layout: Path) -> None:
    """§9.3. `zone` versus `zone-EVIL`, which every lexical check fails. The
    trailing separator is the single difference between the check that got all six
    measured layouts right and the one that got five."""
    evil = layout / "pkg_producer-EVIL"
    evil.mkdir()
    (evil / "readme.md").write_text("# a different package entirely")
    assert reaches(layout / "pkg_producer", evil / "readme.md") is False
    check_separation(
        spec("pkg_producer-EVIL/readme.md"), producer(layout / "pkg_producer"), package_root=layout
    )


def test_the_zone_itself_counts_as_reached(layout: Path) -> None:
    assert reaches(layout / "pkg_producer", layout / "pkg_producer") is True


def test_resolution_may_only_move_a_verdict_toward_accept(layout: Path) -> None:
    """Go's `internal` check resolves symlinks **only to widen** access — a link
    can never turn an allowed import into a denied one. Ours is a *rejection*, so
    the corresponding asymmetry inverts: a link out of the zone is admitted, a
    link into it is not. A deliberate inversion of Go's risk posture."""
    out = layout / "pkg_producer/checks/points_away.md"
    out.symlink_to(layout / "pkg_shared/checks/readme.md")
    # Lexically inside the producer's package; resolved, it is not.
    assert reaches(layout / "pkg_producer", out) is False


def test_the_check_reads_no_field_asserting_independence(layout: Path) -> None:
    """ "Structural" means the layout, not anyone's assertion — which is what makes
    it unfoolable by an author who is simply wrong about their own package."""
    from validator.spec import ValidatorSpec

    assert not {f for f in ValidatorSpec.model_fields if "independ" in f or "isolat" in f}


def test_no_declared_permissions_is_not_a_rejection(layout: Path) -> None:
    """A task that declares no zone reaches nothing, so there is nothing to
    compare. Silence is not a violation."""
    check_separation(spec("pkg_shared/checks/readme.md"), {"name": "t"}, package_root=layout)


def test_permission_shapes_read(layout: Path) -> None:
    """`task_graph` rev. 12's `Permissions` / `Grant` pair is not shipped, and the
    spec key is a *document* either way. Two written shapes are accepted; anything
    that is not a path is ignored rather than guessed at."""
    assert producer_zones({"permissions": ["/a", "/b"]}) == (Path("/a"), Path("/b"))
    assert producer_zones({"permissions": [{"path": "/a", "access": "write"}]}) == (Path("/a"),)
    assert producer_zones({"permissions": {"grants": ["/a"]}}) == (Path("/a"),)
    assert producer_zones({"permissions": [{"no_path": 1}]}) == ()
    assert producer_zones({}) == ()

    # And the shipped object, since a caller may have one to hand. Read through
    # attributes, never imported: `validator` does not own `Permissions`.
    from task_graph.permissions import Grant, Permissions

    assert producer_zones({"permissions": Permissions(grants=(Grant(path="/a"),))}) == (Path("/a"),)
    assert producer_zones({"permissions": Permissions()}) == ()


def test_a_composite_is_not_separation_checked(layout: Path) -> None:
    """A composite's implementation is its members, and each is checked itself."""
    composite = admit(
        validator_record(
            "pair", members=("a",), reduce="all", entry=None, readme="pkg_producer/checks/readme.md"
        ),
        origin="o",
    )
    check_separation(composite, producer(layout / "pkg_producer"), package_root=layout)
