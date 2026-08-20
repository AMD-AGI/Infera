# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
import json
import textwrap

from env_mgr.cli import main


def _recipe(tmp_path, body):
    p = tmp_path / "r.yaml"
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_check_ok_exit_zero(tmp_path, capsys):
    r = _recipe(
        tmp_path,
        f"""
        version: 1
        target: {{kind: repo, name: x, path: {tmp_path}}}
        items:
          - installer: oneline
            importance: suggested
            layer: system
            check_cmd: "true"
            run: "true"
    """,
    )
    rc = main(["check", r])
    assert rc == 0


def test_check_required_missing_exit_two(tmp_path):
    r = _recipe(
        tmp_path,
        f"""
        version: 1
        target: {{kind: repo, name: x, path: {tmp_path}}}
        items:
          - installer: bin
            importance: required
            layer: system
            name: definitely-not-real-xyz
            check_cmd: "definitely-not-real-xyz --version"
    """,
    )
    rc = main(["check", r])
    assert rc == 2


def test_path_override_reaches_runner(tmp_path):
    # recipe target.path is a bogus dir; --path overrides it to a real one where
    # the check_cmd (test -f marker) succeeds, proving the override is applied.
    (tmp_path / "marker").write_text("x")
    r = _recipe(
        tmp_path,
        """
        version: 1
        target: {kind: repo, name: x, path: /nonexistent/bogus/dir}
        items:
          - installer: oneline
            importance: required
            layer: repo
            check_cmd: "test -f marker"
            run: "true"
    """,
    )
    # without override it would fail (bogus cwd); with --path it runs in tmp_path
    assert main(["check", r, "--path", str(tmp_path)]) == 0


def test_workspace_override_accepted(tmp_path):
    r = _recipe(
        tmp_path,
        f"""
        version: 1
        target:
          kind: repo
          name: x
          path: {tmp_path}
          parent: {{workspace: /placeholder}}
        items:
          - installer: oneline
            importance: suggested
            layer: system
            check_cmd: "true"
            run: "true"
    """,
    )
    assert main(["check", r, "--workspace", "/some/ws"]) == 0


def test_json_output_parses(tmp_path, capsys):
    r = _recipe(
        tmp_path,
        f"""
        version: 1
        target: {{kind: repo, name: x, path: {tmp_path}}}
        items:
          - installer: oneline
            importance: suggested
            layer: system
            check_cmd: "true"
            run: "true"
    """,
    )
    main(["check", r, "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["status"] in ("OK", "WARN", "FAIL")
    assert "outcomes" in data


def test_main_malformed_recipe_returns_two(tmp_path):
    # missing the required 'installer' field -> RecipeError, must not crash
    r = _recipe(
        tmp_path,
        f"""
        version: 1
        target: {{kind: repo, name: x, path: {tmp_path}}}
        items:
          - importance: suggested
            layer: system
            run: "true"
    """,
    )
    assert main(["check", r]) == 2


def test_main_unknown_installer_returns_two(tmp_path):
    # installer name is not in the registry -> KeyError, must not crash
    r = _recipe(
        tmp_path,
        f"""
        version: 1
        target: {{kind: repo, name: x, path: {tmp_path}}}
        items:
          - installer: sh
            importance: suggested
            layer: system
            run: "true"
    """,
    )
    assert main(["check", r]) == 2


def test_main_bad_importance_flag_returns_two(tmp_path):
    # --importance bogus reaches _importance_rank -> ValueError, must not crash
    r = _recipe(
        tmp_path,
        f"""
        version: 1
        target: {{kind: repo, name: x, path: {tmp_path}}}
        items:
          - installer: oneline
            importance: suggested
            layer: system
            check_cmd: "true"
            run: "true"
    """,
    )
    assert main(["check", r, "--importance", "bogus"]) == 2


def test_main_missing_recipe_file_returns_two(tmp_path):
    # nonexistent recipe path -> FileNotFoundError normalized to RecipeError -> exit 2
    missing = str(tmp_path / "does-not-exist.yaml")
    assert main(["check", missing]) == 2


def test_main_malformed_yaml_returns_two(tmp_path):
    # syntactically broken YAML -> yaml.YAMLError normalized to RecipeError -> exit 2
    p = tmp_path / "bad.yaml"
    p.write_text("items: [unclosed\n")
    assert main(["check", str(p)]) == 2
