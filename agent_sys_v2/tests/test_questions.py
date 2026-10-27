"""questions.md read as a form: the table agent and user write in."""

from agent_sys_v2 import questions

TABLE = """# 本轮问题

| # | 问题 | 建议 | 你的意见 |
|---|---|---|---|
| 1 | 用哪种调度？ | `--schedule-policy spf`，看 `grep queue | wc -l` | |
| 2 | 要不要开 MTP？ | 先不开 | 开 |
"""


def test_an_unescaped_pipe_before_the_answer_stays_in_its_cell():
    """A command in the suggestion often has a |: the answer is still the last cell, not the command's tail, and
    answering it leaves the rest of the line as the agent wrote it."""
    rows = questions.parse(TABLE)["rows"]
    assert [(r["key"], r["answer"]) for r in rows] == [("1", ""), ("2", "开")]
    assert rows[0]["cells"][2] == "`--schedule-policy spf`，看 `grep queue \\| wc -l`"
    text, filled, total = questions.fill(TABLE, {"1": "用 spf", "2": "开 | 先试"})
    assert (filled, total) == (2, 2)
    assert text == TABLE.replace("wc -l` | |", "wc -l` | 用 spf |").replace(
        "| 开 |", "| 开 \\| 先试 |"
    )
    assert [r["answer"] for r in questions.parse(text)["rows"]] == ["用 spf", "开 | 先试"]


def test_a_row_without_its_answer_cell():
    head = "| # | 问题 | 你的意见 |\n|---|---|---|\n"
    text, _filled, _total = questions.fill(f"{head}| 1 | 要吗？\n", {"1": "要"})
    assert text == f"{head}| 1 | 要吗？ | 要 |\n"
