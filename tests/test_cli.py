"""Tests for the interactive command-line front end."""

from functools import partial
from pathlib import Path

import pytest

from file_organizer import cli
from file_organizer.cli import ask_approval, main, run

from .conftest import ScriptedChatModel, ai, call

REQUEST = {
    "tool_call_id": "c1",
    "tool": "move_file",
    "args": {"source": "a.txt", "destination": "finance"},
    "description": "Move 'a.txt' -> 'finance'",
}


def scripted_input(*answers: str):
    queue = list(answers)
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    fake_input.prompts = prompts
    return fake_input


class TestAskApproval:
    @pytest.mark.parametrize("answer", ["y", "Y", "yes", " yes "])
    def test_yes_approves(self, answer):
        assert ask_approval(REQUEST, scripted_input(answer), lambda *_: None) == (
            {"approved": True, "feedback": None}, False,
        )

    @pytest.mark.parametrize("answer", ["n", "no", "N"])
    def test_no_rejects(self, answer):
        assert ask_approval(REQUEST, scripted_input(answer), lambda *_: None) == (
            {"approved": False, "feedback": None}, False,
        )

    def test_free_text_rejects_with_feedback(self):
        decision, _ = ask_approval(REQUEST, scripted_input("put it in taxes"), lambda *_: None)
        assert decision == {"approved": False, "feedback": "put it in taxes"}

    def test_empty_answer_asks_again(self):
        fake = scripted_input("", "y")
        assert ask_approval(REQUEST, fake, lambda *_: None).decision["approved"] is True
        assert len(fake.prompts) == 2

    def test_shows_the_proposed_change(self):
        shown: list[str] = []
        ask_approval(REQUEST, scripted_input("y"), shown.append)
        assert any("Move 'a.txt' -> 'finance'" in line for line in shown)


def test_run_drives_a_full_session_with_approvals(folder: Path):
    model = ScriptedChatModel(
        responses=[
            ai(call("list_files")),
            ai(call("read_file", path="invoice.txt")),
            ai(call("create_directory", path="finance")),
            ai(
                call("move_file", source="invoice.txt", destination="finance"),
                call("move_file", source="flight.txt", destination="finance"),
            ),
            ai(content="Moved the invoice to finance."),
        ]
    )
    output: list[str] = []
    code = run(folder, model, input_fn=scripted_input("y", "y", "no, that's travel"),
               output_fn=output.append)

    assert code == 0
    assert (folder / "finance" / "invoice.txt").exists()
    assert (folder / "flight.txt").exists()
    text = "\n".join(output)
    assert "read_file" in text  # progress of read-only actions is shown
    assert "Moved the invoice to finance." in text
    rejection = model.received[-1][-1]
    assert "no, that's travel" in rejection.content


def test_run_aborts_cleanly_on_ctrl_c(folder: Path):
    model = ScriptedChatModel(responses=[ai(call("create_directory", path="finance"))])

    def ctrl_c(prompt: str = "") -> str:
        raise KeyboardInterrupt

    output: list[str] = []
    assert run(folder, model, input_fn=ctrl_c, output_fn=output.append) == 130
    assert not (folder / "finance").exists()
    assert any("abort" in line.lower() for line in output)


def test_main_rejects_missing_folder(tmp_path: Path, capsys):
    def factory(name):  # pragma: no cover - must not be reached
        raise AssertionError("model should not be created")

    assert main([str(tmp_path / "missing")], model_factory=factory) == 2
    assert "does not exist" in capsys.readouterr().err


def test_main_uses_model_option(folder: Path, monkeypatch):
    seen = []

    def factory(name):
        seen.append(name)
        return ScriptedChatModel(responses=[ai(content="Nothing to do.")])

    assert main([str(folder), "--model", "openai:gpt-test"], model_factory=factory) == 0
    assert seen == ["openai:gpt-test", "openai:gpt-4.1-mini"]  # main model + cheap summarizer


def test_main_model_defaults_to_env_var(folder: Path, monkeypatch):
    monkeypatch.setenv("FILE_ORGANIZER_MODEL", "anthropic:from-env")
    seen = []

    def factory(name):
        seen.append(name)
        return ScriptedChatModel(responses=[ai(content="ok")])

    main([str(folder)], model_factory=factory)
    assert seen[0] == "anthropic:from-env"


def test_main_reports_model_init_errors(folder: Path, capsys):
    def factory(name):
        raise ValueError("Unable to infer model provider")

    assert main([str(folder), "--model", "bogus"], model_factory=factory) == 2
    err = capsys.readouterr().err
    assert "bogus" in err and "Unable to infer model provider" in err


def test_run_reports_model_failures_without_traceback(folder: Path):
    model = ScriptedChatModel(responses=[])  # raises on the first call

    output: list[str] = []
    assert run(folder, model, input_fn=scripted_input(), output_fn=output.append) == 1
    assert any(line.startswith("error:") and "ran out of responses" in line for line in output)


def _isolate_env(monkeypatch, *names):
    # setenv-then-delenv makes monkeypatch remove anything load_dotenv adds later
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_main_loads_model_and_api_key_from_dotenv(folder: Path, tmp_path_factory, monkeypatch):
    _isolate_env(monkeypatch, "FILE_ORGANIZER_MODEL", "ANTHROPIC_API_KEY")
    cwd = tmp_path_factory.mktemp("cwd")
    (cwd / ".env").write_text("FILE_ORGANIZER_MODEL=anthropic:from-dotenv\nANTHROPIC_API_KEY=sk-test\n")
    monkeypatch.chdir(cwd)
    seen = []

    def factory(name):
        import os
        seen.append((name, os.environ.get("ANTHROPIC_API_KEY")))
        return ScriptedChatModel(responses=[ai(content="ok")])

    main([str(folder)], model_factory=factory)
    assert seen[0] == ("anthropic:from-dotenv", "sk-test")


def test_real_environment_overrides_dotenv(folder: Path, tmp_path_factory, monkeypatch):
    monkeypatch.setenv("FILE_ORGANIZER_MODEL", "anthropic:from-shell")
    cwd = tmp_path_factory.mktemp("cwd")
    (cwd / ".env").write_text("FILE_ORGANIZER_MODEL=anthropic:from-dotenv\n")
    monkeypatch.chdir(cwd)
    seen = []

    def factory(name):
        seen.append(name)
        return ScriptedChatModel(responses=[ai(content="ok")])

    main([str(folder)], model_factory=factory)
    assert seen[0] == "anthropic:from-shell"


def _with_usage(message, input_tokens: int, output_tokens: int):
    message.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return message


def test_run_shows_token_counts_per_call_and_in_total(folder: Path):
    model = ScriptedChatModel(
        responses=[
            _with_usage(ai(call("list_files")), 1000, 20),
            _with_usage(ai(content="Nothing to do."), 1500, 30),
        ]
    )
    output: list[str] = []
    assert run(folder, model, input_fn=scripted_input(), output_fn=output.append) == 0

    assert "  [llm call #1: 1,000 in + 20 out tokens; session total 1,020]" in output
    assert "  [llm call #2: 1,500 in + 30 out tokens; session total 2,550]" in output
    assert "\nTokens used: 2,500 in + 50 out = 2,550 tokens in 2 LLM calls" in output


def test_run_reports_unknown_usage_when_the_model_sends_none(folder: Path):
    model = ScriptedChatModel(responses=[ai(content="Done.")])
    output: list[str] = []
    assert run(folder, model, input_fn=scripted_input(), output_fn=output.append) == 0
    assert "\nTokens used: token usage not reported in 1 LLM call" in output
    assert not any("[llm call" in line for line in output)


def test_ask_approval_offers_all_only_when_more_changes_follow():
    batch = [{"tool_call_id": f"c{i}", "description": f"change {i}"} for i in range(3)]
    fake = scripted_input("a")
    answer = ask_approval({**REQUEST, "batch": batch, "index": 1}, fake, lambda *_: None)
    assert answer == ({"approved": True, "feedback": None}, True)
    assert "[a]ll 2 remaining" in fake.prompts[0]

    fake = scripted_input("all")  # last change: "a"/"all" just approves it (not feedback)
    answer = ask_approval({**REQUEST, "batch": batch, "index": 2}, fake, lambda *_: None)
    assert answer == ({"approved": True, "feedback": None}, False)
    assert "[a]ll" not in fake.prompts[0]


def test_plan_is_shown_first_and_all_approves_the_rest_of_the_batch_only(folder: Path):
    model = ScriptedChatModel(
        responses=[
            ai(
                call("create_directory", path="finance"),
                call("move_file", source="invoice.txt", destination="finance"),
                call("move_file", source="flight.txt", destination="finance"),
            ),
            ai(call("create_directory", path="health")),  # a new batch: asked again
            ai(content="Done."),
        ]
    )
    output: list[str] = []
    fake = scripted_input("n", "all", "y")
    assert run(folder, model, input_fn=fake, output_fn=output.append) == 0

    assert len(fake.prompts) == 3  # 'n', then 'all' (covers flight.txt), then the next batch
    assert not (folder / "finance").exists()  # rejected
    assert (folder / "health").is_dir()
    text = "\n".join(output)
    plan = text.index("Proposed changes (3):")
    assert plan < text.index("? Create directory 'finance'")
    assert "    3. Move 'flight.txt' -> 'finance'" in text
    assert text.count("Proposed changes") == 1  # shown once per batch; single changes need no plan
    assert "✓ Move 'flight.txt' -> 'finance' (approved with 'all')" in text


def test_recursion_limit_option_and_env(folder: Path, monkeypatch, capsys):
    def endless(name):
        return ScriptedChatModel(responses=[ai(call("list_files")) for _ in range(50)])

    assert main([str(folder), "--recursion-limit", "6"], model_factory=endless) == 1
    assert "more than 6 steps" in capsys.readouterr().out

    monkeypatch.setenv("FILE_ORGANIZER_RECURSION_LIMIT", "10")
    main([str(folder), "--state-db", "other.sqlite"], model_factory=endless)  # no resume question
    assert "more than 10 steps" in capsys.readouterr().out


def test_recursion_limit_must_be_positive(folder: Path, capsys):
    with pytest.raises(SystemExit):
        main([str(folder), "--recursion-limit", "0"])
    assert "positive" in capsys.readouterr().err


@pytest.mark.parametrize("args,env,renamed", [([], None, False), (["--rename-on-clash"], None, True),
                                              ([], "true", True), (["--no-rename-on-clash"], "1", False)])
def test_rename_on_clash_option(folder: Path, monkeypatch, args, env, renamed):
    if env:
        monkeypatch.setenv("FILE_ORGANIZER_RENAME_ON_CLASH", env)
    (folder / "finance").mkdir()
    (folder / "finance" / "invoice.txt").write_text("older")
    model = ScriptedChatModel(responses=[ai(call("move_file", source="invoice.txt", destination="finance")),
                                         ai(content="ok")])
    monkeypatch.setattr(cli, "run", partial(cli.run, input_fn=scripted_input("y")))
    assert main([str(folder), *args], model_factory=lambda name: model) == 0
    assert (folder / "finance" / "invoice (2).txt").exists() is renamed
    assert (folder / "invoice.txt").exists() is not renamed
