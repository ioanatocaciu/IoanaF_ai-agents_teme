from pathlib import Path
from types import SimpleNamespace

import tema2
from tema2 import RepositorySearchAgent


def test_llm_agent_calls_tools_in_required_order(tmp_path, monkeypatch):
    calls = []

    def fake_list_files(root):
        calls.append("list")
        return ["pyproject.toml", "src/example.py"]

    def fake_search(files, keyword, root):
        calls.append("search")
        assert files == ["pyproject.toml", "src/example.py"]
        assert keyword == "pytest"
        return ["pyproject.toml"]

    monkeypatch.setattr(tema2, "list_repository_files", fake_list_files)
    monkeypatch.setattr(tema2, "search_keyword_in_files", fake_search)

    tool_calls = [
        SimpleNamespace(
            id="list-call",
            function=SimpleNamespace(
                name="list_repository_files", arguments="{}"
            ),
        ),
        SimpleNamespace(
            id="search-call",
            function=SimpleNamespace(
                name="search_keyword_in_files", arguments='{"keyword":"pytest"}'
            ),
        ),
    ]
    model_calls = []

    def fake_model(messages, tools=None, tool_choice=None):
        model_calls.append((tools, tool_choice))
        if len(model_calls) < 3:
            return SimpleNamespace(tool_calls=[tool_calls[len(model_calls) - 1]])
        return SimpleNamespace(content="Search complete.")

    monkeypatch.setattr(RepositorySearchAgent, "_call_model", lambda self, *args: fake_model(*args))

    result = RepositorySearchAgent(tmp_path).run()

    assert calls == ["list", "search"]
    assert "pyproject.toml" in result
    assert len(model_calls) == 3
    assert model_calls[0][0][0]["function"]["name"] == "list_repository_files"
    assert model_calls[1][0][0]["function"]["name"] == "search_keyword_in_files"


def test_tools_list_files_and_search_case_insensitively(tmp_path):
    (tmp_path / "pyproject.toml").write_text("pytest>=8\n", encoding="utf-8")
    source = tmp_path / "src"
    source.mkdir()
    (source / "example.py").write_text("# PyTest example\n", encoding="utf-8")
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "package.py").write_text("pytest", encoding="utf-8")

    files = tema2.list_repository_files(tmp_path)

    assert files == ["pyproject.toml", "src/example.py"]
    assert tema2.search_keyword_in_files(files, root=tmp_path) == files


def test_agent_reports_project_pytest_files():
    result = RepositorySearchAgent().run_local()

    assert "pyproject.toml" in result
    assert "tests/test_tema2.py" in result
