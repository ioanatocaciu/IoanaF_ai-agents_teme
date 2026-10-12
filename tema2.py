import os
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
EXCLUDED_DIRECTORIES = {".git", ".idea", ".pytest_cache", ".venv", "__pycache__"}
KEYWORD = "pytest"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_repository_files",
            "description": "List all project files, excluding generated and Git directories.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_keyword_in_files",
            "description": "Search the listed project files for the keyword pytest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "enum": [KEYWORD]},
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    },
]


def list_repository_files(root: Path = PROJECT_ROOT) -> list[str]:
    """List project files, excluding Git metadata and generated environments."""
    files: list[str] = []
    for directory, child_directories, file_names in os.walk(root):
        child_directories[:] = sorted(
            name for name in child_directories if name not in EXCLUDED_DIRECTORIES
        )
        files.extend(
            (Path(directory) / file_name).relative_to(root).as_posix()
            for file_name in sorted(file_names)
        )
    return sorted(files)


def search_keyword_in_files(
    files: list[str], keyword: str = KEYWORD, root: Path = PROJECT_ROOT
) -> list[str]:
    """Return listed text files whose contents contain the keyword, case-insensitively."""
    needle = keyword.casefold()
    matches = []
    for relative_path in files:
        content = (root / relative_path).read_text(encoding="utf-8", errors="replace")
        if "\0" not in content and needle in content.casefold():
            matches.append(relative_path)
    return matches


class RepositorySearchAgent:
    """Use OpenRouter tool calling to list files and search for pytest."""

    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = root

    def _call_model(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        model = os.getenv("OPEN_ROUTER_MODEL_NAME")
        if not api_key or not model:
            raise RuntimeError(
                "Set OPEN_ROUTER_API_KEY and OPEN_ROUTER_MODEL_NAME in .env to run the LLM agent."
            )

        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        response = client.chat.completions.create(**request)
        return response.choices[0].message

    @staticmethod
    def _required_tool_call(message: Any, expected_name: str) -> Any:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls or tool_calls[0].function.name != expected_name:
            raise RuntimeError(f"The LLM did not call the required {expected_name} tool.")
        return tool_calls[0]

    def run_local(self) -> str:
        """Run both tools directly without making an LLM API request."""
        files = list_repository_files(self.root)
        matches = search_keyword_in_files(files, KEYWORD, self.root)
        return self._format_results(matches)

    @staticmethod
    def _format_results(matches: list[str]) -> str:
        if not matches:
            return f"No files contained '{KEYWORD}'."
        return f"Files containing '{KEYWORD}':\n" + "\n".join(
            f"- {file}" for file in matches
        )

    def run(self) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a repository search agent. Call the list_repository_files "
                    "tool first, then call search_keyword_in_files with keyword 'pytest'. "
                    "After both tools, report the matching file names."
                ),
            },
            {
                "role": "user",
                "content": "Find every project file containing the keyword pytest.",
            },
        ]
        list_tool = TOOLS[0]
        list_call = self._required_tool_call(
            self._call_model(
                messages,
                [list_tool],
                {"type": "function", "function": {"name": "list_repository_files"}},
            ),
            "list_repository_files",
        )
        files = list_repository_files(self.root)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": list_call.id,
                            "type": "function",
                            "function": {
                                "name": list_call.function.name,
                                "arguments": list_call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": list_call.id,
                    "content": json.dumps(files),
                },
            ]
        )

        search_tool = TOOLS[1]
        search_call = self._required_tool_call(
            self._call_model(
                messages,
                [search_tool],
                {"type": "function", "function": {"name": "search_keyword_in_files"}},
            ),
            "search_keyword_in_files",
        )
        arguments = json.loads(search_call.function.arguments)
        if arguments.get("keyword") != KEYWORD:
            raise RuntimeError(f"The search tool must search for '{KEYWORD}'.")
        matches = search_keyword_in_files(files, KEYWORD, self.root)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": search_call.id,
                            "type": "function",
                            "function": {
                                "name": search_call.function.name,
                                "arguments": search_call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": search_call.id,
                    "content": json.dumps(matches),
                },
            ]
        )
        final_message = self._call_model(messages)
        summary = (getattr(final_message, "content", None) or "").strip()
        results = self._format_results(matches)
        return f"{summary}\n\n{results}" if summary else results


if __name__ == "__main__":
    print(RepositorySearchAgent().run())
