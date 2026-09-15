"""Per-run prompt and tool settings, without modifying the user's CLI files."""

import json
import os
from pathlib import Path
import tempfile

from ...models import AgentContext
from ...prompts import provider_instructions


class GeminiWorkspace:
    def __init__(self, context: AgentContext) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="robot-gemini-")
        self.path = Path(self._directory.name)
        try:
            self.settings = self.path / "settings.json"
            self.system_prompt = self.path / "system.md"
            self.policy = self.path / "deny-tools.toml"
            self.settings.write_text(json.dumps({
                "tools": {"core": [], "discoveryCommand": "", "callCommand": ""},
                "hooksConfig": {"enabled": False},
                "skills": {"enabled": False},
                "admin": {
                    "mcp": {"enabled": False}, "extensions": {"enabled": False},
                    "skills": {"enabled": False},
                },
                "mcp": {"allowed": []},
                "context": {"fileName": ["__robot_no_context__"]},
                "general": {"enableAutoUpdate": False, "enableAutoUpdateNotification": False},
            }), encoding="utf-8")
            self.system_prompt.write_text(
                provider_instructions(context, "Gemini CLI")
                + "\n\nReturn exactly one JSON object matching this schema:\n"
                + json.dumps(context.output_schema, ensure_ascii=False), encoding="utf-8",
            )
            self.policy.write_text(
                '[[rule]]\ntoolName = "*"\ndecision = "deny"\npriority = 999\n',
                encoding="utf-8",
            )
        except BaseException:
            self.close()
            raise

    def environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update({
            "GEMINI_SYSTEM_MD": str(self.system_prompt),
            "GEMINI_CLI_SYSTEM_SETTINGS_PATH": str(self.settings),
            "GEMINI_CLI_SYSTEM_DEFAULTS_PATH": str(self.settings),
            "GEMINI_WRITE_SYSTEM_MD": "false",
        })
        return environment

    def close(self) -> None:
        self._directory.cleanup()
