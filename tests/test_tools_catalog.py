from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gpt_policy.harness.protocol import instructions, output_schema, tool_schemas
from gpt_policy.tools import ToolExecutor, load_tool_catalog


class ToolCatalogTest(unittest.TestCase):
    def test_default_catalog_drives_single_and_bimanual_schemas(self) -> None:
        catalog = load_tool_catalog()

        single = tool_schemas(6, ("left",), catalog)
        bimanual = tool_schemas(6, ("left", "right"), catalog)
        self.assertEqual(
            [tool["function"]["name"] for tool in single],
            ["move_to", "move_eef_chunk", "set_gripper", "check_path", "locate_point", "done", "give_up"],
        )
        single_gripper = single[2]["function"]["parameters"]["properties"]
        bimanual_gripper = bimanual[2]["function"]["parameters"]["properties"]
        self.assertIn("gripper", single_gripper)
        self.assertIn("positions", bimanual_gripper)
        self.assertEqual(
            output_schema(6, ("left", "right"), catalog)["properties"]["name"]["enum"],
            ["move_to", "move_eef_chunk", "set_gripper", "check_path", "locate_point", "done", "give_up"],
        )
        self.assertIn("move_to", instructions("X5", "can1", 6, ("left",), {}, catalog))

    def test_disabled_tool_is_removed_from_all_generated_surfaces(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "tools.json"
            data = json.loads(Path("configs/tools.json").read_text(encoding="utf-8"))
            next(tool for tool in data["tools"] if tool["name"] == "locate_point")["enabled"] = False
            source.write_text(json.dumps(data), encoding="utf-8")
            catalog = load_tool_catalog({"tool_catalog": str(source)})

        names = [tool["function"]["name"] for tool in tool_schemas(6, ("left",), catalog)]
        self.assertNotIn("locate_point", names)
        self.assertNotIn("locate_point", output_schema(6, ("left",), catalog)["properties"]["name"]["enum"])
        self.assertNotIn("locate_point:", instructions("X5", "can1", 6, ("left",), {}, catalog))

    def test_runtime_handlers_are_an_explicit_allowlist(self) -> None:
        catalog = load_tool_catalog()

        class Robot:
            def execute(self, name, arguments):
                return name, arguments

        class Localizer:
            def locate(self, arguments, state, history):
                return arguments, state, history

        executor = ToolExecutor(catalog, ("left",), Robot(), Localizer())
        self.assertEqual(executor.execute("move_to", {"x": 1}, {}, {}), ("move_to", {"x": 1}))
        self.assertTrue(executor.is_terminal("done"))
        with self.assertRaises(ValueError):
            executor.execute("done", {}, {}, {})

    def test_unknown_or_invalid_handler_is_rejected(self) -> None:
        data = json.loads(Path("configs/tools.json").read_text(encoding="utf-8"))
        next(tool for tool in data["tools"] if tool["name"] == "move_to")["handler"] = "os.system"
        with TemporaryDirectory() as directory:
            source = Path(directory) / "tools.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            catalog = load_tool_catalog({"tool_catalog": str(source)})

        class Robot:
            def execute(self, name, arguments):
                return name, arguments

        with self.assertRaises(ValueError):
            ToolExecutor(catalog, ("left",), Robot(), object())


if __name__ == "__main__":
    unittest.main()
