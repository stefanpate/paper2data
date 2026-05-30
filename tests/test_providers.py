"""Provider unit tests — no network. Run: `uv run python -m unittest -v`.

Mocks the anthropic client (claude_api) and asserts the backend returns the
Ollama-shaped dicts the pipeline expects, plus that the pydantic schema is
mapped to a forced-tool input_schema correctly.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from typing import Literal

from pydantic import create_model

import paper2data.llm_providers as P

CLS = create_model("Classification", category=(Literal["sci.space", "rec.autos"], ...))
SCHEMA = CLS.model_json_schema()


class ApiTests(unittest.TestCase):
    def test_chat_forced_tool_use(self):
        tool_block = SimpleNamespace(type="tool_use", input={"category": "rec.autos"})
        resp = SimpleNamespace(
            content=[tool_block],
            usage=SimpleNamespace(input_tokens=20, output_tokens=3,
                                  cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0),
        )
        client = mock.Mock()
        client.messages.create.return_value = resp
        prov = P.ClaudeApiProvider(client=client, model="claude-sonnet-4-6",
                                   use_batch=False)
        out = prov.chat(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "classify"}],
            format=SCHEMA, options={"temperature": 0.0},
        )
        self.assertEqual(CLS.model_validate_json(out["message"]["content"]).category,
                         "rec.autos")
        # forced tool_choice was passed
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"]["type"], "tool")
        self.assertEqual(kwargs["tools"][0]["input_schema"]["additionalProperties"], False)

    def test_generate_text(self):
        text_block = SimpleNamespace(type="text", text="  hello  ")
        resp = SimpleNamespace(content=[text_block], usage=None)
        client = mock.Mock()
        client.messages.create.return_value = resp
        prov = P.ClaudeApiProvider(client=client, model="m", use_batch=False)
        out = prov.generate(model="m", prompt="hi", options={}, max_tokens=32)
        self.assertEqual(out, {"response": "hello"})


class OllamaTests(unittest.TestCase):
    def test_forwarding(self):
        inner = mock.Mock()
        inner.generate.return_value = {"response": "x"}
        inner.chat.return_value = {"message": {"content": "{}"}}
        prov = P.OllamaProvider(inner)
        self.assertFalse(prov.supports_batch())
        prov.generate(model="m", prompt="p", options={"seed": 1}, max_tokens=99)
        inner.generate.assert_called_once_with(model="m", prompt="p",
                                               options={"seed": 1})


if __name__ == "__main__":
    unittest.main()
