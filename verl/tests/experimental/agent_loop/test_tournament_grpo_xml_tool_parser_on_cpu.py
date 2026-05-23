from __future__ import annotations

import pytest

from verl.experimental.agent_loop.tool_parser import TournamentGRPOXMLToolParser


class _DecodeOnlyTokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, ids):
        del ids
        return self.text


@pytest.mark.asyncio
async def test_tournament_grpo_xml_tool_parser_extracts_call_tool_only():
    parser = TournamentGRPOXMLToolParser(
        _DecodeOnlyTokenizer('<think>x</think><call_tool name="google_search">stellar ignition</call_tool>')
    )

    text, function_calls = await parser.extract_tool_calls([1, 2, 3], tools=None)

    assert parser.get_stop_strings() == ["</call_tool>"]
    assert text == "<think>x</think>"
    assert len(function_calls) == 1
    assert function_calls[0].name == "google_search"
    assert function_calls[0].arguments == '{"query": "stellar ignition"}'


@pytest.mark.asyncio
async def test_tournament_grpo_xml_tool_parser_rejects_legacy_tool_call_json():
    legacy_text = '<tool_call>{"name":"google_search","arguments":{"query":"stellar ignition"}}</tool_call>'
    parser = TournamentGRPOXMLToolParser(_DecodeOnlyTokenizer(legacy_text))

    text, function_calls = await parser.extract_tool_calls([1, 2, 3], tools=None)

    assert function_calls == []
    assert text == legacy_text
