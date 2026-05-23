from __future__ import annotations

import tournament_grpo_reward


def test_tournament_grpo_reward_accepts_required_structure():
    transcript = """
<think>Need a source.</think>
<call_tool name="google_search">stellar ignition</call_tool>
<tool_output>
<snippet>
Title: Example
URL: https://example.com
Search Snippet: Example snippet.
</snippet>
</tool_output>
<think>Need to inspect a page.</think>
<call_tool name="browse_webpage">https://example.com</call_tool>
<tool_output>
<webpage>
Title: Example
URL: https://example.com
Content: Example content.
</webpage>
</tool_output>
<think>Stars ignite when gravitational collapse heats the core enough for fusion.</think>
<answer>Stars ignite when gravitational collapse heats the core enough for fusion.</answer>
""".strip()

    result = tournament_grpo_reward.compute_score(
        solution_str=transcript,
        ground_truth={"query": "q", "rubrics": []},
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["format_valid"] == 1
    assert result["tool_call_count"] == 2
    assert result["tool_output_count"] == 2
    assert result["google_search_cycle_count"] == 1
    assert result["browse_webpage_cycle_count"] == 1
    assert result["segment_count"] == 1
    assert result["rollout_answer"] == "Stars ignite when gravitational collapse heats the core enough for fusion."


def test_tournament_grpo_reward_rejects_missing_tool_output_cycle():
    transcript = """
<think>Need sources.</think>
<call_tool name="google_search">stellar ignition</call_tool>
<think>Done searching.</think>
<answer>Done.</answer>
""".strip()

    result = tournament_grpo_reward.compute_score(
        solution_str=transcript,
        ground_truth={"query": "q", "rubrics": []},
        extra_info={},
    )

    assert result["format_reward"] == 0.0
    assert result["format_valid"] == 0
    assert "structure:each_segment_must_start_with_google_search_cycle" in result["format_failure_reasons"] or (
        "structure:invalid_tool_cycle_order" in result["format_failure_reasons"]
    )
    assert result["google_search_call_count"] == 1
    assert result["browse_webpage_call_count"] == 0


def test_tournament_grpo_reward_rejects_unknown_tool_name_and_does_not_count_it():
    transcript = """
<think>Need sources.</think>
<call_tool name="browser">https://example.com</call_tool>
<tool_output>
<webpage>
Title: Example
URL: https://example.com
Content: Example content.
</webpage>
</tool_output>
<think>Done browsing.</think>
<answer>Done.</answer>
""".strip()

    result = tournament_grpo_reward.compute_score(
        solution_str=transcript,
        ground_truth={"query": "q", "rubrics": []},
        extra_info={},
    )

    assert result["format_reward"] == 0.0
    assert result["format_valid"] == 0
    assert "tool:unknown_tool_name" in result["format_failure_reasons"]
    assert result["tool_call_count"] == 0
    assert result["google_search_call_count"] == 0
    assert result["browse_webpage_call_count"] == 0
