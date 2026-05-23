# Copyright 2026 Individual Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import inspect
from collections import defaultdict
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("tournament")
class TournamentRewardManager(RewardManagerBase):
    """Batch reward manager that compares rollouts sharing the same uid."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def _decode_item(self, data_item) -> dict[str, Any]:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        reward_model = data_item.non_tensor_batch.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})

        return {
            "data_source": data_item.non_tensor_batch.get("data_source", "unknown"),
            "solution_str": response_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
        }

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        item = await self._decode_item(data[0])
        if self.is_async_reward_score:
            result = await self.compute_score(**item)
        else:
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(**item))

        if isinstance(result, dict):
            return {"reward_score": result["score"], "reward_extra_info": result}
        return {"reward_score": float(result), "reward_extra_info": {"custom_reward": float(result)}}

    async def run_batch(self, data: DataProto) -> list[dict]:
        uid_to_indices: dict[Any, list[int]] = defaultdict(list)
        for idx in range(len(data)):
            uid_to_indices[data[idx].non_tensor_batch.get("uid", idx)].append(idx)

        outputs: list[dict | None] = [None] * len(data)
        for indices in uid_to_indices.values():
            items = [await self._decode_item(data[idx]) for idx in indices]
            if self.is_async_reward_score:
                results = await self.compute_score(items=items)
            else:
                results = await self.loop.run_in_executor(None, lambda items=items: self.compute_score(items=items))

            for local_idx, global_idx in enumerate(indices):
                result = results[local_idx]
                if isinstance(result, dict):
                    outputs[global_idx] = {"reward_score": result["score"], "reward_extra_info": result}
                else:
                    score = float(result)
                    outputs[global_idx] = {"reward_score": score, "reward_extra_info": {"custom_reward": score}}

        return [output if output is not None else {"reward_score": 0.0, "reward_extra_info": {}} for output in outputs]
