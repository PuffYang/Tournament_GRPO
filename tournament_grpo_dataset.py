import copy
import os

from verl.utils.dataset.rl_dataset import RLHFDataset


class TournamentGRPOPromptDataset(RLHFDataset):
    """Prepend the shared TournamentGRPO system prompt to every training example."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            raise ValueError("TournamentGRPOPromptDataset requires `config`.")

        prompt_file = os.path.expanduser(str(config.get("system_prompt_file", ""))).strip()
        if not prompt_file:
            raise ValueError("`data.system_prompt_file` must be set for TournamentGRPOPromptDataset.")
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"System prompt file does not exist: {prompt_file}")

        with open(prompt_file, encoding="utf-8") as f:
            self.system_prompt_text = f.read().strip()

        if not self.system_prompt_text:
            raise ValueError(f"System prompt file is empty: {prompt_file}")

        self.system_prompt_role = str(config.get("system_prompt_role", "system")).strip() or "system"
        super().__init__(*args, **kwargs)

    def _build_messages(self, example: dict, key: str):
        messages = copy.deepcopy(super()._build_messages(example, key))
        prefix_message = {"role": self.system_prompt_role, "content": self.system_prompt_text}

        if messages and messages[0] == prefix_message:
            return messages
        return [prefix_message, *messages]
