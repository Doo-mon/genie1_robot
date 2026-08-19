"""Per-frame subtask end / subtask prompt lookup driven by ``meta/episodes.jsonl``.

Used by the raw-desk_organization pi0.5 fine-tune flow when the parquet files do
NOT carry a ``macro_end_frame_index`` column. The meta file lists, per episode,
the subtask segmentation with ``start_frame/end_frame/task``, which is enough to
build:

* ``subtask_end[ep][f]``   -> end frame index (inclusive) of the subtask
  containing local frame ``f``.
* ``subtask_prompt[ep][f]``-> natural-language prompt of that subtask.

Loaded once and reused by both the training Dataset wrapper and the offline
norm-stats script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SubtaskMetaIndex:
    """Per-episode lookup for subtask end-frame and subtask prompt."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        episodes_path = self.dataset_root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")

        self._subtask_end: dict[int, np.ndarray] = {}
        self._subtask_prompt: dict[int, list[str]] = {}
        self._episode_len: dict[int, int] = {}

        with episodes_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep_idx = int(rec["episode_index"])
                length = int(rec["length"])
                self._episode_len[ep_idx] = length

                subtasks = rec.get("subtask", {}).get("subtask", [])
                if not subtasks:
                    end_arr = np.full((length,), length - 1, dtype=np.int64)
                    fallback_prompt = rec.get("tasks", ["Tidy up the desk"])[0]
                    prompt_list = [fallback_prompt] * length
                else:
                    # Default: every frame's subtask end is the last frame of the episode.
                    # Subtask segments will overwrite the covered range; uncovered tail
                    # frames (after the last subtask) naturally stay at length-1.
                    end_arr = np.full((length,), length - 1, dtype=np.int64)
                    prompt_list: list[str] = [""] * length
                    for seg in subtasks:
                        s = int(seg["start_frame"])
                        e = int(seg["end_frame"])  # inclusive end (per meta)
                        s = max(0, min(s, length - 1))
                        e = max(s, min(e, length - 1))
                        end_arr[s : e + 1] = e
                        task_text = str(seg.get("task", ""))
                        for k in range(s, e + 1):
                            prompt_list[k] = task_text
                    # Fill any remaining gaps (frames not covered by any subtask)
                    # with the nearest preceding prompt, or the first task.
                    last_prompt = (
                        prompt_list[-1]
                        if prompt_list[-1]
                        else rec.get("tasks", ["Tidy up the desk"])[0]
                    )
                    # Forward-fill: propagate last known prompt to uncovered frames
                    for k in range(length):
                        if prompt_list[k] == "":
                            prompt_list[k] = last_prompt
                        else:
                            last_prompt = prompt_list[k]
                    # Backward-fill: for end_arr, frames after the last subtask
                    # already default to length-1 (correct). Frames before the
                    # first subtask also default to length-1, which is fine as
                    # a conservative fallback.

                self._subtask_end[ep_idx] = end_arr
                self._subtask_prompt[ep_idx] = prompt_list

    # -- accessors -----------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        return len(self._episode_len)

    def episode_length(self, episode_index: int) -> int:
        return self._episode_len[int(episode_index)]

    def subtask_end(self, episode_index: int, frame_index: int) -> int:
        return int(self._subtask_end[int(episode_index)][int(frame_index)])

    def subtask_prompt(self, episode_index: int, frame_index: int) -> str:
        return self._subtask_prompt[int(episode_index)][int(frame_index)]

    def subtask_end_array(self, episode_index: int) -> np.ndarray:
        return self._subtask_end[int(episode_index)]