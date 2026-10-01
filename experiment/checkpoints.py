"""Historical snapshots are ranked exports, never training branches."""
import math
from pathlib import Path

import torch

from .io import save_torch, write_json


class TopK:
    def __init__(self, root, count, metric):
        self.root, self.count, self.metric = Path(root), count, metric
        self.entries = []
        self.resume_paths = set()
        self.root.mkdir(parents=True, exist_ok=True)

    def consider(self, step, prefix, loss):
        if not math.isfinite(loss):
            raise ValueError("保存损失不是有限值")
        if any(row["step"] == step for row in self.entries):
            return False
        order = (loss, step)
        if len(self.entries) >= self.count and order >= max((r["loss"], r["step"]) for r in self.entries):
            return False
        path = self.root / f"step_{step:08d}.pt"
        save_torch(path, {"soft_prefix": prefix.detach().cpu().clone(), "step": step,
                          "checkpoint_loss": loss, "checkpoint_metric": self.metric})
        self.entries.append({"step": step, "loss": loss, "path": path.name})
        self.entries.sort(key=lambda row: (row["loss"], row["step"]))
        while len(self.entries) > self.count:
            retired = self.entries.pop()
            if retired["path"] not in self.resume_paths:
                (self.root / retired["path"]).unlink(missing_ok=True)
        self.export()
        return True

    def export(self):
        write_json(self.root / "manifest.json", {"checkpoint_metric": self.metric, "requested_topk": self.count,
                   "prefix_count": len(self.entries), "prefixes": [{"rank": index + 1, **row} for index, row in enumerate(self.entries)]})

    def state_dict(self):
        return [dict(row) for row in self.entries]

    def commit_resume(self):
        # Call only AFTER latest.pt was atomically replaced. Until then retain
        # snapshots referenced by the last successful state, even if pruned.
        self.resume_paths = {row["path"] for row in self.entries}
        self._prune_files()

    def _prune_files(self):
        keep = self.resume_paths | {row["path"] for row in self.entries}
        for path in self.root.glob("step_*.pt"):
            if path.name not in keep:
                path.unlink(missing_ok=True)

    def load_state_dict(self, state):
        paths = set()
        self.entries = []
        for item in state:
            row = {k: v for k, v in item.items() if k != "payload"}
            if Path(row["path"]).name != row["path"]:
                raise ValueError("续跑前缀文件名无效")
            # Accept older states with embedded payloads, but new states contain
            # metadata only. Never discard a missing reference silently.
            if "payload" in item:
                save_torch(self.root / row["path"], item["payload"])
            elif not (self.root / row["path"]).is_file():
                raise FileNotFoundError(f"续跑缺少前缀文件: {self.root / row['path']}")
            paths.add(row["path"])
            self.entries.append(row)
        self.resume_paths = paths
        self._prune_files()
        self.entries.sort(key=lambda row: (row["loss"], row["step"]))
        self.export()
