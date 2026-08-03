"""Resumable wizard state.

A JSON file recording which steps have completed and which answers were
already collected, so Ctrl-C / a dropped SSH session / a crash partway
through does not mean starting over. Re-running the wizard skips anything
already marked done and pre-fills prompts from previously-collected answers.
"""
import json
import os

DEFAULT_PATH = "/root/.slurm-deck-wizard-state.json"


class State:
    def __init__(self, path: str | None = None):
        # Resolved at call time (not bound as a stale default argument) so
        # SLURM_DECK_WIZARD_STATE genuinely overrides it -- used by tests
        # and for pointing a dry run at a scratch file instead of the real
        # state path.
        self.path = path or os.environ.get("SLURM_DECK_WIZARD_STATE", DEFAULT_PATH)
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                    data.setdefault("done", {})
                    data.setdefault("answers", {})
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"done": {}, "answers": {}}

    def save(self) -> None:
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def is_done(self, step: str) -> bool:
        return bool(self.data["done"].get(step, False))

    def mark_done(self, step: str) -> None:
        self.data["done"][step] = True
        self.save()

    def get_answer(self, key: str, default=None):
        return self.data["answers"].get(key, default)

    def set_answer(self, key: str, value) -> None:
        self.data["answers"][key] = value
        self.save()
