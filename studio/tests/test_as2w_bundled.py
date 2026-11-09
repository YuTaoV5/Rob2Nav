from pathlib import Path
import sys

from studio.config import RunConfig
from studio.experiment_workflow import as2w_commands, as2w_recipes, as2w_resources


def test_bundled_resources_allow_all_four_as2w_preflights(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    for name in ("ROB2NAV_AS2W_TRAINING", "ROB2NAV_AS2W_EVAL", "ROB2NAV_AS2W_MUJOCO_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    training, _, _ = as2w_resources(root, sys.executable)
    assert training == root / "skills/as2w/training"
    recipes = tuple(as2w_recipes())
    assert len(recipes) == 4
    for recipe in recipes:
        cfg = RunConfig.model_validate(recipe["defaults"])
        command, trainer, checkpoint = as2w_commands(cfg, root, tmp_path / "preview", sys.executable, sys.executable)
        assert trainer.is_relative_to(training)
        assert checkpoint.is_relative_to(training)
        assert command[command.index("--sim-python") + 1] == sys.executable
