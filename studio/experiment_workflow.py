"""Reward definitions and the AS2-W adapter for Studio's shared experiment queue."""
import ast
import json
import os
from pathlib import Path
import re
import sys

from .as2w_reward_modules import SOURCE, catalog, defaults, resolve
from .as2w_runner import BASELINE, RECOVERY_TERRAINS, STAIR_BASELINE, STAIR_RECIPES, digest, write
from .as2w_eval.reward_expression import PRESETS, SIGNALS, reward_specs

AS2W_RECIPE = "as2w_recovery_ppo"
AS2W_NAME = "AS2-W 四轮起身与站稳"
AS2W_OPTIONS = {
    "recipe_id": AS2W_RECIPE, "robot_id": "as2w", "algorithm_id": "as2w_recovery_ppo",
    "adapter": "as2w", "capability_id": "", "task": "train", "iterations": 2000,
    "num_envs": 256, "seed": 413, "checkpoint": "r12", "initialization": "initialize",
    "recipe_options": {}, "rewards": {}, "auto_evaluate": True,
}
AS2W_LABELS = {
    "upright_weight": "翻正权重", "pose_weight": "站姿权重", "impact_weight": "撞击惩罚系数",
    "stance_weight": "承重与站稳权重", "effort_base": "基础动作惩罚", "effort_righted": "翻正后的附加动作惩罚",
}


def as2w_recipe():
    return dict(id=AS2W_RECIPE, name=AS2W_NAME, robot="as2w", algorithm=AS2W_RECIPE, adapter="as2w",
                capabilities=[], execution="manual_cli", defaults=AS2W_OPTIONS, option_schema={},
                note="保留原 R13 地形课程与动力学随机化。训练后按固定场景评估平地、坡面和粗糙地面。")


def as2w_recipes():
    yield as2w_recipe()
    for recipe, label in zip(STAIR_RECIPES, ("R14 基础通行", "R15 抬轮净空", "R16 后轮承重课程")):
        yield dict(id=recipe, name="AS2-W 盲走楼梯 · " + label, robot="as2w", algorithm=AS2W_RECIPE,
                   adapter="as2w", capabilities=[], execution="manual_cli", option_schema={},
                   defaults={**AS2W_OPTIONS, "recipe_id": recipe, "checkpoint": "r17"},
                   note="使用原训练器的平地与楼梯课程，并在独立 MuJoCo 楼梯场景评估。")


def terrain_choices(config):
    terrain = config.recipe_options.get("eval_terrain", "stairs" if config.recipe_id in STAIR_RECIPES else "all")
    if config.recipe_id in STAIR_RECIPES:
        if terrain != "stairs":
            raise ValueError("盲走楼梯配方使用楼梯验收协议")
        return ("stairs",)
    if terrain == "all":
        return RECOVERY_TERRAINS
    if terrain not in RECOVERY_TERRAINS:
        raise ValueError("起身配方仅支持平地、坡面和粗糙地面验收")
    return (terrain,)


def profile(config):
    options = dict(config.recipe_options)
    options.pop("eval_terrain", None)
    options.pop("validation_only", None)
    extra = reward_specs(options.pop("custom_rewards", "[]"))
    if config.recipe_id in STAIR_RECIPES:
        if extra or options:
            raise ValueError("楼梯配方使用已注册的训练奖励")
        return {}
    return {**resolve(options), **({"custom_rewards": extra} if extra else {})}


def validate_as2w(config):
    if config.recipe_id not in (AS2W_RECIPE, *STAIR_RECIPES) or (
        config.robot_id, config.adapter, config.algorithm_id, config.task, config.initialization) != (
        "as2w", "as2w", AS2W_RECIPE, "train", "initialize",
    ) or config.rewards or config.capability_id:
        raise ValueError("AS2-W 恢复技能要求匹配的训练配方和仅权重初始化")
    validation = config.recipe_options.get("validation_only", False)
    if type(validation) is not bool:
        raise ValueError("链路验证开关必须为布尔值")
    if validation and (config.recipe_id != AS2W_RECIPE or terrain_choices(config) != ("flat",)
                       or not 3 <= config.iterations <= 32 or config.num_envs != 40):
        raise ValueError("起身链路验证使用 40 个环境、3–32 轮与平地评估")
    if not config.auto_evaluate or not 40 <= config.num_envs <= 2048 or not (validation or 2000 <= config.iterations <= 40000):
        raise ValueError("AS2-W 正式训练需要 40–2048 个环境、2000–40000 轮及独立 MuJoCo 评估")
    if 12001 <= config.seed <= 12004 or 27001 <= config.seed <= 27016:
        raise ValueError("训练种子不能使用 MuJoCo 留出场景的种子")
    terrain_choices(config)
    profile(config)
    return config


def as2w_resources(root, mujoco_python=None):
    bundled = root / "skills/as2w/training"
    training = Path(os.environ.get("ROB2NAV_AS2W_TRAINING") or
                    (bundled if bundled.is_dir() else "/root/as2w-training"))
    harness = Path(os.environ.get("ROB2NAV_AS2W_EVAL", str(root / "studio/as2w_eval")))
    existing_cpu = root.parent / "envs/go2-supermimic/bin/python"
    cpu = Path(os.environ.get("ROB2NAV_AS2W_MUJOCO_PYTHON") or
               (existing_cpu if existing_cpu.is_file() else mujoco_python or existing_cpu))
    return training, harness, cpu


def as2w_checkpoint(config, root, training):
    stair = config.recipe_id in STAIR_RECIPES
    if config.checkpoint == ("r17" if stair else "r12"):
        path = training / (STAIR_BASELINE if stair else BASELINE)
    else:
        pattern = r"outputs/studio/runs/[a-f0-9]{16}/checkpoint/model_[0-9]+\.pt"
        if not re.fullmatch(pattern, config.checkpoint):
            raise ValueError("请选择已评估的 AS2-W 恢复权重")
        path = root / config.checkpoint
        comparison = path.parent.parent / "comparison.json"
        manifest_file = path.parent.parent / "manifest.json"
        source_recipe = json.loads(manifest_file.read_text()).get("recipe", AS2W_RECIPE) if manifest_file.is_file() else None
        if not comparison.is_file() or source_recipe != config.recipe_id or json.loads(comparison.read_text())["candidate"]["checkpoint"] != str(path):
            raise ValueError("权重缺少匹配的 AS2-W 评估记录")
    if not path.is_file():
        raise ValueError("AS2-W 起始权重不可用")
    return path


def as2w_commands(config, root, run, mujoco_python=None, sim_python=None):
    training, harness, cpu = as2w_resources(root, mujoco_python)
    baseline = as2w_checkpoint(config, root, training)
    stair = STAIR_RECIPES.get(config.recipe_id)
    source = training / (stair[0] if stair else SOURCE)
    for path in (source, harness / "evaluate-fall-baseline.py", cpu):
        if not path.is_file():
            raise ValueError(f"缺少 AS2-W 训练或评估资源：{path.name}")
    official = harness / "official-model"
    if not official.is_dir():
        official = training / "unitree_ros"
    contract = harness / "fall-recovery-contract.json"
    if not contract.is_file():
        contract = training / "replay/fall-recovery-contract.json"
    for path in (official / "robots/as2w_description/as2w.urdf", contract,
                 harness / "local-step-diagnostic/as2w_flat.xml"):
        if not path.is_file():
            raise ValueError(f"缺少 AS2-W 仿真资产：{path.name}")
    if not stair:
        catalog(source.read_text())
    argv = [sys.executable, str(Path(__file__).with_name("as2w_runner.py")), str(run),
            str(training), str(harness), str(cpu)]
    if sim_python:
        argv += ["--sim-python", str(sim_python)]
    if os.environ.get("ISAACLAB_ROOT"):
        argv += ["--isaaclab", os.environ["ISAACLAB_ROOT"]]
    return argv, source, baseline


def prepare_as2w(config, root, run, mujoco_python=None, sim_python=None):
    argv, trainer, baseline = as2w_commands(config, root, run, mujoco_python, sim_python)
    run.mkdir(parents=True, exist_ok=False)
    values = profile(config)
    diff = {key: dict(before=value, after=values[key]) for key, value in defaults().items()
            if key in values and value != values[key]}
    write(run / "parameters.json", values)
    write(run / "manifest.json", dict(updates=config.iterations, envs=config.num_envs, seed=config.seed,
          smoke=config.recipe_options.get("validation_only",False), diff=diff, trainer_sha256=digest(trainer), baseline_sha256=digest(baseline),
          baseline_checkpoint=str(baseline), recipe=config.recipe_id, evaluation_terrains=terrain_choices(config)))
    return argv


def as2w_progress(folder):
    trace = folder / "reward.jsonl"
    if not trace.is_file():
        return {}
    with trace.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 16384))
        lines = stream.read().split(b"\n")
    for line in reversed(lines[:-1]):
        try:
            sample = json.loads(line)
        except ValueError:
            continue
        completed = sample["step"] // 24
        return dict(iteration=completed - 1) if completed else {}
    return {}


def manifest(root, config, recipe):
    if config.recipe_id in STAIR_RECIPES:
        file, flags = STAIR_RECIPES[config.recipe_id]
        source = (as2w_resources(root)[0] / file).read_text()
        names = {"traversal_reward"}
        if "--stair-clearance-reward" in flags:
            names.add("stair_clearance_reward")
        if "--stair-transfer-curriculum" in flags:
            names.add("stair_transfer_reward")
        terms = [dict(id=node.name, label=node.name, kind="mixed", group="盲走楼梯",
                      formula="见原训练函数", code=ast.get_source_segment(source, node),
                      description="使用该楼梯配方的固定训练公式。", source=file, available=True, parameters=[])
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
        return dict(terms=terms, note="楼梯配方的奖励按原函数执行，可查看源码。")
    if config.recipe_id == AS2W_RECIPE:
        source = (as2w_resources(root)[0] / SOURCE).read_text()
        values = profile(config)
        result = catalog(source, values)
        return dict(sourceCode=result["code"], sourceVersion=digest(as2w_resources(root)[0] / SOURCE), custom=dict(signals=SIGNALS, presets=PRESETS), terms=[
            dict(id=term["id"], label=term["name"], kind=term["kind"], group="起身与站稳",
                 formula=term["formula"], description=term["help"], source=result["source"],
                 available=True, parameters=[dict(id=p["id"], label=AS2W_LABELS[p["id"]], value=values[p["id"]],
                     default=p["default"], min=p["min"], max=p["max"], editable=True) for p in term["parameters"]])
            for term in result["terms"]
        ])
    if recipe and recipe["execution"] != "native":
        from .experiment_native_rewards import native_manifest
        return native_manifest(root, config)
    from rob2nav.experiment_schema import reward_schema
    return dict(terms=[
        dict(id=key, label=t["label"], kind="penalty" if (t["template_weight"] or t["weight"]) < 0 else
             "mixed" if key in ("forward_progress", "feet_air_time", "landing_support", "supported_contact") else "reward",
             group=t["category"], unit=t["unit"], formula=t["formula"], description=t["description"],
             source=t["source"], available=t["available"],
             parameters=[dict(id=key, label="权重", value=t["effective_weight"] or 0, default=t["template_weight"] or 0,
                              min=-1000., max=1000., editable=t["available"])])
        for key, t in reward_schema(config).items()
    ])
