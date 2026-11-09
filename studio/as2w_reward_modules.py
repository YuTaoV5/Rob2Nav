"""AS2-W R13 reward modules, shared by Studio and the instrumented trainer."""
import ast
import json
import os
import sys
from types import ModuleType
from pathlib import Path
if __package__:
    from .as2w_eval.reward_expression import SIGNALS, expression, reward_specs
else:
    from as2w_eval.reward_expression import SIGNALS, expression, reward_specs

SOURCE = "train_as2w_fall_mixed_terrain.py"
FUNCTION = "recovery_reward"
ORIGINAL = """active * (1.5 * upright + 2. * standing - .0005 * impact.clamp(max=2000.)
                         + righted * stance
                         - (.015 + .06 * righted) * action_cost)"""

# Each expression is evaluated inside the original reward function, after its gates.
TERMS = (
    ("upright", "翻正", "mixed", "upright", "机身朝上时加分，倒扣时扣分。", (("upright_weight", 1.5, 6.),)),
    ("pose", "站姿", "reward", "standing", "腿姿越接近默认站姿、机身高度越接近 0.36 m，得分越高。", (("pose_weight", 2., 8.),)),
    ("impact", "非轮撞击", "penalty", "impact.clamp(max=2000.)", "取底盘等非轮部位的最大接触力，上限 2000 N。", (("impact_weight", .0005, .01),)),
    ("stance", "承重与站稳", "reward", "righted * stance", "翻正后才逐步计分，合并减速、四轮承重、离地高度与连续站稳时间。", (("stance_weight", 1., 4.),)),
    ("effort", "动作幅度", "penalty", "action_cost", "对动作平方取平均，翻正后逐渐提高惩罚，帮助收住动作。", (("effort_base", .015, .25), ("effort_righted", .06, .5))),
)


def defaults():
    return {key: default for *_, params in TERMS for key, default, _ in params}


def resolve(overrides):
    values = defaults()
    if not isinstance(overrides, dict):
        raise ValueError("参数必须为对象")
    limits = {key: high for *_, params in TERMS for key, _, high in params}
    for key, value in overrides.items():
        if key not in limits or type(value) not in (int, float) or not 0 <= value <= limits[key]:
            raise ValueError(f"无效参数或超出范围：{key}")
        values[key] = float(value)
    return values


def formulas(p):
    return (
        f"{p['upright_weight']!r} * upright",
        f"{p['pose_weight']!r} * standing",
        f"-{p['impact_weight']!r} * impact.clamp(max=2000.)",
        f"{p['stance_weight']!r} * (righted * stance)",
        f"-({p['effort_base']!r} + {p['effort_righted']!r} * righted) * action_cost",
    )


def locate(source):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == FUNCTION)
    expected = ast.dump(ast.parse(ORIGINAL, mode="eval").body)
    matches = [node for node in ast.walk(function) if isinstance(node, ast.Return) and node.value and ast.dump(node.value) == expected]
    if len(matches) != 1:
        raise ValueError("服务器训练函数与 R13 已审核公式不匹配")
    return tree, function, matches[0]


def catalog(source, values=None):
    _, function, _ = locate(source)
    return {
        "source": SOURCE, "function": FUNCTION, "line": function.lineno,
        "code": ast.get_source_segment(source, function),
        "terms": [
            {"id": key, "name": name, "kind": kind, "raw": raw, "help": help_text,
             "formula": f"active * ({formula})",
             "parameters": [{"id": param, "default": default, "min": 0., "max": high}
                            for param, default, high in params]}
            for (key, name, kind, raw, help_text, params), formula in zip(TERMS, formulas(values or defaults()))
        ],
    }


def instrument(source, values):
    tree, _, target = locate(source)
    # Keep the original arithmetic order and active gate. When weights are
    # unchanged, the original expression remains the returned reward.
    u, pose, impact, stance, effort = formulas(values)
    expression = ORIGINAL if values == defaults() else f"active * ({u} + {pose} - ({impact[1:]}) + {stance} - ({effort[1:]}))"
    raw = ", ".join(term[3] for term in TERMS)
    weighted = ", ".join(f"active * ({term})" for term in formulas(values))
    target.value = ast.parse(
        f"_studio_reward_sample(env, active, ({expression}), lambda: ({raw}), lambda: ({weighted}))",
        mode="eval",
    ).body
    return ast.fix_missing_locations(tree)


def execute_trainer(source, tree, sample=None):
    directory = os.environ.get("ROB2NAV_AS2W_USD_DIR")
    if directory:
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "UrdfFileCfg"]
        if len(calls) != 1 or any(kw.arg == "usd_dir" for kw in calls[0].keywords):
            raise ValueError("AS2-W asset conversion layout changed")
        calls[0].keywords.append(ast.keyword(arg="usd_dir", value=ast.Constant(directory)))
    code = compile(ast.fix_missing_locations(tree), str(source), "exec")
    original_main = sys.modules["__main__"]
    trainer = ModuleType("__main__")
    trainer.__dict__.update(__file__=str(source), _studio_reward_sample=sample)
    sys.modules["__main__"] = trainer
    try:
        exec(code, vars(trainer))
    finally:
        sys.modules["__main__"] = original_main


def run_training(source, profile, trace):
    source = Path(source).resolve()
    profile = json.loads(Path(profile).read_text(encoding="utf-8"))
    extra = reward_specs(profile.pop("custom_rewards", []))
    weights = resolve(profile)
    compiled = [(expression(term["formula"]), term["weight"]) for term in extra]
    tick = 0
    with Path(trace).open("x", encoding="utf-8", buffering=1) as output:
        def sample(env, active, reward, raw, weighted):
            nonlocal tick
            tick += 1
            added_raw, added_weighted = [], []
            if compiled:
                raw_values = raw()
                signals = dict(zip(SIGNALS, raw_values))
                for fn, weight in compiled:
                    value = (reward.new_zeros(reward.shape) + fn(signals)).clamp(-10., 10.)
                    contribution = value * weight * active
                    reward = reward + contribution
                    added_raw.append(value)
                    added_weighted.append(contribution)
            if tick == 1 or tick % 24 == 0:
                import torch
                with torch.no_grad():
                    # A single transfer per sample. Contributions are before
                    # IsaacLab RewardManager's multiplication by step_dt.
                    means = torch.stack([x.float().mean() for x in (*raw(), *added_raw, *weighted(), *added_weighted, reward)])
                    values = means.cpu().tolist()
                if all(abs(value) < float("inf") for value in values):
                    ids = [term[0] for term in TERMS] + [f"custom_{i}" for i in range(len(extra))]
                    terms = {key: {"raw": values[i], "weighted": values[len(ids)+i]}
                             for i, key in enumerate(ids)}
                    output.write(json.dumps({"step": tick, "dt": env.step_dt, "total": values[-1], "terms": terms}) + "\n")
            return reward

        execute_trainer(source, instrument(source.read_text(encoding="utf-8"), weights), sample)
