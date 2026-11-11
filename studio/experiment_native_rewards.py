"""Reviewed mapping from the pinned Go2 flip trainer to the common reward editor."""
import ast
import hashlib
from pathlib import Path

FLIP = frozenset(("backflip_native_ppo", "frontflip_native_ppo"))
LABELS = {
    "termination": "失败终止", "ang_vel_x": "横滚角速度", "ang_vel_y": "翻转俯仰角速度",
    "ang_vel_z": "偏航角速度", "lin_vel_z": "起跳竖直速度", "gravity_y": "横向重力分量",
    "rotation_progress": "翻转进度", "flip_completion": "完成翻转", "flip_success": "翻转后站稳",
    "landing_impact": "落地冲击", "action_rate": "动作变化率", "recovery_upright": "落地恢复直立",
    "recovery_height": "落地恢复高度", "recovery_still": "落地减速",
}
FIXSTAND_DEFAULTS = {"recovery_default_pose": 8., "recovery_still": 6., "recovery_dof_velocity": -4.}


def native_manifest(root, config):
    if config.recipe_id not in FLIP or not config.recipe_options.get("fixstand_roundtrip"):
        return dict(terms=[], note="此配方保留原训练设置，尚未提供统一奖励函数适配。")
    if config.recipe_options.get("rebound_cycles", 1) != 1:
        return dict(terms=[], note="连续翻转配方保留原技能参数。单次翻转的奖励编辑器不适用于多阶段动作。")
    from skills.backflip.reward_options import NAMES, parse_scales
    source = root.parent / "Rob2Nav-upstreams/GO2_backflip/isaaclab_backflip/tasks/go2_backflip"
    cfg = ast.parse((source / "go2_backflip_env_cfg.py").read_text())
    entries = [ast.literal_eval(node.value) for node in ast.walk(cfg) if isinstance(node, ast.AnnAssign)
               and isinstance(node.target, ast.Name) and node.target.id == "scales" and isinstance(node.value, ast.Dict)]
    if len(entries) != 1:
        raise ValueError("Go2 奖励配置结构与已审核版本不一致")
    defaults = dict(entries[0], **FIXSTAND_DEFAULTS)
    front = config.recipe_id == "frontflip_native_ppo"
    if front:
        defaults.setdefault("ang_vel_x", 0.)
    overrides = parse_scales(config.recipe_options.get("reward_scales_json", "{}"))
    if set(overrides) - set(defaults):
        raise ValueError("该翻转方向未提供所选奖励项")
    text = (source / "go2_backflip_env.py").read_text()
    functions = {node.name.removeprefix("_reward_"): ast.get_source_segment(text, node)
                 for node in ast.walk(ast.parse(text)) if isinstance(node, ast.FunctionDef) and node.name.startswith("_reward_")}
    if front:
        from skills.frontflip.task import roll_rate_penalty
        import inspect
        functions["ang_vel_x"] = inspect.getsource(roll_rate_penalty)
    terms = []
    for key, default in defaults.items():
        code = functions.get(key)
        if key not in NAMES or not code:
            continue
        formula = next((line.strip().removeprefix("return ") for line in code.splitlines() if line.strip().startswith("return ")), key)
        if front and key in ("ang_vel_y", "orientation_control"):
            # The frontend must identify the mirrored source rather than display the backflip sign as effective.
            formula = "前翻方向由 skills/frontflip/task.py 变换俯仰符号，展开查看原函数与变换。"
            code += "\n\n" + (root / "skills/frontflip/task.py").read_text()
        terms.append(dict(id=key, label=LABELS.get(key, key), kind="mixed" if key in ("ang_vel_y", "lin_vel_z") else
             "penalty" if default < 0 or key=="ang_vel_x" else "reward", group="原生翻转",
             formula=formula, description="按该技能的实际尺度与控制步长计入训练奖励。",
             source=source.name + "/go2_backflip_env.py", code=code, available=True,
             parameters=[dict(id=key,label="权重",value=overrides.get(key,default),default=default,min=-1000.,max=1000.,editable=True)]))
    fingerprint=hashlib.sha256()
    for file in (source/"go2_backflip_env.py",source/"go2_backflip_env_cfg.py",root/"skills/backflip/launcher_patch.py"):
        fingerprint.update(file.read_bytes())
    if front:
        fingerprint.update((root/"skills/frontflip/task.py").read_bytes())
    return dict(terms=terms, sourceVersion=fingerprint.hexdigest(),
                note="单次 FIXSTAND 配方。落地端点附加项由训练器固定计算。")
