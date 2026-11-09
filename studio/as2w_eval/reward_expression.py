"""Small arithmetic reward language shared by training and MuJoCo."""
import ast
import json
import math
import operator

SIGNALS = ("upright", "standing", "impact", "stance", "action_cost")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
PRESETS = (
    ("保持直立", "upright", .2),
    ("额外承重奖励", "stance", .1),
    ("动作幅度惩罚", "-action_cost", .05),
    ("非轮撞击惩罚", "-0.001 * impact", .1),
)


def expression(text):
    if not isinstance(text, str) or not 1 <= len(text) <= 160:
        raise ValueError("公式长度须为 1–160 字符")
    try:
        node = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError) as exc:
        raise ValueError("公式语法无效") from exc

    def compile_node(n, depth):
        if depth > 8:
            raise ValueError("公式嵌套过深")
        if isinstance(n, ast.Name) and n.id in SIGNALS:
            return lambda values: values[n.id]
        if (isinstance(n, ast.Constant) and type(n.value) in (int, float)
                and -1000 <= n.value <= 1000):
            return lambda values: n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            inner = compile_node(n.operand, depth + 1)
            return lambda values: inner(values) * (-1 if isinstance(n.op, ast.USub) else 1)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            left = compile_node(n.left, depth + 1)
            right = compile_node(n.right, depth + 1)
            return lambda values: OPS[type(n.op)](left(values), right(values))
        raise ValueError("公式仅支持列出的信号、数字、括号和加减乘")

    return compile_node(node, 0)


def reward_specs(raw):
    if isinstance(raw, str):
        if len(raw) > 4096:
            raise ValueError("自定义函数配置过长")
        raw = json.loads(raw)
    if not isinstance(raw, list) or len(raw) > 8:
        raise ValueError("最多添加 8 个自定义函数")
    result = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"name", "formula", "weight"}:
            raise ValueError("函数需要名称、公式和权重")
        name, formula, weight = (item[k] for k in ("name", "formula", "weight"))
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40:
            raise ValueError("函数名称须为 1–40 字符")
        if type(weight) not in (int, float) or not -4 <= weight <= 4 or not math.isfinite(weight):
            raise ValueError("自定义函数权重范围为 -4 至 4")
        expression(formula)
        result.append(dict(name=name.strip(), formula=formula, weight=float(weight)))
    return result
