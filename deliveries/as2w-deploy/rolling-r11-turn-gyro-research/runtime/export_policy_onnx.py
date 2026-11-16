import io, os, json, hashlib
import torch, torch.nn as nn

H = "/root/gpufree-data/as2w-runs/r11_fanziqi_lab/yaw-arms/armE-support-20261116-034514"
OUT = "/root/gpufree-data/Rob2Nav/deliveries/as2w-deploy/rolling-r11-turn-gyro-research"
CKPT = os.path.join(H, "r11-final.pt")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["actor_state_dict"]

# keys are mlp.<idx>.<weight|bias>; distribution.std_param is the log-std used only when
# SAMPLING.  Deployment uses the deterministic mean, so it is not part of the exported graph.
lin = {}
for k, v in sd.items():
    if not k.startswith("mlp."):
        continue
    _, idx, param = k.split(".")
    if param in ("weight", "bias"):
        lin.setdefault(int(idx), {})[param] = v
order = sorted(lin)
print("linear layers:", [(i, tuple(lin[i]["weight"].shape)) for i in order])

layers = []
for n, idx in enumerate(order):
    w = lin[idx]["weight"]; b = lin[idx]["bias"]
    m = nn.Linear(w.shape[1], w.shape[0]); m.weight.data = w.clone(); m.bias.data = b.clone()
    layers.append(m)
    if n != len(order) - 1:
        layers.append(nn.ELU())
net = nn.Sequential(*layers).eval()
print("rebuilt:", net)

OBS, ACT = net[0].in_features, net[-1].out_features
assert (OBS, ACT) == (65, 16), "unexpected interface %dx%d" % (OBS, ACT)

torch.manual_seed(0)
x = torch.randn(4096, OBS)
with torch.no_grad():
    y = net(x)

onnx_path = os.path.join(OUT, "weights", "policy.onnx")
torch.onnx.export(net, (x[:1],), onnx_path,
    input_names=["obs"], output_names=["action"], opset_version=17,
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, do_constant_folding=True)
print("exported:", onnx_path, os.path.getsize(onnx_path), "bytes")

import onnx
m = onnx.load(onnx_path); onnx.checker.check_model(m)

import numpy as np, onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
yo = sess.run(["action"], {"obs": x.numpy()})[0]
err = np.abs(yo - y.numpy())
rep = {
  "checkpoint": CKPT,
  "checkpoint_sha256": hashlib.sha256(io.open(CKPT, "rb").read()).hexdigest(),
  "onnx_sha256": hashlib.sha256(io.open(onnx_path, "rb").read()).hexdigest(),
  "opset": int(m.opset_import[0].version),
  "onnx_size_bytes": os.path.getsize(onnx_path),
  "input": {"name": "obs", "shape": ["batch", OBS], "dtype": "float32"},
  "output": {"name": "action", "shape": ["batch", ACT], "dtype": "float32"},
  "control_period_s": 0.02,
  "architecture": "MLP 65-128-128-128-16, ELU hidden, deterministic mean (std_param excluded)",
  "internal_state": None,
  "tolerance_max_abs": 1e-5,
  "comparison": {
    "n_samples": int(x.shape[0]),
    "max_abs_err": float(err.max()),
    "mean_abs_err": float(err.mean()),
    "torch_abs_mean": float(np.abs(y.numpy()).mean()),
    "onnx_abs_mean": float(np.abs(yo).mean()),
    "max_abs_err_rel": float(err.max() / max(1e-9, float(np.abs(y.numpy()).max()))),
  },
}
rep["comparison"]["pass"] = bool(rep["comparison"]["max_abs_err"] <= rep["tolerance_max_abs"])
io.open(os.path.join(OUT, "evidence", "C2_ONNX_PARITY.json"), "w", encoding="utf-8").write(
    json.dumps(rep, indent=1))
print(json.dumps(rep["comparison"], indent=1))
print("ONNX_SHA", rep["onnx_sha256"][:16], "PARITY_PASS:", rep["comparison"]["pass"])
