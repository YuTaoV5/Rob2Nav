"""Test whether proprioceptive history can identify AS2-W stair engagement."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import evaluate_omni_stairs as evaluation
from train_omni_mujoco import OmniEnvironment
from train_recovery_student import make_actor, numpy_actor


HISTORY = 7
FRAME_INDICES = (0, 2, 4, 6)
FEATURES = np.r_[3:9, 10:58]
WIDTH = len(FRAME_INDICES) * len(FEATURES)


def contact_gate():
    return nn.Sequential(nn.Linear(WIDTH, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1))


def touched_riser(env):
    if not env.stairs:
        return False
    for index, contact in enumerate(env.data.contact):
        surface, other = contact.geom
        if surface not in env.surfaces:
            surface, other = other, surface
        if env.surfaces.get(surface, 0) < 1:
            continue
        body = env.model.geom_bodyid[other]
        if body in env.wheel_index:
            evaluation.sim.mj.mj_contactForce(env.model, env.data, index, env.force)
            if env.force[0] >= 8.:
                return True
    return False


def trajectory(env, seed, actor, target=None, demo=None):
    options = {"scene_seed": seed}
    if target is not None:
        options["target"] = target
    obs, _ = env.reset(options=options)
    history = deque([obs[FEATURES]] * HISTORY, maxlen=HISTORY)
    features, labels = [], []
    engaged = False
    horizon = len(demo) if demo is not None else 450
    for tick in range(horizon):
        engaged |= touched_riser(env)
        features.append(np.concatenate([history[index] for index in FRAME_INDICES]))
        labels.append(engaged)
        action = demo[tick] if demo is not None else actor(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        history.append(obs[FEATURES])
        if terminated or truncated:
            break
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=bool)


def collect(results, actors, start, count):
    flat = OmniEnvironment(results, 1)
    stairs = OmniEnvironment(results, 0)
    rng = np.random.default_rng(start)
    commands = tuple(evaluation.FLAT_COMMANDS.values())
    episodes = []
    for i in range(count):
        actor = actors[i % len(actors)]
        if i % 3 == 0:
            target = (.4, 0., 0.)
        elif i % 3 == 1:
            target = commands[rng.integers(len(commands))]
        else:
            target = rng.uniform((-.6, -.35, -.8), (.6, .35, .8))
        episodes.append(("flat", start + 2 * i, *trajectory(flat, start + 2 * i, actor, target)))
        episodes.append(("stairs", start + 2 * i + 1, *trajectory(stairs, start + 2 * i + 1, actor)))
    return episodes


def score(episodes, predict, threshold, persistence):
    rows = []
    for scene, seed, features, labels in episodes:
        high = predict(features) >= threshold
        consecutive = np.convolve(
            high.astype(np.int8), np.ones(persistence, dtype=np.int8), mode="full",
        )[:len(high)] == persistence
        contact = np.flatnonzero(labels)
        detection = np.flatnonzero(consecutive)
        rows.append({
            "scene": scene, "seed": seed,
            "contact_tick": int(contact[0]) if len(contact) else None,
            "detection_tick": int(detection[0]) if len(detection) else None,
        })
    no_contact = [row for row in rows if row["contact_tick"] is None]
    contacted = [row for row in rows if row["contact_tick"] is not None]
    delays = [
        row["detection_tick"] - row["contact_tick"] for row in contacted
        if row["detection_tick"] is not None and row["detection_tick"] >= row["contact_tick"]
    ]
    return {
        "false_episodes": sum(row["detection_tick"] is not None for row in no_contact),
        "no_contact_episodes": len(no_contact),
        "early_episodes": sum(row["detection_tick"] is not None
                              and row["detection_tick"] < row["contact_tick"] for row in contacted),
        "contact_episodes": len(contacted),
        "detected_within_025s": sum(0 <= delay <= 12 for delay in delays),
        "median_delay_s": float(np.median(delays) * .02) if delays else None,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--actors", nargs="+", type=Path, required=True)
    parser.add_argument("--demos", nargs="+", type=Path, default=())
    parser.add_argument("--mine-from-gate", type=Path,
                        help="Replay false activations from the previous gate's development split")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(460)
    root = args.results.resolve()
    actors = [
        numpy_actor(make_actor(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]))
        for path in args.actors
    ]
    train = collect(root, actors, 96501, 48)
    reference = OmniEnvironment(root, 0)
    for path in args.demos:
        with np.load(path) as archive:
            for seed in np.unique(archive["seed"]):
                demo = archive["actions"][archive["seed"] == seed]
                train.append(("demo", int(seed), *trajectory(reference, int(seed), actors[0], demo=demo)))
    hard = []
    if args.mine_from_gate:
        previous = torch.load(args.mine_from_gate, map_location="cpu", weights_only=True)
        previous_model = contact_gate().eval()
        previous_model.load_state_dict(previous["gate_state_dict"])

        @torch.inference_mode()
        def previous_predict(features):
            inputs = (torch.from_numpy(features) - previous["center"]) / previous["scale"]
            return previous_model(inputs.clamp(-8., 8.))[:, 0].sigmoid().numpy()

        for episode in collect(root, actors, 97501, 28):
            _, _, features, labels = episode
            if labels.any():
                continue
            scores = previous_predict(features)
            if (scores >= previous["threshold"]).sum() >= previous.get("required_consecutive", 3):
                hard.append(features)
        dev = collect(root, actors, 99501, 28)
        test = collect(root, actors, 100501, 36)
    else:
        dev = collect(root, actors, 97501, 28)
        test = collect(root, actors, 98501, 36)
    x = torch.from_numpy(np.concatenate([episode[2] for episode in train]))
    y = torch.from_numpy(np.concatenate([episode[3] for episode in train])).float()
    center = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(.04)
    hard_x = torch.from_numpy(np.concatenate(hard)) if hard else None
    if hard_x is not None:
        hard_x = ((hard_x - center) / scale).clamp(-8., 8.)
    x = ((x - center) / scale).clamp(-8., 8.)
    positive = torch.where(y > .5)[0]
    negative = torch.where(y < .5)[0]
    if not len(positive) or not len(negative):
        raise ValueError("Contact training requires both engaged and unengaged frames")
    model = contact_gate()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
    for _ in range(550):
        indices = torch.cat((positive[torch.randint(len(positive), (128,))],
                             negative[torch.randint(len(negative), (256,))]))
        loss = nn.functional.binary_cross_entropy_with_logits(model(x[indices])[:, 0], y[indices])
        if hard_x is not None:
            logits = model(hard_x[torch.randint(len(hard_x), (96,))])[:, 0]
            loss += .7 * nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @torch.inference_mode()
    def predict(features):
        normalized = ((torch.from_numpy(features) - center) / scale).clamp(-8., 8.)
        return model(normalized)[:, 0].sigmoid().numpy()

    development = [
        (threshold, persistence, score(dev, predict, threshold, persistence))
        for threshold in (.5, .65, .75, .85, .92, .97, .99)
        for persistence in ((3, 5, 7, 9) if args.mine_from_gate else (3,))
    ]
    threshold, persistence, dev_result = min(development, key=lambda item: (
        item[2]["false_episodes"] + item[2]["early_episodes"], -item[2]["detected_within_025s"], item[1], item[0],
    ))
    test_result = score(test, predict, threshold, persistence)
    torch.save({"gate_state_dict": model.state_dict(), "center": center, "scale": scale,
                "threshold": threshold, "required_consecutive": persistence},
               args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps({
        "threshold": threshold, "train_frames": len(x), "positive_train_frames": len(positive),
        "required_consecutive": persistence, "hard_negative_episodes": len(hard),
        "dev": dev_result, "test": test_result,
    }, indent=2) + "\n")
    print(json.dumps({"threshold": threshold, **{k: v for k, v in test_result.items() if k != "rows"}}))


if __name__ == "__main__":
    main()
