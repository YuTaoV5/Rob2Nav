"""Train and validate a recurrent stair-top detector without base linear velocity."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


class TopDetector(torch.nn.Module):
    def __init__(self, mean, scale, hidden_size=64):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.input = torch.nn.Linear(len(mean), hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, observations, hidden=None):
        features = torch.nn.functional.elu(self.input((observations - self.mean) / self.scale))
        features, hidden = self.gru(features, hidden)
        return self.output(features).squeeze(-1), hidden


def detection_metrics(probabilities, labels, relative_x, threshold, streak):
    rows = []
    for probability, label, x in zip(probabilities, labels, relative_x):
        count = 0
        trigger = None
        for index, value in enumerate(probability):
            count = count + 1 if value >= threshold else 0
            if count >= streak:
                trigger = index - streak + 1
                break
        positive = bool(label.any())
        true_index = int(np.argmax(label)) if positive else None
        trigger_x = float(x[trigger]) if trigger is not None else None
        rows.append({
            "positive": positive, "true_index": true_index, "trigger": trigger,
            "trigger_x": trigger_x,
            "false_early": trigger_x is not None and trigger_x < .35,
            "safe_trigger": trigger_x is not None and trigger_x >= .35,
            "delay_frames": None if trigger is None or true_index is None else trigger - true_index,
        })
    positive_rows = [row for row in rows if row["positive"]]
    delays = [row["delay_frames"] for row in positive_rows if row["delay_frames"] is not None]
    return {
        "threshold": threshold, "streak": streak,
        "positive_episodes": len(positive_rows),
        "triggered": sum(row["trigger"] is not None for row in rows),
        "safe_triggers": sum(row["safe_trigger"] for row in rows),
        "false_early": sum(row["false_early"] for row in rows),
        "positive_detected": sum(row["trigger"] is not None for row in positive_rows),
        "negative_triggered": sum(row["trigger"] is not None for row in rows if not row["positive"]),
        "median_delay_frames": float(np.median(delays)) if delays else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--positive-weight", type=float, default=4.)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=240001)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.train) as archive:
        train_x = archive["observations"].astype(np.float32)
        train_y = archive["labels"].astype(np.float32)
    with np.load(args.validation) as archive:
        validation_x = archive["observations"].astype(np.float32)
        validation_y = archive["labels"].astype(np.uint8)
        validation_position = archive["relative_x"].astype(np.float32)
    if np.max(np.abs(train_x[:, :, :3])) != 0.:
        raise ValueError("Training data unexpectedly contains base linear velocity")
    mean = train_x.mean(axis=(0, 1))
    scale = np.maximum(train_x.std(axis=(0, 1)), 1e-3)
    model = TopDetector(mean, scale, args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    positive_weight = torch.tensor(args.positive_weight, device=device)
    losses = []
    for epoch in range(args.epochs):
        order = np.random.permutation(len(train_x))
        epoch_losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            x = torch.as_tensor(train_x[indices], device=device)
            y = torch.as_tensor(train_y[indices], device=device)
            logits, _ = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, pos_weight=positive_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        print(json.dumps({"epoch": epoch + 1, "loss": losses[-1]}), flush=True)

    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(validation_x), args.batch_size):
            x = torch.as_tensor(validation_x[start:start + args.batch_size], device=device)
            logits, _ = model(x)
            chunks.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(chunks)
    metrics = [
        detection_metrics(probabilities, validation_y, validation_position, threshold, streak)
        for threshold in (.80, .90, .95, .98, .99, .995)
        for streak in (1, 3, 5, 10)
    ]
    ranked = sorted(
        metrics,
        key=lambda row: (row["false_early"], row["negative_triggered"],
                         -row["safe_triggers"], -row["positive_detected"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "hidden_size": args.hidden_size,
        "input_size": train_x.shape[-1],
        "seed": args.seed,
    }, args.output)
    report = {
        "device": str(device), "epochs": args.epochs, "losses": losses,
        "positive_weight": args.positive_weight,
        "base_linear_velocity_max_abs": float(np.max(np.abs(train_x[:, :, :3]))),
        "best": ranked[:8], "all": metrics,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "best": ranked[:8]}))


if __name__ == "__main__":
    main()

