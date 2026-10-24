"""Persist Trainer history and render portable training curves."""

import json
from pathlib import Path


def series(log_history, key):
    """Return numeric (step, value) pairs for one Trainer log key."""
    points = []
    for index, entry in enumerate(log_history):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            points.append((entry.get("step", index), value))
    return points


def save_training_artifacts(output_dir, log_history, summary):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_history.json").write_text(
        json.dumps(log_history, indent=2) + "\n", encoding="utf-8"
    )
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = [
        ("loss", "Training loss"),
        ("eval_loss", "Validation loss"),
        ("learning_rate", "Learning rate"),
        ("mean_token_accuracy", "Training token accuracy"),
        ("eval_mean_token_accuracy", "Validation token accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (("loss", "Training"), ("eval_loss", "Validation"), "Loss"),
        (("mean_token_accuracy", "Training"),
         ("eval_mean_token_accuracy", "Validation"), "Token accuracy"),
        (("learning_rate", "Learning rate"), None, "Learning rate"),
    ]
    for axis, panel in zip(axes.flat, panels):
        first, second, title = panel
        for item in (first, second):
            if item is None:
                continue
            points = series(log_history, item[0])
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=item[1])
        axis.set_title(title); axis.set_xlabel("Optimizer step"); axis.grid(alpha=0.25)
        if len(axis.lines) > 1:
            axis.legend()
    axes.flat[3].axis("off")
    details = [f"{label}: {len(series(log_history, key))} points" for key, label in plots]
    axes.flat[3].text(0, 1, "Recorded series\n\n" + "\n".join(details), va="top")
    fig.suptitle("Fine-tuning history")
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
