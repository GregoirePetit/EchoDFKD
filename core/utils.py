import json
import os


def save_scores(scores, xp_name, model_name, metrics_dir):
    target_dir = os.path.join(metrics_dir, xp_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    metrics_file = os.path.join(target_dir, f"{xp_name}_{model_name}_metrics.json")

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            existing_metrics = json.load(f)
    else:
        existing_metrics = {}

    existing_metrics.update(scores)

    with open(metrics_file, "w") as f:
        json.dump(existing_metrics, f, indent=4)
