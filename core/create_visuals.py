from matplotlib import pyplot as plt
import os
import json
import sys
import argparse
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
root_dir_path = os.path.dirname(dir_path)

sys.path.append(root_dir_path)
import settings


def main(xp_name, metrics_dir=settings.METRICS_DIR, visuals_dir=settings.VISUALS_DIR):
    xp_metrics = os.path.join(metrics_dir, xp_name)
    all_infos = {}
    for file_path in os.listdir(xp_metrics):
        model_name = file_path.split("_")[-2]

        if not model_name in all_infos:
            all_infos[model_name] = {}

        with open(os.path.join(xp_metrics, file_path), "r") as f:
            content = json.load(f)

        if "_metrics" in file_path:
            all_infos[model_name]["metrics"] = content
        elif "_weights" in file_path:
            all_infos[model_name]["weights"] = content
        else:
            raise Exception

    sorted_keys = sorted(all_infos.keys())
    weights = [all_infos[key]["weights"] for key in sorted_keys]
    metric_names = list(
        next(iter(all_infos.values()))["metrics"].keys()
    )  # métriques disponibles
    metric_to_XY = {}
    for metric in metric_names:
        metric_values = [all_infos[key]["metrics"][metric] for key in sorted_keys]
        metric_to_XY[metric] = (weights, metric_values)

    target_dir = os.path.join(visuals_dir, xp_name)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    for metric in metric_to_XY:
        (weights, metric_values) = metric_to_XY[metric]
        logweights = np.log(weights)
        logscores = np.log(metric_values)
        plt.figure(figsize=(10, 6))
        plt.scatter(logweights, logscores, label=metric)
        plt.xlabel("Weights")
        plt.ylabel("Score")
        plt.title(f"{metric} function of weight")
        plt.legend()
        plt.grid(False)
        file_path = os.path.join(target_dir, f"{metric}.png")
        plt.savefig(file_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xp_name", type=str, default=None, help="Experiment name.")
    args = parser.parse_args()
    main(args.xp_name)
