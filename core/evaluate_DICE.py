import numpy as np
import scipy.signal
import sys
import os
import utils
import argparse


core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
sys.path.append(root_dir)
import echonet_a4c_example
import settings


def yield_gt(dataset, phase):
    """
    For now we select the first label when we find several labelers,
    Notice that we could merge labels instead.
    """
    for example in dataset:
        dia_index, sys_index = echonet_a4c_example.Example(example).get_traced_frames()
        if phase == "ED":
            labels = echonet_a4c_example.Example(example).get_diastol_labels()
        elif phase == "ES":
            labels = echonet_a4c_example.Example(example).get_systol_labels()
        labels = labels[0]  # we select the first labeler
        gt = echonet_a4c_example.mask_from_trace(labels)
        yield gt


def mean_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    iou = intersection / union
    return iou


def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    sum_masks = mask1.sum() + mask2.sum()
    if sum_masks == 0:
        return 1.0 if intersection == 0 else 0.0
    dice = 2 * intersection / sum_masks
    return dice


seg_loss_functions = {"DICE": dice_coefficient, "IoU": mean_iou}


def yield_scores(gt_loader, outputs_loader, loss_function, threshold=None):
    for gt, outputs in zip(gt_loader, outputs_loader):
        if not threshold is None:
            outputs = outputs > threshold
        loss = loss_function(outputs, gt)
        yield loss


def main(
    xp_name,
    tested_model,
    reference,
    example_set,
    metrics_dir=settings.METRICS_DIR,
    threshold=settings.ARBITRARY_THRESHOLD,
):
    if reference == "human":
        reference_EF_ES = yield_gt(dataset=example_set, phase="ES")
        reference_EF_ED = yield_gt(dataset=example_set, phase="ED")
    else:
        raise NotImplementedError  # TODO mettre ici le jugement de la qualité du masque via prompts echoclip

    ED_outputs = echonet_a4c_example.yield_outputs(
        xp_name=xp_name,
        model_subname=tested_model,
        examples=example_set,
        phase="ED",
    )

    ES_outputs = echonet_a4c_example.yield_outputs(
        xp_name=xp_name,
        model_subname=tested_model,
        examples=example_set,
        phase="ES",
    )

    ED_loss_generator = yield_scores(
        reference_EF_ED, ED_outputs, dice_coefficient, threshold
    )
    ES_loss_generator = yield_scores(
        reference_EF_ES, ES_outputs, dice_coefficient, threshold
    )

    ED_DICE = np.mean([example_loss for example_loss in ED_loss_generator])
    ES_DICE = np.mean([example_loss for example_loss in ES_loss_generator])

    results = {"ED_DICE": ED_DICE, "ES_DICE": ES_DICE}

    utils.save_scores(results, xp_name, tested_model, metrics_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Path of a file which contains list of examples on which to infer",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="human",
        help="What brings the values considered as ground truth",
    )
    args = parser.parse_args()

    xp_name = args.xp_name
    model_name = args.model_name
    examples = args.examples
    reference = args.reference

    if examples is None:
        example_names = echonet_a4c_example.test_examples
    else:
        with open(examples, "r") as f:
            example_names = [x for x in f.read().split("\n") if x]

    main(
        xp_name=xp_name,
        tested_model=model_name,
        reference=reference,
        example_set=example_names,
    )
