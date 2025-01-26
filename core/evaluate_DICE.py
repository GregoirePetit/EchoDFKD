import numpy as np
import scipy.signal
import sys
import os

core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
sys.path.append(root_dir)
import echonet_a4c_example


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


if __name__ == "__main__":
    tested_models = ["43"]
    xp_name = "multi/"
    threshold = 80
    phase = "ES"
    loss_function_name = "DICE"
    test_examples = echonet_a4c_example.test_examples
    results = []
    for model_name in tested_models:
        diff_scores = []
        outputs_generator = echonet_a4c_example.yield_outputs(
            xp_name=xp_name,
            model_subname=model_name,
            examples=test_examples,
            phase=phase,
        )
        gt_generator = yield_gt(dataset=test_examples, phase=phase)
        loss_generator = yield_scores(
            gt_loader=gt_generator,
            outputs_loader=outputs_generator,
            loss_function=seg_loss_functions[loss_function_name],
            threshold=threshold,
        )
        for loss in loss_generator:
            diff_scores.append(loss)
        print(np.mean(diff_scores))
