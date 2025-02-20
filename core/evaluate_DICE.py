import numpy as np
import scipy.signal
import sys
import os
import utils
import argparse

# Set up paths for importing modules
core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
sys.path.append(root_dir)
import echonet_a4c_example
import settings

def yield_gt(dataset, phase):
    """
    Generator function to yield ground truth masks for each example in the dataset.
    For now, it selects the first label when multiple labelers are present.
    Args:
        dataset: List of examples.
        phase: String indicating the phase ("ED" or "ES").
    Yields:
        gt: Ground truth mask for the example.
    """
    for example in dataset:
        # Get the indices for diastole and systole frames
        dia_index, sys_index = echonet_a4c_example.Example(example).get_traced_frames()
        # Get the labels based on the phase
        if phase == "ED":
            labels = echonet_a4c_example.Example(example).get_diastol_labels()
        elif phase == "ES":
            labels = echonet_a4c_example.Example(example).get_systol_labels()
        labels = labels[0]  # Select the first labeler
        gt = echonet_a4c_example.mask_from_trace(labels)  # Generate mask from trace
        yield gt

def mean_iou(mask1, mask2):
    """
    Calculate the mean Intersection over Union (IoU) between two masks.
    Args:
        mask1: First mask.
        mask2: Second mask.
    Returns:
        iou: Mean IoU value.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    iou = intersection / union
    return iou

def dice_coefficient(mask1, mask2):
    """
    Calculate the Dice coefficient between two masks.
    Args:
        mask1: First mask.
        mask2: Second mask.
    Returns:
        dice: Dice coefficient value.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    sum_masks = mask1.sum() + mask2.sum()
    if sum_masks == 0:
        return 1.0 if intersection == 0 else 0.0
    dice = 2 * intersection / sum_masks
    return dice

# Dictionary mapping loss function names to their corresponding functions
seg_loss_functions = {"DICE": dice_coefficient, "IoU": mean_iou}

def yield_scores(gt_loader, outputs_loader, loss_function, threshold=None):
    """
    Generator function to yield loss scores for each pair of ground truth and output masks.
    Args:
        gt_loader: Generator for ground truth masks.
        outputs_loader: Generator for output masks.
        loss_function: Function to calculate the loss (e.g., dice_coefficient).
        threshold: Optional threshold to binarize the output masks.
    Yields:
        loss: Loss value for the pair of masks.
    """
    for gt, outputs in zip(gt_loader, outputs_loader):
        if not threshold is None:
            outputs = outputs > threshold  # Apply threshold if provided
        loss = loss_function(outputs, gt)  # Calculate loss
        yield loss

def main(
    xp_name,
    tested_model,
    reference,
    example_set,
    metrics_dir=settings.METRICS_DIR,
    threshold=settings.ARBITRARY_THRESHOLD,
):
    """
    Main function to evaluate the model using Dice coefficient.
    Args:
        xp_name: Experiment name.
        tested_model: Name of the model to be tested.
        reference: Reference type (e.g., "human").
        example_set: List of examples to be evaluated.
        metrics_dir: Directory to save the metrics.
        threshold: Threshold to binarize the output masks.
    Returns:
        results: Dictionary containing the evaluation results.
    """
    if reference == "human":
        # Generate ground truth masks for ED and ES phases
        reference_EF_ES = yield_gt(dataset=example_set, phase="ES")
        reference_EF_ED = yield_gt(dataset=example_set, phase="ED")
    else:
        raise NotImplementedError  # TODO: Implement mask quality judgment via prompts echoclip

    # Generate output masks for ED and ES phases
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

    # Calculate Dice coefficient for ED and ES phases
    ED_loss_generator = yield_scores(
        reference_EF_ED, ED_outputs, dice_coefficient, threshold
    )
    ES_loss_generator = yield_scores(
        reference_EF_ES, ES_outputs, dice_coefficient, threshold
    )

    # Compute mean Dice coefficient for ED and ES phases
    ED_DICE = np.mean([example_loss for example_loss in ED_loss_generator])
    ES_DICE = np.mean([example_loss for example_loss in ES_loss_generator])

    # Store results in a dictionary
    results = {"ED_DICE": ED_DICE, "ES_DICE": ES_DICE}

    # Save the results to a file
    utils.save_scores(results, xp_name, tested_model, metrics_dir)

    return results

if __name__ == "__main__":
    # Parse command-line arguments
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

    # Load example names from file if provided, otherwise use default test examples
    if examples is None:
        example_names = echonet_a4c_example.test_examples
    else:
        with open(examples, "r") as f:
            example_names = [x for x in f.read().split("\n") if x]

    # Run the main evaluation function
    main(
        xp_name=xp_name,
        tested_model=model_name,
        reference=reference,
        example_set=example_names,
    )
