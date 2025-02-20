# Import necessary modules and libraries
import utils
import numpy as np
import sys
import os
from scipy.stats import pearsonr
import argparse

# Set up directory paths
core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
sys.path.append(root_dir)

# Import custom modules
import echonet_a4c_example
import settings

# Function to yield ground truth ejection fraction (EF) values from the dataset
def yield_ef_gt(dataset):
    for example in dataset:
        yield echonet_a4c_example.Example(example).EF

# Function to calculate the Pearson correlation coefficient between two arrays
def corr_coef(x, y):
    corr_coef, _ = pearsonr(x, y)
    return corr_coef

# Function to yield naive EF estimations based on mask outputs
def yield_naive_EF_estimation(outputs_generator, threshold=settings.ARBITRARY_THRESHOLD):
    for ED_o, ES_o in outputs_generator:
        ED_area = (ED_o > threshold).sum()
        ES_area = (ES_o > threshold).sum()
        if ED_area > 0:
            downstream_naive_EF_estimation = (ED_area - ES_area) / ED_area
        else:
            downstream_naive_EF_estimation = 0
        downstream_naive_EF_estimation = 100 * downstream_naive_EF_estimation
        yield downstream_naive_EF_estimation

# Function to get all ground truth EF values from the dataset
def get_EF_GT(dataset):
    """
    Load all ground truth ejection fraction values.
    We load only 1 value per example, thus we can load everything into memory at once.
    """
    ef_gt_generator = yield_ef_gt(dataset=dataset)
    return np.array([x for x in ef_gt_generator])

# Function to get all EchoCLIP EF values from the dataset
def get_EF_EchoCLIP(dataset):
    """
    Load all EchoCLIP LVEF. In this quick PoC, we load only 1 value per example, so we can load everything into memory at once.
    It would be interesting to load the whole EchoCLIP array for all EF candidate values (instead of loading only the argmax value), and to train a model that learns a mapping between these outputs and a distribution over possible EF.
    """
    return np.array([
        echonet_a4c_example.Example(x).get_echoclip_features()["ejection_fraction"]
        for x in dataset
    ])

# Function to get naive EF estimations from mask outputs
def get_naive_EF_estimation(xp_name, model_name, dataset):
    """
    Load all estimations of EF from mask outputs.
    We load only one value per example, thus we can load everything into memory at once.
    Notice that we could add a learnable layer, or even finetune the encoder model to make it predict the LVEF, if we wanted better estimations
    But this is not the goal here ; we instead want to evaluate the model's knowledge with external criteria.
    Estimating the fraction by comparing ED mask area with ES mask area is quite naive, but we actually cherish this naivety
    """
    outputs_generator_ED = echonet_a4c_example.yield_outputs(
        xp_name=xp_name,
        model_subname=model_name,
        examples=dataset,
        phase="ED",
    )
    outputs_generator_ES = echonet_a4c_example.yield_outputs(
        xp_name=xp_name,
        model_subname=model_name,
        examples=dataset,
        phase="ES",
    )
    naive_EF_estimation_generator = yield_naive_EF_estimation(
        zip(outputs_generator_ED, outputs_generator_ES)
    )
    downstream_naive_EF_estimations = [x for x in naive_EF_estimation_generator]
    return downstream_naive_EF_estimations

# Main function to evaluate the model
def main(xp_name, tested_model, reference, example_set, metrics_dir=settings.METRICS_DIR):
    if reference == "echoclip":
        reference_EF = get_EF_EchoCLIP(dataset=example_set)
    elif reference == "ground_truth":
        reference_EF = get_EF_GT(dataset=example_set)
    else:
        raise NotImplementedError  # TODO targets from other model estimations

    tested_model_estimations = get_naive_EF_estimation(
        xp_name, tested_model, example_set
    )

    results = {"LVEF_correlation": corr_coef(tested_model_estimations, reference_EF)}

    utils.save_scores(results, xp_name, tested_model, metrics_dir)

    return results

# Entry point of the script
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
        default="ground_truth",
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
