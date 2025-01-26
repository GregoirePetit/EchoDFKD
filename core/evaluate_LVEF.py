""" 
This experiment is not described in the paper, but it allows us, similar to the one concerning the localization of the frames of interest ES and ED,
to evaluate the model using a score on an external task that is not the one used to train it.
We can verify that, once again, comparing the estimations to those of EchoCLIP allows us to accurately assess the performance of the models (we obtain the same ranking as when we use the ground truth EFs provided by the original EchoNet Dynamic repo).
"""


import numpy as np
import scipy.signal
import sys
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
sys.path.append(root_dir)
import echonet_a4c_example
import settings
from scipy.stats import pearsonr


def yield_ef_gt(dataset):
    for example in dataset:
        yield echonet_a4c_example.Example(example).EF


def corr_coef(x, y):
    corr_coef, _ = pearsonr(x, y)
    return corr_coef


def yield_naive_EF_estimation(
    outputs_generator, threshold=settings.ARBITRARY_THRESHOLD
):
    for ED_o, ES_o in outputs_generator:
        ED_area = (ED_o > threshold).sum()
        ES_area = (ES_o > threshold).sum()
        if ED_area > 0:
            downstream_naive_EF_estimation = (ED_area - ES_area) / ED_area
        else:
            downstream_naive_EF_estimation = 0
        downstream_naive_EF_estimation = 100 * downstream_naive_EF_estimation
        yield downstream_naive_EF_estimation


def get_EF_GT(dataset):
    """
    Load all ground truth ejection fraction values.
    We load only 1 value per example, thus we can load everything into memory at once.
    """
    ef_gt_generator = yield_ef_gt(dataset=dataset)
    return np.array([x for x in ef_gt_generator])


def get_EF_EchoCLIP(dataset):
    """
    Load all EchoCLIP LVEF. In this quick PoC, we load only 1 value per example, so we can load everything into memory at once.
    It would be interesting to load the whole EchoCLIP array for all EF candidate values (instead of loading only the argmax value), and to train a model that learns a mapping between these outputs and a distribution over possible EF.
    """
    return np.array(
        [
            echonet_a4c_example.Example(x).get_echoclip_features()["ejection_fraction"]
            for x in dataset
        ]
    )


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


if __name__ == "__main__":
    model_name = "32"
    xp_name = "multi/"
    test_examples = echonet_a4c_example.test_examples

    all_EF_gt = get_EF_GT(dataset=test_examples)
    all_echoclip_EF = get_EF_EchoCLIP(dataset=test_examples)
    downstream_naive_EF_estimations = get_naive_EF_estimation(
        xp_name=xp_name, model_name=model_name, dataset=test_examples
    )

    print(corr_coef(downstream_naive_EF_estimations, all_echoclip_EF))
