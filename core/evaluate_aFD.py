import numpy as np
import scipy.signal
import sys
import os

core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
sys.path.append(root_dir)
import echonet_a4c_example


def get_peaks(
    example_aperture_gt,
    minimal_distance_between_peaks=10,
    minimal_peak_prominence=1.5,
):
    syspeaks, _ = scipy.signal.find_peaks(
        -example_aperture_gt,
        distance=minimal_distance_between_peaks,
        prominence=minimal_peak_prominence,
    )
    if len(syspeaks) == 0:
        return [], [], np.argmin(example_aperture_gt)
    ref_peak_indexindex = np.argmin([example_aperture_gt[p] for p in syspeaks])
    ref_peak_index = syspeaks[ref_peak_indexindex]
    diapeaks, _ = scipy.signal.find_peaks(
        example_aperture_gt,
        distance=minimal_distance_between_peaks,
        prominence=minimal_peak_prominence,
    )
    return syspeaks, diapeaks, ref_peak_index


def find_best_index_in_closest_block(mask_size_signal, reference_frame):
    median_val = np.median(mask_size_signal)

    # Find connected blocks of True values in below_median
    below_median = mask_size_signal < median_val
    if not below_median.any():
        below_median = mask_size_signal <= median_val

    blocks = []
    start = None
    for i in range(len(mask_size_signal)):
        if below_median[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                blocks.append((start, i - 1))
                start = None
    if start is not None:
        blocks.append((start, len(mask_size_signal) - 1))

    # Find the block closest to the reference frame
    closest_block = None
    min_distance = float("inf")
    for start, end in blocks:
        distances = [abs(start - reference_frame), abs(end - reference_frame)]
        min_block_distance = min(distances)
        if min_block_distance < min_distance:
            min_distance = min_block_distance
            closest_block = (start, end)

    # Find the index of the minimal value within the closest block
    start, end = closest_block
    min_index_in_block = np.argmin(mask_size_signal[start : end + 1]) + start
    return min_index_in_block


def find_reference_frame(
    example,
    reference,
    evaluated_class,
    reverse,
):
    if reference == "echoclip":
        example_aperture_gt = example.get_xp_aperture_gt()
        example_aperture_gt -= example_aperture_gt.mean()
        example_aperture_gt /= example_aperture_gt.std()
        _, _, ref_peak_index = get_peaks(reverse * example_aperture_gt)
    elif reference == "human":
        human_selected_dia, human_selected_sys = example.get_traced_frames()
        if evaluated_class == "ES":
            ref_peak_index = human_selected_sys
        elif evaluated_class == "ED":
            ref_peak_index = human_selected_dia
    elif reference == "deeplabV3":
        # Get apertures from deeplabV3 model
        deeplab_apertures = example.get_deeplab_aperture()
        if evaluated_class == "ES":
            # Get systolic frames predicted by deeplabV3
            deeplab_sysframes = example.get_deeplab_sys_frames()
            # Get apertures at systolic frames
            sys_apertures = [deeplab_apertures[x] for x in deeplab_sysframes]
            if sys_apertures:
                # Find the frame with minimum aperture (smallest heart volume during systole)
                sys_indexindex = np.argmin(sys_apertures)
                ref_peak_index = deeplab_sysframes[sys_indexindex]
            else:
                # If no systolic frames detected, use the frame with minimum aperture overall
                ref_peak_index = np.argmin(deeplab_apertures)
        elif evaluated_class == "ED":
            # Get diastolic frames predicted by deeplabV3
            deeplab_diaframes = example.get_deeplab_dia_frames()
            # Get apertures at diastolic frames
            dia_apertures = [deeplab_apertures[x] for x in deeplab_diaframes]
            if dia_apertures:
                # Find the frame with maximum aperture (largest heart volume during diastole)
                dia_indexindex = np.argmax(dia_apertures)
                ref_peak_index = deeplab_diaframes[dia_indexindex]
            else:
                # If no diastolic frames detected, use the frame with maximum aperture overall
                ref_peak_index = np.argmax(deeplab_apertures)
    else:
        raise ValueError(f"Unknown reference: {reference}")
    return ref_peak_index


def get_example_apert(
    example,
    tested_model,
    xp_name,
    outputs_threshold,
):
    if tested_model == "echoclip":
        example_apert = example.get_xp_aperture_gt()
    elif tested_model == "deeplabV3":
        example_apert = example.get_deeplab_aperture()
    else:
        xp_outputs = example.get_outputs(xp_name=xp_name, subdir=tested_model)
        example_apert = (xp_outputs > outputs_threshold).sum(axis=(-1, -2))
    example_apert = example_apert - example_apert.mean()
    example_apert_std = example_apert.std()
    if example_apert_std == 0:
        example_apert.fill(0.0)
    else:
        example_apert /= example_apert_std
    return example_apert


def evaluate_model_afd(
    xp_name,
    tested_model,
    reference,
    example_set,
    outputs_threshold,
    shift_parameter=0,
    evaluated_class="ES",
):
    if evaluated_class == "ES":
        reverse = 1
    elif evaluated_class == "ED":
        reverse = -1
    else:
        raise ValueError("evaluated_class must be 'ES' or 'ED'")

    delay_collection = []
    example_collection = []

    for example in example_set:
        example_collection.append(example)
        example = echonet_a4c_example.Example(example)
        ref_peak_index = find_reference_frame(
            example, reference, evaluated_class, reverse
        )
        example_apert = get_example_apert(
            example,
            tested_model,
            xp_name,
            outputs_threshold,
        )
        syst_estimation = find_best_index_in_closest_block(
            reverse * example_apert,
            ref_peak_index,
        )

        # Calculate the delay between the estimated frame and the reference frame
        delay = shift_parameter + syst_estimation - ref_peak_index
        delay_collection.append(delay)

    return np.array(delay_collection), example_collection


if __name__ == "__main__":

    tested_models = ["43"]
    xp_name = "multi/"
    output_threshold = 25

    complete_test_set = echonet_a4c_example.test_examples

    results = []
    for model_name in tested_models:
        results.append(
            evaluate_model_afd(
                xp_name=xp_name,
                tested_model=model_name,
                reference="echoclip",
                outputs_threshold=output_threshold,
                shift_parameter=0,
                evaluated_class="ED",
                example_set=complete_test_set,
            )[0]
        )
    results_ED = np.abs(np.array(results))
    ED = np.mean(results_ED, 1)
    print(ED)
