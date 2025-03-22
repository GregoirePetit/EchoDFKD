import numpy as np
import sys
import os
import argparse
import json
import cv2
from matplotlib import pyplot as plt

core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)

sys.path.append(root_dir)

import echonet_a4c_example
import settings
import torch
import torchvision


def loadvideo(filename: str) -> np.ndarray:
    """from the original repository"""
    assert isinstance(filename, str)
    if not filename[-4:] == settings.VIDEO_EXTENSION:
        filename += settings.VIDEO_EXTENSION
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_width, frame_height, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count] = frame

    v = v.transpose((3, 0, 1, 2))

    return v


def apply_sigmoid_and_convert(npyseg):
    return (torch.sigmoid(torch.Tensor(npyseg)).numpy() * 255).astype(np.uint8)


def load_deeplabV3_model(
    model_weights_path,
    device,
):
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=False, aux_loss=False
    )
    model.classifier[-1] = torch.nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size,
    )
    checkpoint = torch.load(model_weights_path, map_location=device)["state_dict"]
    checkpoint = {".".join(k.split(".")[1:]): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_and_preprocess_video(video_path):
    """load and preprocess for deeplabV3"""
    input_video = np.array(loadvideo(video_path), dtype=np.float32)
    mean = np.array([32.085175, 32.591923, 33.37932], dtype=np.float32)
    std = np.array([50.129288, 50.487717, 51.34619], dtype=np.float32)
    input_video -= mean.reshape(3, 1, 1, 1)
    input_video /= std.reshape(3, 1, 1, 1)
    input_video = torch.Tensor(input_video).float().transpose(0, 1)
    return input_video


def process_one_example(example_name, model, loader, device):
    """process one example"""
    example = echonet_a4c_example.Example(example_name)
    video_path = example.get_video_path()
    video = loader(video_path).to(device)
    output = model(video)["out"]
    return apply_sigmoid_and_convert(output.detach().numpy())


def main(process_one_example, device, example_names, model, loader):
    for example_name in example_names:
        print(example_name)
        output = process_one_example(example_name, model, loader, device)
        output_path = os.path.join(
            output_dir, example_name + settings.TEACHER_MASK_EXTENSION
        )
        np.savez_compressed(output_path, output)


if __name__ == "__main__":
    print("PRODUCE TARGETS ")
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_name", type=str, default="echonet_deeplabV3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Path of a file which contains list of examples on which to infer",
    )
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    teacher_name = args.teacher_name
    device = args.device
    examples = args.examples
    output_dir = args.output_dir

    if examples is None:
        example_names = echonet_a4c_example.test_examples_synthetic
    else:
        with open(examples, "r") as f:
            example_names = [x for x in f.read().split("\n") if x]

    if teacher_name == settings.ORIGINAL_ECHONET_DYNAMIC_TEACHER:
        teacher_name_path = settings.ORIGINAL_ECHONET_DYNAMIC_WEIGHTS
        model = load_deeplabV3_model(teacher_name_path, device)
        loader = load_and_preprocess_video
    else:
        raise Exception("WIP")

    if output_dir is None:
        output_dir = os.path.join(settings.OUTPUT_DIR, teacher_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(process_one_example, device, example_names, model, loader)
