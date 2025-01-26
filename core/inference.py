import numpy as np
import sys
import os
import argparse
import json

core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
convLSTM_dir = os.path.join(root_dir, "ConvLSTM_Segmentation")


assert os.path.isfile(
    os.path.join(convLSTM_dir, "models.py")
), "ConvLSTM_Segmentation/models.py not found, have you initiated the submodule ?"

sys.path.append(root_dir)
sys.path.append(convLSTM_dir)


from models import ConvLSTM2Dlightning

import echonet_a4c_example
import settings
import torch


def apply_sigmoid_and_convert(npyseg):
    return (torch.sigmoid(torch.Tensor(npyseg)).numpy() * 255).astype(np.uint8)


def load_model(
    checkpoint_path,
    num_blocks,
    num_layers_per_block,
    input_shape=(None, settings.IMG_SIZE, settings.IMG_SIZE, 1),
    num_filters=12,
    kernel_size=(3, 3),
    device=torch.device("cpu"),
):
    """
    Redondant with ConvLSTM_Segmentation/inference/model_loader.py but better fits our needs now
    We will modify the code to have only one function
    """
    model = ConvLSTM2Dlightning.load_from_checkpoint(
        checkpoint_path,
        input_shape=input_shape,
        num_filters=num_filters,
        kernel_size=kernel_size,
        num_blocks=num_blocks,
        num_layers_per_block=num_layers_per_block,
    )
    model.eval()
    model.to(device)
    return model


def infer(model, example, device=torch.device("cpu")):
    example = echonet_a4c_example.Example(example)
    inputs = example.get_video()
    inputs = torch.Tensor(inputs)
    inputs = inputs.unsqueeze(0)  # batch of one example
    inputs = inputs.permute(0, 1, 4, 2, 3)
    inputs = inputs.to(device)
    outputs = model(inputs)
    return outputs.detach().numpy()


def main(model, examples, target_dir, device):
    try:
        import tqdm

        g = tqdm.tqdm(examples)
    except:
        g = examples
    for example in g:
        outputs = infer(model, example, device=device)
        output_path = os.path.join(target_dir, example) + ".seg"
        if os.path.isfile(output_path):
            continue
        np.savez_compressed(output_path, apply_sigmoid_and_convert(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--xp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=3,
        help="Number of blocks in the ConvLSTM model.",
    )
    parser.add_argument(
        "--num_layers_per_block",
        type=int,
        default=3,
        help="Number of layers per block in the ConvLSTM model.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Where should we do the computations"
    )
    parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Path of a file which contains list of examples on which to infer",
    )

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    xp_name = args.xp_name
    num_blocks = args.num_blocks
    num_layers_per_block = args.num_layers_per_block
    device = args.device
    examples = args.examples

    if examples is None:
        example_names = echonet_a4c_example.test_examples
    else:
        with open(examples, "r") as f:
            example_names = [x for x in f.read().split("\n") if x]

    model_name = str(num_blocks) + str(num_layers_per_block)
    if checkpoint_path is None:
        """
        by default we guess model file name, for instance models/multi/13.ckpt
        """
        postfix = model_name
        if not xp_name is None:
            postfix = os.path.join(xp_name, postfix)
        if postfix[:-5] != ".ckpt":
            postfix = postfix + ".ckpt"
        checkpoint_path = os.path.join(settings.MODELS_DIR, postfix)

    model = load_model(
        checkpoint_path=checkpoint_path,
        num_blocks=num_blocks,
        num_layers_per_block=num_layers_per_block,
        device=device,
    )

    target_dir = os.path.join(settings.OUTPUT_DIR, xp_name, model_name)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    # Save the weights of the model for later plots
    trainable_params_target_dir = os.path.join(settings.METRICS_DIR, xp_name)
    if not os.path.isdir(trainable_params_target_dir):
        os.makedirs(trainable_params_target_dir)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nweights_file = os.path.join(
        trainable_params_target_dir, f"{xp_name}_{model_name}_weights.json"
    )
    with open(nweights_file, "w") as f:
        json.dump(trainable_params, f)

    main(model=model, examples=example_names, target_dir=target_dir, device=device)
