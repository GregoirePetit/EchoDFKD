import numpy as np
import sys
import os
import argparse
import json

# Define the core directory and root directory paths
core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
convLSTM_dir = os.path.join(root_dir, "ConvLSTM_Segmentation")

# Ensure that the ConvLSTM_Segmentation/models.py file exists
assert os.path.isfile(
    os.path.join(convLSTM_dir, "models.py")
), "ConvLSTM_Segmentation/models.py not found, have you initiated the submodule?"

# Add the root directory and ConvLSTM directory to the system path
sys.path.append(root_dir)
sys.path.append(convLSTM_dir)

# Import necessary modules
from models import ConvLSTM2Dlightning
import echonet_a4c_example
import settings
import torch # type: ignore


def apply_sigmoid_and_convert(npyseg):
    """
    Apply sigmoid activation to the segmentation output and convert it to uint8 format.
    Args:
        npyseg: Numpy array of segmentation output.
    Returns:
        Converted numpy array in uint8 format.
    """
    return (torch.sigmoid(torch.Tensor(npyseg)).numpy() * 255).astype(np.uint8)


def load_model(
    checkpoint_path,
    num_blocks,
    num_layers_per_block,
    input_shape,
    num_filters,
    kernel_size,
    device,
):
    """
    Load the ConvLSTM model from a checkpoint.
    Args:
        checkpoint_path: Path to the model checkpoint.
        num_blocks: Number of blocks in the ConvLSTM model.
        num_layers_per_block: Number of layers per block in the ConvLSTM model.
        input_shape: Shape of the input data.
        num_filters: Number of filters in the ConvLSTM model.
        kernel_size: Size of the convolutional kernel.
        device: Device to load the model on (e.g., 'cuda' or 'cpu').
    Returns:
        Loaded model.
    """
    model = ConvLSTM2Dlightning.load_from_checkpoint(
        checkpoint_path,
        input_shape=input_shape,
        num_filters=num_filters,
        kernel_size=kernel_size,
        num_blocks=num_blocks,
        num_layers_per_block=num_layers_per_block,
    )
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the specified device
    return model


def infer(model, example):
    """
    Perform inference on a single example using the loaded model.
    Args:
        model: Loaded ConvLSTM model.
        example: Example data to perform inference on.
    Returns:
        Inference output as a numpy array.
    """
    example = echonet_a4c_example.Example(example)
    inputs = example.get_video()  # Get the video data from the example
    inputs = torch.Tensor(inputs)  # Convert the inputs to a Torch tensor
    inputs = inputs.unsqueeze(0)  # Add a batch dimension
    inputs = inputs.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to match model input
    inputs = inputs.to(device)  # Move the inputs to the specified device
    outputs = model(inputs)  # Perform inference
    return outputs.detach().numpy()  # Convert the outputs to a numpy array


def main(model, examples, target_dir, device):
    """
    Main function to perform inference on a list of examples and save the results.
    Args:
        model: Loaded ConvLSTM model.
        examples: List of examples to perform inference on.
        target_dir: Directory to save the inference results.
        device: Device to perform inference on (e.g., 'cuda' or 'cpu').
    """
    try:
        import tqdm # type: ignore

        g = tqdm.tqdm(examples)  # Use tqdm for progress bar if available
    except:
        g = examples  # Fallback to plain list if tqdm is not available

    for example in g:
        output_path = (
            os.path.join(target_dir, example) + settings.TRAINED_MODEL_EXTENSION
        )
        if os.path.isfile(output_path):
            continue  # Skip if the output file already exists
        outputs = infer(model, example)  # Perform inference
        np.savez_compressed(
            output_path, apply_sigmoid_and_convert(outputs)
        )  # Save the results


if __name__ == "__main__":
    # Parse command-line arguments
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

    # Load example names from file if provided, otherwise use default test examples
    if examples is None:
        example_names = echonet_a4c_example.test_examples
    else:
        with open(examples, "r") as f:
            example_names = [x for x in f.read().split("\n") if x]

    model_name = str(num_blocks) + str(num_layers_per_block)
    if checkpoint_path is None:
        """
        By default, guess model file name, for instance models/multi/13.ckpt
        """
        extension = settings.MODEL_WEIGHTS_EXTENSION
        postfix = model_name
        if not xp_name is None:
            postfix = os.path.join(xp_name, postfix)
        if postfix[: -len(extension)] != extension:
            postfix = postfix + extension
        checkpoint_path = os.path.join(settings.MODELS_DIR, postfix)

    # Load hyperparameters from the corresponding file
    hyperparameters_path = checkpoint_path.split(".")[0] + ".hyperparameters"
    with open(hyperparameters_path, "r") as f:
        hyperparameters = json.load(f)

    # Load the model with the specified hyperparameters
    model = load_model(
        checkpoint_path=checkpoint_path,
        num_blocks=num_blocks,
        num_layers_per_block=num_layers_per_block,
        input_shape=hyperparameters["input_shape"],
        num_filters=hyperparameters["num_filters"],
        kernel_size=hyperparameters["kernel_size"],
        device=device,
    )

    # Define the target directory to save the inference results
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

    # Run the main inference function
    main(model=model, examples=example_names, target_dir=target_dir, device=device)
