import numpy as np
import sys
import os
import argparse
import json
from pytorch_lightning.callbacks import EarlyStopping


core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
convLSTM_dir = os.path.join(root_dir, "ConvLSTM_Segmentation")


assert os.path.isfile(
    os.path.join(convLSTM_dir, "models.py")
), "ConvLSTM_Segmentation/models.py not found, have you initiated the git submodule ?"

sys.path.append(root_dir)
sys.path.append(convLSTM_dir)


from models import ConvLSTM2Dlightning

import echonet_a4c_example
import settings
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from data_loader import VideoFrameDataset


def main(
    training_examples,
    val_examples,
    num_blocks,
    num_layers_per_block,
    hyperparameters,
    model_name,
    checkpoint_path,
    device,
    skip_if_already_exists,
):

    if os.path.isfile(checkpoint_path) and skip_if_already_exists:
        return None

    hyperparameters_trace_path = checkpoint_path.split(".")[0] + ".hyperparameters"
    with open(hyperparameters_trace_path, "w") as f:
        json.dump(hyperparameters, f)

    """
    LOAD MODEL
    """

    model = ConvLSTM2Dlightning(
        input_shape=hyperparameters["input_shape"],
        num_filters=hyperparameters["num_filters"],
        kernel_size=hyperparameters["kernel_size"],
        num_blocks=num_blocks,
        num_layers_per_block=num_layers_per_block,
    )
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params : ", trainable_params)

    """
    PREPARE DATA
    """

    tr_videos = [
        os.path.join(video_dir, x + settings.VIDEO_EXTENSION) for x in training_examples
    ]
    tr_labels = [
        os.path.join(teacher_dir, x + settings.TEACHER_MASK_EXTENSION)
        for x in training_examples
    ]
    val_videos = [
        os.path.join(video_dir, x + settings.VIDEO_EXTENSION) for x in val_examples
    ]
    val_labels = [
        os.path.join(teacher_dir, x + settings.TEACHER_MASK_EXTENSION)
        for x in val_examples
    ]

    train_gen = VideoFrameDataset(
        tr_videos,
        tr_labels,
        sequence_length=hyperparameters["sequence_length"],
        frames_size="halfed",
    )
    val_gen = VideoFrameDataset(
        val_videos,
        val_labels,
        sequence_length=hyperparameters["sequence_length"],
        frames_size="halfed",
    )

    train_loader = torch.utils.data.DataLoader(
        train_gen, batch_size=hyperparameters["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_gen, batch_size=hyperparameters["batch_size"], shuffle=False
    )

    """
    TRAIN
    """

    logger = TensorBoardLogger("models/", name=f"{xp_name}", version=f"{model_name}")

    callback = ModelCheckpoint(
        dirpath=os.path.join("models/", xp_name),
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        filename=f"{model_name}",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=hyperparameters["early_stopping_patience"],
        mode="min",
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=hyperparameters["max_epochs"],
        devices=1,
        accelerator=device,
        callbacks=[callback, early_stopping_callback],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    """
    RESOLVE ALL CONFIG VARIABLES
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--xp_name", type=str, default="my_xp")
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
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--training_examples",
        type=str,
        default=None,
    )
    parser.add_argument("--val_examples", type=str, default=None)
    parser.add_argument(
        "--teacher_dir", type=str, default=settings.ORIGINAL_OUTPUTS_DIR
    )
    parser.add_argument("--skip_if_already_exists", type=str, default="False")
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default=settings.DEFAULT_HYPERPARAMETERS_PATH,
        help="If the corresponding file .ckpt already exists, we assume the model was already trained and take this file as the result of the call to this python file",
    )

    args = parser.parse_args()
    xp_name = args.xp_name
    num_blocks = args.num_blocks
    num_layers_per_block = args.num_layers_per_block
    device = args.device
    training_examples = args.training_examples
    val_examples = args.val_examples
    teacher_dir = args.teacher_dir
    video_dir = settings.VIDEO_DIR
    hyperparameters_dir = args.hyperparameters
    skip_if_already_exists = args.skip_if_already_exists
    if skip_if_already_exists == "True":
        skip_if_already_exists = True
    else:
        skip_if_already_exists = False

    with open(hyperparameters_dir, "r") as f:
        hyperparameters = json.load(f)

    if training_examples is None:
        training_examples = echonet_a4c_example.train_examples
    else:
        with open(training_examples, "r") as f:
            training_examples = [x for x in f.read().split("\n") if x]

    if val_examples is None:
        val_examples = echonet_a4c_example.val_examples
    else:
        with open(val_examples, "r") as f:
            val_examples = [x for x in f.read().split("\n") if x]

    model_name = str(num_blocks) + str(num_layers_per_block)
    checkpoint_path = os.path.join(settings.MODELS_DIR, xp_name, f"{model_name}.ckpt")

    main(
        training_examples,
        val_examples,
        num_blocks,
        num_layers_per_block,
        hyperparameters,
        model_name,
        checkpoint_path,
        device,
        skip_if_already_exists,
    )
