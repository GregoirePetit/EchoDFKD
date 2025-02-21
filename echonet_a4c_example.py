import os
import numpy as np
import pandas as pd
import json
import cv2
import skimage
import settings


def mask_from_trace(trace, handle_incomplete_labels=False):
    """
    From original repository
    """
    if trace is None:
        return None
    if not handle_incomplete_labels:
        if trace.shape[0] != settings.STANDARD_TRACE_SEGMENTS:
            return None
    x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))
    r, c = skimage.draw.polygon(
        np.rint(y).astype(int),
        np.rint(x).astype(int),
        (settings.IMG_SIZE, settings.IMG_SIZE),
    )
    mask = np.zeros((settings.IMG_SIZE, settings.IMG_SIZE), np.float32)
    mask[r, c] = 1
    return mask


fixed_examples = {"systol": {}, "diastol": {}}
for x in os.listdir(settings.REPAIRED_LABELS):
    frame_type = x.split(".")[-3]
    example_name = x.split(".")[0]
    fixed_examples[frame_type].setdefault(example_name, []).append(
        os.path.join(settings.REPAIRED_LABELS, x)
    )


volumetracing_df = pd.read_csv(settings.volumeTracing_path)
volumetracing_dict = (
    volumetracing_df.groupby("FileName")
    .apply(lambda x: x.to_dict(orient="records"))
    .to_dict()
)


EF_df = pd.read_csv(settings.EFDF_PATH)
EF_df_synthetic = pd.read_csv(settings.SYNTHETIC_EFDF_PATH)

EF_dict = {row["FileName"]: row.to_dict() for index, row in EF_df.iterrows()}
EF_dict.update(
    {row["FileName"]: row.to_dict() for index, row in EF_df_synthetic.iterrows()}
)

train_EF = EF_df[EF_df["Split"] == "TRAIN"]
val_EF = EF_df[EF_df["Split"] == "VAL"]
test_EF = EF_df[EF_df["Split"] == "TEST"]
train_examples = train_EF["FileName"].values.tolist()
val_examples = val_EF["FileName"].values.tolist()
test_examples = test_EF["FileName"].values.tolist()
# We remove the 6 examples that have no entry in Volumetrace
train_examples.remove("0X2DC68261CBCC04AE")
train_examples.remove("0X6C435C1B417FDE8A")
train_examples.remove("0X234005774F4CB5CD")
train_examples.remove("0X5515B0BD077BE68A")
train_examples.remove("0X35291BE9AB90FB89")
test_examples.remove("0X5DD5283AC43CCDD1")

train_EF = EF_df[EF_df["Split"] == "TRAIN"]
val_EF = EF_df[EF_df["Split"] == "VAL"]
test_EF = EF_df[EF_df["Split"] == "TEST"]
train_examples = train_EF["FileName"].values.tolist()
val_examples = val_EF["FileName"].values.tolist()
test_examples = test_EF["FileName"].values.tolist()

train_EF_synthetic = EF_df_synthetic[EF_df_synthetic["Split"] == "TRAIN"]
val_EF_synthetic = EF_df_synthetic[EF_df_synthetic["Split"] == "VAL"]
test_EF_synthetic = EF_df_synthetic[EF_df_synthetic["Split"] == "TEST"]
train_examples_synthetic = train_EF_synthetic["FileName"].values.tolist()
val_examples_synthetic = val_EF_synthetic["FileName"].values.tolist()
test_examples_synthetic = test_EF_synthetic["FileName"].values.tolist()

echonet_deeplab_aperture = pd.read_csv(settings.APERTURE)


def load_video(avi_file):
    cap = cv2.VideoCapture(avi_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to (112, 112)
        frame = cv2.resize(frame, (settings.IMG_SIZE, settings.IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame[..., np.newaxis])
    cap.release()
    frames = np.array(frames)[np.newaxis, ...]
    return frames


class Example:
    def __init__(self, example_name, video_dir=settings.VIDEO_DIR):
        self.example_name = example_name
        self.video_dir = video_dir
        # Assign ef_info_dict and tracing_info_dict directly
        self.ef_info_dict = EF_dict[example_name]
        
        example_entry = example_name + settings.EXAMPLES_SUFFIX_IN_VOLUMETRACING
        if example_entry in volumetracing_dict:
            self.tracing_info_dict = volumetracing_dict[example_entry]
            self.traces = self._build_trace_dict()
        # Dynamically set attributes for ef_info
        for key, value in self.ef_info_dict.items():
            setattr(self, key, value)

    def _build_trace_dict(self):
        frame_dict = {}
        for entry in self.tracing_info_dict:
            frame = entry["Frame"]
            if frame not in frame_dict:
                frame_dict[frame] = {"X1": [], "Y1": [], "X2": [], "Y2": []}
            frame_dict[frame]["X1"].append(entry["X1"])
            frame_dict[frame]["Y1"].append(entry["Y1"])
            frame_dict[frame]["X2"].append(entry["X2"])
            frame_dict[frame]["Y2"].append(entry["Y2"])
        return frame_dict

    def __repr__(self):
        return self.example_name

    def get_video_path(self):
        video_file = self.example_name
        extension = settings.VIDEO_EXTENSION
        if not video_file[-len(settings.VIDEO_EXTENSION) :] == settings.VIDEO_EXTENSION:
            video_file += settings.VIDEO_EXTENSION

        for candidate_video_path in settings.VIDEO_PATHS:
            matching_path = os.path.join(candidate_video_path, video_file)
            if os.path.isfile(matching_path):
                video_path = matching_path
                break
        else:
            raise Exception("video path not found")

        return video_path

    def get_video(self):
        """
        load the clip
        """
        video_path = self.get_video_path()
        assert os.path.isfile(video_path), "not found : " + video_path
        return load_video(video_path)[0, ...]

    def get_traced_frames(self):
        """
        Return the indexes of the chosen ES frame and ED frame from Echonet Dynamic human labels
        """
        return list(self.traces.keys())

    def get_traces(self, frame_index):
        """
        Get trace and groups labels that seem to come from the same annotator together."
        """
        trace = self.traces.get(frame_index, None)
        trace = np.column_stack((trace["X1"], trace["Y1"], trace["X2"], trace["Y2"]))
        if trace.shape[0] <= settings.STANDARD_TRACE_SEGMENTS:  # single annotator
            return [trace]
        else:
            raise Exception("shouldn't need to handle multiple annotators any more")

    def get_trace_masks(self, frame_number):
        """
        Provides the masks as they are constructed in the original Echonet Dynamic repository.
        """
        traces = self.get_traces(frame_number)
        return [mask_from_trace(t) for t in traces]

    def get_diastol_labels(self):
        if (
            self.example_name in fixed_examples["diastol"]
        ):  # we check if there's an explicit label file
            found_labels = [
                np.load(x) for x in fixed_examples["diastol"][self.example_name]
            ]
        else:
            diastol_index, _ = self.get_traced_frames()
            found_labels = self.get_traces(diastol_index)
        return found_labels

    def get_systol_labels(self):
        if (
            self.example_name in fixed_examples["systol"]
        ):  # we check if there's an explicit label file
            found_labels = [
                np.load(x) for x in fixed_examples["systol"][self.example_name]
            ]
        else:
            _, systol_index = self.get_traced_frames()
            found_labels = self.get_traces(systol_index)
        return found_labels

    def get_echoclip_diasys(self, source_dir=settings.ECHOCLIP_DIASYS):
        file_name = self.example_name + ".json"
        file_path = os.path.join(source_dir, file_name)
        with open(file_path, "r") as f:
            content = json.load(f)
        return content

    def get_echoclip_features(self, source_dir=settings.ECHOCLIP_DIR):
        """
        Assumes that the outputs of echoclip have been saved in a JSON file.
        Collecte les outputs echoclip de différentes expériences en parcourant les sous-dossiers d'ECHOCLIP_DIR
        """
        file_name = self.example_name + ".json"
        dirs = os.listdir(source_dir)
        found_features = {}
        for subdir in dirs:
            partial_file = os.path.join(source_dir, subdir, file_name)
            with open(partial_file, "r") as f:
                content = json.load(f)
            found_features = {**found_features, **content}
        return found_features

    def get_deeplab_aperture(self):
        """
        only available for test examples
        """
        corresponding_entry = self.example_name + settings.EXAMPLES_SUFFIX_IN_APERTURE
        example_data = echonet_deeplab_aperture[
            echonet_deeplab_aperture["Filename"] == corresponding_entry
        ]
        area = example_data["Size"].values
        return area

    def get_deeplab_sys_frames(self):
        """
        only available for test examples
        """
        corresponding_entry = self.example_name + settings.EXAMPLES_SUFFIX_IN_APERTURE
        example_data = echonet_deeplab_aperture[
            echonet_deeplab_aperture["Filename"] == corresponding_entry
        ]
        small = example_data["ComputerSmall"].values
        return [index for index, value in enumerate(small) if value == 1]

    def get_outputs(self, xp_name, subdir=None):
        """
        Try to load outputs of a specific model from (a subdir of) OUTPUT_DIR
        Tries different formats
        """
        if subdir is not None:
            target_dir = os.path.join(settings.OUTPUT_DIR, xp_name, subdir)
        else:
            target_dir = os.path.join(settings.OUTPUT_DIR, xp_name)
        name = os.path.join(target_dir, self.example_name)
        if os.path.isfile(name + ".seg.npz"):
            content = np.load(name + ".seg.npz")
            if "arr_0" in content:
                mask = content["arr_0"]
            else:
                raise Exception
        elif os.path.isfile(name + "_predictions.npz"):
            mask = np.load(name + "_predictions.npz")["arr_0"]
        elif os.path.isfile(name + ".npz"):
            mask = np.load(name + ".npz")["arr_0"]
        else:
            raise Exception("not found : " + name)
        return mask.squeeze()

    def get_xp_aperture_gt(self):
        """
        load echoclip signal about phases
        """
        target_dir = settings.APERTURE_ECHOCLIP
        target_path = os.path.join(
            target_dir, self.example_name + settings.APERTURE_ECHOCLIP_EXTENSION
        )
        return np.load(target_path)


def yield_outputs(xp_name, model_subname, examples, phase):
    """
    Function for inference.
    """
    for example in examples:
        dia_index, sys_index = Example(example).get_traced_frames()
        xp_outputs = Example(example).get_outputs(xp_name=xp_name, subdir=model_subname)
        if phase == "ED":
            try:
                xp_outputs = xp_outputs[dia_index]
            except IndexError:
                print(
                    "can't open ",
                    dia_index,
                    " nth frame in mask tensor of shape ",
                    xp_outputs.shapef,
                )
                raise IndexError
        elif phase == "ES":
            xp_outputs = xp_outputs[sys_index]
        else:
            raise Exception("phase must be ES or ED")
        yield xp_outputs
