import os
import numpy as np
import pandas as pd
import json
import cv2

import settings



def mask_from_trace(trace, handle_incomplete_labels = False):
    """
    From original repository
    """
    if trace is None:
        return None
    if not handle_incomplete_labels:
        if trace.shape[0] != 21:
            return None
    x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))
    r, c = skimage.draw.polygon(
        np.rint(y).astype(int), np.rint(x).astype(int), (112, 112)
    )
    mask = np.zeros((112, 112), np.float32)
    mask[r, c] = 1
    return mask


fixed_examples = {"systol":{}, "diastol":{}}
for x in os.listdir(settings.REPAIRED_LABELS):
    frame_type = x.split(".")[-3]
    example_name = x.split(".")[0]
    fixed_examples[frame_type].setdefault(example_name, []).append( os.path.join(settings.REPAIRED_LABELS, x))

        
volumetracing_df = pd.read_csv(settings.volumeTracing_path)
volumetracing_dict = (
    volumetracing_df.groupby("FileName")
    .apply(lambda x: x.to_dict(orient="records"))
    .to_dict()
)


EF_df = pd.read_csv(settings.EFDF_PATH)
EF_dict = {row["FileName"]: row.to_dict() for index, row in EF_df.iterrows()}

echonet_deeplab_aperture = pd.read_csv(settings.APERTURE)

def load_video(avi_file):
    cap = cv2.VideoCapture(avi_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to (112, 112)
        frame = cv2.resize(frame, (112, 112))
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
        self.tracing_info_dict = volumetracing_dict[example_name + ".avi"]
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
    def get_video(self):
        """
        load the clip
        """
        video_file = self.example_name
        if not video_file[-4:] == ".avi":
            video_file += ".avi"
        video_path = os.path.join(self.video_dir, video_file)
        return load_video(video_path)[0,...]
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
        if trace.shape[0] <= 21: # single annotator
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
        if self.example_name in fixed_examples["diastol"]: # on vérifie si les labels ne sont pas déjà présents sous leur forme décortiquée
            found_labels = [np.load(x) for x in fixed_examples["diastol"][self.example_name]]
        else:
            diastol_index, _ = self.get_traced_frames()
            found_labels = self.get_traces(diastol_index)
        return found_labels
    def get_systol_labels(self):
        if self.example_name in fixed_examples["systol"]: # on vérifie si les labels ne sont pas déjà présents sous leur forme décortiquée
            found_labels = [np.load(x) for x in fixed_examples["systol"][self.example_name]]
        else:
            _, systol_index = self.get_traced_frames()
            found_labels = self.get_traces(systol_index)
        return found_labels
    def get_echoclip_diasys(self, source_dir = settings.ECHOCLIP_DIASYS):
        file_name = self.example_name + ".json"
        file_path = os.path.join(source_dir, file_name)
        with open(file_path, "r") as f:
            content = json.load(f)
        return content
    def get_echoclip_features(self, source_dir = settings.ECHOCLIP_DIR):
        file_name = self.example_name + ".json"
        dirs = os.listdir(source_dir)
        found_features = {}
        for subdir in dirs:
            partial_file = os.path.join(source_dir, subdir, file_name)
            with open(partial_file, "r") as f:
                content = json.load(f)
            found_features = {**found_features , **content} 
        return found_features
    def get_deeplab_aperture(self):
        """ 
        only available for test examples 
        """
        corresponding_entry = self.example_name + ".avi"
        example_data = echonet_deeplab_aperture[echonet_deeplab_aperture['Filename'] == corresponding_entry]
        area = example_data['Size'].values
        return area
    def get_deeplab_sys_frames(self):
        """ 
        only available for test examples 
        """
        corresponding_entry = self.example_name + ".avi"
        example_data = echonet_deeplab_aperture[echonet_deeplab_aperture['Filename'] == corresponding_entry]
        small = example_data['ComputerSmall'].values
        return [index for index, value in enumerate(small) if value == 1]
    def get_outputs(self, xp_name, subdir = None):
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
            mask = np.load(name + ".seg.npz")["arr_0"]
        elif os.path.isfile(name + "_predictions.npz"):
            mask = np.load(name + "_predictions.npz")["arr_0"]
        elif os.path.isfile(name + ".npz"):
            mask = np.load(name + ".npz")["arr_0"]
        else:
            raise Exception("not found : "+name)
        return mask
    def get_xp_aperture_gt(self):
        """ 
        load echoclip signal about phases 
        """
        target_dir = settings.APERTURE_ECHOCLIP
        target_path = os.path.join(target_dir, self.example_name + "_aperture.npy")
        return np.load(target_path )

