import os


file_path = os.path.realpath(__file__)
dir_path = os.path.dirname(file_path)


DATA_DIR = os.path.join(dir_path, "a4c-video-dir")
VIDEO_DIR_DEFAULT = os.path.join(DATA_DIR, "Videos")
VIDEO_DIR = os.getenv("VIDEO_DIR", VIDEO_DIR_DEFAULT)
OUTPUT_DIR = os.path.join(dir_path, "Output")
MODELS_DIR = os.path.join(dir_path, "models")
ORIGINAL_OUTPUTS_DIR = os.path.join(OUTPUT_DIR, "echonet_deeplabV3")

ECHOCLIP_DIR = os.path.join(dir_path, "echoclip")
ECHOCLIP_DIASYS = os.path.join(dir_path, "diasto_systo")


CORRUPTED = os.path.join(dir_path, "data", "corrupted_examples_or_labels")
CURED0 = os.path.join(dir_path, "data", "cured0")

REPAIRED_LABELS = os.path.join(dir_path, "data", "repaired_labels")


APERTURE = os.path.join( dir_path, "echonet_deeplab_dir", "size.csv" )
EXAMPLES_SUFFIX_IN_APERTURE = ".avi"

APERTURE_ECHOCLIP = os.path.join(dir_path, "data", "diasys_frames_echoclip_auto_test_set")
APERTURE_ECHOCLIP_EXTENSION = ".npy"

volumeTracing_path = os.path.join(dir_path, "a4c-video-dir", "VolumeTracings.csv")
EXAMPLES_SUFFIX_IN_VOLUMETRACING = ".avi"
EFDF_PATH = os.path.join(dir_path, "a4c-video-dir", "FileList.csv")


METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
VISUALS_DIR = os.path.join(OUTPUT_DIR, "visuals")

HYPERPARAMETERS_DIR = os.path.join(dir_path, "hyperparameters")
DEFAULT_HYPERPARAMETERS_PATH = os.path.join(HYPERPARAMETERS_DIR, "default_hyperparameters.json")

VIDEO_EXTENSION = ".avi"
TEACHER_MASK_EXTENSION = ".seg.npz"
TRAINED_MODEL_EXTENSION = ".seg.npz"
MODEL_WEIGHTS_EXTENSION = ".ckpt"

ARBITRARY_THRESHOLD = 80
IMG_SIZE = 112

STANDARD_TRACE_SEGMENTS = 21
