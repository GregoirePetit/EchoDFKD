import os


file_path = os.path.realpath(__file__)
dir_path = os.path.dirname(file_path)


DATA_DIR = os.path.join(dir_path, "a4c-video-dir")
VIDEO_DIR = os.path.join(DATA_DIR, "Videos")
OUTPUT_DIR = os.path.join(dir_path, "Output")
ORIGINAL_OUTPUTS_DIR = os.path.join(OUTPUT_DIR, "echonet_deeplabV3")

ECHOCLIP_DIR = os.path.join(dir_path, "echoclip")
ECHOCLIP_DIASYS = os.path.join(dir_path, "diasto_systo")


CORRUPTED = os.path.join(dir_path, "data", "corrupted_examples_or_labels")
CURED0 = os.path.join(dir_path, "data", "cured0")

REPAIRED_LABELS = os.path.join(dir_path, "data", "repaired_labels")


APERTURE = os.path.join( dir_path, "output", "segmentation", "deeplabv3_resnet50_random", "size.csv" )
APERTURE_ECHOCLIP = os.path.join(dir_path, "diasys_frames_auto")
