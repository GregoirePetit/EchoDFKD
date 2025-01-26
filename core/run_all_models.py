import subprocess
import os
import argparse
import sys


core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)

sys.path.append(root_dir)


import settings

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--script_path', type=str, default='core/inference.py', help='Path to the main inference script.')
parser.add_argument('--xp_name', type=str, default='multi', help='Experiment name for model checkpoints.')
parser.add_argument('--device', type=str, default='cuda', help='Device to run on, e.g. "cuda" or "cpu".')
parser.add_argument('--max_size', type=int, default=4)
parser.add_argument('--examples', type=str, default=None, help='Path of a file which contains list of examples on which to infer')
args = parser.parse_args()

script_path = args.script_path
xp_name = args.xp_name
device = args.device
max_size = args.max_size
examples = args.examples

# Iterate over all combinations of k (blocks) and n (layers per block) from 1 to max_size
for k in range(1, max_size+1):
    for n in range(1, max_size+1):
        # Construct the default checkpoint filename (e.g., "13.ckpt" for k=1, n=3)
        model_name = f"{k}{n}.ckpt"
        checkpoint_path = os.path.join(settings.MODELS_DIR, xp_name, model_name)
        
        if not os.path.isfile(checkpoint_path):
            print(f"The file {checkpoint_path} does not exist. Please check that your checkpoints are present.")
            continue
        
        # Build the command to execute the main inference script
        cmd = [
            "python3",
            script_path,
            "--checkpoint_path", checkpoint_path,
            "--xp_name", xp_name,
            "--num_blocks", str(k),
            "--num_layers_per_block", str(n),
            "--device", device,
            "--examples", examples
        ]
        
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        
        """ 
        TODO introduce here evaluation command
        """

