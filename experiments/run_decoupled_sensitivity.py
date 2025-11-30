import argparse
import sys
import os
sys.path.append(os.getcwd())

from src.methods.research.decoupled_sensitivity import DecoupledSensitivityProbe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    probe = DecoupledSensitivityProbe(args.model)
    probe.run_benchmark(args.image, "Describe this.")