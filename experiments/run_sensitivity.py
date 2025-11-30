import argparse
import sys
import os
sys.path.append(os.getcwd())

from src.methods.research.sensitivity_probe import SensitivityProbe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (e.g., 4 or 8)")
    args = parser.parse_args()

    probe = SensitivityProbe(args.model)
    probe.run_benchmark(args.image, "Describe this.", n_bits=args.bits)