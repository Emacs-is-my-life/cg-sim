import sys
import argparse

from sim.core import Simulator

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
args, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args

input_config_path = args.input

if input_config_path is None:
    raise Exception("No simulation input file is supplied.")

sim = Simulator(input_config_path)
sim.run()
