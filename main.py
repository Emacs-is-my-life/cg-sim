import argparse

from sim.core import Simulator

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
args = parser.parse_args()

input_config_path = args.input

if input_config_path is None:
    raise Exception("No simulation input file is supplied.")

sim = Simulator(input_config_path)
sim.run()
