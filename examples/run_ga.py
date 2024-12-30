import random
import numpy as np
import argparse

from ga.run import run_ga
from ppo.args import add_ppo_args

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    parser = argparse.ArgumentParser(description='Arguments for ga script')
    parser.add_argument('--exp-name', type=str, default='blah10', help='Name of the experiment (default: test_ga)')
    parser.add_argument('--env-name-1', type=str, default='Carrier-v0', help='Name of the environment 1 (default: Walker-v0)') # harish add two environments names
    parser.add_argument('--env-name-2', type=str, default='Walker-v0', help='Name of the environment 2 (default: Carrier-v1)')
    parser.add_argument('--pop-size', type=int, default=25, help='Population size (default: 3)')
    parser.add_argument('--structure_shape', type=tuple, default=(5,5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max-evaluations', type=int, default=250, help='Maximum number of robots that will be evaluated (default: 6)')
    parser.add_argument('--num-cores', type=int, default=8, help='Number of robots to evaluate simultaneously (default: 3)')
    add_ppo_args(parser)
    args = parser.parse_args()
    
    run_ga(args)