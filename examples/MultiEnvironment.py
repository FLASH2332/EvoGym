import gymnasium as gym
import numpy as np
import os
from evogym import WorldObject
from stable_baselines3 import PPO
import evogym.envs


# Load the saved controller
def load_controller(controller_path):
    model = PPO.load(controller_path)
    return model

# Load the saved structure
def load_structure(structure_path):
    structure = np.load(structure_path, allow_pickle=True).item()
    return structure

def read_robot_from_file(file_name):
    possible_paths = [
        os.path.join(file_name),
        os.path.join(f'{file_name}.npz'),
        os.path.join(f'{file_name}.json'),
        os.path.join('world_data', file_name),
        os.path.join('world_data', f'{file_name}.npz'),
        os.path.join('world_data', f'{file_name}.json'),
    ]

    best_path = None
    for path in possible_paths:
        if os.path.exists(path):
            best_path = path
            break

    if best_path.endswith('json'):
        robot_object = WorldObject.from_json(best_path)
        return (robot_object.get_structure(), robot_object.get_connections())
    if best_path.endswith('npz'):
        structure_data = np.load(best_path)
        structure = []
        for key, value in structure_data.items():
            structure.append(value)
        return tuple(structure)
    return None

# Set up the robot in the environment (this needs to be customized based on your environment)
def setup_robot(env, structure):
    read_robot_from_file(structure)  # Replace with actual method to set the structure in your environment

# Paths to saved controller and structure
controller_path = r"C:\Users\jayad\Desktop\evorobotics\Scripts\evogym\examples\saved_data\THROWER\generation_0\controller\0.zip"
structure_path = r"C:\Users\jayad\Desktop\evorobotics\Scripts\evogym\examples\saved_data\THROWER\generation_0\structure\0.npz"

# Load the controller and structure
controller = load_controller(controller_path)
structure = load_structure(structure_path)

# Initialize the environment
env_name = 'BridgeWalker-v0'  # Replace with your environment
env = gym.make(env_name)

# Set up the robot in the environment
setup_robot(env, structure)

# Attach the environment to the controller
controller.set_env(env)

# Retrain the robot
total_timesteps = 10000  # Set the number of timesteps you want to train
controller.learn(total_timesteps=total_timesteps)

# Save the retrained controller and structure in the same location
controller.save(r"C:\Users\jayad\Desktop\evorobotics\Scripts\evogym\examples\saved_data\THROWER\generation_0\controller\0_retrained.zip")
np.save(r"C:\Users\jayad\Desktop\evorobotics\Scripts\evogym\examples\saved_data\THROWER\generation_0\structure\0_retrained.npy", structure)
