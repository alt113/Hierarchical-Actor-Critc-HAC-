"""This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.

The below script processes the command-line options specified by the user and
instantiates the environment and agent.
"""
from hac.design_agent_and_env import design_agent_and_env
from hac.options import parse_options
from hac.run_HAC import run_HAC

# Determine training options specified by user.  The full list of available
# options can be found in "options.py" file.
flags = parse_options()

# Instantiate the agent and Mujoco environment.  The designer must assign
# values to the hyperparameters listed in the "design_agent_and_env.py" file.
agent, env = design_agent_and_env(flags)

# Begin training
run_HAC(flags, env, agent)
