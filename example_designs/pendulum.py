"""
This file provides the template for designing the agent and environment.

The below hyperparameters must be assigned to a value for the algorithm to work
properly.
"""

import numpy as np
from hac.environment import Pendulum
from hac.agent import Agent


def design_agent_and_env(flags):
    """Instantiate the Agent and Environment classes for training.

    TODO

    Parameters
    ----------
    flags : argparse.Namespace
        the parsed arguments from the command line (see options.py)

    Returns
    -------
    hac.Agent
        the agent class
    hac.Environment
        the training environment
    """
    # ======================================================================= #
    # Step 1. Design agent                                                    #
    #                                                                         #
    # The key hyperparameters for agent construction are                      #
    #                                                                         #
    #  a. Number of levels in agent hierarchy                                 #
    #  b. Max sequence length in which each policy will specialize            #
    #  c. Max number of atomic actions allowed in an episode                  #
    #  d. Environment timesteps per atomic action                             #
    #                                                                         #
    # See Section 3 of this file for other agent hyperparameters that can be  #
    # configured.                                                             #
    # ======================================================================= #

    # Enter max number of atomic actions. This will typically be
    # flags.time_scale**(flags.layers).
    max_actions = 1000

    # Provide the number of time steps per atomic action.
    timesteps_per_action = 10

    # ======================================================================= #
    # Step 2. Design environment                                              #
    #                                                                         #
    #  a. Designer must provide the original UMDP (S,A,T,G,R).                #
    #     - The S,A,T components can be fulfilled by providing the Mujoco     #
    #       model.                                                            #
    #     - The user must separately specifiy the initial state space.        #
    #     - G can be provided by specifying the end goal space.               #
    #     - R, which by default uses a shortest path {-1,0} reward function,  #
    #       can be implemented by specifying two components: (i) a function   #
    #       that maps the state space to the end goal space and (ii) the end  #
    #       goal achievement thresholds for each dimensions of the end goal.  #
    #                                                                         #
    #  b. In order to convert the original UMDP into a hierarchy of k UMDPs,  #
    #     the designer must also provide                                      #
    #     - The subgoal action space, A_i, for all higher-level UMDPs i > 0   #
    #     - R_i for levels 0 <= i < k-1 (i.e., all levels that try to achieve #
    #       goals in the subgoal space). As in the original UMDP, R_i can be  #
    #       implemented by providing two components: (i) a function that maps #
    #       the state space to the subgoal space and (ii) the subgoal         #
    #       achievement thresholds.                                           #
    #                                                                         #
    #  c. Designer should also provide subgoal and end goal visualization     #
    #     functions in order to show video of training. These can be updated  #
    #     in "display_subgoal" and "display_end_goal" methods in the          #
    #     "environment.py" file.                                              #
    # ======================================================================= #

    # Provide file name of Mujoco model(i.e., "pendulum.xml").  Make sure file
    # is stored in "mujoco_files" folder
    model_name = "pendulum.xml"

    # Provide initial state space consisting of the ranges for all joint angles
    # and velocities. In the inverted pendulum task, we randomly sample from
    # the below initial joint position and joint velocity ranges. These values
    # are then converted to the actual state space, which is
    # [cos(pendulum angle), sin(pendulum angle), pendulum velocity].
    initial_state_space = [(np.pi/4, 7*np.pi/4), (-0.05, 0.05)]

    # Provide end goal space. The code supports two types of end goal spaces if
    # user would like to train on a larger end goal space. If user needs to
    # make additional customizations to the end goals, the "get_next_goal"
    # method in "environment.py" can be updated.

    # In the inverted pendulum environment, the end goal will be the desired
    # joint angle and joint velocity for the pendulum.
    goal_space_train = [(np.deg2rad(-16), np.deg2rad(16)), (-0.6, 0.6)]
    goal_space_test = [(0, 0), (0, 0)]

    # Provide a function that maps from the state space to the end goal space.
    # This is used to determine whether the agent should be given the sparse
    # reward. It is also used for Hindsight Experience Replay to determine
    # which end goal was achieved after a sequence of actions.

    # Supplemental function that converts angle to between [-pi,pi]
    def bound_angle(angle):
        bounded_angle = angle % (2 * np.pi)
        if np.absolute(bounded_angle) > np.pi:
            bounded_angle = -(np.pi - bounded_angle % np.pi)
        return bounded_angle

    def project_state_to_end_goal(sim, state):
        return np.array([bound_angle(sim.data.qpos[0]), 15 if state[2] > 15
                         else -15 if state[2] < -15 else state[2]])

    # Set end goal achievement thresholds. If the agent is within the threshold
    # for each dimension, the end goal has been achieved and the reward of 0 is
    # granted.
    end_goal_thresholds = np.array([np.deg2rad(9.5), 0.6])

    # Provide range for each dimension of subgoal space in order to configure
    # subgoal actor networks. Subgoal space can be the same as the state space
    # or some other projection out of the state space. In our implementation of
    # the inverted pendulum task, the subgoal space is the concatenation
    # pendulum position and velocity. This is slightly different than the state
    # space, which is
    # [cos(pendulum angle), sin(pendulum angle), pendulum velocity].
    subgoal_bounds = np.array([[-np.pi, np.pi], [-15, 15]])

    # Provide state to subgoal projection function.
    def project_state_to_subgoal(sim, state):
        return np.array([bound_angle(sim.data.qpos[0]), 15 if state[2] > 15
                         else -15 if state[2] < -15 else state[2]])

    # Set subgoal achievement thresholds
    subgoal_thresholds = np.array([np.deg2rad(9.5), 0.6])

    # To properly visualize goals, update "display_end_goal" and
    # "display_subgoals" methods in "environment.py"

    # ======================================================================= #
    # Step 3. Set miscellaneous hyperparameters                               #
    #                                                                         #
    # Below are some other agent hyperparameters that can affect results,     #
    # including                                                               #
    #                                                                         #
    #  a. Subgoal testing percentage                                          #
    #  b. Subgoal penalty                                                     #
    #  c. Exploration noise                                                   #
    #  d. Replay buffer size                                                  #
    #                                                                         #
    # For other relevant agent hyperparameters, refer to the "agent.py" and   #
    # "layer.py" files.                                                       #
    # ======================================================================= #

    agent_params = {
        # Define percentage of actions that a subgoal level (i.e. level i > 0)
        # will test subgoal actions
        "subgoal_test_perc": 0.3,

        # Define subgoal penalty for missing subgoal. Please note that by
        # default the Q value target for missed subgoals does not include
        # Q-value of next state (i.e, discount rate = 0). As a result, the
        # Q-value target for missed subgoal just equals penalty. For instance
        # in this 3-level pendulum implementation, if a level proposes a
        # subgoal and misses it, the Q target value for this action would be
        # -10. To incorporate the next state in the penalty, go to the
        # "penalize_subgoal" method in the "layer.py" file.
        "subgoal_penalty": -flags.time_scale,

        # Define exploration noise that is added to both subgoal actions and
        # atomic actions.
        # Noise added is Gaussian N(0, noise_percentage * action_dim_range)
        "atomic_noise": [0.1 for _ in range(1)],
        "subgoal_noise": [0.1 for _ in range(2)],

        # Define number of episodes of transitions to be stored by each level
        # of the hierarchy
        "episodes_to_store": 200,

        # Provide training schedule for agent. Training by default will
        # alternate between exploration and testing. Hyperparameter below
        # indicates number of exploration episodes. Testing occurs for 100
        # episodes. To change number of testing episodes, go to "ran_HAC.py".
        "num_exploration_episodes": 50
    }

    # ======================================================================= #
    # Step 4: Instantiate and return agent and environment.                   #
    # ======================================================================= #

    env = Pendulum(model_name, goal_space_train, goal_space_test,
                   project_state_to_end_goal, end_goal_thresholds,
                   initial_state_space, subgoal_bounds,
                   project_state_to_subgoal, subgoal_thresholds,
                   max_actions, timesteps_per_action, flags.show)

    agent = Agent(flags, env, agent_params)

    return agent, env
