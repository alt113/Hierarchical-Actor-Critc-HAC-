import numpy as np
import gym
from hac.utils import check_validity
from hac.agent import Agent

try:
    import mujoco_py
except ImportError:
    # for testing purposes
    import hac.utils.dummy_mujoco as mujoco_py


class Environment(gym.Env):
    """Base environment class.

    TODO

    Attributes
    ----------
    name : str
        name of the environment; adopted from the name of the model
    model : TODO
        the imported MuJoCo model
    sim : mujoco_py.MjSim
        TODO
    observation_space : gym.spaces.*
        TODO
    action_space : gym.spaces.*
        TODO
    end_goal_dim : int
        TODO
    subgoal_dim : int
        TODO
    subgoal_bounds : array_like
        range for each dimension of subgoal space
    project_state_to_end_goal : function
        function that maps from the state space to the end goal space
    project_state_to_subgoal : function
        state to subgoal projection function
    subgoal_bounds_symmetric : array_like
        TODO
    subgoal_bounds_offset : array_like
        TODO
    end_goal_thresholds : array_like
        goal achievement thresholds. If the agent is within the threshold for
        each dimension, the end goal has been achieved and the reward of 0 is
        granted.
    subgoal_thresholds : array_like
        subgoal achievement thresholds
    initial_state_space : list of (float, float)
        bounds for the initial values for all elements in the state space.
        This is achieved during the reset procedure.
    goal_space_train : list of (float, float)
        upper and lower bounds of each element of the goal space during
        training
    goal_space_test : list of (float, float)
        upper and lower bounds of each element of the goal space during
        evaluation
    subgoal_colors : list of str
        colors that are assigned to the subgoal points during visualization
    max_actions : int
        maximum number of atomic actions. This will typically be
        flags.time_scale**(flags.layers).
    visualize : bool
        specifies whether to render the environment
    viewer : mujoco_py.MjViewer
        a display GUI showing the scene of an MjSim object
    num_frames_skip : int
        number of time steps per atomic action
    num_steps : int
        number of steps since the start of the current rollout
    """

    def __init__(self,
                 model_name,
                 goal_space_train,
                 goal_space_test,
                 project_state_to_end_goal,
                 end_goal_thresholds,
                 initial_state_space,
                 subgoal_bounds,
                 project_state_to_subgoal,
                 subgoal_thresholds,
                 max_actions=1200,
                 num_frames_skip=10,
                 show=False):
        """Instantiate the Environment object.

        Parameters
        ----------
        model_name : str
            name of the xml file in './mujoco_files/' that the model is
            generated from
        goal_space_train : list of (float, float)
            upper and lower bounds of each element of the goal space during
            training
        goal_space_test : list of (float, float)
            upper and lower bounds of each element of the goal space during
            evaluation
        project_state_to_end_goal : function
            function that maps from the state space to the end goal space
        end_goal_thresholds : array_like
            goal achievement thresholds. If the agent is within the threshold
            for each dimension, the end goal has been achieved and the reward
            of 0 is granted.
        initial_state_space : list of (float, float)
            bounds for the initial values for all elements in the state space.
            This is achieved during the reset procedure.
        subgoal_bounds : array_like
            range for each dimension of subgoal space
        project_state_to_subgoal : function
            state to subgoal projection function
        subgoal_thresholds : array_like
            subgoal achievement thresholds
        max_actions : int, optional
            maximum number of atomic actions. Defaults to 1200.
        num_frames_skip : int, optional
            number of time steps per atomic action. Defaults to 10.
        show : bool, optional
            specifies whether to render the environment. Defaults to False.
        """
        # Ensure environment customization have been properly entered.
        check_validity(model_name, goal_space_train, goal_space_test,
                       end_goal_thresholds, initial_state_space,
                       subgoal_bounds, subgoal_thresholds, max_actions,
                       num_frames_skip)

        self.name = model_name

        # Create Mujoco Simulation
        self.model = mujoco_py.load_model_from_path(
            "./mujoco_files/" + model_name)
        self.sim = mujoco_py.MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to
        # configure actor/critic networks
        self.end_goal_dim = len(goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_subgoal = project_state_to_subgoal

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to
        # properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = \
                (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]) / 2
            self.subgoal_bounds_offset[i] = \
                self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # End goal/subgoal thresholds
        self.end_goal_thresholds = end_goal_thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set initial state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = [
            "Magenta", "Green", "Red", "Blue", "Cyan", "Orange", "Maroon",
            "Gray", "White", "Black"]

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.viewer = None
        self.num_frames_skip = num_frames_skip

        self.num_steps = 0

    def get_state(self):
        """Get state, which concatenates joint positions and velocities."""
        raise NotImplementedError

    def reset(self):
        """Reset simulation to state within initial state specified by user.

        Returns
        -------
        array_like
            the initial observation
        """
        # Reset the time counter.
        self.num_steps = 0

        # Reset joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(
                self.initial_state_space[i][0], self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(
                self.initial_state_space[len(self.sim.data.qpos) + i][0],
                self.initial_state_space[len(self.sim.data.qpos) + i][1])

        # Return state
        return self.get_state()

    def step(self, action):
        """Advance the simulation by one step.

        This method executes the low-level action. This is done for number of
        frames specified by num_frames_skip.

        Parameters
        ----------
        action : array_like
            the low level primitive action

        Returns
        -------
        array_like
            the next observation
        float
            reward
        bool
            done mask
        dict
            extra info (set to an empty dictionary by default)
        """
        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            self.num_steps += 1
            if self.visualize:
                self.render()

        # TODO
        """
                Should return 
               (observation, reward, done, info)
        """
        return self.get_state(), None, None, {}

    def display_end_goal(self, end_goal):
        """Visualize end goal.

        The goal can be visualized by changing the location of the relevant
        site object.

        Parameters
        ----------
        end_goal : array_like
            the desired end goals to be displayed
        """
        raise NotImplementedError

    def get_next_goal(self, test):
        """Return an end goal.

        Parameters
        ----------
        test : bool
            False if training, True if performing evaluations

        Returns
        -------
        array_like
            TODO
        """
        raise NotImplementedError

    def display_subgoals(self, subgoals):
        """Visualize all subgoals.

        Parameters
        ----------
        subgoals : array_like
            the subgoals to be displayed (e.g. desired positions)
        """
        raise NotImplementedError

    def render(self, mode='human'):  # TODO: make better
        self.viewer.render()

    """
    ################
    ################
    ################
    """
    # TODO fix documentation of below function
    def get_random_action(self, layer_number):
        """Select random action.

        Parameters
        ----------
        layer_number : hac.Environment
            the training environment

        Returns
        -------
        array_like
            a random action, within the bounds that are specified in the
            environment
        """
        if layer_number == 0:
            action = np.zeros(self.action_space.shape[0])
        else:
            action = np.zeros(self.subgoal_dim)

        # Each dimension of random action should take some value in the
        # dimension's range
        for i in range(len(action)):
            if layer_number == 0:
                ac_space = self.action_space
                action_bounds = (ac_space.high - ac_space.low) / 2
                action_offset = (ac_space.high + ac_space.low) / 2

                action[i] = np.random.uniform(
                    - action_bounds[i] + action_offset[i],
                    + action_bounds[i] + action_offset[i])
            else:
                action[i] = np.random.uniform(
                    self.subgoal_bounds[i][0], self.subgoal_bounds[i][1])

        return action

    def choose_action(self,
                      agent,
                      actor,
                      subgoal_test,
                      layer_number,
                      current_state,
                      goal,
                      noise_perc):
        """Select action using an epsilon-greedy policy.

        Parameters
        ----------
        agent : hac.Agent
            the agent class
        env : hac.Environment
            the training environment
        subgoal_test : bool
            TODO

        Returns
        -------
        array_like
            the action by the agent
        str
            the action type, one of: {"Policy", "Noise Policy", "Random"}
        bool
            specifies whether to perform evaluation on the next subgoal
        """
        # If testing mode or testing subgoals, action is output of actor
        # network without noise
        if agent.flags.test or subgoal_test:
            return actor.get_action(
                np.reshape(current_state,
                           (1, len(current_state))),
                np.reshape(goal, (1, len(goal)))
            )[0], "Policy", subgoal_test

        else:
            if np.random.random_sample() > 0.2:
                # Choose noisy action
                action = self.add_noise(actor.get_action(
                    np.reshape(current_state,
                               (1, len(current_state))),
                    np.reshape(goal, (1, len(goal))))[0],
                                        layer_number,
                                        noise_perc)
                action_type = "Noisy Policy"

            # Otherwise, choose random action
            else:
                action = self.get_random_action(layer_number)
                action_type = "Random"

            # Determine whether to test upcoming subgoal
            if np.random.random_sample() < agent.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False

            return action, action_type, next_subgoal_test

    def add_noise(self,
                  action,
                  layer_number,
                  noise_perc):
        """Add noise to provided action.

        Parameters
        ----------
        action : hac.Agent
            the agent class

        Returns
        -------
        array_like
            the action with noise
        """
        # Noise added will be percentage of range
        if layer_number == 0:
            ac_space = self.action_space
            action_bounds = (ac_space.high - ac_space.low) / 2
            action_offset = (ac_space.high + ac_space.low) / 2
        else:
            action_bounds = self.subgoal_bounds_symmetric
            action_offset = self.subgoal_bounds_offset

        assert len(action) == len(action_bounds), \
            "Action bounds must have same dimension as action"
        assert len(action) == len(noise_perc), \
            "Noise percentage vector must have same dimension as action"

        # Add noise to action and ensure remains within bounds
        for i in range(len(action)):
            action[i] += np.random.normal(
                0, noise_perc[i] * action_bounds[i])

            action[i] = max(min(action[i], action_bounds[i]+action_offset[i]),
                            -action_bounds[i]+action_offset[i])

        return action



class UR5(Environment):
    """TODO

    TODO
    """

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-1, high=1,  # TODO: bounds?
            shape=(len(self.sim.data.qpos) + len(self.sim.data.qvel),)
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-self.sim.model.actuator_ctrlrange[:, 1],
            high=self.sim.model.actuator_ctrlrange[:, 1]
        )

    def get_state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def get_next_goal(self, test):
        end_goal = np.zeros(shape=(self.end_goal_dim,))
        goal_possible = False

        while not goal_possible:
            end_goal = np.zeros(shape=(self.end_goal_dim,))

            end_goal[0] = np.random.uniform(self.goal_space_test[0][0],
                                            self.goal_space_test[0][1])
            end_goal[1] = np.random.uniform(self.goal_space_test[1][0],
                                            self.goal_space_test[1][1])
            end_goal[2] = np.random.uniform(self.goal_space_test[2][0],
                                            self.goal_space_test[2][1])

            # Next need to ensure chosen joint angles result in achievable
            # task (i.e., desired end effector position is above ground)
            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            # upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
            forearm_pos_3 = np.array([0.425, 0, 0, 1])
            wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

            # Transformation matrix from shoulder to base reference frame
            t_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0.089159], [0, 0, 0, 1]])

            # Transformation matrix from upper arm to shoulder reference frame
            t_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                              [np.sin(theta_1), np.cos(theta_1), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from forearm to upper arm reference frame
            t_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                              [0, 1, 0, 0.13585],
                              [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            t_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                              [0, 1, 0, 0],
                              [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                              [0, 0, 0, 1]])

            forearm_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(
                forearm_pos_3)[:3]
            wrist_1_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(t_4_3).dot(
                wrist_1_pos_4)[:3]

            # Make sure wrist 1 pos is above ground so can actually be reached
            if np.absolute(end_goal[0]) > np.pi / 4 \
                    and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                goal_possible = True

        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal

    def display_end_goal(self, end_goal):
        theta_1 = end_goal[0]
        theta_2 = end_goal[1]
        theta_3 = end_goal[2]

        # shoulder_pos_1 = np.array([0,0,0,1])
        upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
        forearm_pos_3 = np.array([0.425, 0, 0, 1])
        wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

        # Transformation matrix from shoulder to base reference frame
        t_1_0 = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0.089159],
                          [0, 0, 0, 1]])

        # Transformation matrix from upper arm to shoulder reference frame
        t_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                          [np.sin(theta_1), np.cos(theta_1), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        # Transformation matrix from forearm to upper arm reference frame
        t_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                          [0, 1, 0, 0.13585],
                          [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                          [0, 0, 0, 1]])

        # Transformation matrix from wrist 1 to forearm reference frame
        t_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                          [0, 1, 0, 0],
                          [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                          [0, 0, 0, 1]])

        # Determine joint position relative to original reference frame
        # shoulder_pos = T_1_0.dot(shoulder_pos_1)
        upper_arm_pos = t_1_0.dot(t_2_1).dot(upper_arm_pos_2)[:3]
        forearm_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(forearm_pos_3)[:3]
        wrist_1_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(t_4_3).dot(
            wrist_1_pos_4)[:3]

        joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

        for i in range(3):
            self.sim.data.mocap_pos[i] = joint_pos[i]

    def display_subgoals(self, subgoals):
        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1, min(len(subgoals), 11)):
            theta_1 = subgoals[subgoal_ind][0]
            theta_2 = subgoals[subgoal_ind][1]
            theta_3 = subgoals[subgoal_ind][2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
            forearm_pos_3 = np.array([0.425, 0, 0, 1])
            wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

            # Transformation matrix from shoulder to base reference frame
            t_1_0 = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0.089159],
                              [0, 0, 0, 1]])

            # Transformation matrix from upper arm to shoulder reference frame
            t_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                              [np.sin(theta_1), np.cos(theta_1), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from forearm to upper arm reference frame
            t_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                              [0, 1, 0, 0.13585],
                              [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            t_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                              [0, 1, 0, 0],
                              [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                              [0, 0, 0, 1]])

            # Determine joint position relative to original reference frame
            # shoulder_pos = T_1_0.dot(shoulder_pos_1)
            upper_arm_pos = t_1_0.dot(t_2_1).dot(upper_arm_pos_2)[:3]
            forearm_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(
                forearm_pos_3)[:3]
            wrist_1_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(t_4_3).dot(
                wrist_1_pos_4)[:3]

            joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

            # Designate site position for upper arm, forearm and wrist
            for j in range(3):
                self.sim.data.mocap_pos[3 + 3 * (i - 1) + j] = \
                    np.copy(joint_pos[j])
                self.sim.model.site_rgba[3 + 3 * (i - 1) + j][3] = 1

            subgoal_ind += 1


class Pendulum(Environment):
    """TODO

    TODO
    """

    @property
    def observation_space(self):
        # State will include (i) joint angles and (ii) joint velocities
        return gym.spaces.Box(
            low=0, high=1,  # TODO: bounds?
            shape=(2 * len(self.sim.data.qpos) + len(self.sim.data.qvel),)
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-self.sim.model.actuator_ctrlrange[:, 1],
            high=self.sim.model.actuator_ctrlrange[:, 1]
        )

    def get_state(self):
        return np.concatenate(
            [np.cos(self.sim.data.qpos), np.sin(self.sim.data.qpos),
             self.sim.data.qvel]
        )

    def get_next_goal(self, test):
        end_goal = np.zeros((len(self.goal_space_test)))

        if not test and self.goal_space_train is not None:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0],
                                                self.goal_space_train[i][1])
        else:
            assert self.goal_space_test is not None, \
                "Need goal space for testing. Set goal_space_test variable " \
                "in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(
                    self.goal_space_test[i][0], self.goal_space_test[i][1])

        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal

    def display_end_goal(self, end_goal):
        self.sim.data.mocap_pos[0] = np.array(
            [0.5 * np.sin(end_goal[0]), 0, 0.5 * np.cos(end_goal[0]) + 0.6])

    def display_subgoals(self, subgoals):
        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1, min(len(subgoals), 11)):
            self.sim.data.mocap_pos[i] = np.array(
                [0.5 * np.sin(subgoals[subgoal_ind][0]), 0,
                 0.5 * np.cos(subgoals[subgoal_ind][0]) + 0.6])

            # Visualize subgoal
            self.sim.model.site_rgba[i][3] = 1
            subgoal_ind += 1
