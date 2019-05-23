import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer


class Environment:
    """TODO

    TODO

    Attributes
    ----------
    name : TODO
        TODO
    model : TODO
        TODO
    sim : TODO
        TODO
    state_dim : TODO
        TODO
    action_dim : TODO
        TODO
    action_bounds : TODO
        TODO
    action_offset : TODO
        TODO
    end_goal_dim : TODO
        TODO
    subgoal_dim : TODO
        TODO
    subgoal_bounds : TODO
        TODO
    project_state_to_end_goal : TODO
        TODO
    project_state_to_subgoal : TODO
        TODO
    subgoal_bounds_symmetric : TODO
        TODO
    subgoal_bounds_offset : TODO
        TODO
    end_goal_thresholds : TODO
        TODO
    subgoal_thresholds : TODO
        TODO
    initial_state_space : TODO
        TODO
    goal_space_train : TODO
        TODO
    goal_space_test : TODO
        TODO
    subgoal_colors : TODO
        TODO
    max_actions : TODO
        TODO
    visualize : TODO
        TODO
    viewer : TODO
        TODO
    num_frames_skip : TODO
        TODO
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
            TODO
        goal_space_train : TODO
            TODO
        goal_space_test : TODO
            TODO
        project_state_to_end_goal : TODO
            TODO
        end_goal_thresholds : TODO
            TODO
        initial_state_space : TODO
            TODO
        subgoal_bounds : TODO
            TODO
        project_state_to_subgoal : TODO
            TODO
        subgoal_thresholds : TODO
            TODO
        max_actions : TODO
            TODO
        num_frames_skip : TODO
            TODO
        show : TODO
            TODO
        """
        self.name = model_name

        # Create Mujoco Simulation
        self.model = load_model_from_path("./mujoco_files/" + model_name)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to
        # configure actor/critic networks
        if model_name == "pendulum.xml":
            self.state_dim = \
                2 * len(self.sim.data.qpos) + len(self.sim.data.qvel)
        else:
            # State will include (i) joint angles and (ii) joint velocities
            self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel)
        # low-level action dim
        self.action_dim = len(self.sim.model.actuator_ctrlrange)
        # low-level action bounds
        self.action_bounds = self.sim.model.actuator_ctrlrange[:, 1]
        # Assumes symmetric low-level action ranges
        self.action_offset = np.zeros((len(self.action_bounds)))
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
                (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
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
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

    def get_state(self):
        """Get state, which concatenates joint positions and velocities."""
        if self.name == "pendulum.xml":
            return np.concatenate(
                [np.cos(self.sim.data.qpos), np.sin(self.sim.data.qpos),
                 self.sim.data.qvel]
            )
        else:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def reset_sim(self):
        """Reset simulation to state within initial state specified by user."""
        # Reset joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(
                self.initial_state_space[i][0], self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(
                self.initial_state_space[len(self.sim.data.qpos) + i][0],
                self.initial_state_space[len(self.sim.data.qpos) + i][1])

        self.sim.step()

        # Return state
        return self.get_state()

    def execute_action(self, action):
        """Execute low-level action.

        This is done for number of frames specified by num_frames_skip.

        Parameters
        ----------
        action : TODO
            TODO

        Returns
        -------
        TODO
            TODO
        """
        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state()

    def display_end_goal(self, end_goal):
        """Visualize end goal.

        This function may need to be adjusted for new environments.

        Parameters
        ----------
        end_goal : TODO
            TODO

        Raises
        ------
        AssertionError
            TODO
        """
        # Goal can be visualized by changing the location of the relevant site
        # object.
        if self.name == "pendulum.xml":
            self.sim.data.mocap_pos[0] = np.array(
                [0.5*np.sin(end_goal[0]), 0, 0.5*np.cos(end_goal[0])+0.6])
        elif self.name == "ur5.xml":
            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
            forearm_pos_3 = np.array([0.425, 0, 0, 1])
            wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

            # Transformation matrix from shoulder to base reference frame
            T_1_0 = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0.089159],
                              [0, 0, 0, 1]])

            # Transformation matrix from upper arm to shoulder reference frame
            T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                              [np.sin(theta_1), np.cos(theta_1), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from forearm to upper arm reference frame
            T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                              [0, 1, 0, 0.13585],
                              [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            T_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                              [0, 1, 0, 0],
                              [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                              [0, 0, 0, 1]])

            # Determine joint position relative to original reference frame
            # shoulder_pos = T_1_0.dot(shoulder_pos_1)
            upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
            forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(
                wrist_1_pos_4)[:3]

            joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

            """
            print("\nEnd Goal Joint Pos: ")
            print("Upper Arm Pos: ", joint_pos[0])
            print("Forearm Pos: ", joint_pos[1])
            print("Wrist Pos: ", joint_pos[2])
            """

            for i in range(3):
                self.sim.data.mocap_pos[i] = joint_pos[i]
        else:
            assert False, \
                "Provide display end goal function in environment.py file"

    def get_next_goal(self, test):
        """Return an end goal.

        Parameters
        ----------
        test : TODO
            TODO

        Returns
        -------
        TODO
            TODO
        """
        end_goal = np.zeros((len(self.goal_space_test)))

        if self.name == "ur5.xml":

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
                T_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                  [0, 0, 1, 0.089159], [0, 0, 0, 1]])

                # Transformation matrix from upper arm to shoulder reference
                # frame
                T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                                  [np.sin(theta_1), np.cos(theta_1), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

                # Transformation matrix from forearm to upper arm reference
                # frame
                T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                                  [0, 1, 0, 0.13585],
                                  [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                                  [0, 0, 0, 1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                                  [0, 1, 0, 0],
                                  [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                                  [0, 0, 0, 1]])

                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(
                    forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(
                    wrist_1_pos_4)[:3]

                # Make sure wrist 1 pos is above ground so can actually be
                # reached
                if np.absolute(end_goal[0]) > np.pi/4 \
                        and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                    goal_possible = True

        elif not test and self.goal_space_train is not None:
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

    def display_subgoals(self, subgoals):
        """Visualize all subgoals.

        Parameters
        ----------
        subgoals : TODO
            TODO
        """
        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1, min(len(subgoals), 11)):
            if self.name == "pendulum.xml":
                self.sim.data.mocap_pos[i] = np.array(
                    [0.5*np.sin(subgoals[subgoal_ind][0]), 0,
                     0.5*np.cos(subgoals[subgoal_ind][0])+0.6])

                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1
            elif self.name == "ur5.xml":
                theta_1 = subgoals[subgoal_ind][0]
                theta_2 = subgoals[subgoal_ind][1]
                theta_3 = subgoals[subgoal_ind][2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
                forearm_pos_3 = np.array([0.425, 0, 0, 1])
                wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0.089159],
                                  [0, 0, 0, 1]])

                # Transformation matrix from upper arm to shoulder reference
                # frame
                T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                                  [np.sin(theta_1), np.cos(theta_1), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

                # Transformation matrix from forearm to upper arm reference
                # frame
                T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                                  [0, 1, 0, 0.13585],
                                  [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                                  [0, 0, 0, 1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                                  [0, 1, 0, 0],
                                  [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                                  [0, 0, 0, 1]])

                # Determine joint position relative to original reference frame
                # shoulder_pos = T_1_0.dot(shoulder_pos_1)
                upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(
                    forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(
                    wrist_1_pos_4)[:3]

                joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

                """
                print("\nSubgoal %d Joint Pos: " % i)
                print("Upper Arm Pos: ", joint_pos[0])
                print("Forearm Pos: ", joint_pos[1])
                print("Wrist Pos: ", joint_pos[2])
                """

                # Designate site position for upper arm, forearm and wrist
                for j in range(3):
                    self.sim.data.mocap_pos[3 + 3*(i-1) + j] = \
                        np.copy(joint_pos[j])
                    self.sim.model.site_rgba[3 + 3*(i-1) + j][3] = 1

                subgoal_ind += 1
            else:
                # Visualize desired gripper position, which is elements 18-21
                # in subgoal vector
                self.sim.data.mocap_pos[i] = subgoals[subgoal_ind]
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1
