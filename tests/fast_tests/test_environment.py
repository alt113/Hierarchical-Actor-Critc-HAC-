import unittest
import numpy as np
from hac.environment import Pendulum, UR5


# class TestEnv(unittest.TestCase):
#     """Tests the base environment class."""
#
#     def test_init(self):
#         self.assertEqual(
#             self.env.subgoal_colors,
#             ['Magenta', 'Green', 'Red', 'Blue', 'Cyan', 'Orange', 'Maroon',
#              'Gray', 'White', 'Black']
#         )


class TestUR5(unittest.TestCase):
    """Tests the UR5 environment class."""

    def setUp(self):
        """Create the UR5 environment.

        This follows the ur5.py example located in example_designs/.
        """
        def bound_angle(angle):
            bounded_angle = np.absolute(angle) % (2 * np.pi)
            if angle < 0:
                bounded_angle = -bounded_angle
            return bounded_angle

        def project_state_to_end_goal(sim, *_):
            return np.array([bound_angle(sim.data.qpos[i])
                             for i in range(len(sim.data.qpos))])

        def project_state_to_subgoal(sim, *_):
            return np.concatenate((
                np.array([bound_angle(sim.data.qpos[i])
                          for i in range(len(sim.data.qpos))]),
                np.array([4 if sim.data.qvel[i] > 4 else -4
                          if sim.data.qvel[i] < -4 else sim.data.qvel[i]
                          for i in range(len(sim.data.qvel))])
            ))

        initial_joint_pos = np.array(
            [5.96625837e-03, 3.22757851e-03, -1.27944547e-01])
        initial_joint_pos = np.reshape(
            initial_joint_pos, (len(initial_joint_pos), 1))

        initial_joint_ranges = np.concatenate(
            (initial_joint_pos, initial_joint_pos), 1)
        initial_joint_ranges[0] = np.array([-np.pi / 8, np.pi / 8])

        self.env = UR5(
            model_name="ur5.xml",
            goal_space_train=[(-np.pi, np.pi),
                              (-np.pi/4, 0),
                              (-np.pi/4, np.pi/4)],
            goal_space_test=[(-np.pi, np.pi),
                             (-np.pi/4, 0),
                             (-np.pi/4, np.pi/4)],
            project_state_to_end_goal=project_state_to_end_goal,
            end_goal_thresholds=np.array(
                [np.deg2rad(10) for _ in range(3)]),
            initial_state_space=np.concatenate(
                (initial_joint_ranges,
                 np.zeros((len(initial_joint_ranges), 2))), 0),
            subgoal_bounds=np.array([[-2 * np.pi, 2 * np.pi],
                                     [-2 * np.pi, 2 * np.pi],
                                     [-2 * np.pi, 2 * np.pi],
                                     [-4, 4],
                                     [-4, 4],
                                     [-4, 4]]),
            project_state_to_subgoal=project_state_to_subgoal,
            subgoal_thresholds=np.concatenate(
                (np.array([np.deg2rad(10) for _ in range(3)]),
                 np.array([2 for _ in range(3)]))),
            max_actions=600,
            num_frames_skip=15,
            show=False
        )

    def tearDown(self):
        del self.env

    def test_init(self):
        self.assertEqual(self.env.name, 'ur5.xml')
        self.assertEqual(self.env.observation_space.shape[0], 6)
        self.assertEqual(self.env.action_space.shape[0], 3)
        np.testing.assert_array_almost_equal(
            (self.env.action_space.high - self.env.action_space.low) / 2,
            [3.15, 5.00, 3.15])
        np.testing.assert_array_almost_equal(
            (self.env.action_space.high + self.env.action_space.low) / 2,
            [0.00, 0.00, 0.00])
        self.assertEqual(self.env.end_goal_dim, 3)
        self.assertEqual(self.env.subgoal_dim, 6)
        np.testing.assert_array_almost_equal(self.env.subgoal_bounds,
                                             [[-6.28318531, 6.28318531],
                                              [-6.28318531, 6.28318531],
                                              [-6.28318531, 6.28318531],
                                              [-4, 4], [-4, 4], [-4, 4]])
        np.testing.assert_array_almost_equal(
            self.env.subgoal_bounds_symmetric,
            [6.28318531, 6.28318531, 6.28318531, 4, 4, 4])
        np.testing.assert_array_almost_equal(
            self.env.subgoal_bounds_offset, [0, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(
            self.env.end_goal_thresholds, [0.17453293 for _ in range(3)])
        np.testing.assert_array_almost_equal(
            self.env.subgoal_thresholds,
            [0.17453293, 0.17453293, 0.17453293, 2, 2, 2])
        np.testing.assert_array_almost_equal(
            self.env.initial_state_space,
            [[-0.39269908, 0.39269908], [0.00322758, 0.00322758],
             [-0.12794455, -0.12794455], [0, 0], [0, 0], [0, 0]])
        self.assertEqual(self.env.goal_space_train,
                         [(-np.pi, np.pi), (-np.pi/4, 0), (-np.pi/4, np.pi/4)])
        self.assertEqual(self.env.goal_space_test,
                         [(-np.pi, np.pi), (-np.pi/4, 0), (-np.pi/4, np.pi/4)])
        self.assertEqual(self.env.max_actions, 600)
        self.assertEqual(self.env.visualize, False)
        self.assertEqual(self.env.viewer, None)
        self.assertEqual(self.env.num_frames_skip, 15)

    def test_step(self):
        pass

    def test_reset(self):
        pass

    def test_display_end_goal(self):
        pass

    def test_get_next_goal(self):
        pass

    def test_display_subgoal(self):
        pass


class TestPendulum(unittest.TestCase):
    """Tests the Pendulum environment class."""

    def setUp(self):
        """Create the UR5 environment.

        This follows the pendulum.py example located in example_designs/.
        """
        def bound_angle(angle):
            bounded_angle = angle % (2 * np.pi)
            if np.absolute(bounded_angle) > np.pi:
                bounded_angle = -(np.pi - bounded_angle % np.pi)
            return bounded_angle

        def project_state_to_end_goal(sim, state):
            return np.array([bound_angle(sim.data.qpos[0]), 15 if state[2] > 15
                             else -15 if state[2] < -15 else state[2]])

        def project_state_to_subgoal(sim, state):
            return np.array([bound_angle(sim.data.qpos[0]), 15 if state[2] > 15
                             else -15 if state[2] < -15 else state[2]])

        self.env = Pendulum(
            model_name="pendulum.xml",
            goal_space_train=[(np.deg2rad(-16), np.deg2rad(16)), (-0.6, 0.6)],
            goal_space_test=[(0, 0), (0, 0)],
            project_state_to_end_goal=project_state_to_end_goal,
            end_goal_thresholds=np.array([np.deg2rad(9.5), 0.6]),
            initial_state_space=np.array([[np.pi/4, 7*np.pi/4],
                                          [-0.05, 0.05]]),
            subgoal_bounds=np.array([[-np.pi, np.pi], [-15, 15]]),
            project_state_to_subgoal=project_state_to_subgoal,
            subgoal_thresholds=np.array([np.deg2rad(9.5), 0.6]),
            max_actions=1000,
            num_frames_skip=10,
            show=False
        )

    def tearDown(self):
        del self.env

    def test_init(self):
        self.assertEqual(self.env.name, 'pendulum.xml')
        self.assertEqual(self.env.observation_space.shape[0], 3)
        self.assertEqual(self.env.action_space.shape[0], 1)
        np.testing.assert_array_almost_equal(
            (self.env.action_space.high - self.env.action_space.low) / 2, [2])
        np.testing.assert_array_almost_equal(
            (self.env.action_space.high + self.env.action_space.low) / 2, [0])
        self.assertEqual(self.env.end_goal_dim, 2)
        self.assertEqual(self.env.subgoal_dim, 2)
        np.testing.assert_array_almost_equal(
            self.env.subgoal_bounds, [[-3.14159265 , 3.14159265], [-15, 15.]])
        np.testing.assert_array_almost_equal(
            self.env.subgoal_bounds_symmetric, [3.14159265, 15])
        np.testing.assert_array_almost_equal(
            self.env.subgoal_bounds_offset, [0, 0])
        np.testing.assert_array_almost_equal(
            self.env.end_goal_thresholds, [0.16580628, 0.6])
        np.testing.assert_array_almost_equal(
            self.env.subgoal_thresholds, [0.16580628, 0.6])
        np.testing.assert_array_almost_equal(
            self.env.initial_state_space,
            [[0.78539816, 5.49778714], [-0.05, 0.05]])
        self.assertEqual(
            self.env.goal_space_train,
            [(-0.2792526803190927, 0.2792526803190927), (-0.6, 0.6)])
        self.assertEqual(self.env.goal_space_test, [(0, 0), (0, 0)])
        self.assertEqual(self.env.max_actions, 1000)
        self.assertEqual(self.env.visualize, False)
        self.assertEqual(self.env.viewer, None)
        self.assertEqual(self.env.num_frames_skip, 10)

    def test_step(self):
        pass

    def test_reset(self):
        pass

    def test_display_end_goal(self):
        pass

    def test_get_next_goal(self):
        pass

    def test_display_subgoal(self):
        pass


if __name__ == '__main__':
    unittest.main()
