"""Several methods for creating environments, etc...

This allows us to reduce the number of times these features are specified when
creating new tests, as all tests follow approximately the same format.
"""
from hac.environment import UR5
import numpy as np


def setup_ur5():
    """Create a generic UR5 environment.

    This is used to add some random environment as an input parameter to some
    of the tests.

    Returns
    -------
    hac.Environment
        an output environment
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

    env = UR5(
        model_name="ur5.xml",
        goal_space_train=[(-np.pi, np.pi),
                          (-np.pi / 4, 0),
                          (-np.pi / 4, np.pi / 4)],
        goal_space_test=[(-np.pi, np.pi),
                         (-np.pi / 4, 0),
                         (-np.pi / 4, np.pi / 4)],
        project_state_to_end_goal=project_state_to_end_goal,
        end_goal_thresholds=np.array(
            [np.deg2rad(10) for _ in range(3)]),
        initial_state_space=[(-0.39269908, 0.39269908),
                             (0.00322758, 0.00322758),
                             (-0.12794455, -0.12794455),
                             (0, 0), (0, 0), (0, 0)],
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

    return env
