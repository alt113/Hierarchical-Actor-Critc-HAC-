import unittest
import numpy as np
import tensorflow as tf
# from setup_scripts import setup_ur5
from hac.layer import Layer
from hac.options import parse_options
from tests.fast_tests.setup_scripts import setup_ur5


class TestLayer(unittest.TestCase):
    """Tests the Layer class."""

    def setUp(self):
        tf.reset_default_graph()
        env = setup_ur5()
        flags = parse_options(args=['ur5'])
        self.sess = tf.Session()
        self.layer = Layer(
            env=env,
            sess=self.sess,
            layer_number=0,
            flags=flags,
            agent_params={
                "subgoal_test_perc": 0.3,
                "subgoal_penalty": -flags.time_scale,
                "atomic_noise": [0.1 for _ in range(3)],
                "subgoal_noise": [0.03 for _ in range(6)],
                "episodes_to_store": 500,
                "num_exploration_episodes": 50
            }
        )

    def tearDown(self):
        self.sess.close()
        del self.layer

    def test_init(self):
        self.assertEqual(self.layer.layer_number, 0)
        self.assertEqual(self.layer.time_limit, 600)
        self.assertEqual(self.layer.current_state, None)
        self.assertEqual(self.layer.goal, None)
        self.assertEqual(self.layer.buffer_size_ceiling, 10**7)
        self.assertEqual(self.layer.episodes_to_store, 500)
        self.assertEqual(self.layer.num_replay_goals, 3)
        self.assertEqual(self.layer.trans_per_attempt, 2400)
        self.assertEqual(self.layer.buffer_size, 1200000)
        self.assertEqual(self.layer.batch_size, 1024)
        self.assertEqual(self.layer.temp_goal_replay_storage, [])
        np.testing.assert_array_equal(self.layer.noise_perc, [0.1, 0.1, 0.1])
        self.assertEqual(self.layer.maxed_out, False)
        self.assertEqual(self.layer.subgoal_penalty, -10)

    def test_add_noise(self):
        pass

    def test_get_random_action(self):
        pass

    def test_choose_action(self):
        pass

    def test_perform_action_replay(self):
        pass

    def test_create_prelim_goal_replay_trans(self):
        pass

    def test_get_reward(self):
        pass

    def test_finalize_goal_replay(self):
        pass

    def test_penalize_subgoal(self):
        pass

    def test_return_to_higher_level(self):
        pass

    def test_train(self):
        pass

    def test_learn(self):
        pass


if __name__ == '__main__':
    unittest.main()
