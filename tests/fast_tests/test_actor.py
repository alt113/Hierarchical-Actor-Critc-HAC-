import unittest
import numpy as np
import tensorflow as tf
from setup_scripts import setup_ur5
from hac.actor import Actor
from hac.options import parse_options


class TestActor(unittest.TestCase):
    """Tests the Actor class."""

    def setUp(self):
        tf.reset_default_graph()
        env = setup_ur5()
        flags = parse_options(args=[])
        self.sess = tf.Session()
        self.actor = Actor(
            env=env,
            sess=self.sess,
            batch_size=128,
            layer_number=0,
            flags=flags,
            learning_rate=0.001,
            tau=0.05
        )

    def tearDown(self):
        self.sess.close()
        del self.actor

    def test_init(self):
        """Validate the initialization method of the Actor class.

        This test ensures that the variables of the class are being initialized
        to their current values (given a specific environment and set of input
        parameters).

        This test also implicitly validates that the `create_nn` network is
        functioning properly by identifying that the names and weights of the
        created elements in the graph match what is expected from the
        aforementioned method.
        """
        # some tests to validate that the things that are acquired from the
        # environment are correct
        np.testing.assert_array_almost_equal(
            self.actor.action_space_bounds, [3.15, 5.00, 3.15])
        np.testing.assert_array_almost_equal(
            self.actor.action_offset, [0.00, 0.00, 0.00])
        self.assertEqual(self.actor.action_space_size, 3)
        self.assertEqual(self.actor.goal_dim, 3)
        self.assertEqual(self.actor.state_dim, 6)

        # tests for class-specific parameters
        self.assertEqual(self.actor.actor_name, "actor_0")
        self.assertEqual(self.actor.learning_rate, 0.001)
        self.assertEqual(self.actor.tau, 0.05)
        self.assertEqual(self.actor.batch_size, 128)

        # check the length, shape, and name of the weights in the infer network
        self.assertEqual(len(self.actor.weights), 8)
        self.assertEqual(
            [w.shape for w in self.actor.weights],
            [(9, 64), (64,), (64, 64), (64,), (64, 64), (64,), (64, 3), (3,)]
        )
        self.assertEqual(
            [w.name for w in self.actor.weights],
            ['actor_0_fc_1/weights:0', 'actor_0_fc_1/biases:0',
             'actor_0_fc_2/weights:0', 'actor_0_fc_2/biases:0',
             'actor_0_fc_3/weights:0', 'actor_0_fc_3/biases:0',
             'actor_0_fc_4/weights:0', 'actor_0_fc_4/biases:0']
        )

        # check the length, shape, and name of the weights in the target
        # network
        self.assertEqual(len(self.actor.target_weights), 8)
        self.assertEqual(
            [w.shape for w in self.actor.target_weights],
            [(9, 64), (64,), (64, 64), (64,), (64, 64), (64,), (64, 3), (3,)]
        )
        self.assertEqual(
            [w.name for w in self.actor.target_weights],
            ['actor_0_target_fc_1/weights:0', 'actor_0_target_fc_1/biases:0',
             'actor_0_target_fc_2/weights:0', 'actor_0_target_fc_2/biases:0',
             'actor_0_target_fc_3/weights:0', 'actor_0_target_fc_3/biases:0',
             'actor_0_target_fc_4/weights:0', 'actor_0_target_fc_4/biases:0']
        )

        # check the dimension of features_ph
        self.assertEqual(
            self.actor.obs_ph.shape[1] + self.actor.goal_ph.shape[1],
            self.actor.features_ph.shape[1]
        )

    # def test_get_action(self):
    #     # TODO: test this in get_action / create_nn
    #     print(self.actor.infer)
    #
    # def test_get_target_action(self):
    #     # TODO: test this in get_target_action / create_nn
    #     print(self.actor.target)
    #
    # def test_update(self):
    #     # TODO: test in update?
    #     print(self.actor.update_target_weights)
    #     print(self.actor.action_derivs)
    #     print(self.actor.unnormalized_actor_gradients)
    #     print(self.actor.policy_gradient)
    #     print(self.actor.train)


if __name__ == '__main__':
    unittest.main()
