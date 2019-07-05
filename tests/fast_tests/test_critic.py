import unittest
import tensorflow as tf
# from setup_scripts import setup_ur5
from hac.critic import Critic
from hac.options import parse_options
from tests.fast_tests.setup_scripts import setup_ur5


class TestCritic(unittest.TestCase):
    """Tests the Critic class."""

    def setUp(self):
        tf.reset_default_graph()
        env = setup_ur5()
        flags = parse_options(args=['ur5'])
        self.sess = tf.Session()
        self.critic = Critic(
            sess=self.sess,
            env=env,
            layer_number=0,
            flags=flags,
            learning_rate=0.001,
            gamma=0.98,
            tau=0.05
        )

    def tearDown(self):
        self.sess.close()
        del self.critic

    def test_init(self):
        """Validate the initialization method of the Critic class.

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
        self.assertEqual(self.critic.goal_dim, 3)
        self.assertEqual(self.critic.state_dim, 6)

        # tests for class-specific parameters
        self.assertEqual(self.critic.critic_name, "critic_0")
        self.assertEqual(self.critic.learning_rate, 0.001)
        self.assertEqual(self.critic.gamma, 0.98)
        self.assertEqual(self.critic.tau, 0.05)
        self.assertEqual(self.critic.q_limit, -10)
        self.assertEqual(self.critic.loss_val, 0)
        self.assertEqual(self.critic.q_init, -0.067)
        self.assertEqual(self.critic.q_offset, -4.9989252068243895)

        # check the length, shape, and name of the weights in the infer network
        self.assertEqual(len(self.critic.weights), 8)
        self.assertEqual(
            [w.shape for w in self.critic.weights],
            [(12, 64), (64,), (64, 64), (64,), (64, 64), (64,), (64, 1), (1,)]
        )
        self.assertEqual(
            [w.name for w in self.critic.weights],
            ['critic_0_fc_1/weights:0', 'critic_0_fc_1/biases:0',
             'critic_0_fc_2/weights:0', 'critic_0_fc_2/biases:0',
             'critic_0_fc_3/weights:0', 'critic_0_fc_3/biases:0',
             'critic_0_fc_4/weights:0', 'critic_0_fc_4/biases:0']
        )

        # check the length, shape, and name of the weights in the target
        # network
        self.assertEqual(len(self.critic.target_weights), 8)
        self.assertEqual(
            [w.shape for w in self.critic.target_weights],
            [(12, 64), (64,), (64, 64), (64,), (64, 64), (64,), (64, 1), (1,)]
        )
        self.assertEqual(
            [w.name for w in self.critic.target_weights],
            ['critic_0_target_fc_1/weights:0', 'critic_0_target_fc_1/biases:0',
             'critic_0_target_fc_2/weights:0', 'critic_0_target_fc_2/biases:0',
             'critic_0_target_fc_3/weights:0', 'critic_0_target_fc_3/biases:0',
             'critic_0_target_fc_4/weights:0', 'critic_0_target_fc_4/biases:0']
        )

        # check the dimension of features_ph
        self.assertEqual(
            self.critic.features_ph.shape[1],
            self.critic.obs_ph.shape[1] + self.critic.goal_ph.shape[1] +
            self.critic.action_ph.shape[1]
        )

    def test_get_q_value(self):
        pass

    def test_get_target_q_value(self):
        pass

    def test_update(self):
        pass

    def test_get_gradients(self):
        pass


if __name__ == '__main__':
    unittest.main()
