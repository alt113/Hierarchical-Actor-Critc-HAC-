import unittest
import tensorflow as tf
from hac.utils import layer, check_validity


class TestUtils(unittest.TestCase):
    """Tests for the utility method in utils.py."""

    def setUp(self):
        # create a tensorflow session
        self.sess = tf.Session()

    def tearDown(self):
        # close the tensorflow session
        self.sess.close()

    def test_layer(self):
        # create a random placeholder
        input_ph = tf.placeholder(tf.float32, shape=(None, 2))

        # create a layer with is_output set to False
        with tf.variable_scope("layer1"):
            next_layer = layer(input_ph, num_next_neurons=5, is_output=False)

        # create a layer with is_output set to True
        with tf.variable_scope("layer2"):
            final_layer = layer(next_layer, num_next_neurons=1, is_output=True)

        self.sess.run(tf.global_variables_initializer())

        out1, out2 = self.sess.run([next_layer, final_layer],
                                   feed_dict={input_ph: [[0, 1], [2, 3]]})

        # tests the shapes and that the ReLU worked
        self.assertEqual(out1.shape, (2, 5))
        self.assertEqual(out2.shape, (2, 1))
        self.assertTrue((out2 >= 0).all())

    def test_check_validity(self):
        bad_model_name = "bad"
        model_name = "good.xml"

        bad_goal_space_train = [(-1, 1), (2, 1)]
        goal_space_train = [(-1, 1), (1, 2)]

        bad_goal_space_test = [(-1, 1), (2, 1)]
        goal_space_test = [(-1, 1), (1, 2)]

        bad_end_goal_thresholds = [(0, 0)]
        end_goal_thresholds = [(0, 0), (0, 0)]

        bad_initial_state_space = [(-1, 1), (2, 1)]
        initial_state_space = [(-1, 1), (1, 2)]

        bad_subgoal_bounds = [(-1, 1), (2, 1)]
        subgoal_bounds = [(-1, 1), (1, 2)]

        bad_subgoal_thresholds = [(0, 0)]
        subgoal_thresholds = [(0, 0), (0, 0)]

        bad_max_actions = -100
        max_actions = 100

        bad_timesteps_per_action = -100
        timesteps_per_action = 100

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=bad_model_name,
            goal_space_train=goal_space_train,
            goal_space_test=goal_space_test,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=initial_state_space,
            subgoal_bounds=subgoal_bounds,
            subgoal_thresholds=subgoal_thresholds,
            max_actions=max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            goal_space_train=bad_goal_space_train,
            goal_space_test=goal_space_test,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=initial_state_space,
            subgoal_bounds=subgoal_bounds,
            subgoal_thresholds=subgoal_thresholds,
            max_actions=max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            goal_space_train=goal_space_train,
            goal_space_test=bad_goal_space_test,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=initial_state_space,
            subgoal_bounds=subgoal_bounds,
            subgoal_thresholds=subgoal_thresholds,
            max_actions=max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            goal_space_train=goal_space_train,
            goal_space_test=goal_space_test,
            end_goal_thresholds=bad_end_goal_thresholds,
            initial_state_space=initial_state_space,
            subgoal_bounds=subgoal_bounds,
            subgoal_thresholds=subgoal_thresholds,
            max_actions=max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            goal_space_train=goal_space_train,
            goal_space_test=goal_space_test,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=bad_initial_state_space,
            subgoal_bounds=subgoal_bounds,
            subgoal_thresholds=subgoal_thresholds,
            max_actions=max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            goal_space_train=goal_space_train,
            goal_space_test=goal_space_test,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=initial_state_space,
            subgoal_bounds=bad_subgoal_bounds,
            subgoal_thresholds=subgoal_thresholds,
            max_actions=max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            goal_space_train=goal_space_train,
            goal_space_test=goal_space_test,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=initial_state_space,
            subgoal_bounds=subgoal_bounds,
            subgoal_thresholds=bad_subgoal_thresholds,
            max_actions=max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            goal_space_train=goal_space_train,
            goal_space_test=goal_space_test,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=initial_state_space,
            subgoal_bounds=subgoal_bounds,
            subgoal_thresholds=subgoal_thresholds,
            max_actions=bad_max_actions,
            timesteps_per_action=timesteps_per_action
        )

        self.assertRaises(
            AssertionError,
            check_validity,
            model_name=model_name,
            goal_space_train=goal_space_train,
            goal_space_test=goal_space_test,
            end_goal_thresholds=end_goal_thresholds,
            initial_state_space=initial_state_space,
            subgoal_bounds=subgoal_bounds,
            subgoal_thresholds=subgoal_thresholds,
            max_actions=max_actions,
            timesteps_per_action=bad_timesteps_per_action
        )


if __name__ == '__main__':
    unittest.main()
