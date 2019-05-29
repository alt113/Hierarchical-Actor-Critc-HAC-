"""Utility methods for the training procedure."""
import tensorflow as tf


def layer(input_layer, num_next_neurons, is_output=False):
    """Create a fully connected layer.

    Parameters
    ----------
    input_layer : tf.placeholder or tf.Tensor
        the input to the neural network layer
    num_next_neurons : int
        the number of output elements from this layer
    is_output : bool, optional
        specifies whether the current layer is an output layer or not. This
        affects how the weights and biases of the layer is initialized, and
        whether a ReLU nonlinearity is added to the output of the layer

    Returns
    -------
    tf.Tensor
        the output from the neural network layer
    """
    num_prev_neurons = int(input_layer.shape[1])
    shape = [num_prev_neurons, num_next_neurons]

    if is_output:
        weight_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        bias_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    else:
        fan_in_init = 1 / num_prev_neurons ** 0.5
        weight_init = tf.random_uniform_initializer(minval=-fan_in_init,
                                                    maxval=fan_in_init)
        bias_init = tf.random_uniform_initializer(minval=-fan_in_init,
                                                  maxval=fan_in_init)

    weights = tf.get_variable("weights", shape, initializer=weight_init)
    biases = tf.get_variable("biases", [num_next_neurons],
                             initializer=bias_init)

    dot = tf.matmul(input_layer, weights) + biases

    if is_output:
        return dot
    else:
        return tf.nn.relu(dot)


def check_validity(model_name,
                   goal_space_train,
                   goal_space_test,
                   end_goal_thresholds,
                   initial_state_space,
                   subgoal_bounds,
                   subgoal_thresholds,
                   max_actions,
                   timesteps_per_action):
    """Ensure environment configurations were properly entered.

    This is done via a sequence of assertions.

    Parameters
    ----------
    model_name : str
        name of the Mujoco model file
    goal_space_train : list of (float, float)
        upper and lower bounds of each element of the goal space during
        training
    goal_space_test : list of (float, float)
        upper and lower bounds of each element of the goal space during
        evaluation
    end_goal_thresholds : array_like
        goal achievement thresholds. If the agent is within the threshold for
        each dimension, the end goal has been achieved and the reward of 0 is
        granted.
    initial_state_space : array_like
        initial values for all elements in the state space
    subgoal_bounds : array_like
        range for each dimension of subgoal space
    subgoal_thresholds : array_like
        subgoal achievement thresholds
    max_actions : int
        maximum number of atomic actions. This will typically be
        flags.time_scale**(flags.layers).
    timesteps_per_action : int
        number of time steps per atomic action
    """
    # Ensure model file is an ".xml" file
    assert model_name[-4:] == ".xml", "Mujoco model must be an \".xml\" file"

    # Ensure upper bounds of range is >= lower bound of range
    if goal_space_train is not None:
        for i in range(len(goal_space_train)):
            assert goal_space_train[i][1] >= goal_space_train[i][0], \
                "In the training goal space, upper bound must be >= lower " \
                "bound"

    if goal_space_test is not None:
        for i in range(len(goal_space_test)):
            assert goal_space_test[i][1] >= goal_space_test[i][0], \
                "In the training goal space, upper bound must be >= lower " \
                "bound"

    for i in range(len(initial_state_space)):
        assert initial_state_space[i][1] >= initial_state_space[i][0], \
            "In initial state space, upper bound must be >= lower bound"

    for i in range(len(subgoal_bounds)):
        assert subgoal_bounds[i][1] >= subgoal_bounds[i][0], \
            "In subgoal space, upper bound must be >= lower bound"

    # Make sure end goal spaces and thresholds have same first dimension
    if goal_space_train is not None and goal_space_test is not None:
        assert len(goal_space_train) == len(goal_space_test) \
               == len(end_goal_thresholds), \
               "End goal space and thresholds must have same first dimension"

    # Makde sure suboal spaces and thresholds have same dimensions
    assert len(subgoal_bounds) == len(subgoal_thresholds), \
        "Subgoal space and thresholds must have same first dimension"

    # Ensure max action and timesteps_per_action are positive integers
    assert max_actions > 0, "Max actions should be a positive integer"

    assert timesteps_per_action > 0, \
        "Timesteps per action should be a positive integer"
