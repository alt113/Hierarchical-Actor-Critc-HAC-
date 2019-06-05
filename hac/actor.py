"""Contains the Actor object."""
import tensorflow as tf
from hac.utils import layer


class Actor:
    """Base actor class.

    TODO

    Attributes
    ----------
    sess : tf.Session
        the tensorflow session
    action_space_bounds : array_like
        scaling term to the indivudal actions
    action_offset : array_like
        offset term to the indivudal actions
    action_space_size : int
        dimension of the action space by the actor class. Set to the
        environment action dimension if at the lowest level of the hierarchy,
        and to the sub_goal dimension if you are at higher levels.
    actor_name : str
        the default base term within the scope of the actor components in the
        computation graph
    goal_dim : int
        number of elements in the goal vector
    state_dim : int
        number of elements in the environment states
    tau : float
        actor target update rate
    batch_size : int
        SGD batch size
    obs_ph : tf.placeholder
        placeholder for the environment observations
    goal_ph : tf.placeholder
        placeholder from the goals that are provided from the layer that is one
        level above the current layer
    features_ph : tf.placeholder
        feature placeholder, consisting of the placeholders for the states and
        goals
    infer : tf.Tensor
        the actor network
    weights : list of tf.Variable
        trainable parameters of the actor network
    target : tf.Tensor
        the target actor network
    target_weights : list of tf.Variable
        trainable parameters of the target actor network
    update_target_weights : list of tf.Operation
        the processes that are used to perform soft target updates
    action_derivs : tf.placeholder
        TODO
    unnormalized_actor_gradients : TODO
        TODO
    policy_gradient : TODO
        TODO
    train : tf.Operation
        a tensorflow operation for applying the gradients to the parameters of
        the critic
    """

    def __init__(self,
                 sess,
                 env,
                 batch_size,
                 layer_number,
                 flags,
                 learning_rate=0.001,
                 tau=0.05):
        """Instantiate the Actor object.

        Parameters
        ----------
        sess : tf.Session
            the tensorflow session
        env : hac.Environment
            the training environment
        batch_size : int
            SGD batch size
        layer_number : int
            the level of the layer (0 being the lowest)
        flags : argparse.Namespace
            the parsed arguments from the command line (see options.py)
        learning_rate : float, optional
            actor learning rate. Defaults to 0.001
        tau : float, optional
            actor target update rate. Defaults to 0.05
        """
        self.sess = sess

        # Determine range of actor network outputs.  This will be used to
        # configure outer layer of neural network
        if layer_number == 0:
            ac_space = env.action_space
            self.action_space_bounds = (ac_space.high - ac_space.low) / 2
            self.action_offset = (ac_space.high + ac_space.low) / 2
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = env.subgoal_bounds_symmetric
            self.action_offset = env.subgoal_bounds_offset

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.action_space_size = env.action_space.shape[0]
        else:
            self.action_space_size = env.subgoal_dim

        self.actor_name = 'actor_{}'.format(layer_number)

        # Dimensions of goal placeholder will differ depending on layer level
        self.goal_dim = env.end_goal_dim \
            if layer_number == flags.layers - 1 \
            else env.subgoal_dim
        self.state_dim = env.observation_space.shape[0]

        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.obs_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))
        self.features_ph = tf.concat([self.obs_ph, self.goal_ph], axis=1)

        # Create actor network
        self.infer = self.create_nn(self.features_ph)

        # Target network code "re-purposed" from Patrick Emani :^)
        self.weights = [v for v in tf.trainable_variables()
                        if self.actor_name in v.op.name]
        # self.num_weights = len(self.weights)

        # Create target actor network
        self.target = self.create_nn(self.features_ph,
                                     name=self.actor_name+'_target')
        self.target_weights = \
            [v for v in tf.trainable_variables()
             if self.actor_name in v.op.name][len(self.weights):]

        self.update_target_weights = \
            [self.target_weights[i].assign(
                tf.multiply(self.weights[i], self.tau) +
                tf.multiply(self.target_weights[i], 1. - self.tau))
                for i in range(len(self.target_weights))]

        self.action_derivs = tf.placeholder(
            tf.float32, shape=(None, self.action_space_size))
        self.unnormalized_actor_gradients = tf.gradients(
            self.infer, self.weights, -self.action_derivs)
        self.policy_gradient = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.train = tf.train.AdamOptimizer(learning_rate).apply_gradients(
            zip(self.policy_gradient, self.weights))

    def get_action(self, state, goal):
        """Compute the action.

        Parameters
        ----------
        state : array_like
            observation array
        goal : array_like
            goal array

        Returns
        -------
        array_like
            the output action
        """
        return self.sess.run(self.infer, feed_dict={
            self.obs_ph: state,
            self.goal_ph: goal
        })

    def get_target_action(self, state, goal):
        """Compute the action from the target network.

        Parameters
        ----------
        state : array_like
            observation array
        goal : array_like
            goal array

        Returns
        -------
        array_like
            the output action
        """
        return self.sess.run(self.target, feed_dict={
            self.obs_ph: state,
            self.goal_ph: goal
        })

    def update(self, state, goal, action_derivs):
        """Perform an gradient update step to the weights of the actor network.

        Parameters
        ----------
        state : array_like
            observation array
        goal : array_like
            goal array
        action_derivs : TODO
            TODO

        Returns
        -------
        int
            the number of trainable variables
        """
        weights, policy_grad, _ = self.sess.run(
            [self.weights, self.policy_gradient, self.train], feed_dict={
                self.obs_ph: state,
                self.goal_ph: goal,
                self.action_derivs: action_derivs
            }
        )

        return len(weights)

    def create_nn(self, features, name=None):
        """Create the graph for the actor function.

        Parameters
        ----------
        features : tf.placeholder
            the input (feature) placeholder
        name : str, optional
            the base term within the scope of the actor components in the
            computation graph. Defaults to the `actor_name` variable within
            the class.

        Returns
        -------
        tf.Tensor
            the output from the actor network
        """
        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.tanh(fc4) * self.action_space_bounds + self.action_offset

        return output
