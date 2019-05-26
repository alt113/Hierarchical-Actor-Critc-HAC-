import tensorflow as tf
import numpy as np
from hac.utils import layer


class Critic:
    """Base critic class.

    TODO

    Attributes
    ----------
    sess: tf.Session
        the tensorflow session
    critic_name : str
        the default base term within the scope of the critic components in the
        computation graph
    learning_rate : float
        critic learning rate
    gamma : float
        critic discount factor
    tau : float
        critic target update rate
    q_limit : float
        largest valid value by the Q-function
    goal_dim : int
        number of elements in the goal vector
    loss_val : float
        the most recent value from the loss function
    state_dim : int
        number of elements in the environment states
    state_ph : tf.placeholder
        placeholder for the environment states
    goal_ph : tf.placeholder
        placeholder from the goals that are provided from the layer that is one
        level above the current layer
    action_ph : tf.placeholder
        placeholder for the actions performed by the current layer
    features_ph : tf.placeholder
        feature placeholder, consisting of the placeholders for the states,
        goals, and actions
    q_init : float
        Initial Q values
    q_offset : float
        an offset term to the Q function. This critic optimistic initialization
        near q_init
    infer : tf.Tensor
        the critic network
    weights : list of of tf.Variable
        the trainable variables of the critic network
    target : tf.Tensor
        the target critic network
    target_weights : list of of tf.Variable
        the trainable variables of the target critic network
    update_target_weights : list of tf.Operation
        the processes that are used to perform soft target updates
    wanted_qs : TODO
        TODO
    loss : tf.Tensor
        the loss function
    train : tf.Operation
        a tensorflow operation for applying the gradients to the parameters of
        the critic
    gradient : TODO
        TODO
    """

    def __init__(self,
                 sess,
                 env,
                 layer_number,
                 flags,
                 learning_rate=0.001,
                 gamma=0.98,
                 tau=0.05):
        """Instantiate the Critic object.

        Parameters
        ----------
        sess : tf.Session
            the tensorflow session
        env : hac.Environment
            the training environment
        layer_number : int
            the level of the layer (0 being the lowest)
        flags : argparse.Namespace
            the parsed arguments from the command line (see options.py)
        learning_rate : float
            critic learning rate
        gamma : float
            critic discount factor
        tau : float
            critic target update rate
        """
        self.sess = sess
        self.critic_name = 'critic_{}'.format(layer_number)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        self.q_limit = -flags.time_scale

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == flags.layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.loss_val = 0
        self.state_dim = env.state_dim
        self.state_ph = tf.placeholder(
            tf.float32, shape=(None, env.state_dim), name='state_ph')
        self.goal_ph = tf.placeholder(
            tf.float32, shape=(None, self.goal_dim))

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            action_dim = env.action_dim
        else:
            action_dim = env.subgoal_dim

        self.action_ph = tf.placeholder(
            tf.float32, shape=(None, action_dim), name='action_ph')

        self.features_ph = tf.concat([self.state_ph, self.goal_ph,
                                      self.action_ph], axis=1)

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -np.log(self.q_limit/self.q_init - 1)

        # Create critic network graph
        self.infer = self.create_nn(self.features_ph)
        self.weights = [v for v in tf.trainable_variables()
                        if self.critic_name in v.op.name]

        # Create target critic network graph.  Please note that by default the
        # critic networks are not used and updated. To use critic networks
        # please follow instructions in the "update" method in this file and
        # the "learn" method in the "layer.py" file.

        # Target network code "re-purposed" from Patrick Emani :^)
        self.target = self.create_nn(self.features_ph,
                                     name=self.critic_name + '_target')
        self.target_weights = \
            [v for v in tf.trainable_variables()
             if self.critic_name in v.op.name][len(self.weights):]

        self.update_target_weights = \
            [self.target_weights[i].assign(
                tf.multiply(self.weights[i], self.tau)
                + tf.multiply(self.target_weights[i], 1. - self.tau))
                for i in range(len(self.target_weights))]

        self.wanted_qs = tf.placeholder(tf.float32, shape=(None, 1))

        self.loss = tf.reduce_mean(tf.square(self.wanted_qs - self.infer))

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.gradient = tf.gradients(self.infer, self.action_ph)

    def get_q_value(self, state, goal, action):
        """Compute the Q-value.

        Parameters
        ----------
        state : array_like
            observation array
        goal : array_like
            goal array
        action : array_like
            action array

        Returns
        -------
        float
            the output Q-value
        """
        return self.sess.run(self.infer, feed_dict={
            self.state_ph: state,
            self.goal_ph: goal,
            self.action_ph: action
        })[0]

    def get_target_q_value(self, state, goal, action):
        """Compute the Q-value from the target network.

        Parameters
        ----------
        state : array_like
            observation array
        goal : array_like
            goal array
        action : array_like
            action array

        Returns
        -------
        float
            the output Q-value
        """
        return self.sess.run(self.target, feed_dict={
            self.state_ph: state,
            self.goal_ph: goal,
            self.action_ph: action
        })[0]

    def update(self,
               old_states,
               old_actions,
               rewards,
               new_states,
               goals,
               new_actions,
               is_terminals):
        """TODO

        Parameters
        ----------
        old_states : array_like
            TODO
        old_actions : array_like
            TODO
        rewards : array_like
            TODO
        new_states : array_like
            TODO
        goals : array_like
            TODO
        new_actions : array_like
            TODO
        is_terminals : array_like
            TODO
        """
        # Be default, repo does not use target networks. To use target
        # networks, comment out "wanted_qs" line directly below and uncomment
        # next "wanted_qs" line.  This will let the Bellman update use
        # Q(next state, action) from target Q network instead of the regular
        # Q network. Make sure you also make the updates specified in the
        # "learn" method in the "layer.py" file.
        wanted_qs = self.sess.run(self.infer, feed_dict={
            self.state_ph: new_states,
            self.goal_ph: goals,
            self.action_ph: new_actions
        })

        """
        # Uncomment to use target networks
        wanted_qs = self.sess.run(self.target,
                feed_dict={
                    self.state_ph: new_states,
                    self.goal_ph: goals,
                    self.action_ph: new_actions
                })
        """

        for i in range(len(wanted_qs)):
            if is_terminals[i]:
                wanted_qs[i] = rewards[i]
            else:
                wanted_qs[i] = rewards[i] + self.gamma * wanted_qs[i][0]

            # Ensure Q target is within bounds [-self.time_limit,0]
            wanted_qs[i] = max(min(wanted_qs[i], 0), self.q_limit)
            assert 0 >= wanted_qs[i] >= self.q_limit, \
                "Q-Value target not within proper bounds"

        self.loss_val, _ = self.sess.run([self.loss, self.train], feed_dict={
            self.state_ph: old_states,
            self.goal_ph: goals,
            self.action_ph: old_actions,
            self.wanted_qs: wanted_qs
        })

    def get_gradients(self, state, goal, action):
        """Compute the gradients for the critic w.r.t. the loss function.

        Parameters
        ----------
        state : array_like
            observation array
        goal : array_like
            goal array
        action : array_like
            action array

        Returns
        -------
        array_like
            the gradient values for each trainable variable
        """
        grads = self.sess.run(self.gradient, feed_dict={
            self.state_ph: state,
            self.goal_ph: goal,
            self.action_ph: action
        })

        return grads[0]  # TODO: why bother? Isn't this slower?

    def create_nn(self, features, name=None):
        """Create the graph for the critic function.

        The output uses a sigmoid, which bounds the Q-values to between
        [-Policy Length, 0].

        Parameters
        ----------
        features : tf.placeholder
            the input (feature) placeholder
        name : str, optional
            the base term within the scope of the critic components in the
            computation graph. Defaults to the `critic_name` variable within
            the class.

        Returns
        -------
        tf.Tensor
            the output from the critic network
        """
        if name is None:
            name = self.critic_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, 1, is_output=True)

            # A q_offset is used to give the critic function an optimistic
            # initialization near 0
            output = tf.sigmoid(fc4 + self.q_offset) * self.q_limit

        return output
