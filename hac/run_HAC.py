""" This script executes the training schedule for the agent.

By default, the agent will alternate between exploration and testing phases.
The number of episodes in the exploration phase can be configured in section 3
of the design files. If the user prefers to only explore or only test, the user
can enter the command-line options ""--train_only" or "--test", respectively.
The full list of command-line options is available in the "options.py" file.
"""
NUM_BATCH = 1000
TEST_FREQ = 2
NUM_TEST_EPISODES = 100


def run_HAC(flags, env, agent):
    """TODO

    TODO: detailed description

    Parameters
    ----------
    flags : argparse.Namespace
        the parsed arguments from the command line (see options.py)
    env : hac.Environment
        the training environment
    agent : hac.Agent
        the agent class
    """
    # Reset successful episode counter
    successful_episodes = 0

    # Print task summary
    print("\n---------------------")
    print("Task Summary: ", "\n")
    print("Environment: ", env.name)
    print("Number of Layers: ", flags.layers)
    print("Time Limit per Layer: ", flags.time_scale)
    print("Max Episode Time Steps: ", env.max_actions)
    print("Retrain: ", flags.retrain)
    print("Test: ", flags.test)
    print("Visualize: ", flags.show)
    print("---------------------", "\n\n")

    # Determine training mode. If not testing and not solely training,
    # interleave training and testing to track progress
    mix_train_test = False
    if not flags.test and not flags.train_only:
        mix_train_test = True

    for batch in range(NUM_BATCH):
        # Evaluate policy every TEST_FREQ batches if interleaving training and
        # testing
        if mix_train_test and batch % TEST_FREQ == 0:
            print("\n--- TESTING ---")
            agent.flags.test = True
            num_episodes = NUM_TEST_EPISODES
        else:
            num_episodes = agent.other_params["num_exploration_episodes"]

        for episode in range(num_episodes):
            print("\nBatch %d, Episode %d" % (batch, episode))

            # Train for an episode
            success = agent.train(env, episode)

            # Increment successful episode counter if applicable
            if success and mix_train_test and batch % TEST_FREQ == 0:
                successful_episodes += 1

        # Save agent
        agent.save_model(batch)

        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:
            # Log performance
            success_rate = successful_episodes / NUM_TEST_EPISODES * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            agent.log_performance(success_rate, batch)  # FIXME: batch
            agent.flags.test = False
            print("\n--- END TESTING ---\n")
