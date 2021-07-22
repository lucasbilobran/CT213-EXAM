import gym
import numpy as np
import os
from utils import reward_engineering
import matplotlib.pyplot as plt
from sarsa_agent import Sarsa, QLearning, greedy_action


NUM_EPISODES = 50000 # Number of episodes used for training
RENDER = False  # If the Environment should be rendered
rom = 'CartPole-v1'
#rom = 'MountainCar-v0'
#rom = 'Assault-ram-v0'

fig_format = 'png'
# fig_format = 'eps'
# fig_format = 'svg'


# Initiating the Environment
env = gym.make(rom)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
epsilon = 0.1  # epsilon of epsilon-greedy
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor

# Creating the Sarsa agent
agent = Sarsa(rom, state_size, action_size, epsilon, alpha, gamma)

# Checking if weights from previous learning session exists
if os.path.exists('../models/SARSA-' + rom + '.h5'):
    print('Loading weights from previous learning session.')
    agent.load('../models/SARSA-' + rom + '.h5')
else:
    print('No weights found from previous learning session.')

return_history = []

# training
for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state = env.reset()
    # This reshape is needed to keep compatibility with Keras
    state = np.reshape(state, [1, state_size])
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    # Select action
    action = agent.get_exploratory_action(agent.map_state_to_table(state[0]))
    for mytime in range(1, 500):
        if RENDER:
            env.render()  # Render the environment for visualization
        # Take action, observe reward and new state
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        # Making reward engineering to allow faster training
        reward = reward_engineering(state[0], action, reward, next_state[0], done, mytime)
        next_action = agent.get_exploratory_action(agent.map_state_to_table(next_state[0]))
        agent.learn(agent.map_state_to_table(state[0]), action, reward, agent.map_state_to_table(next_state[0]),
                    next_action)
        state = next_state
        action = next_action
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, mytime, cumulative_reward, agent.epsilon))
            break
    return_history.append(cumulative_reward)

    if episodes % 500 == 0:
        # Plot Results
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.show(block=False)
        plt.pause(0.1)
        plt.savefig('../plots/sarsa_training_' + rom + '.' + fig_format)

    if episodes % 10000 == 0:
        # Saving the model to disk
        agent.save('../models/SARSA-' + rom + '.h5')

