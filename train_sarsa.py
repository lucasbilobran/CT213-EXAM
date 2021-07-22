import gym
import numpy as np
from utils import reward_engineering_mountain_car
from sarsa_agent import Sarsa, QLearning, greedy_action


NUM_EPISODES = 140 # Number of episodes used for training
RENDER = False  # If the Mountain Car environment should be rendered

# Initiating the Mountain Car environment
rom = 'CartPole-v1'
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
epsilon = 0.1  # epsilon of epsilon-greedy
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor

# Creating the Sarsa agent
agent = Sarsa(rom, state_size, action_size, epsilon, alpha, gamma)

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
        reward = reward_engineering_mountain_car(state[0], action, reward, next_state[0], done, mytime)
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

