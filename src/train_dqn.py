import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import reward_engineering
import tensorflow as tf
import time


NUM_EPISODES = 140 # Number of episodes used for training
RENDER = False  # If the Environment should be rendered

rom = 'CartPole-v1'
#rom = 'MountainCar-v0'
#rom = 'Assault-ram-v0'

fig_format = 'png'
# fig_format = 'eps'
# fig_format = 'svg'

# Comment this line to enable training using your GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.compat.v1.disable_eager_execution()

# Initiating the Environment
env = gym.make(rom)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Creating the DQN agent
agent = DQNAgent(state_size, action_size)

# Checking if weights from previous learning session exists
if os.path.exists('../models/DQN-' + rom + '.h5'):
    print('Loading weights from previous learning session.')
    agent.load('../models/DQN-' + rom + '.h5')
else:
    print('No weights found from previous learning session.')
done = False
batch_size = 32  # batch size used for the experience replay
return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state = env.reset()
    # This reshape is needed to keep compatibility with Keras
    state = np.reshape(state, [1, state_size])
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    # Time it
    start_episode = time.time()
    for mytime in range(1, 500):
        if RENDER:
            env.render()  # Render the environment for visualization
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, reward, done, _ = env.step(action)
        # Reshaping to keep compatibility with Keras
        next_state = np.reshape(next_state, [1, state_size])
        # Making reward engineering to allow faster training
        reward = reward_engineering(state[0], action, reward, next_state[0], done, mytime)
        # Appending this experience to the experience replay buffer
        agent.append_experience(state, action, reward, next_state, done)
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, mytime, cumulative_reward, agent.epsilon))
            break
        # We only update the policy if we already have enough experience in memory
        if len(agent.replay_buffer) > 2 * batch_size:
            loss = agent.replay(batch_size)
    end_episode = time.time()
    print("It took {} to process the entire EPISODE".format(end_episode - start_episode))
    # print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
    #       .format(episodes, NUM_EPISODES, mytime, cumulative_reward, agent.epsilon))
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    # Every 20 episodes, update the plot for training monitoring
    if episodes % 20 == 0:
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.show(block=False)
        plt.pause(0.1)
        plt.savefig('../plots/dqn_training_' + rom + '.' + fig_format)
        # Saving the model to disk
        agent.save('../models/DQN-' + rom + '.h5')
plt.pause(1.0)
