import gym
import config
from time import sleep

env = gym.make('Assault-ram-v0')

print('==== INFO ABOUT THE ENVIRONMENT ====')
print('Action Space: {}'.format(env.action_space))
print('Observation Space: {}'.format(env.observation_space))
print('====================================')
print('')

print('===== INFO ABOUT THE EPISODES =====')
for i_episode in range(config.NUM_EPISODES):
    observation = env.reset()
    print('Episode {} ---------'.format(i_episode))
    for t in range(config.TIMESTEPS_PER_EPISODE):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print('Timestamp {} | Reward: {} | Info: {}'.format(t, reward, info))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        sleep(0.1)
env.close()
