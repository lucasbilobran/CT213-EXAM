import numpy as np
from utils import build_state_table_size
import config

def compute_greedy_policy_as_table(q):
    """
    Computes the greedy policy as a table.

    :param q: action-value table.
    :type q: bidimensional numpy array.
    :return: greedy policy table.
    :rtype: bidimensional numpy array.
    """
    policy = np.zeros(q.shape)
    for s in range(q.shape[0]):
        policy[s, greedy_action(q, s)] = 1.0
    return policy


def epsilon_greedy_action(q, state, epsilon):
    """
    Computes the epsilon-greedy action.

    :param q: action-value table.
    :type q: bidimensional numpy array.
    :param state: current state.
    :type state: int.
    :param epsilon: probability of selecting a random action.
    :type epsilon: float.
    :return: epsilon-greedy action.
    :rtype: int.
    """
    greedy_act = greedy_action(q, state)
    if np.random.random(1) > epsilon:
        return greedy_act

    random_action = np.random.randint(0, range(q.shape[1]).stop)
    while random_action is greedy_act:
        random_action = np.random.randint(0, range(q.shape[1]).stop)

    return random_action


def greedy_action(q, state):
    """
    Computes the greedy action.

    :param q: action-value table.
    :type q: bidimensional numpy array.
    :param state: current state.
    :type state: int.
    :return: greedy action.
    :rtype: int.
    """
    greedy_act = 0
    q_max = q[state][greedy_act]
    for action in range(q.shape[1]):
        val = q[state][action]
        if val > q_max:
            greedy_act = action
            q_max = val

    return greedy_act


class RLAlgorithm:
    """
    Represents a model-free reinforcement learning algorithm.
    """
    def __init__(self, rom, num_states, num_actions, epsilon, alpha, gamma):
        """
        Creates a model-free reinforcement learning algorithm.

        :param num_states: number of states of the MDP.
        :type num_states: int.
        :param num_actions: number of actions of the MDP.
        :type num_actions: int.
        :param epsilon: probability of selecting a random action in epsilon-greedy policy.
        :type epsilon: float.
        :param alpha: learning rate.
        :type alpha: float.
        :param gamma: discount factor.
        :type gamma: float.
        """
        self.rom = rom
        self.q = np.zeros((build_state_table_size(num_states, rom), num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_num_states(self):
        """
        Returns the number of states of the MDP.

        :return: number of states.
        :rtype: int.
        """
        return self.q.shape[0]

    def get_num_actions(self):
        """
        Returns the number of actions of the MDP.

        :return: number of actions.
        :rtype: int.
        """
        return self.q.shape[1]

    def get_exploratory_action(self, state):
        """
        Returns an exploratory action using epsilon-greedy policy.

        :param state: current state.
        :type state: int.
        :return: exploratory action.
        :rtype: int.
        """
        return epsilon_greedy_action(self.q, state, self.epsilon)

    def get_greedy_action(self, state):
        """
        Returns a greedy action considering the policy of the RL algorithm.

        :param state: current state.
        :type state: int.
        :return: greedy action considering the policy of the RL algorithm.
        :rtype: int.
        """
        raise NotImplementedError('Please implement this method')

    def learn(self, state, action, reward, next_state, next_action):
        raise NotImplementedError('Please implement this method')

    def map_state_to_table(self, state):
        if self.rom == 'CartPole-v1':
            cart_pos = int((state[0] + config.CART_POSITION)/config.CART_POSITION_RESOLUTION)
            cart_vel = int((state[1] + config.CART_VELOCITY)/config.CART_VELOCITY_RESOLUTION)
            pole_ang = int((state[2] + config.POLE_ANGLE)/config.POLE_ANGLE_RESOLUTION)
            pole_vel = int((state[3] + config.POLE_ANGLE_VELOCITY)/config.POLE_ANGLE_VELOCITY_RESOLUTION)

            if cart_pos > 96:
                cart_pos = 96

            if cart_vel > 100:
                cart_vel = 50

            if pole_ang > 167:
                pole_ang = 167

            if pole_vel > 100:
                pole_vel = 100

            b1 = 1 + 96
            b2 = b1*(1 + 100)
            b3 = b2*(1 + 167)
            return cart_pos + cart_vel*b1 + pole_ang*b2 + pole_vel*b3

        return state


class Sarsa(RLAlgorithm):
    def __init__(self, rom, num_states, num_actions, epsilon, alpha, gamma):
        super().__init__(rom, num_states, num_actions, epsilon, alpha, gamma)

    def get_greedy_action(self, state):
        """
        Notice that Sarsa is an on-policy algorithm, so it uses the same epsilon-greedy
        policy for learning and execution.

        :param state: current state.
        :type state: int.
        :return: epsilon-greedy action of Sarsa's execution policy.
        :rtype: int.
        """
        return epsilon_greedy_action(self.q, state, self.epsilon)

    def learn(self, state, action, reward, next_state, next_action):
        q_s_a = self.q[state][action]
        self.q[state][action] = q_s_a  + \
                                self.alpha*(reward+self.gamma*self.q[next_state][next_action]-q_s_a)


class QLearning(RLAlgorithm):
    def __init__(self, rom, num_states, num_actions, epsilon, alpha, gamma):
        super().__init__(rom, num_states, num_actions, epsilon, alpha, gamma)

    def get_greedy_action(self, state):
        return greedy_action(self.q, state)

    def learn(self, state, action, reward, next_state, next_action):
        q_action = greedy_action(self.q, next_state)
        q_val = self.q[next_state, q_action]
        q_s_a = self.q[state][action]
        self.q[state][action] = q_s_a + self.alpha * (reward + self.gamma * q_val - q_s_a)
