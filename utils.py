START_POSITION_CAR = -0.5


def reward_engineering_mountain_car(state, action, reward, next_state, done, episode_length):
    """
    Makes reward engineering to allow faster training in the Mountain Car environment.

    :param state: state.
    :type state: NumPy array with dimension (1, 2).
    :param action: action.
    :type action: int.
    :param reward: original reward.
    :type reward: float.
    :param next_state: next state.
    :type next_state: NumPy array with dimension (1, 2).
    :param done: if the simulation is over after this experience.
    :type done: bool.
    :return: modified reward for faster training.
    :rtype: float.
    """
    # se ta caindo pra esquerda e ta empurrando o carro pra direita, ta errado
    # o mesmo se ta caindo pra direita e empurrando pra esquerda
    if (state[2] < -0.005 and state[3] < 0 and action is 1) or (state[2] > 0.005 and state[3] > 0 and action is 0):
        reward -= state[2]*state[2]*200

    # se a posição angular diminuiu, entao bizu
    dif = abs(next_state[2]) - abs(state[2])
    if dif < 0:
        reward += (0.418-next_state[2]) * (0.418-next_state[2]) * 20

    # quanto mais longe do centro pior. Só conta a partir de 1.5
    pos = state[0]*state[0]
    if pos > 0.3:
        reward -= pos

    # quanto menor a velocidade melhor
    vel = state[1] * state[1]
    reward -= vel

    # essas são s estados alvo
    if abs(next_state[0]) < 0.010 and abs(next_state[1]) < 0.010 and abs(next_state[2]) < 0.01 and \
            abs(next_state[3]) < 0.01:
        reward += 100

    if episode_length == 199:
        print("UHUUL")

    return reward
