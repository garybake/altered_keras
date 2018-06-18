#! /usr/bin/env python

import random
import json
from collections import deque

import numpy as np

from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam

from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import sobel
# from skimage import io

import retro

# NUM_ACTIONS = 12  # ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
# ACTION_LIST = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C']
# NUM_ACTIONS = 9

ACTION_LIST = ['B', 'A', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C']
NUM_ACTIONS = 7


MAX_MEMORY = 50000
EPOCHS = 1000

STACK_SIZE = 4
IMG_ROWS = 80
IMG_COLS = 160

OBSERVATIONS_BEFORE_TRAINING = 1000
BATCH_SIZE = 32
GAMMA = 0.99  # decay rate of past observations
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EXPLORE = 3000  # frames over which to anneal epsilon TODO


def get_action_str(action):
    actions = []
    for idx, val in enumerate(action):
        if val:
            actions.append(ACTION_LIST[idx])
    return actions


def get_preprocessed_frame(observation):
    """
    """
    observation = resize(observation, (122, 160))
    grey = rgb2gray(observation)
    grey = sobel(grey)
    return grey[20:160 - 60, :]


def fix_actions(action):
    out = action[:2]
    out = np.append(out, [0, 0])
    out = np.append(out, action[-5:])
    return out


def build_model():
    print()
    print("Building the model ")

    model = Sequential()
    model.add(layers.Convolution2D(
        32, 8, 8, subsample=(4, 4), border_mode='same',
        input_shape=(IMG_ROWS, IMG_COLS, STACK_SIZE)))
    model.add(layers.Activation('relu'))
    model.add(layers.Convolution2D(
        64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Convolution2D(
        64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(NUM_ACTIONS))

    adam = Adam(lr=1e-6)
    model.compile(
        loss='mse',
        optimizer=adam)
    print("Finished building the model")
    print(model.summary())
    return model


def get_empty_action_space(num_actions):
    return [0] * num_actions  # len(env.BUTTONS)

if __name__ == "__main__":

    state = 'run'

    model = build_model()
    model.load_weights("models/model.h5")
    print ("Weight load successfully")

    # Define environment/game
    env = retro.make(game='AlteredBeast-Genesis')
    env.reset()
    game_over = False

    # get initial input
    obs_t, reward, game_over, info = env.step(
        get_empty_action_space(NUM_ACTIONS))

    obs_t = get_preprocessed_frame(obs_t)
    state_stack = np.stack((obs_t,) * STACK_SIZE, axis=2)
    state_stack = state_stack.reshape(
        1, state_stack.shape[0], state_stack.shape[1], state_stack.shape[2])
    tick = 0
    epsilon = 0

    health = 12
    tot_reward = 0
    while not game_over:

        tick += 1
        action = get_empty_action_space(NUM_ACTIONS)
        q = model.predict(state_stack)

        max_Q = np.argmax(q)
        action_index = max_Q
        action[max_Q] = 1

        # apply action, get rewards and new state
        action_fix = fix_actions(action)
        obs_t, reward, game_over, info = env.step(action_fix)
        if info['health'] < health:
            reward -= 50
            health = info['health']

        # store the transition in the replay memory
        obs_t = get_preprocessed_frame(obs_t)
        obs_t = obs_t.reshape(
            1, obs_t.shape[0], obs_t.shape[1], 1)
        state_stack_t1 = np.append(
            obs_t, state_stack[:, :, :, :STACK_SIZE-1], axis=3)

        env.render()

        action_string = get_action_str(action)
        print('T: {} | State: {} | E: {:.8f} | Action: {} | Reward: {} | TotReward: {} | Q_Max: {:.8f} '.format(
            tick, state, epsilon, action_string, reward, tot_reward, np.max(max_Q)))

        state_stack = state_stack_t1
        tot_reward += reward
    print("Total reward: ", tot_reward)
