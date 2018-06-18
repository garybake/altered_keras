import random
import json
from collections import deque

import numpy as np

import time
import retrowrapper

from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam

from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage import io

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

MAX_EPISODES = 15
OBSERVATIONS_BEFORE_TRAINING = 1000
# OBSERVATIONS_BEFORE_TRAINING = 32
BATCH_SIZE = 32
GAMMA = 0.99  # decay rate of past observations
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EXPLORE = 3000  # frames over which to anneal epsilon TODO


def save_screen(screen, folder='/tmp/', timestamp=True):
    """
    Output screen as image file
    """
    timestr = 'x'
    if timestamp:
        timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = '{}/boxing_{}.png'.format(folder, timestr)
    io.imsave(filename, screen)


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = deque()
        self.discount = discount

    def push(self, states):
        self.memory.append(states)
        if len(self.memory) > self.max_memory:
            self.memory.popleft()

    def size(self):
        return len(self.memory)

    def get_batch(self, batch_size):
        return random.sample(self.memory, batch_size)


def get_action_str(action):
    actions = []
    for idx, val in enumerate(action):
        if val:
            actions.append(ACTION_LIST[idx])
    return actions


def get_preprocessed_frame(observation):
    """
    Clean up/optimise the screen image
    - Resize to remove the top and bottom portions
    - Change to grey scale
    - Use sobel funct to emphasize the edges

    """
    observation = resize(observation, (122, 160))
    grey = rgb2gray(observation)
    grey = sobel(grey)
    return grey[20:160 - 60, :]


def fix_actions(action):
    """
    The action list doesn't include XYZ and Start/Stop actions
    This puts them back in (empty) as the env expects them.
    """
    out = action[:2]
    out = np.append(out, [0, 0])
    out = np.append(out, action[-5:])
    return out


def get_reward(base_reward, info):
    """
    Allows us to modify the reward
    TODO: this can be done inside openai
    """
    if info['health'] < health:
        return base_reward - 50
    return base_reward


def build_model():
    """
    Build the NN model
    """
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
    """
    Returns an action space with nothing selected.
    """
    return [0] * num_actions  # len(env.BUTTONS)

if __name__ == "__main__":
    epsilon = .1  # exploration

    env = retrowrapper.RetroWrapper(
        game='AlteredBeast-Genesis',
        state='Level1'
    )

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=MAX_MEMORY)

    model = build_model()
    # Uncomment the line below to continue training
    model.load_weights("model.h5")

    # Train
    tick = 0  # Frame count
    loss = 0  # Cumulative loss 
    q_max = 0
    print('starting')
    state = 'exploring'
    for episode in range(MAX_EPISODES):
        print('Episode: ', episode)
        health = 12
        loss = 0.
        env.reset()
        game_over = False

        # get initial input
        obs_t, reward, game_over, info = env.step(
            get_empty_action_space(NUM_ACTIONS))

        # The network input is the current frame and past 3 frames
        # Build a stack initiall of the initial frame
        obs_t = get_preprocessed_frame(obs_t)
        state_stack = np.stack((obs_t,) * STACK_SIZE, axis=2)
        state_stack = state_stack.reshape(
            1, state_stack.shape[0], state_stack.shape[1], state_stack.shape[2])

        while not game_over:

            # Generate next action
            action = get_empty_action_space(NUM_ACTIONS)

            if random.random() <= epsilon:
                # Randomly pick a random action
                action = np.random.randint(0, 2, size=NUM_ACTIONS)
            else:
                # input a stack of 4 images and the NN returns its predicted
                # best action
                q = model.predict(state_stack)
                max_Q = np.argmax(q)
                action_index = max_Q
                action[max_Q] = 1

            # Reduced epsilon gradually (epsilon greedy)
            if epsilon > FINAL_EPSILON and tick > OBSERVATIONS_BEFORE_TRAINING:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # Apply action, get rewards and new state
            action_fix = fix_actions(action)
            base_obs_t, base_reward, game_over, info = env.step(action_fix)

            # Process the outcomes
            reward = get_reward(base_reward, info)
            health = info['health']
            tick += 1
            obs_t = get_preprocessed_frame(base_obs_t)
            env.render()

            # Add the observation to the current stack and replay stack
            obs_t = obs_t.reshape(
                1, obs_t.shape[0], obs_t.shape[1], 1)
            state_stack_t1 = np.append(
                obs_t, state_stack[:, :, :, :STACK_SIZE-1], axis=3)
            exp_replay.push(
                (state_stack, action, reward, state_stack_t1, game_over))

            # Learn something from the actions
            if tick > OBSERVATIONS_BEFORE_TRAINING:
                state = 'Learning'

                # Get a random series of observations from the history
                minibatch = exp_replay.get_batch(BATCH_SIZE)

                # Input to the NN
                inputs = np.zeros(
                    (BATCH_SIZE, state_stack.shape[1], state_stack.shape[2],
                        state_stack.shape[3]))   # 32, 80, 80, 4
                # What the reward output was from executing those actions
                targets = np.zeros((inputs.shape[0], NUM_ACTIONS))  # 32, 2

                # Now we do the experience replay
                for i in range(len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]  # This is action index
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal_t = minibatch[i][4]

                    inputs[i:i + 1] = state_t  # I saved down state_stack

                    targets[i] = model.predict(state_t)  # Probability of hitting each button
                    q_max = model.predict(state_t1)

                    if terminal_t:
                        # if gameover just use immediate the reward
                        targets[i, action_t] = reward_t
                    else:
                        # else calculate the reward from the history
                        targets[i, action_t] = reward_t + GAMMA * np.max(q_max)

                # targets2 = normalize(targets)
                # Train the model on the batch of envents/screens and check how
                # close the rewards where to the expected rewards
                loss += model.train_on_batch(inputs, targets)

            if tick % 1000 == 0:
                # Every so often save the model and output the current status
                print("Saving model", tick)
                model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)

                q_max = [1]
                action_string = get_action_str(action)
                print('T: {} | State: {} | E: {:.8f} | Action: {} | Reward: {} | Q_Max: {:.8f} | Loss: {:.8f}'.format(
                    tick, state, epsilon, action_string, reward, np.max(q_max), loss))

            # x = [1]
            # action_string = get_action_str(action)
            # print('T: {} | State: {} | E: {:.8f} | Action: {} | Reward: {} | Q_Max: {:.8f} | Loss: {:.8f}'.format(
            #     tick, state, epsilon, action_string, reward, np.max(q_max), loss))

            state_stack = state_stack_t1
