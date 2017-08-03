#!/usr/bin/env python
import sys
import random
import logging
from collections import defaultdict

import numpy as np
import click
from tqdm import tqdm

from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, tocode,\
    agent_by_mark, next_mark, check_game_status, O_REWARD,\
    X_REWARD
from examples.human_agent import HumanAgent


DEFAULT_VALUE = 0
MAX_EPISODE = 10000
MODEL_FILE = 'td_agent.dat'
EPSILON = 0.1
ALPHA = 0.4

st_values = {}
st_visits = defaultdict(lambda: 0)


def set_state_value(state, value):
    st_visits[state] += 1
    st_values[state] = value


class TDAgent(object):
    def __init__(self, action_space, mark, epsilon, alpha):
        self.action_space = action_space
        self.mark = mark
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode = 0

    def act(self, state, ava_actions):
        return self.egreedy_policy(state, ava_actions)

    def egreedy_policy(self, state, ava_actions):
        """Returns action by Epsilon greedy policy.

        Return random action with epsilon probability or best action.

        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions

        Returns:
            int: Selected action.
        """
        logging.debug("egreedy_policy for '{}'".format(self.mark))
        e = random.random()
        if e < self.epsilon:
            logging.debug("Explore with eps {}".format(self.epsilon))
            action = self.random_action(ava_actions)
        else:
            logging.debug("Exploit with eps {}".format(self.epsilon))
            action = self.greedy_action(state, ava_actions)
        return action

    def random_action(self, ava_actions):
        return random.choice(ava_actions)

    def greedy_action(self, state, ava_actions):
        """Return best action by current state value.

        Evaluate each action, select best one. Tie-breaking is random.

        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions

        Returns:
            int: Selected action
        """
        assert len(ava_actions) > 0

        ava_values = []
        for action in ava_actions:
            nstate = self.after_action_state(state, action)
            nval = self.ask_value(nstate)
            ava_values.append(nval)
            vcnt = st_visits[nstate]
            logging.debug("  nstate {} val {:0.2f} visits {}".
                          format(nstate, nval, vcnt))

        # select most right action for 'O' or 'X'
        val_arr = np.array(ava_values)
        if self.mark == 'O':
            midx = np.argwhere(val_arr == np.max(val_arr))
        else:
            midx = np.argwhere(val_arr == np.min(val_arr))

        # tie breaking by random choice
        aidx = np.random.choice(midx.ravel())
        logging.debug("greedy_action mark {} val_arr {} midx {} aidx {}".
                      format(self.mark, val_arr, midx.ravel(), aidx))

        action = ava_actions[aidx]

        return action

    def after_action_state(self, state, action):
        """Execute an action and returns resulted state.

        Args:
            state (tuple): Board status + mark
            action (int): Action to run

        Returns:
            tuple: New state
        """

        board, mark = state
        nboard = list(board[:])
        nboard[action] = tocode(mark)
        nboard = tuple(nboard)
        return nboard, next_mark(mark)

    def ask_value(self, state):
        """Returns value of given state.

        If state is not exists, set it as default value.

        Args:
            state (tuple): State.

        Returns:
            float: Value of a state.
        """
        if state not in st_values:
            logging.debug("ask_value - new state {}".format(state))
            gstatus = check_game_status(state[0])
            val = DEFAULT_VALUE
            # win
            if gstatus > 0:
                val = O_REWARD if self.mark == 'O' else X_REWARD
            set_state_value(state, val)
        return st_values[state]

    def backup(self, state, nstate, reward):
        """Backup value by difference and step size.

        Execute an action then backup Q by best value of next state.

        Args:
            state (tuple): Current state
            nstate (tuple): Next state
            reward (int): Immediate reward from action
        """
        logging.debug("backup state {} nstate {} reward {}".
                      format(state, nstate, reward))

        val = self.ask_value(state)
        nval = self.ask_value(nstate)
        diff = nval - val
        val2 = val + self.alpha * diff

        logging.debug("  value from {:0.2f} to {:0.2f}".format(val, val2))
        set_state_value(state, val2)


@click.group()
@click.option('-v', '--verbose', count=True, help="Increase verbosity.")
@click.pass_context
def cli(ctx, verbose):
    global tqdm

    set_log_level_by(verbose)
    if verbose > 0:
        tqdm = lambda x: x  # NOQA


@cli.command(help="Learn and save the model.")
@click.option('-e', '--episode', "max_episode", default=MAX_EPISODE,
              show_default=True, help="Episode count.")
@click.option('-x', '--exploring-factor', "epsilon", default=EPSILON,
              show_default=True, help="Exploring factor.")
@click.option('-s', '--step-size', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--save-file', default=MODEL_FILE, show_default=True,
              help="Save file name.")
def learn(max_episode, epsilon, alpha, save_file):
    _learn(max_episode, epsilon, alpha, save_file)


def _learn(max_episode, epsilon, alpha, save_file):
    env = TicTacToeEnv()
    agents = [TDAgent(env.action_space, 'O', epsilon, alpha),
              TDAgent(env.action_space, 'X', epsilon, alpha)]

    start_mark = 'O'
    for i in tqdm(range(max_episode)):
        episode = i + 1
        env.show_episode(False, episode)

        # reset agent for new episode
        for agent in agents:
            agent.episode = episode

        env.set_start_mark(start_mark)
        obs = env.reset()
        _, mark = obs
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            env.show_turn(False, mark)
            action = agent.act(obs, ava_actions)

            # update (no rendering)
            nobs, reward, done, info = env.step(action)
            agent.backup(obs, nobs, reward)

            if done:
                env.show_result(False, mark, reward)
                # set terminal state value
                set_state_value(obs, reward)

            _, mark = obs = nobs

        # rotate start
        start_mark = next_mark(start_mark)

    # save states
    save_states(save_file)


def save_states(save_file):
    with open(save_file, 'wt') as f:
        for state, value in st_values.items():
            vcnt = st_visits[state]
            f.write('{}\t{:0.3f}\t{}\n'.format(state, value, vcnt))


def load_states(filename):
    with open(filename, 'rb') as f:
        for line in f:
            elms = line.decode('ascii').split('\t')
            state = eval(elms[0])
            val = eval(elms[1])
            vcnt = eval(elms[2])
            st_values[state] = val
            st_visits[state] = vcnt


@cli.command(help="Play with saved model.")
@click.option('-f', '--load-file', default=MODEL_FILE, show_default=True,
              help="Load file name.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number in the board.")
def play(load_file, show_number):
    _play(load_file, show_number)


def _play(load_file, show_number):
    load_states(load_file)
    env = TicTacToeEnv(show_number=show_number)
    td_agent = TDAgent(env.action_space, 'X', 0, 0)  # prevent exploring
    start_mark = 'O'
    agents = [HumanAgent(env.action_space, 'O'), td_agent]

    while True:
        # start agent rotation
        env.set_start_mark(start_mark)
        obs = env.reset()
        _, mark = obs
        done = False

        # show start board for human agent
        if mark == 'O':
            env.render(mode='human')

        while not done:
            agent = agent_by_mark(agents, mark)
            human = isinstance(agent, HumanAgent)

            env.show_turn(True, mark)
            ava_actions = env.available_actions()
            if human:
                action = agent.act(ava_actions)
                if action is None:
                    sys.exit()
            else:
                action = agent.act(obs, ava_actions)

            obs, reward, done, info = env.step(action)

            env.render(mode='human')
            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = obs

        # rotation start
        start_mark = next_mark(start_mark)


@cli.command(help="Learn and play")
@click.option('-e', '--episode', "max_episode", default=MAX_EPISODE,
              show_default=True, help="Episode count.")
@click.option('-x', '--exploring-factor', "epsilon", default=EPSILON,
              show_default=True, help="Exploring factor.")
@click.option('-s', '--step-size', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--model-file', default=MODEL_FILE, show_default=True,
              help="Model file name.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number in the board.")
def learnplay(max_episode, epsilon, alpha, model_file, show_number):
    _learn(max_episode, epsilon, alpha, model_file)
    _play(model_file, show_number)


if __name__ == '__main__':
    cli()
