#!/usr/bin/env python
import sys
import random
import logging
import math
from collections import defaultdict

import numpy as np
import click
from tqdm import tqdm

from gym_tictactoe.envs import TicTacToeEnv, set_log_level_by, tocode,\
    agent_by_mark, next_mark, check_game_status, tomark, O_REWARD,\
    X_REWARD
from examples.human_agent import HumanAgent


DEFAULT_VALUE = 0
MAX_EPISODE = 10000
MODEL_FILE = 'td_agent.dat'
EPSILON = 0.2

st_values = {}
st_visits = defaultdict(lambda: 0)
ucb_visits = defaultdict(lambda: defaultdict(int))


def set_state_value(state, value):
    st_visits[state] += 1
    st_values[state] = value


class TDAgent(object):
    def __init__(self, action_space, mark, policy, epsilon=EPSILON, alpha=0.4,
                 gamma=0.9):
        self.action_space = action_space
        self.mark = mark
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode = 0
        self.turn_cnt = 0

    def act(self, state, ava_actions):
        if self.policy == 'egreedy':
            return self.egreedy_policy(state, ava_actions)
        else:
            return self.ucb_policy(state, ava_actions)

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

    def ucb_policy(self, state, ava_actions):
        """Returns action by Upper confidence bound  policy.

        Return random action with epsilon probability or best action.

        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions

        Returns:
            int: Selected action.
        """
        maxk = None
        max_action = None
        c = 5.0  # degree of exploration
        t = self.turn_cnt

        for action in ava_actions:
            nstate = self.after_action_state(state, action)
            nval = abs(self.ask_value(nstate))
            ucnt = ucb_visits[state][action]
            if ucnt == 0:
                max_action = action
                break

            k = nval + c * math.sqrt(math.log(t+1) / float(ucnt+1))
            logging.debug("ucb_policy state {} action {} nval {:0.2f} ucnt {}"
                          " k {:0.2f}".format(nstate, action, nval, ucnt, k))
            if maxk is None or k > maxk:
                max_action, maxk = action, k

        if max_action is not None:
            logging.debug("  selected action {} k {}".format(max_action, maxk))
            ucb_visits[state][max_action] += 1

        return max_action

    def random_action(self, ava_actions):
        return random.choice(ava_actions)

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
            if gstatus > 0:
                # always called by self
                assert tomark(gstatus) == self.mark
                val = O_REWARD if self.mark == 'O' else X_REWARD
            set_state_value(state, val)
        return st_values[state]

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

    def backup(self, state, nstate, reward):
        """Backup value by difference and learning rate.

        Execute an action then backup Q by best value of next state.

        Args:
            state (tuple): Current state
            nstate (tuple): Next state
            reward (int): Immediate reward from action. Default is 0.
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


@cli.command(help="Learn TD agent.")
@click.option('-e', '--episode', "max_episode", default=MAX_EPISODE,
              show_default=True, help="Episode count.")
@click.option('-p', '--epsilon', default=EPSILON, show_default=True,
              help="Exploring factor.")
@click.option('-f', '--save-file', default=MODEL_FILE, show_default=True,
              help="Save file name.")
def learn(max_episode, epsilon, save_file):
    _learn(max_episode, epsilon, save_file)


def _learn(max_episode, epsilon, save_file):
    env = TicTacToeEnv()
    agents = [TDAgent(env.action_space, 'O', 'egreedy', epsilon),
              TDAgent(env.action_space, 'X', 'egreedy', epsilon)]

    start_mark = 'O'
    for i in tqdm(range(max_episode)):
        episode = i + 1
        env.show_episode(False, episode)

        # reset agent for new episode
        for agent in agents:
            agent.episode = episode
            agent.turn_cnt = 0

        env.set_start_mark(start_mark)
        obs = env.reset()
        _, mark = obs
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            agent.turn_cnt += 1
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


@cli.command(help="Play with model.")
@click.option('-f', '--load-file', default=MODEL_FILE, show_default=True,
              help="Load file name.")
def play(load_file):
    _play(load_file)


def _play(load_file):
    load_states(load_file)
    env = TicTacToeEnv()
    td_agent = TDAgent(env.action_space, 'X', 'egreedy')
    td_agent.epsilon = 0  # prevent exploring
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
@click.option('-p', '--epsilon', default=EPSILON, show_default=True,
              help="Exploring factor.")
@click.option('-f', '--model-file', default=MODEL_FILE, show_default=True,
              help="Model file name.")
def learnplay(max_episode, epsilon, model_file):
    _learn(max_episode, epsilon, model_file)
    _play(model_file)


if __name__ == '__main__':
    cli()
