import logging

import gym
from gym import spaces
import click

CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
NUM_LOC = 9

LOG_FMT = logging.Formatter('%(asctime)-15s %(levelname)s '
                            '[%(filename)s:%(lineno)d] %(message)s',
                            '%Y-%m-%d %H:%M:%S')


def tomark(code):
    return CODE_MARK_MAP[code]


def tocode(mark):
    return 1 if mark == 'O' else 2


def check_game_status(board):
    """Return game status.

    Args:
        board (list): Current board state

    Returns:
        int: -1 for game in progress, 0 for draw game, 1/2 for finished
            game(winer code).
    """
    for t in [1, 2]:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j+3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j+3] == t and board[j+6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            # still playing
            return -1

    # draw game
    return 0


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=0.02, show_number=True):
        self.action_space = spaces.Discrete(NUM_LOC)
        self.observation_space = spaces.Discrete(NUM_LOC)
        self.alpha = alpha
        self._seed()
        self._reset()
        self.show_number = show_number

    def _reset(self):
        self.board = [0] * NUM_LOC
        self.to_move = 1
        self.done = False
        return self._get_obs()

    def _step(self, action):
        """Step environment by action.

        Reward rule:
            -1: if agent lose or misplace
            0: game in progress or draw
            1: if agent win the game.

        Args:
            action (int): Location

        Returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action)
        loc = action
        if self.done:
            return self._get_obs(), 0, True, None

        reward = 0
        if self.board[loc] != 0:
            # misplace
            self.done = True
            reward = -1
        else:
            # place
            self.board[loc] = self.to_move
            # decide
            status = check_game_status(self.board)
            logging.debug("check_game_status board {} status"
                          " {}".format(self.board, status))
            if status >= 0:
                self.done = True
                reward = 1 if status in [1, 2] else 0
        # switch turn
        self.to_move = 2 if self.to_move == 1 else 1
        return self._get_obs(), reward, self.done, None

    def _get_obs(self):
        return tuple(self.board), self.to_move

    def _render(self, mode='human', close=False):
        if close:
            return
        self.render_board()
        print('')

    def render_board(self):
        for j in range(0, 9, 3):
            def mark(i):
                return tomark(self.board[i]) if not self.show_number or\
                    self.board[i] != 0 else str(i+1)
            print('|'.join([mark(i) for i in range(j, j+3)]))
            if j < 6:
                print('-----')

    def render_turn(self, to_move):
        print("{}'s turn.".format(tomark(to_move)))

    def render_result(self, to_move, reward):
        if reward == 0:
            print("==== Finished: Draw ====")
        else:
            result = 'win' if reward == 1 else 'lose'
            print("==== Finished: {} {}! ====".format(tomark(to_move), result))

    def empty_locs(self):
        return [i for i, c in enumerate(self.board) if c == 0]


def set_log_level_by(verbosity):
    if verbosity == 0:
        level = 40
    elif verbosity == 1:
        level = 20
    elif verbosity >= 2:
        level = 10

    logger = logging.getLogger()
    logger.setLevel(level)
    if len(logger.handlers):
        handler = logger.handlers[0]
    else:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(LOG_FMT)
    return level
