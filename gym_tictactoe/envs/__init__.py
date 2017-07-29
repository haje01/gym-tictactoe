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
        int: -1 for game in progress, 0 for draw game, 1/2 for finished game(winer code).
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

    def __init__(self, alpha=0.02):
        self.action_space = spaces.Discrete(NUM_LOC)
        self.observation_space = spaces.Discrete(NUM_LOC)
        self.alpha = alpha
        self._seed()
        self._reset()

    def _reset(self):
        self.board = [0] * NUM_LOC
        self.to_move = 0
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
            self.board[loc] = self.to_move + 1
            # decide
            status = check_game_status(self.board)
            if status >= 0:
                self.done = True
                reward = 1 if status in [1, 2] else 0
        # switch turn
        self.to_move = 1 - self.to_move

        return self._get_obs(), reward, self.done, None

    def _get_obs(self):
        return self.board, self.to_move

    def _render(self, mode='human', close=False):
        if close:
            return

        for j in range(0, 9, 3):
            print('|'.join([tomark(self.board[i]) for i in range(j, j+3)]))
            if j < 6:
                print('-----')
        print('')

    def empty_locs(self):
        return [i for i, c in enumerate(self.board) if c == 0]

    def render_turn(self, agent):
        print("{}'s turn.".format(agent.mark))

    def render_result(self, agent, reward):
        if reward == 0:
            print("==== Finished: Draw ====")
        else:
            result = 'win' if reward == 1 else 'lose'
            print("==== Finished: {} {} ====".format(agent.mark, result))


def set_log_level_by(verbosity):
    if verbosity == 0:
        level = 40
    elif verbosity == 1:
        level =  20
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


@click.group()
@click.option('-v', '--verbose', count=True, help="Increase verbosity.")
@click.pass_context
def cli(ctx, verbose):
    level = set_log_level_by(verbose)
    logging.debug("log level {}".format(level))


@cli.command(help="Play Human vs Humna")
@click.pass_context
def playhuman(ctx):
    env = TicTacToeEnv()
    env.board = [1, 0, 2, 0, 1, 0, 2, 0, 1]
    env.render()


if __name__ == '__main__':
    cli(obj={})
