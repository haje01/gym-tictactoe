#!/usr/bin/env python
import sys
import logging

import click

from gym_tictactoe.envs import TicTacToeEnv, set_log_level_by


class HumanAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        loc = input("Enter location[1-9], q for quit: ")
        if loc.lower() == 'q':
            return None
        return int(loc) - 1


@click.group()
@click.option('-v', '--verbose', count=True, help="Increase verbosity.")
@click.pass_context
def cli(ctx, verbose):
    level = set_log_level_by(verbose)
    logging.debug("log level {}".format(level))



@cli.command(help="Play human agent.")
@click.option('-e', '--episode', "max_episode", default=10, show_default=True,
              help="Episode count.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number in the board.")
def play(max_episode, show_number):
    episode = 0
    env = TicTacToeEnv(show_number=show_number)
    agents = [HumanAgent(env.action_space),
              HumanAgent(env.action_space)]

    while episode < max_episode:
        obs = env.reset()
        _, to_move = obs
        done = False
        env.render()
        while not done:
            agent = agents[to_move - 1]
            env.render_turn(to_move)
            action = agent.act()
            if action is None:
                sys.exit()

            obs, reward, done, info = env.step(action)

            print('')
            env.render()
            if done:
                env.render_result(to_move, reward)
                break
            else:
                _, to_move = obs
        episode += 1


if __name__ == '__main__':
    cli()
