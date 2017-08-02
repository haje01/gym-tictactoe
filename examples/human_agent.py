#!/usr/bin/env python
import sys
import logging

import click

from gym_tictactoe.envs import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark


class HumanAgent(object):
    def __init__(self, action_space, mark):
        self.action_space = action_space
        self.mark = mark

    def act(self, ava_actions):
        while True:
            uloc = input("Enter location[1-9], q for quit: ")
            if uloc.lower() == 'q':
                return None
            try:
                action = int(uloc) - 1
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal location: '{}'".format(uloc))
            else:
                break

        return action


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
    agents = [HumanAgent(env.action_space, 'O'),
              HumanAgent(env.action_space, 'X')]

    while episode < max_episode:
        obs = env.reset()
        _, mark = obs
        done = False
        env.render()
        while not done:
            agent = agent_by_mark(agents, next_mark(mark))
            env.print_turn(mark)
            action = agent.act()
            if action is None:
                sys.exit()

            obs, reward, done, info = env.step(action)

            print('')
            env.render()
            if done:
                env.print_result(mark, reward)
                break
            else:
                _, mark = obs
        episode += 1


if __name__ == '__main__':
    cli()
