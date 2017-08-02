#!/usr/bin/env python
import click
import random

from gym_tictactoe.envs import TicTacToeEnv, agent_by_mark


class RandomAgent(object):
    def __init__(self, action_space, mark):
        self.action_space = action_space
        self.mark = mark

    def act(self, ava_actions):
        return random.choice(ava_actions)


@click.command(help="Play random agent.")
@click.option('-e', '--episode', "max_episode", default=10, show_default=True,
              help="Episode count.")
def play(max_episode):
    episode = 0
    env = TicTacToeEnv()
    agents = [RandomAgent(env.action_space, 'O'),
              RandomAgent(env.action_space, 'X')]

    while episode < max_episode:
        obs = env.reset()
        _, mark = obs
        done = False
        while not done:
            env.show_turn(True, mark)

            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(ava_actions)
            obs, reward, done, info = env.step(action)
            env.render()

            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = obs
        episode += 1


if __name__ == '__main__':
    play()
