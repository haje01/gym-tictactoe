#!/usr/bin/env python
import click

from gym_tictactoe.envs import TicTacToeEnv


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


@click.command(help="Play random agent.")
@click.option('-e', '--episode', "max_episode", default=10, show_default=True,
              help="Episode count.")
@click.option('-s', '--hide', is_flag=True, show_default=True,  help="Hide"
              " progress.")
def play(max_episode, hide):
    episode = 0
    env = TicTacToeEnv()
    agents = [RandomAgent(env.action_space),
              RandomAgent(env.action_space)]

    while episode < max_episode:
        obs = env.reset()
        _, to_move = obs
        done = False
        while not done:
            if not hide:
                env.render_turn(to_move)

            agent = agents[to_move - 1]
            action = agent.act()
            obs, reward, done, info = env.step(action)
            if not hide:
                env.render()

            if done:
                if not hide:
                    env.render_result(to_move, reward)
                break
            else:
                _, to_move = obs
        episode += 1


if __name__ == '__main__':
    play()
