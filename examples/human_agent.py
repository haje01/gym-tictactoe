import sys

import click

from gym_tictactoe.envs import TicTacToeEnv


class HumanAgent(object):
    def __init__(self, action_space, mark):
        self.action_space = action_space
        self.mark = mark

    def act(self):
        loc = input("Enter location[1-9], q for quit: ")
        if loc.lower() == 'q':
            return None
        return int(loc) - 1


@click.command(help="Run random agent.")
@click.option('-e', '--episode', "max_episode", default=10, show_default=True,
              help="Episode count.")
def run(max_episode):
    episode = 0
    env = TicTacToeEnv()
    agents = [HumanAgent(env.action_space, 'O'),
              HumanAgent(env.action_space, 'X')]

    while episode < max_episode:
        obs = env.reset()
        _, to_move = obs
        done = False
        env.render()
        while not done:
            agent = agents[to_move]
            env.render_turn(agent)
            action = agent.act()
            if action is None:
                sys.exit()

            obs, reward, done, info = env.step(action)
            _, to_move = obs

            print('')
            env.render()
            if done:
                env.render_result(agent, reward)
                break
        episode += 1


if __name__ == '__main__':
    run()
