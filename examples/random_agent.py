import click

from gym_tictactoe.envs import TicTacToeEnv


class RandomAgent(object):
    def __init__(self, action_space, mark):
        self.action_space = action_space
        self.mark = mark

    def act(self):
        return self.action_space.sample()


@click.command(help="Run random agent.")
@click.option('-e', '--episode', "max_episode", default=10, show_default=True,
              help="Episode count.")
@click.option('-s', '--hide', is_flag=True, show_default=True,  help="Hide"
              " progress.")
def run(max_episode, hide):
    episode = 0
    env = TicTacToeEnv()
    agents = [RandomAgent(env.action_space, 'O'),
              RandomAgent(env.action_space, 'X')]

    while episode < max_episode:
        obs = env.reset()
        _, to_move = obs
        done = False
        while not done:
            agent = agents[to_move]
            action = agent.act()
            obs, reward, done, info = env.step(action)
            _, to_move = obs
            if not hide:
                print('')
                env.render_turn(agent)
                env.render()

            if done:
                if not hide:
                    env.render_result(agent, reward, done)
                break
        episode += 1


if __name__ == '__main__':
    run()
