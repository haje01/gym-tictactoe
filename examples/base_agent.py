#!/usr/bin/env python
import random

from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark


class BaseAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, obs, ava_actions):
        for action in ava_actions:
            nstate = after_action_state(obs, action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                if tomark(gstatus) == self.mark:
                    return action
        return random.choice(ava_actions)


def play(max_episode=10):
    episode = 0
    start_mark = 'O'
    env = TicTacToeEnv()
    agents = [BaseAgent('O'),
              BaseAgent('X')]

    while episode < max_episode:
        env.set_start_mark(start_mark)
        obs = env.reset()
        _, mark = obs
        done = False
        while not done:
            env.show_turn(True, mark)

            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(obs, ava_actions)
            obs, reward, done, info = env.step(action)
            env.render()

            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = obs

        # rotate start
        start_mark = next_mark(start_mark)
        episode += 1


if __name__ == '__main__':
    play()
