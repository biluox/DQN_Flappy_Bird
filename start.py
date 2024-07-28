import numpy as np
import pygame

import game.wrapped_flappy_bird as game

actions = 2


def play_game():
    game_state = game.GameState()

    while True:
        a_t = np.array([1, 0])
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                a_t = np.array([0, 1])
        # 需要传入动作数组 [1,0] 下降 [0,1]上升（鼠标点击）one-hot编码
        _, _, terminal = game_state.frame_step(a_t)
        # if terminal:
        #     break


def main():
    play_game()


if __name__ == '__main__':
    main()
