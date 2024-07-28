import argparse

import torch

from src.utils import resize_and_bgr2gray, image_to_tensor
from train import get_args
import game.wrapped_flappy_bird as game


def test(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if torch.cuda.is_available():
        model = torch.load("{}/flappy_bird".format(args.saved_path)).cuda()
    else:
        model = torch.load("{}/flappy_bird".format(args.saved_path), map_location=lambda storage, loc: storage)
    model.eval()

    game_state = game.GameState()
    action = torch.tensor([1, 0], dtype=torch.float32)
    image_data, reward, terminal = game_state.frame_step(action)

    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)

    if torch.cuda.is_available():
        image_data = image_data.cuda()
    state = torch.cat([image_data for i in range(4)]).unsqueeze(0)

    while True:
        # 预测Q值
        prediction = model(state)[0]
        # 构造行为
        action_index = torch.argmax(prediction).item()
        action = torch.zeros(2, dtype=torch.float32)
        action[action_index] = 1
        # 和环境互动
        # game_state
        image_data, reward, terminal = game_state.frame_step(action)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)

        if torch.cuda.is_available():
            image_data = image_data.cuda()
        next_state = torch.cat((state.squeeze(0)[1:, :, :], image_data)).unsqueeze(0)
        state = next_state


if __name__ == '__main__':
    args = get_args()
    test(args)
