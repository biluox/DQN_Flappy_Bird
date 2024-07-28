import argparse
from random import random, randint, sample

import cv2
import numpy as np
import torch

import game.wrapped_flappy_bird as game
from src.DQNNetwork import DQNNetwork
from src.utils import resize_and_bgr2gray, image_to_tensor


def get_args():
    parser = argparse.ArgumentParser("DQN Flappy Bird")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=1e7)
    parser.add_argument("--replay_memory_size", type=int, default=50000, help='轮次的数量，数据池')
    parser.add_argument("--saved_path", type=str, default='./saved')
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam'])
    args = parser.parse_args()
    return args


def train(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # model = DQNNetwork()
    model = torch.load("{}/flappy_bird".format(args.saved_path)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    game_state = game.GameState()
    action = torch.tensor([1, 0], dtype=torch.float32)
    image_data, reward, terminal = game_state.frame_step(action)
    # 预处理图像 改编大小和彩色变黑白
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)

    if torch.cuda.is_available():
        image_data = image_data.cuda()
        model.cuda()
    # state 被设计为4帧图片，但是一开始只有一帧，我们直接复制一样的，让模型跑起来
    state = torch.cat([image_data for i in range(4)]).unsqueeze(0)

    # 多次迭代训练
    replay_memory = []
    iter = 0
    while iter < args.num_iters:
        prediction = model(state)[0]
        action = torch.zeros(2, dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()
        epsilon = args.final_epsilon + (
                (args.num_iters - iter) * (args.initial_epsilon - args.final_epsilon) / args.num_iters)
        # if random() <= epsilon:
        #     # 随机操作
        #     print('采取随机动作')
        #     action_index = randint(0, 1)
        # else:
        #     print('采取Q动作')
        action_index = torch.argmax(prediction).item()
        action[action_index] = 1
        image_data, reward, terminal = game_state.frame_step(action)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)
        if torch.cuda.is_available():
            image_data = image_data.cuda()
        # 之前4帧取3帧和最新的一帧拼接
        next_state = torch.cat((state.squeeze(0)[1:, :, :], image_data)).unsqueeze(0)
        # 有了一次状态转换，加入replay中
        replay_memory.append([state, action, reward, next_state, terminal])
        # 如果replay_memory满了，就清除最早的
        if len(replay_memory) > args.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), args.batch_size))

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.cat(tuple(state for state in action_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        # Q(S,A) model预测的是这个状态动作的得分
        current_prediction_batch = model(state_batch)
        # Q(S',A)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(tuple(reward if terminal else reward + args.gamma * torch.max(prediction) for
                                  reward, terminal, prediction in
                                  zip(reward_batch, terminal_batch, next_prediction_batch)))
        # 模型的预测值 action_batch是one-hot编码，所以相乘定价等同于获取某个action对应模型预测的Q值
        q_value = torch.sum(current_prediction_batch * action_batch.view(-1, 2), dim=1)

        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1
        print("Iteration: {}/{}, Loss: {}, Reward: {}, Q-value: {}".format(iter + 1, args.num_iters,
                                                                     loss,
                                                                     reward,
                                                                     torch.max(prediction)))

        # 间隔一段跌点保存模型
        if (iter + 1) % 100000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(args.saved_path, iter + 1))
    torch.save(model, "{}/flappy_bird".format(args.saved_path))


if __name__ == '__main__':
    args = get_args()
    train(args)
