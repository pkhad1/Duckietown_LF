import numpy as np
import torch

from args import get_ddpg_args_test
from ddpg import DDPG
from env import launch_env
from wrappers import ActionWrapper, ImgWrapper, NormalizeWrapper, ResizeWrapper

policy_name = "DDPG"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_ddpg_args_test()

file_name = "{}_{}".format(policy_name, args.seed)

env = launch_env()

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")

policy.load(file_name, directory="./pytorch_models")

cutoff = 256

with torch.no_grad():
    while True:
        obs = env.reset()
        env.render()
        rewards = []
        steps = 0
        while True:
            action = policy.predict(np.array(obs))
            r = action

            print(action)
           
            #action[0] is the linear velocity and action [1] is angular velocity
            r = [0.5 , 0.5] # reduce the angular speed and angular velocity by 0.5 each
            
            obs, rew, done, misc = env.step(r)
            rewards.append(rew)
            env.render()
            steps += 1
            #if done or steps >= cutoff:
            if done:
                break
        print("mean episode reward:", np.mean(rewards))
