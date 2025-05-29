# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Visualize Avoid environment from D3IL in MuJoCo GUI
"""

import gym
import gym_avoiding
import imageio

# from gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
# from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
from gym.envs import make as make_
import numpy as np

# env = ObstacleAvoidanceEnv(render=False)
env = make_("avoiding-v0", render=True)
# env.start()   # no need to start() any more, already run in init() now
env.reset()
print(env.action_space)

# video_writer = imageio.get_writer("test_d3il.mp4", fps=30)
while 1:
    obs, reward, done, info = env.step(np.array([0.02, 0.1]))
    print("Reward:", reward)
    # video_img = env.render(
    #     mode="rgb_array",
    #     # height=640,
    #     # width=480,
    #     # camera_name=self.render_camera_name,
    # )
    # video_writer.append_data(video_img)
    if input("Press space to stop, or any other key to continue") == " ":
        break
# video_writer.close()
