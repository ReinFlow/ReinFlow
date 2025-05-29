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
Test Robomimic rendering, no GUI

"""

import os
import time
from gym import spaces
import robosuite as suite

os.environ["MUJOCO_GL"] = "egl"
if __name__ == "__main__":
    env = suite.make(
        env_name="TwoArmTransport",
        robots=["Panda", "Panda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_heights=96,
        camera_widths=96,
        camera_names="shouldercamera0",
        render_gpu_device_id=0,
        horizon=20,
    )
    obs, done = env.reset(), False
    print("Finished resetting!")
    low, high = env.action_spec
    action_space = spaces.Box(low=low, high=high)
    steps, time_stamp = 0, time.time()
    while True:
        while not done:
            obs, reward, done, info = env.step(action_space.sample())
            steps += 1
        obs, done = env.reset(), False
        print(f"FPS: {steps / (time.time() - time_stamp)}")
        steps, time_stamp = 0, time.time()
