import os
os.environ['MUJOCO_GL'] = 'osmesa'
import robosuite
env = robosuite.make(
    "PickPlaceCan",
    robots="Panda",  # Specify the robot (e.g., "Panda", "Sawyer", etc.)
    has_renderer=False,
    has_offscreen_renderer=True,
    render_gpu_device_id=7
)
env.reset()
print("Environment initialized successfully!")



