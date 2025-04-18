import numpy as np
import pandas as pd
import os
import time
import imageio
import pybullet as p
from safe_control_gym.envs.gym_control.cartpole import CartPole

# === Setup paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = "metamlp_seed41.csv"  # 🔁 Change filename here
csv_path = os.path.join(script_dir, csv_filename)
video_dir = os.path.join(script_dir, "replay_videos")
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, f"{os.path.splitext(csv_filename)[0]}_replay_HD.mp4")

fps = 50  # Match your control frequency

# === Load data ===
df = pd.read_csv(csv_path)
assert all(col in df.columns for col in ["x", "x_dot", "theta", "theta_dot", "u"]), "Missing required columns in CSV"

# === Environment setup ===
env_config = {
    'gui': True,
    'ctrl_freq': fps,
    'pyb_freq': fps,
    'done_on_out_of_bound': False,
    'inertial_prop': {
        'pole_length': 0.5,
        'cart_mass': 1.0,
        'pole_mass': 0.1
    }
}
env = CartPole(**env_config)

# Set higher resolution BEFORE reset
env.RENDER_WIDTH = 1280   # HD
env.RENDER_HEIGHT = 720
env.reset(seed=42)
client = env.PYB_CLIENT
dt = 1.0 / fps

# === Video Writer ===
frames = []
render_width = env.RENDER_WIDTH
render_height = env.RENDER_HEIGHT

# === Hide overlays
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client)

# === Camera view
p.resetDebugVisualizerCamera(
    cameraDistance=2.5, cameraYaw=0, cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0], physicsClientId=client
)

# === Red static goal marker
goal_pos = [0, 0, 0]
goal_vis_id = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=0.1,
    rgbaColor=[1, 0, 0, 1],
    physicsClientId=client
)
p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=goal_vis_id,
    basePosition=goal_pos,
    physicsClientId=client
)

# === Red vertical reference line at x = 0
ref_x = 0.0
ref_z_start = 0.0
ref_z_end = 1.5
ref_height = ref_z_end - ref_z_start
ref_thickness = 0.02

ref_box_vis_id = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[ref_thickness / 2, ref_thickness / 2, ref_height / 2],
    rgbaColor=[1, 0, 0, 1],  # Red
    physicsClientId=client
)

ref_box_id = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=ref_box_vis_id,
    basePosition=[ref_x, 0, ref_z_start + ref_height / 2],
    physicsClientId=client
)



print(f"Replaying and recording trajectory (GUI view) to {video_path}...")

for i in range(len(df)):
    row = df.iloc[i]
    x, x_dot = row["x"], row["x_dot"]
    theta, theta_dot = row["theta"], row["theta_dot"]
    u = row["u"]

    # Reset joint states
    p.resetJointState(env.CARTPOLE_ID, 0, x, x_dot, physicsClientId=client)
    p.resetJointState(env.CARTPOLE_ID, 1, theta, theta_dot, physicsClientId=client)

    # Visual-only force
    p.setJointMotorControl2(
        env.CARTPOLE_ID,
        jointIndex=0,
        controlMode=p.TORQUE_CONTROL,
        force=u,
        physicsClientId=client
    )

    p.stepSimulation(physicsClientId=client)

    # Use the GUI camera (visualizer) for recording
    cam_info = p.getDebugVisualizerCamera(physicsClientId=client)
    view_matrix = cam_info[2]
    projection_matrix = cam_info[3]

    # Capture frame from GUI view
    _, _, px, _, _ = p.getCameraImage(
        width=render_width,
        height=render_height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        physicsClientId=client
    )
    frame = np.reshape(px, (render_height, render_width, 4))[:, :, :3]
    frames.append(frame)

    time.sleep(dt)  # Real-time pacing

# === Save video ===
print(f"Saving {len(frames)} frames to {video_path}...")
imageio.mimsave(video_path, frames, fps=fps)
print("✅ Done! HD MP4 saved to:", video_path)

env.close()
