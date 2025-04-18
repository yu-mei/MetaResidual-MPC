import numpy as np
import pandas as pd
import os
import time
import imageio
import pybullet as p
from tqdm import tqdm
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor

# === Setup paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = "metamlp_seed41.csv"  # Change this to your desired file
csv_path = os.path.join(script_dir, csv_filename)
video_dir = os.path.join(script_dir, "replay_videos")
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, f"{os.path.splitext(csv_filename)[0]}_replay_HD.mp4")

# === Constants ===
fps = 50
dt = 1.0 / fps

# === Load CSV ===
df = pd.read_csv(csv_path)
assert all(col in df.columns for col in ["x", "x_dot", "z", "z_dot", "theta", "theta_dot", "u1", "u2"])

# === Color code based on file name ===
def get_color_from_name(name):
    if name.startswith("nominal"):
        return [0.12, 0.47, 0.71]  # C0
    elif name.startswith("lightmlp"):
        return [1.00, 0.50, 0.05]  # C1
    elif name.startswith("metamlp"):
        return [0.17, 0.63, 0.17]  # C2
    else:
        return [0.5, 0.5, 0.5]     # Default gray

traj_color = get_color_from_name(csv_filename)

# === Env setup ===
env_config = {
    'gui': True,
    'ctrl_freq': fps,
    'pyb_freq': fps,
    'done_on_out_of_bound': False,
    'mass': df["mass"].iloc[0] if "mass" in df.columns else 0.775,
}
env = Quadrotor(**env_config)
env.reset(seed=42)
client = env.PYB_CLIENT
drone_id = env.DRONE_IDS[0]

# === Hide GUI overlays ===
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client)

# === Camera & render resolution ===
render_width, render_height = 1280, 720
env.RENDER_WIDTH = render_width
env.RENDER_HEIGHT = render_height
p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=0,
    cameraPitch=0,
    cameraTargetPosition=[0, 0, 1],
    physicsClientId=client
)

# === Draw origin frame ===
axis_len = 0.8
p.addUserDebugLine([0, 0, 0], [axis_len, 0, 0], [1, 0, 0], 2, 0, client)
p.addUserDebugLine([0, 0, 0], [0, axis_len, 0], [0, 1, 0], 2, 0, client)
p.addUserDebugLine([0, 0, 0], [0, 0, axis_len], [0, 0, 1], 2, 0, client)

# === Reference trajectory setup (circle) ===
circle_radius = 0.5
circle_center_x = 0.0
circle_center_z = 1.0
circle_period = 15
omega = 2 * np.pi / circle_period

# === Red goal marker ===
goal_marker_vis_id = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=0.01,
    rgbaColor=[1, 0, 0, 1],
    physicsClientId=client
)
goal_marker_id = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=goal_marker_vis_id,
    basePosition=[0, 0, 0],
    physicsClientId=client
)

# === Helper to draw capsule between two points ===
def create_capsule_segment(from_xyz, to_xyz, color, radius=0.001):
    vec = np.array(to_xyz) - np.array(from_xyz)
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return
    mid = (np.array(from_xyz) + np.array(to_xyz)) / 2
    vis_id = p.createVisualShape(
        shapeType=p.GEOM_CAPSULE,
        radius=radius,
        length=length,
        rgbaColor=color + [1],
        physicsClientId=client
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vis_id,
        basePosition=mid.tolist(),
        baseOrientation=[0, 0, 0, 1],
        physicsClientId=client
    )

# === Start recording ===
frames = []
prev_ref_pos, prev_real_pos = None, None

print(f"📽️ Recording to {video_path}...")
for i in tqdm(range(len(df)), desc="Replaying"):
    row = df.iloc[i]
    x, x_dot, z, z_dot = row["x"], row["x_dot"], row["z"], row["z_dot"]
    theta, theta_dot = row["theta"], row["theta_dot"]
    u1, u2 = row["u1"], row["u2"]

    # Reference pos
    t_now = i * dt
    ref_x = circle_center_x + circle_radius * np.cos(omega * t_now)
    ref_z = circle_center_z + circle_radius * np.sin(omega * t_now)
    ref_pos = [ref_x, 0, ref_z]
    if prev_ref_pos is not None:
        create_capsule_segment(prev_ref_pos, ref_pos, [1, 0, 0], radius=0.005)
    prev_ref_pos = ref_pos
    p.resetBasePositionAndOrientation(goal_marker_id, ref_pos, [0, 0, 0, 1], client)

    # Real pos
    real_pos = [x, 0, z]
    if prev_real_pos is not None:
        create_capsule_segment(prev_real_pos, real_pos, traj_color, radius=0.005)
    prev_real_pos = real_pos

    # Drone pose & velocity
    quat = p.getQuaternionFromEuler([0, theta, 0])
    p.resetBasePositionAndOrientation(drone_id, [x, 0, z], quat, client)
    p.resetBaseVelocity(drone_id, [x_dot, 0, z_dot], [0, theta_dot, 0], client)
    p.applyExternalForce(drone_id, -1, [0, 0, u1 + u2], [x, 0, z], p.WORLD_FRAME, client)

    # Render
    p.stepSimulation(client)
    view_matrix = p.getDebugVisualizerCamera()[2]
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=render_width/render_height, nearVal=0.1, farVal=100.0)
    _, _, px, _, _ = p.getCameraImage(
        width=render_width, height=render_height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client
    )
    frame = np.reshape(px, (render_height, render_width, 4))[:, :, :3]
    frames.append(frame)
    time.sleep(dt)

# Save video
print(f"📼 Saving to {video_path}...")
with imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8) as writer:
    for frame in tqdm(frames, desc="Writing"):
        writer.append_data(frame)

print("✅ Done. Video saved to:", video_path)
env.close()
