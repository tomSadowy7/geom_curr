import argparse
import json
import time
import mujoco

from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as pb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

#mug lifting env used for both train_all.py (difficulty labeling training) and experiments

#path for 7 DOF franka arm in mujoco menageria submodule repo
PANDA_MODEL_DIR = Path(__file__).parent.parent / "mujoco_menagerie" / "franka_emika_panda"
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]
GRIPPER_ACT = "actuator8" 

#JOINT LIMITS FOR FRANKA ARM (slightly higher than online definitions for additional mobility 
J_LOW = np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671])
J_HIGH = np.array([2.9671, 1.8326, 2.9671, 0.0000, 2.9671, 3.8223, 2.9671])
J_REST = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
J_PREGRASP = np.array([-0.383, -0.488, 0.406, -2.878, 0.186, 3.729, 0.958])

#simulation and control timing (run control action for several Mujoco steps)
MAX_JVEL = 0.5   
SIM_HZ = 500     
CTRL_HZ = 20     
SIM_PER_CTRL = SIM_HZ // CTRL_HZ

#starting pos for table, platform, franka arm
TABLE_POS = np.array([0.55, 0.0, 0.20])
TABLE_HALF_EXT = np.array([0.30, 0.30, 0.20])
TABLE_TOP_Z = float(TABLE_POS[2] + TABLE_HALF_EXT[2])
PLATFORM_HEIGHT = 0.08  #8 cm platform allows for easier grasping affordance (very tricky w/o this)
PLATFORM_TOP_Z = float(TABLE_TOP_Z + PLATFORM_HEIGHT)
LIFT_GOAL_M = 0.05 #5 cm lift -> success for mug lift.
PREGRASP_BACKOFF = 0.14 #how far we start behind cup
GRIPPER_OPEN = 255.0

#appends a JSON object as a line to a JSONL file
def append_jsonl(path, data):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")

#split VHACD convex pieces in one .obj into individual .objs for easier Mujoco loading
def split_vhacd(col_obj, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True) #ensure dir exists
    existing = sorted(out_dir.glob("part_*.obj"))
    if existing:
        return existing

    verts = []
    groups = []
    name, faces = "part_000", []

    with open(col_obj) as fp:
        raw_lines = fp.readlines()

    #parse Infinigen obj file
    for raw in raw_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            verts.append(line)
        elif line.startswith(("o ", "g ")):
            if faces:
                groups.append((name, faces))
            name, faces = line.split(None, 1)[1].replace("/", "_"), []
        elif line.startswith("f "):
            faces.append(line)
    if faces:
        groups.append((name, faces))

    if not groups:
        dst = out_dir / "part_000.obj"
        dst.write_text(col_obj.read_text())
        return [dst]

    paths = []
    for i, (_, fcs) in enumerate(groups):
        #remap vertix indices
        used = sorted({int(t.split("/")[0]) for f in fcs for t in f.split()[1:]})
        remap = {o: n for n, o in enumerate(used, 1)}
        dst = out_dir / f"part_{i:03d}.obj"
        with open(dst, "w") as fp:
            for old in used:
                fp.write(verts[old - 1] + "\n")
            for f in fcs:
                toks = f.split()[1:]
                new = []
                for t in toks:
                    p = t.split("/")
                    p[0] = str(remap[int(p[0])])
                    new.append("/".join(p))
                fp.write("f " + " ".join(new) + "\n")
        paths.append(dst)
    return paths


#helper to figure out bottom of mug mesh to properly place on platform
#instead of having it sink through floor
def mesh_axis_min(obj_path, axis):
    best = float("inf")
    for line in open(obj_path):
        if line.startswith("v "):
            best = min(best, float(line.split()[axis + 1]))
    return best if best != float("inf") else 0.0

#build XML scene to load into Mujoco
def build_scene(cup_dir, mug_params, grip_scale, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)

    col_obj = cup_dir / "cup_collision.obj"
    parts = split_vhacd(col_obj, cup_dir / "mujoco_parts")
    vis_obj = cup_dir / "cup_visual.obj"

    scale = mug_params.get("scale", 0.20)
    depth = mug_params.get("depth", 0.35)
    h = depth * scale * grip_scale
    cup_z = PLATFORM_TOP_Z + h / 2.0

    panda_xml_src = (PANDA_MODEL_DIR / "panda.xml").resolve()
    assets_dir = (PANDA_MODEL_DIR / "assets").resolve()

    #create patched panda.xml to resolve asset patches 
    panda_patched = (work_dir / "panda_patched.xml").resolve()
    patched_text = panda_xml_src.read_text().replace(
        'meshdir="assets"', f'meshdir="{assets_dir}"')
    
    #replace end effector gripper collision friction
    patched_text = patched_text.replace('<geom type="mesh" group="3"/>', '<geom type="mesh" group="3" friction="6 0.5 0.01" condim="6"/>')
    patched_text = patched_text.replace('<general class="panda" name="actuator8" tendon="split" forcerange="-100 100" ctrlrange="0 255"\n'
        '      gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>',
        '<general class="panda" name="actuator8" tendon="split" forcerange="-300 300" ctrlrange="0 255"\n'
        '      gainprm="0.04705882353 0 0" biasprm="0 -300 -30"/>')
    patched_text = patched_text.replace('<geom class="fingertip_pad_collision_5"/>\n'
        '                      </body>\n'
        '                      <body name="right_finger"',
        '<geom class="fingertip_pad_collision_5"/>\n'
        '                        <site name="left_fingertip" '
        'pos="0 0.0055 0.0445" size="0.004" rgba="0 1 0 1"/>\n'
        '                      </body>\n'
        '                      <body name="right_finger"', 1)
    patched_text = patched_text.replace('<geom class="fingertip_pad_collision_5"/>\n'
        '                      </body>\n'
        '                    </body>',
        '<geom class="fingertip_pad_collision_5"/>\n'
        '                        <site name="right_fingertip" '
        'pos="0 0.0055 0.0445" size="0.004" rgba="0 1 0 1"/>\n'
        '                      </body>\n'
        '                    </body>', 1)
    panda_patched.write_text(patched_text)
    panda_xml = panda_patched

    #assets for table and mug collision/visual mesh geometries
    asset_lines = ['<material name="table_mat"  rgba="0.55 0.35 0.10 1"/>', '<material name="cup_vis_mat" rgba="0.2 0.5 0.9 1"/>']
    grip_scale_str = " ".join(f"{float(x):.6f}" for x in [grip_scale] * 3)
    for i, part in enumerate(parts):
        asset_lines.append(f'<mesh name="hull_{i:03d}" file="{part.resolve()}" '
            f'scale="{grip_scale_str}"/>')
    if vis_obj.exists():
        asset_lines.append(f'<mesh name="cup_vis" file="{vis_obj.resolve()}" '
            f'scale="{grip_scale_str}"/>')

    #build XML lines for mug's collision geometries and visual geometries
    cup_geoms = []
    for i in range(len(parts)):
        cup_geoms.append(f'<geom name="cup_col_{i:03d}" type="mesh" mesh="hull_{i:03d}" '
            f'friction="4 0.02 0.002" solref="0.004 1" solimp="0.95 0.99 0.001" '
            f'condim="6" rgba="1.0 0.2 0.2 0.4"/>')
    if vis_obj.exists():
        cup_geoms.append('<geom name="cup_vis_geom" type="mesh" mesh="cup_vis" '
            'contype="0" conaffinity="0" material="cup_vis_mat" group="2"/>')

    #mujoco spacing helper
    asset_str = "\n    ".join(asset_lines)
    cup_geom_str = "\n        ".join(cup_geoms)

    #format vectors for XML
    table_pos_str = " ".join(f"{float(x):.6f}" for x in TABLE_POS)
    table_extent_str = " ".join(f"{float(x):.6f}" for x in TABLE_HALF_EXT)
    platform_pos_str = " ".join(f"{float(x):.6f}" for x in [TABLE_POS[0] + 0.10, TABLE_POS[1], TABLE_TOP_Z + PLATFORM_HEIGHT/2.0])
    platform_size_str = " ".join(f"{float(x):.6f}" for x in [0.10, 0.10, PLATFORM_HEIGHT/2.0])
    cup_pos_str = " ".join(f"{float(x):.6f}" for x in [TABLE_POS[0], TABLE_POS[1], cup_z])

    #build env XML for Mujoco to load
    xml = f"""<mujoco model="mug_lift">
        <include file="{panda_xml}"/>

        <option timestep="{1.0/SIM_HZ:.6f}" gravity="0 0 -9.81"
                integrator="implicitfast" iterations="20" cone="elliptic"/>

        <asset>
            {asset_str}
        </asset>

        <worldbody>
            <light name="scene_light" pos="0.5 0 2.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
            <camera name="task_cam"
                    pos="1.20 -0.90 0.95"
                    xyaxes="0.707 0.707 0 -0.354 0.354 0.866"
                    fovy="55"/>

            <geom name="floor" type="plane" size="2 2 0.05"
                rgba="0.3 0.3 0.3 1" friction="1 0.01 0.001"/>

            <body name="table" pos="{table_pos_str}">
            <geom name="table_geom" type="box" size="{table_extent_str}"
                    material="table_mat" friction="1 0.01 0.001"/>
            </body>

            <body name="platform" pos="{platform_pos_str}">
            <geom name="platform_geom" type="box" size="{platform_size_str}"
                    rgba="0.4 0.4 0.4 1" friction="1 0.01 0.001"/>
            </body>

            <body name="cup" pos="{cup_pos_str}">
            <freejoint name="cup_joint"/>
            <inertial pos="0 0 0" mass="0.30" diaginertia="0.005 0.005 0.008"/>
            {cup_geom_str}
            </body>
        </worldbody>
        </mujoco>"""

    out = work_dir / "scene.xml"
    out.write_text(xml)
    return out


#stable baselines callback for tracking training statistics
class LiveLogger(BaseCallback):
    def __init__(self, log_freq=5_000, events_path=None):
        super().__init__()
        self.log_freq = log_freq
        self.events_path = events_path
        self.reward_buffer = []
        self.last_log_step = 0
        self.successes = 0

    def _on_step(self):

        for info in self.locals.get("infos", []):
            if info.get("is_success", False):
                self.successes += 1
                lift_cm = 100.0 * float(info.get("lift_m", 0.0))
                print(f"SUCCESS #{self.successes} at step {self.num_timesteps:,} lifted {lift_cm:.1f} cm")
                append_jsonl(self.events_path, {
                    "type": "success",
                    "step": self.num_timesteps,
                    "success_count": self.successes,
                    "lift_cm": lift_cm,
                })
            if "episode" in info:
                ep_r = info["episode"]["r"]
                self.reward_buffer.append(ep_r)

        if self.num_timesteps - self.last_log_step >= self.log_freq and self.reward_buffer:
            mean_r = float(np.mean(self.reward_buffer))
            print(f"step {self.num_timesteps:>8,} | mean ep reward = {mean_r:+.4f}  ({len(self.reward_buffer)} eps)")
            append_jsonl(self.events_path, {
                "type": "mean_ep_reward",
                "step": self.num_timesteps,
                "mean_ep_reward": mean_r,
                "episodes": len(self.reward_buffer),
            })
            self.reward_buffer.clear()
            self.last_log_step = self.num_timesteps
        return True

#record eval videos during training
class RolloutVideoCallback(BaseCallback):
    W, H = 640, 480

    def __init__(self, env_kwargs, rollout_freq, video_dir):
        super().__init__()
        self.env_kwargs = env_kwargs
        self.rollout_freq = rollout_freq
        self.video_dir = video_dir
        self.last_rollout = 0

    def _on_step(self):
        if self.rollout_freq and self.num_timesteps - self.last_rollout >= self.rollout_freq:
            self.record_rollout()
            self.last_rollout = self.num_timesteps
        return True

    def record_rollout(self):
        
        env = MujocoMugLiftEnv(**self.env_kwargs)
        obs, _ = env.reset()
        frames, total_r = [], 0.0
        terminated, truncated = False, False
        last_info = {}
        
        #step through env
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True) #rollouts -> deterministic, trianning -> stochastic
            obs, r, terminated, truncated, last_info = env.step(action)
            total_r += r
            frames.append(env.render_frame(self.W, self.H))
        env.close()

        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        #explicitly name successes for better accessibility to successful runs
        success_tag = "_success" if last_info.get("is_success", False) else ""
        out = self.video_dir / (f"step_{self.num_timesteps:08d}_r{total_r:.2f}{success_tag}.mp4")
        imageio.mimwrite(str(out), frames, fps=15, quality=5, format="ffmpeg")
        print(f"video saved at {out.name} | reward {total_r:.3f})")

#create mug lifting env
class MujocoMugLiftEnv(gym.Env):
    OBS_DIM = 41 #privileged env state
    ACT_DIM = 8  #7 franka arm joints + open/close end-effector
    MAX_STEPS = 300 #number of env steps
    #cup upright quaternion
    CUP_QUAT = np.array([0.7071068, 0.7071068, 0.0, 0.0])  #[qw, qx, qy, qz]

    #map between name and joint idx
    def joint_id(self, name):
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return joint_id

    def __init__(self, scene_xml, cup_dir, mug_params,
                 grip_scale=1.0, real_time=False):
        
        super().__init__()
        self.mug_params = mug_params
        self.grip_scale = grip_scale
        self.real_time_enabled = real_time

        #observation and action spaces used by PPO.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.ACT_DIM,), dtype=np.float32)

        #load the MuJoCo model and simulation state.
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)

        #arm joint IDs and addresses in MuJoCo arrays.
        self.arm_jids = [self.joint_id(n) for n in ARM_JOINT_NAMES]
        self.arm_qpos_adrs = np.array([self.model.jnt_qposadr[j] for j in self.arm_jids])
        self.arm_dof_adrs = np.array([self.model.jnt_dofadr[j] for j in self.arm_jids])

        #finger joint IDs and addresses.
        self.finger_jids = [self.joint_id(n) for n in FINGER_JOINT_NAMES]
        self.finger_qpos_adrs = [self.model.jnt_qposadr[j] for j in self.finger_jids]
        self.finger_dof_adrs = [self.model.jnt_dofadr[j] for j in self.finger_jids]

        #arm actuators 0-6, gripper actuator 7
        self.gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)

        #body and site IDs
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.left_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_fingertip")
        self.right_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_fingertip")

        #cup body and e joint locations.
        self.cup_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.cup_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_joint")
        self.cup_qpos_adr = self.model.jnt_qposadr[self.cup_jid]
        self.cup_dof_adr = self.model.jnt_dofadr[self.cup_jid]

        self.renderer = None

        scale = mug_params.get("scale", 0.20)
        depth = mug_params.get("depth", 0.35)
        x_end = mug_params.get("x_end", 0.25)
        self.cup_h = depth * scale * grip_scale
        self.cup_r = x_end * scale * grip_scale

        #rotate cup mesh 90 degrees around X so local Y maps to world Z
        #(weird Mujoco localization)
        parts_dir = cup_dir / "mujoco_parts"
        part_files = sorted(parts_dir.glob("part_*.obj"))
        if part_files:
            y_min_local = min(mesh_axis_min(p, 1) for p in part_files)
            y_max_local = max(max(float(line.split()[2]) for line in open(p) if line.startswith("v ")) for p in part_files)
        else:
            vis_obj = cup_dir / "cup_visual.obj"
            col_obj = cup_dir / "cup_collision.obj"
            src = col_obj if col_obj.exists() else vis_obj
            y_min_local = mesh_axis_min(src, 1) if src.exists() else 0.0
            y_max_local = 0.0
            if src.exists():
                for line in open(src):
                    if line.startswith("v "):
                        y_max_local = max(y_max_local, float(line.split()[2]))

        #place the cup so its lowest mesh point sits on the platform
        self.cup_body_z = PLATFORM_TOP_Z - y_min_local * grip_scale
        self.cup_height = (y_max_local - y_min_local) * grip_scale
        self.handle_location = mug_params.get("handle_location", 0.5)

        #values reset or updated during each episode.
        self.step_count = 0
        self.cup_start_z = 0.0
        self.goal_z = 0.0
        self.target_q = J_REST.copy()
        self.prev_potential = 0.0
        self.prev_pair_dist = 0.0


    #return the midpoint between the two end-effector finger sites
    def ee_pos(self):
        left = self.data.site_xpos[self.left_tip_site_id]
        right = self.data.site_xpos[self.right_tip_site_id]
        return 0.5 * (left + right)

    #compute the avg position Jacobian of the end effector tips
    def ee_jac_pos(self):
        jac_left = np.zeros((3, self.model.nv))
        jac_right = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_left, None, self.left_tip_site_id)
        mujoco.mj_jacSite(self.model, self.data, jac_right, None, self.right_tip_site_id)
        return 0.5 * (jac_left + jac_right)

    #return the current world position of the cup body.
    def cup_pos(self):
        return self.data.xpos[self.cup_body_id].copy()

    #pick grasp point halfway up the cup.
    def grasp_target(self, cup):
        grasp_z = cup[2] + 0.5 * self.cup_height
        return np.array([cup[0], cup[1], grasp_z], dtype=np.float64)

    #compute target points on opposite sides of the cup for the end effector fingertips
    def finger_side_targets(self, cup):
        grasp_z = cup[2] + 0.5 * self.cup_height
        side_offset = self.cup_r + 0.006
        plus_y = np.array([cup[0], cup[1] + side_offset, grasp_z], dtype=np.float64)
        minus_y = np.array([cup[0], cup[1] - side_offset, grasp_z], dtype=np.float64)
        return plus_y, minus_y

    #measure how close the end-effector fingertips are to opposite sides of the cup
    def finger_pair_distance(self, cup):
        left_tip = self.data.site_xpos[self.left_tip_site_id]
        right_tip = self.data.site_xpos[self.right_tip_site_id]
        plus_y, minus_y = self.finger_side_targets(cup)
        direct = np.linalg.norm(left_tip - plus_y) + np.linalg.norm(right_tip - minus_y)
        swapped = np.linalg.norm(left_tip - minus_y) + np.linalg.norm(right_tip - plus_y)
        return float(min(direct, swapped))

    #cllect all body IDs under mujoco body
    def body_subtree_ids(self, root_body_id):
        ids = {root_body_id}
        changed = True
        while changed:
            changed = False
            for body_id in range(self.model.nbody):
                parent_id = int(self.model.body_parentid[body_id])
                if body_id not in ids and parent_id in ids:
                    ids.add(body_id)
                    changed = True
        return ids

    #cache collision geom IDs for the cup and gripper fingers for faster loading
    def init_contact_geom_sets(self):

        cup_bodies = self.body_subtree_ids(self.cup_body_id)
        left_bodies = self.body_subtree_ids(self.left_finger_body_id)
        right_bodies = self.body_subtree_ids(self.right_finger_body_id)

        self.cup_geom_ids = set()
        self.left_finger_geoms = set()
        self.right_finger_geoms = set()
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id in cup_bodies:
                self.cup_geom_ids.add(geom_id)
            if body_id in left_bodies:
                self.left_finger_geoms.add(geom_id)
            if body_id in right_bodies:
                self.right_finger_geoms.add(geom_id)
        self.finger_geom_ids = self.left_finger_geoms | self.right_finger_geoms

    #setter function for the cup position, orientation, and joint velocity
    def set_cup_pose(self, pos):
        a = self.cup_qpos_adr
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = self.CUP_QUAT
        self.data.qvel[self.cup_dof_adr:self.cup_dof_adr + 6] = 0.0

    #setter function for the arm pose and zero its velocity
    def set_arm(self, q):
        q = np.clip(q, J_LOW, J_HIGH)
        self.data.qpos[self.arm_qpos_adrs] = q
        self.data.qvel[self.arm_dof_adrs] = 0.0
        self.data.ctrl[:7] = q

    #setter function for open/close command to the gripper actuator
    def set_gripper(self, val):
        self.data.ctrl[self.gripper_act_id] = val

    #INVERSE KINEMATICS FOR end-effector motion
    def ik(self, target, q0=None, n_iter=100, tol=5e-3):
        q = (q0 if q0 is not None else J_REST).copy()
        lam = 0.05
        for _ in range(n_iter):
            self.data.qpos[self.arm_qpos_adrs] = np.clip(q, J_LOW, J_HIGH)
            mujoco.mj_forward(self.model, self.data)
            err = target - self.ee_pos()
            if np.linalg.norm(err) < tol:
                break
            jacp = self.ee_jac_pos()
            J = jacp[:, self.arm_dof_adrs]
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
            q = np.clip(q + dq, J_LOW, J_HIGH)
        return q

    #target rotation for handle approach
    PREGRASP_ROT = np.array([[0., 0., 1.], [-1., 0., 0.], [0.,  1., 0.]], dtype=np.float64)

    #solve position-and-orientation inverse kinemaitcs for end effector
    def ik_6dof(self, target_pos, target_rot, q0=None, n_iter=600, lam=0.05):
        q = (q0 if q0 is not None else J_REST).copy()
        for _ in range(n_iter):
            self.data.qpos[self.arm_qpos_adrs] = np.clip(q, J_LOW, J_HIGH)
            mujoco.mj_forward(self.model, self.data)
            pos = self.ee_pos()
            R = self.data.xmat[self.hand_body_id].reshape(3, 3)
            ep = target_pos - pos
            Rerr = target_rot @ R.T
            ea = np.array([Rerr[2, 1] - Rerr[1, 2],Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]]) * 0.5
            if np.linalg.norm(ep) < 0.008 and np.linalg.norm(ea) < 0.05:
                break
            
            err = np.concatenate([ep, ea * 0.3])
            jacp = self.ee_jac_pos()
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, None, jacr, pos, self.hand_body_id)
            J = np.vstack([jacp[:, self.arm_dof_adrs], jacr[:, self.arm_dof_adrs] * 0.3])
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            q = np.clip(q + dq, J_LOW, J_HIGH)
        return q


    #GYMNASIUM API functions for stepping through env

    #reset the episode with the cup near the platform center and end-effector pre-positioned
    def reset(self, *, seed=None, options=None):
        super().reset()
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        dx = float(np.random.uniform(-0.02, 0.02))
        dy = float(np.random.uniform(-0.02, 0.02))
        cup_pos = np.array([TABLE_POS[0] + 0.10 + dx, TABLE_POS[1] + dy, self.cup_body_z])

        #place cup on the platform.
        self.set_cup_pose(cup_pos)
        #set gripper fully open: both fingers at max travel (0.04 m)
        
        for addr in self.finger_qpos_adrs:
            self.data.qpos[addr] = 0.04
        self.set_gripper(GRIPPER_OPEN)

        grasp_pos = self.grasp_target(cup_pos) + np.array([-self.cup_r - PREGRASP_BACKOFF, 0.0, 0.0])
        q_grasp = self.ik_6dof(grasp_pos, self.PREGRASP_ROT, q0=J_PREGRASP)
        self.set_arm(q_grasp)
        self.target_q = q_grasp.copy()

        mujoco.mj_forward(self.model, self.data)
        self.data.qvel[self.cup_dof_adr:self.cup_dof_adr + 6] = 0.0
        self.data.qvel[self.arm_dof_adrs] = 0.0
        self.data.ctrl[self.gripper_act_id] = GRIPPER_OPEN

        self.cup_start_z = float(self.cup_pos()[2])
        self.goal_z = self.cup_start_z + LIFT_GOAL_M

        #init potential for reward shaping
        mujoco.mj_forward(self.model, self.data)
        ee = self.ee_pos()
        cup = self.cup_pos()
        finger_pos = self.data.qpos[self.finger_qpos_adrs[0]]
        gripper_closed_frac = 1.0 - (finger_pos / 0.04)
        cup_quat = self.data.qpos[self.cup_qpos_adr + 3:self.cup_qpos_adr + 7]
        quat_dot = float(np.abs(np.dot(cup_quat, self.CUP_QUAT)))
        self.prev_potential = self.potential_score(
            ee, cup, gripper_closed_frac, False, 0.0, quat_dot)
        self.prev_pair_dist = self.finger_pair_distance(cup)

        return self.get_obs(), {}

    #apply one policy action, advance Mujoco, and return the Gymnasium step tuple
    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        #action[:7] = joint velocity deltas.
        #action[7] = gripper command
        self.target_q = np.clip(self.target_q + action[:7] * MAX_JVEL / CTRL_HZ, J_LOW, J_HIGH)

        gripper_cmd = float((action[7] + 1.0) / 2.0 * 255.0)

        t0 = time.time()
        for _ in range(SIM_PER_CTRL):
            self.data.ctrl[:7] = self.target_q
            self.data.ctrl[self.gripper_act_id] = gripper_cmd
            mujoco.mj_step(self.model, self.data)
        if self.real_time_enabled:
            dt = 1.0 / CTRL_HZ - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

        obs = self.get_obs()
        reward, info = self.compute_reward(action)
        terminated = info["goal_bonus"] > 0.0
        truncated = self.step_count >= self.MAX_STEPS
        return obs, reward, terminated, truncated, info


    #build the 41-dimensional privileged observation vector for PPO
    def get_obs(self):
        q = self.data.qpos[self.arm_qpos_adrs].astype(np.float32)
        qv = self.data.qvel[self.arm_dof_adrs].astype(np.float32)
        q_norm = (2.0 * (q - J_LOW) / (J_HIGH - J_LOW) - 1.0).astype(np.float32)

        ee = self.ee_pos().astype(np.float32)
        cup = self.cup_pos().astype(np.float32)
        goal = cup.copy()
        goal[2] = float(self.goal_z)

        #grasp target
        grasp_target = self.grasp_target(cup).astype(np.float32)

        #rel geometry
        ee_to_grasp = (grasp_target - ee).astype(np.float32)
        ee_to_cup = (cup - ee).astype(np.float32)
        cup_to_goal = (goal - cup).astype(np.float32)

        #lift
        lift = np.array([cup[2] - self.cup_start_z], dtype=np.float32)

        #end effector state
        finger_left = np.array([self.data.qpos[self.finger_qpos_adrs[0]] / 0.04], dtype=np.float32)
        finger_right = np.array([self.data.qpos[self.finger_qpos_adrs[1]] / 0.04], dtype=np.float32)
        finger_vel_left = np.array([self.data.qvel[self.finger_dof_adrs[0]]], dtype=np.float32)
        finger_vel_right = np.array([self.data.qvel[self.finger_dof_adrs[1]]], dtype=np.float32)

        #contact state
        has_contact = self.has_finger_cup_contact()
        left_contact = self.check_finger_cup_contact(0)  # left finger
        right_contact = self.check_finger_cup_contact(1)  # right finger
        num_contacts = float(sum([left_contact, right_contact]))

        #estimate normal force using contact state and closure as a proxy
        gripper_closed_frac = (
            1.0 - self.data.qpos[self.finger_qpos_adrs[0]] / 0.04)
        contact_force = float(has_contact) * gripper_closed_frac * 100.0

        #cup motion stuff
        cup_vel = self.data.qvel[self.cup_dof_adr:self.cup_dof_adr + 3].astype(np.float32)
        cup_ang_vel = self.data.qvel[self.cup_dof_adr + 3:self.cup_dof_adr + 6].astype(np.float32)

        # how upright is the cup
        cup_quat = self.data.qpos[self.cup_qpos_adr + 3:self.cup_qpos_adr + 7]
        quat_dot = float(np.abs(np.dot(cup_quat, self.CUP_QUAT)))
        uprightness = np.array([quat_dot], dtype=np.float32)

        #grasp quality proxies.
        two_sided_contact = np.array([
            float(left_contact and right_contact)], dtype=np.float32)
        closed_while_contacting = np.array([
            float(has_contact and gripper_closed_frac > 0.5)], dtype=np.float32)
        
        #7 + 7+ 3 + 3 + 3 + 1 + 2 + 2 + 1 + 1 + 1 + 1 + 3 + 3 + 1 + 1 + 1 = 41 dim
        return np.concatenate([q_norm, qv, ee_to_grasp, ee_to_cup, cup_to_goal, lift,                           
            finger_left, finger_right, finger_vel_left, finger_vel_right, np.array([left_contact], dtype=np.float32), 
            np.array([right_contact], dtype=np.float32), np.array([num_contacts], dtype=np.float32), 
            np.array([contact_force], dtype=np.float32), cup_vel, cup_ang_vel, uprightness,              
            two_sided_contact, closed_while_contacting]).astype(np.float32)


    #helper score used for reward shaping based on reach distance and cup uprightness.
    #not exactly potential function since we don't multiply by gamma downstream
    #multiplying by gamma actually hurt here for some reason...?
    def potential_score(self, ee, cup, gripper_closed_frac, has_contact,
                   cup_height_lifted, quat_dot):
        pair_dist = self.finger_pair_distance(cup)
        reach_pot = 5.0 * np.exp(-10.0 * pair_dist**2)
        contact_pot = 0.0
        closure_pot = 0.0
        lift_pot = 0.0
        upright_pot = 1.0 * quat_dot

        return float(reach_pot + contact_pot + closure_pot + lift_pot + upright_pot)

    #is either end of end-effector touching mug
    def has_finger_cup_contact(self):
        self.init_contact_geom_sets()
        for contact in self.data.contact[:self.data.ncon]:
            b1, b2 = contact.geom1, contact.geom2
            if (b1 in self.cup_geom_ids and b2 in self.finger_geom_ids) or (b1 in self.finger_geom_ids and b2 in self.cup_geom_ids):
                return True
        return False
    
    #is particular finger of end-effector touching mug?
    def check_finger_cup_contact(self, finger_idx):
        self.init_contact_geom_sets()
        target_geoms = (self.left_finger_geoms if finger_idx == 0 else self.right_finger_geoms)

        for contact in self.data.contact[:self.data.ncon]:
            b1, b2 = contact.geom1, contact.geom2
            if (b1 in self.cup_geom_ids and b2 in target_geoms) or (b1 in target_geoms and b2 in self.cup_geom_ids):
                return True
        return False

    #DENSE REWARD
    def compute_reward(self, action):
        #current end effector and cup positions
        ee = self.ee_pos()
        cup = self.cup_pos()

        #contact and grasp state info
        grasp_target = self.grasp_target(cup)
        pair_dist = self.finger_pair_distance(cup)
        has_contact = self.has_finger_cup_contact()
        left_contact = self.check_finger_cup_contact(0)
        right_contact = self.check_finger_cup_contact(1)
        two_sided_contact = left_contact and right_contact
        finger_pos = self.data.qpos[self.finger_qpos_adrs[0]]
        gripper_closed_frac = 1.0 - (finger_pos / 0.04)

        #lift amount and cup orientation
        lift = max(0.0, float(cup[2]) - self.cup_start_z)
        cup_quat = self.data.qpos[self.cup_qpos_adr + 3:self.cup_qpos_adr + 7]
        quat_dot = float(np.abs(np.dot(cup_quat, self.CUP_QUAT)))

        #reaching reward: approach from the front and move fingers toward the cup sides
        midline_error = abs(float(ee[1] - cup[1]))
        front_approach_ok = ee[0] < cup[0] + 0.02
        approach_corridor = (float(np.exp(-200.0 * midline_error**2)) if front_approach_ok else 0.0)
        finger_reach = approach_corridor * (1.0 - np.tanh(3.0 * pair_dist))
        raw_pair_progress = self.prev_pair_dist - pair_dist
        pair_progress = (approach_corridor * max(0.0, raw_pair_progress) + min(0.0, raw_pair_progress))

        #check whether the end effector fingertips form a valid side grasp
        left_tip = self.data.site_xpos[self.left_tip_site_id]
        right_tip = self.data.site_xpos[self.right_tip_site_id]
        tip_mid = 0.5 * (left_tip + right_tip)
        midpoint_centered_xy = np.linalg.norm(tip_mid[:2] - cup[:2]) < 0.018
        left_height_ok = abs(float(left_tip[2] - grasp_target[2])) < 0.025
        right_height_ok = abs(float(right_tip[2] - grasp_target[2])) < 0.025
        side_height_ok = left_height_ok and right_height_ok
        below_rim = left_tip[2] < cup[2] + 0.75 * self.cup_height and right_tip[2] < cup[2] + 0.75 * self.cup_height
        
        
        tips_straddle_cup = (left_tip[1] - cup[1]) * (right_tip[1] - cup[1]) < 0.0 and abs(float(left_tip[1] - right_tip[1])) > 0.015
        valid_two_sided_contact = (two_sided_contact and side_height_ok and below_rim
            and tips_straddle_cup and midpoint_centered_xy)

        contact_reward = 0.0
        grasp = 0.0
 
        #lifting reward and success bonus
        lift_reward = min(lift / LIFT_GOAL_M, 1.0)
        goal_bonus = 5.0 if lift >= LIFT_GOAL_M else 0.0

        #penalize tipping away from upright
        tilt = 1.0 - quat_dot
        tip_penalty = -0.5 * tilt 

        #penalize low, side, or wrong direction approaches to our mug
        too_low = max(0.0, float((grasp_target[2] - 0.04) - ee[2]))
        below_platform = max(0.0, float((PLATFORM_TOP_Z + 0.015) - ee[2]))
        low_penalty = -8.0 * too_low - 12.0 * below_platform
        side_route_penalty = -15.0 * max(0.0, midline_error - 0.025)
        wrong_side_penalty = (-4.0 if not front_approach_ok and not valid_two_sided_contact else 0.0)

        #shaping term based on improvement in the progress score -> not exactly potential bc not gamma
        #gamma broke reward before
        potential = self.potential_score(ee, cup, gripper_closed_frac, has_contact, lift, quat_dot)
        potential_delta = potential - self.prev_potential
        self.prev_potential = potential
        self.prev_pair_dist = pair_dist

        #minor penalty for large actions
        act_penalty = -0.002 * float(np.sum(action ** 2))
        
        #full reward function
        reward = (2.0 * finger_reach + 20.0 * pair_progress + 4.0 * lift_reward + 0.15 * potential_delta + goal_bonus + tip_penalty + low_penalty + side_route_penalty + wrong_side_penalty + act_penalty)

        #dict used for naming episode and knowing if success hapepned
        return float(reward), {"goal_bonus": goal_bonus, "is_success": goal_bonus > 0.0, "lift_m": lift}
        
    #render frame in video
    def render_frame(self, width=640, height=480):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        elif self.renderer.width != width or self.renderer.height != height:
            self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.renderer.update_scene(self.data, camera="task_cam")
        return self.renderer.render()
    
    #close renderer
    def close(self):
        try:
            if self.renderer is not None:
                self.renderer.close()
        except Exception:
            pass

def make_env(env_kwargs):
    def init_env():
        kwargs = dict(env_kwargs)
        return Monitor(MujocoMugLiftEnv(**kwargs))
    return init_env


#train PPO
def train(args):
    mug_dir = Path(args.mug)
    with open(mug_dir / "params.json") as f:
        mug_params = json.load(f)

    scale = mug_params.get("scale", 0.20)
    x_end = mug_params.get("x_end", 0.25)
    actual_diam = 2 * x_end * scale
    grip_scale = min(1.0, 0.076 / actual_diam)
    grip_scale *= args.mug_scale

    work_dir = Path(args.work_dir)
    scene_xml = build_scene(mug_dir, mug_params, grip_scale, work_dir)
    env_kwargs = dict(scene_xml=scene_xml, cup_dir=mug_dir, mug_params=mug_params,
                      grip_scale=grip_scale, real_time=args.real_time)

    env_fns = [make_env(env_kwargs) for _ in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns) #dummyEnv doesn't have speedsup here bc its sequential

    #load PPO
    model = PPO("MlpPolicy", vec_env, verbose=0,
        n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)

    events_path = Path(args.work_dir).parent / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [LiveLogger(log_freq=args.log_freq, events_path=events_path)]
    if args.rollout_freq:
        callbacks.append(RolloutVideoCallback(env_kwargs, args.rollout_freq, Path(args.video_dir)))

    print("Starting Training")
    #ppo wraps training process
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    vec_env.close()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.save)
        print(f"Model saved -> {args.save}")
    print("Training finished")

#read args and start training loop
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mug", default="CupFactory_001")
    ap.add_argument("--mug_scale",   type=float, default=0.50)
    ap.add_argument("--work_dir", default="_mujoco_scene")
    ap.add_argument("--timesteps", type=int, default=2000000)
    ap.add_argument("--log_freq", type=int, default=10_000)
    ap.add_argument("--rollout_freq", type=int, default=5000)
    ap.add_argument("--num_envs", type=int, default=4,
                    help="Number of parallel environments for PPO rollouts")
    ap.add_argument("--video_dir", default="videos_mujoco")
    ap.add_argument("--real_time", action="store_true")
    ap.add_argument("--save", default="")
    
    #start training loop
    train(ap.parse_args())
