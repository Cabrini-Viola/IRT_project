#!/usr/bin/env python3
"""Twist task: tighten a cube by goal_deg with the Shadow Hand fingers.

The cube is either a yaw hinge (locked, 1D) or a free body (3D). Which one is decided by the scene XML.

  python twist.py selftest
  python twist.py progress MODEL.zip VECNORM.pkl [SYNERGY.pkl ACT_SCALE]
  python twist.py video MODEL.zip VECNORM.pkl OUT.mp4 [EPISODES]     (needs MUJOCO_GL=egl)
  python twist.py --task train --env-id HandTwistCube-v0 [any sac_her_pipeline.py option]

Environment variables read when the environment is registered:
  TWIST_XML         scene file (default demo/assets/hand/manipulate_block_touch_sensors_ws.xml)
  TWIST_GOAL_DEG    goal angle, default 360
  TWIST_REWARD      sparse | dense, default sparse
  TWIST_STALL       extra cost per step without new progress, default 0
  TWIST_STALL_STEPS grace period before the stall cost starts, default 15
"""
import os
import sys
import time

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(ROOT, "demo", "assets", "hand", "manipulate_block_touch_sensors_ws.xml")
ENV_ID = "HandTwistCube-v0"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TwistEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, xml_path=DEFAULT_XML, goal_deg=360.0, tighten_sign=-1, lock_wrist=True,
                 random_yaw=True, n_settle=10, dense=False, stall_penalty=0.0, stall_steps=15,
                 max_tilt_deg=30.0, max_drift_cm=5.0, render_mode=None):
        super().__init__()
        m = self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(m)
        self.render_mode = render_mode
        self.n_sub = max(1, int(round(0.04 / m.opt.timestep)))  # one env step = 0.04 s
        self.goal = np.radians(goal_deg)
        self.sign, self.lock_wrist = tighten_sign, lock_wrist
        self.random_yaw, self.n_settle = random_yaw, n_settle
        self.dense = dense
        self.stall_penalty, self.stall_steps, self.idle = stall_penalty, stall_steps, 0
        self.cos_tilt = np.cos(np.radians(max_tilt_deg))
        self.max_drift = max_drift_cm / 100.0

        # cube: yaw hinge (locked, 1D) or free joint (3D)
        J = mujoco.mjtObj.mjOBJ_JOINT
        jid = mujoco.mj_name2id(m, J, "object:yaw")
        self.free = jid < 0
        if self.free:
            jid = mujoco.mj_name2id(m, J, "object:joint")
            if jid < 0:
                raise ValueError("no cube joint ('object:yaw' or 'object:joint') in the XML")
            assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
        self.cube_q, self.cube_v = m.jnt_qposadr[jid], m.jnt_dofadr[jid]

        hand = [j for j in range(m.njnt)
                if (mujoco.mj_id2name(m, J, j) or "").startswith("robot0:")
                and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.hand_q = np.array([m.jnt_qposadr[j] for j in hand])
        self.hand_v = np.array([m.jnt_dofadr[j] for j in hand])

        # touch sensors only (the XML also contains other sensors)
        touch = [i for i in range(m.nsensor) if m.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH]
        self.touch_idx = np.array([m.sensor_adr[i] + k for i in touch for k in range(m.sensor_dim[i])],
                                  dtype=int)
        self.n_touch = len(self.touch_idx)

        # actuators: 20 position actuators, absolute control around the mid-range
        self.nu = m.nu
        assert self.nu == 20, f"expected 20 actuators, found {self.nu}"
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(self.nu)]
        self.wrist = np.array([i for i, n in enumerate(names) if "WRJ" in n])
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.center, self.half = (self.hi + self.lo) / 2, (self.hi - self.lo) / 2

        cube_dim = 18 if self.free else 3
        dim = len(self.hand_q) * 2 + cube_dim + self.n_touch
        inf = np.inf
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-inf, inf, (dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-inf, inf, (1,), dtype=np.float32),
            "desired_goal": spaces.Box(-inf, inf, (1,), dtype=np.float32),
        })
        self.yaw0, self.yaw_prev, self.yaw_acc, self.best = 0.0, 0.0, 0.0, 0.0
        self.start_pos, self.start_xy = np.zeros(3), np.zeros(2)
        self._renderer = self._viewer = None

    # ---- helpers ----
    def _cube_pose(self):
        q = self.data.qpos[self.cube_q:self.cube_q + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[3:7])
        return q[:3].copy(), R.reshape(3, 3)

    def _track(self):
        """Free cube: update the accumulated yaw. Returns True if progress may count."""
        if not self.free:
            return True
        pos, R = self._cube_pose()
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        valid = bool(R[2, 2] > self.cos_tilt and np.linalg.norm(pos[:2] - self.start_xy) < self.max_drift)
        if valid:
            self.yaw_acc += wrap(yaw - self.yaw_prev)
        self.yaw_prev = yaw
        return valid

    def _angle(self):
        """Tightening angle in radians since reset (positive = tightening)."""
        if self.free:
            return self.sign * self.yaw_acc
        return self.sign * (self.data.qpos[self.cube_q] - self.yaw0)

    def _apply(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        ctrl = self.center + a * self.half
        if self.lock_wrist:
            ctrl[self.wrist] = 0.0
        self.data.ctrl[:] = np.clip(ctrl, self.lo, self.hi)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)

    def _obs(self):
        d = self.data
        if self.free:
            pos, R = self._cube_pose()
            cube = np.concatenate([pos - self.start_pos, R.ravel(), d.qvel[self.cube_v:self.cube_v + 6]])
        else:
            yaw = d.qpos[self.cube_q]
            cube = np.array([np.sin(yaw), np.cos(yaw), d.qvel[self.cube_v]])
        touch = np.log1p(np.maximum(d.sensordata[self.touch_idx], 0.0))
        vec = np.concatenate([d.qpos[self.hand_q], d.qvel[self.hand_v], cube, touch]).astype(np.float32)
        return {"observation": vec,
                "achieved_goal": np.array([self.best], dtype=np.float32),
                "desired_goal": np.array([self.goal], dtype=np.float32)}

    # ---- gym API ----
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        ach, des = np.asarray(achieved_goal)[..., 0], np.asarray(desired_goal)[..., 0]
        if self.dense:
            r = -np.clip(des - ach, 0.0, 2 * np.pi) / (2 * np.pi)
        else:
            r = np.where(ach >= des - 1e-6, 0.0, -1.0)
        if self.stall_penalty > 0 and info is not None:
            try:
                if isinstance(info, dict):
                    stall = float(info.get("stall", 0.0))
                else:
                    stall = np.array([float(i.get("stall", 0.0)) for i in info])
                r = r - self.stall_penalty * stall
            except Exception:
                pass
        return np.asarray(r, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.random_yaw:
            th = self.np_random.uniform(-np.pi, np.pi)
            if self.free:
                self.data.qpos[self.cube_q + 3:self.cube_q + 7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
            else:
                self.data.qpos[self.cube_q] = th
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.n_settle):
            self._apply(np.zeros(self.nu))
        if self.free:
            pos, R = self._cube_pose()
            self.start_pos, self.start_xy = pos, pos[:2].copy()
            self.yaw_prev, self.yaw_acc = float(np.arctan2(R[1, 0], R[0, 0])), 0.0
        else:
            self.yaw0 = float(self.data.qpos[self.cube_q])
        self.best, self.idle = 0.0, 0
        return self._obs(), {"is_success": 0.0}

    def step(self, action):
        self._apply(action)
        valid = self._track()
        ang = self._angle()
        if valid and ang > self.best + np.radians(0.1):
            self.idle = 0
        else:
            self.idle += 1
        if valid:
            self.best = max(self.best, ang)
        obs = self._obs()
        stall = float(self.idle > self.stall_steps and self.best < self.goal)
        info = {"is_success": float(self.best >= self.goal), "stall": stall, "valid": float(valid),
                "angle_deg": float(np.degrees(ang)), "best_deg": float(np.degrees(self.best))}
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        return obs, reward, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
                self._cam = mujoco.MjvCamera()
                self._cam.lookat[:] = [1.0, 0.87, 0.05]
                self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.55, 135, -30
            self._renderer.update_scene(self.data, camera=self._cam)
            return self._renderer.render()
        if self.render_mode == "human":
            from mujoco import viewer as mj_viewer
            if self._viewer is None:
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=TwistEnv, max_episode_steps=150,
                 kwargs={"xml_path": os.path.abspath(os.environ.get("TWIST_XML", DEFAULT_XML)),
                         "goal_deg": float(os.environ.get("TWIST_GOAL_DEG", "360")),
                         "dense": os.environ.get("TWIST_REWARD", "sparse") == "dense",
                         "stall_penalty": float(os.environ.get("TWIST_STALL", "0")),
                         "stall_steps": int(os.environ.get("TWIST_STALL_STEPS", "15"))})


def selftest():
    env = gym.make(ENV_ID)
    base = env.unwrapped
    obs, _ = env.reset(seed=0)
    print("observation:", obs["observation"].shape, "| action space:", env.action_space.shape)
    print(f"cube: {'FREE (3D)' if base.free else 'locked (yaw only)'} | sensors in the XML: "
          f"{base.model.nsensor}, touch sensors used: {base.n_touch}")
    print(f"reward: {'dense' if base.dense else 'sparse'}, stall penalty {base.stall_penalty}, "
          f"goal {np.degrees(base.goal):.0f} deg")
    t0, n = time.time(), 0
    for ep in range(3):
        env.reset(seed=ep)
        done = False
        while not done:
            _, r, term, trunc, info = env.step(env.action_space.sample())
            done, n = term or trunc, n + 1
        print(f"random actions, episode {ep}: best={info['best_deg']:.0f} deg, "
              f"final={info['angle_deg']:+.0f} deg, cube still valid={info['valid']:.0f}")
    print(f"{n / (time.time() - t0):.0f} env steps per second")
    env.close()
    if base.free:
        print("free cube: replay check skipped")
        return

    path = os.path.join(ROOT, "workspace", "best_motion.npy")
    if not os.path.exists(path):
        print("no workspace/best_motion.npy, skipping the replay check")
        return
    x = np.load(path)
    fing = [i for i in range(base.nu) if i not in base.wrist]
    nf = len(fing)
    if x.size != 1 + 3 * nf:
        print("best_motion.npy has an unexpected size, skipping the replay check")
        return
    f, b, A, ph = x[0], x[1:1 + nf], x[1 + nf:1 + 2 * nf], x[1 + 2 * nf:]
    for label, kw in (("with settle steps and random start", {}),
                      ("without settle steps (like rotate_test.py)", dict(n_settle=0, random_yaw=False))):
        e = gym.make(ENV_ID, **kw)
        e.reset(seed=0)
        at = {}
        for t in range(150):
            action = np.zeros(base.nu)
            action[fing] = np.clip(b + A * np.sin(2 * np.pi * f * t * 0.04 + ph), -1, 1)
            _, _, _, _, info = e.step(action)
            at[t + 1] = info["angle_deg"]
        print(f"replay {label}: {at[100]:+.0f} deg after 100 steps, {at[150]:+.0f} deg after 150, "
              f"best {info['best_deg']:.0f} deg")
        e.close()


def progress():
    """python twist.py progress MODEL.zip VECNORM.pkl [SYNERGY.pkl ACT_SCALE]"""
    import sac_her_pipeline as P
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    a = sys.argv[2:]
    model_zip, vec_pkl = a[0], a[1]
    if len(a) > 2:
        raw = DummyVecEnv([P.make_synergy_env(ENV_ID, P.load_synergy_bundle(a[2]),
                                              act_scale=float(a[3]) if len(a) > 3 else 3.0)])
    else:
        raw = DummyVecEnv([P.make_full_env(ENV_ID)])
    env = VecNormalize.load(vec_pkl, raw)
    env.training, env.norm_reward = False, False
    model = SAC.load(model_zip, env=env, device="cpu")
    best, final, succ = [], [], []
    for _ in range(20):
        obs, done = env.reset(), False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(act)
            done = bool(dones[0])
        best.append(infos[0]["best_deg"])
        final.append(infos[0]["angle_deg"])
        succ.append(infos[0]["is_success"])
    print(f"20 episodes: best angle mean {np.mean(best):.0f} deg, median {np.median(best):.0f}, "
          f"max {np.max(best):.0f} | final angle mean {np.mean(final):+.0f} | success {np.mean(succ):.0%}")


def video():
    """python twist.py video MODEL.zip VECNORM.pkl OUT.mp4 [EPISODES]   (set MUJOCO_GL=egl)"""
    import imageio
    from PIL import Image, ImageDraw
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model_zip, vec_pkl, out = sys.argv[2:5]
    n_ep = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    raw = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="rgb_array")])
    env = VecNormalize.load(vec_pkl, raw)
    env.training, env.norm_reward = False, False
    model = SAC.load(model_zip, env=env, device="cpu")
    frames = []
    for ep in range(n_ep):
        obs, done = env.reset(), False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(act)
            done, info = bool(dones[0]), infos[0]
            if not done:
                img = Image.fromarray(raw.envs[0].render())
                ImageDraw.Draw(img).text(
                    (10, 10), f"episode {ep + 1}   cube turned {info['angle_deg']:+.0f} deg   goal 360",
                    fill=(255, 255, 255))
                frames.append(np.asarray(img))
        print(f"episode {ep + 1}: best {info['best_deg']:.0f} deg, final {info['angle_deg']:+.0f} deg, "
              f"success {info['is_success']:.0f}")
    imageio.mimsave(out, frames, fps=25)
    print("saved", out, f"({len(frames)} frames)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "progress":
        progress()
    elif cmd == "video":
        video()
    else:
        if float(os.environ.get("TWIST_STALL", "0")) > 0:
            # make hindsight replay store and pass the info dicts (with the stall flag) to compute_reward
            from stable_baselines3 import HerReplayBuffer
            _orig_init = HerReplayBuffer.__init__

            def _init(self, *a, **kw):
                kw.setdefault("copy_info_dict", True)
                _orig_init(self, *a, **kw)

            HerReplayBuffer.__init__ = _init
        import sac_her_pipeline
        sac_her_pipeline.main()