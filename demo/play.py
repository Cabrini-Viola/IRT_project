"""Play or record a Shadow Hand agent. Usage examples are in the chat."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
try:
    import sac_her_pipeline  # also loads mujoco_compat
except Exception as e:
    print("note: could not import sac_her_pipeline:", e)

p = argparse.ArgumentParser()
p.add_argument("--env-id", default="HandManipulateEggRotate_ContinuousTouchSensors-v1")
p.add_argument("--model-dir", default=None, help="folder with best_model.zip + vecnorm_best.pkl; omit for random actions")
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--video", default=None, help="save an mp4 here instead of opening a window")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

env = gym.make(args.env_id, render_mode="rgb_array" if args.video else "human")

model = norm = None
if args.model_dir:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    dummy = DummyVecEnv([lambda: gym.make(args.env_id)])
    norm = VecNormalize.load(os.path.join(args.model_dir, "vecnorm_best.pkl"), dummy)
    norm.training, norm.norm_reward = False, False
    model = SAC.load(os.path.join(args.model_dir, "best_model.zip"), env=norm, device="cpu")
    print("agent action space:", model.action_space.shape)

frames, wins = [], 0
for ep in range(args.episodes):
    obs, info = env.reset(seed=args.seed + ep)
    done, steps = False, 0
    while not done:
        if model is not None:
            action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
        else:
            action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1
        if args.video:
            frames.append(env.render())
        else:
            time.sleep(0.03)
    ok = bool(info.get("is_success", False))
    wins += ok
    print(f"episode {ep+1}: steps={steps} success={ok}")
print(f"successes: {wins}/{args.episodes}")

if args.video:
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    imageio.mimsave(args.video, frames, fps=25)
    print("saved", args.video)
env.close()
