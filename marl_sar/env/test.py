from stable_baselines3.common.env_checker import check_env

from marl_sar.env.env import SAREnv
env = SAREnv(grid_size=6, max_steps=80, obstacle_ratio=0.0, scan_radius=2, auto_discover=True)
obs, info = env.reset(seed=42)

done = False

print('============================================')
print('=Search and Rescue Single Agent Environment=')
print('============================================')

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Action: {action}')
    env.render()

check_env(env, warn=True, skip_render_check=True)