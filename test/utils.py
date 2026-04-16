import os
import torch

def make_eval_video(
    env,
    policy,
    filename="videos/eval.mp4",
    eval_steps=600,
    fps=50,
    deterministic=True,
):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    policy.eval()
    obs = env.reset()

    cam = env.camera
    cam.start_recording()

    episode_reward = 0.0

    with torch.no_grad():
        for step in range(eval_steps):
            if deterministic:
                action_mean, _, _ = policy.forward(obs)
                actions = torch.clamp(action_mean, -1.0, 1.0)
            else:
                actions, _ = policy.get_actions(obs)
                actions = torch.clamp(actions, -1.0, 1.0)

            obs, reward, done, info = env.step(actions)
            reward, _ = reward
            episode_reward += reward.mean().item()

            cam.render()

            if done[0].item():
                break

    cam.stop_recording(save_to_filename=filename, fps=fps)
    policy.train()

    return {
        "video_path": filename,
        "episode_reward": episode_reward,
        "steps": eval_steps,
    }

def save_checkpoint(path, policy, optim, update, avg_rew, extra=None):
    checkpoint = {
        "update": update,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "avg_reward": float(avg_rew),
    }
    if extra is not None:
        checkpoint.update(extra)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)

def adjust_motion_command(total_lin_reward, lin_reward_t, i, lin_vel_x, path, env, buffer, optim=None, policy=None, ):
    counter = i % 20 # should restart and replace list every 20 update iterations
    total_lin_reward[counter] = lin_reward_t
    # get mean from lin_vel_reward and adjust the command accordingly
    mean_lin_vel_reward = total_lin_reward.mean().item()
    # if devieates
    if mean_lin_vel_reward >= 0.85:
        lin_vel_x += 0.05
        env.set_commands(lin_vel_x=lin_vel_x, lin_vel_y=0.0, ang_vel_yaw=0.0)
        ##print(save_checkpoint(
        #    path=f"{path}/checkpoints/Safe_Before_Change_go2_update_{i}.pt",
        #    policy=policy,
        #    optim=optim,
        #    update=i,
        #    avg_rew=buffer.rewards.mean().item()))
    elif i% 500 == 0 and i != 0: 
        lin_vel_x += 0.05
        env.set_commands(lin_vel_x=lin_vel_x, lin_vel_y=0.0, ang_vel_yaw=0.0)
        ##print(save_checkpoint(
        #    path=f"{path}/checkpoints/Safe_Reset_go2_update_{i}.pt",
        #    policy=policy,
        #    optim=optim,
        #    update=i,
        #    avg_rew=buffer.rewards.mean().item()))
    return total_lin_reward, lin_vel_x