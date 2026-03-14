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