import genesis as gs
import logging
import utils
import torch
import os
import argparse
import json
from torch.utils.tensorboard import SummaryWriter

gs.init(logging_level=logging.WARNING, backend=gs.cuda)

from buffer import Buffer
from network import Network
from make_environment import Go2WalkingEnv
from reward import Rewards


def main(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--num_envs", type=int, default=4096)
    parser.add_argument("-u", "--num_updates", type=int, default=4000)
    parser.add_argument("-b", "--num_batches", type=int, default=4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-g", "--gamma", type=float, default=0.99)
    parser.add_argument("-r", "--start_update", type=int, default=0)
    parser.add_argument("-o", "--run_name", type=str, default="go2_run")
    args = parser.parse_args()

    path = os.getcwd()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Running on: {device}")
    lin_vel_x = config["env_cfg"]["movement"]["lin_vel_x"]
    writer = SummaryWriter(f"runs/{args.run_name}")

    env = Go2WalkingEnv(
        num_envs=args.num_envs,
        device=device,
        show_viewer=False,
        use_terrain=False,
        episode_length_s=config["env_cfg"]["episode_length_s"],
        reward_fn=Rewards(scales=config["reward_cfg"], tracking_sigma=config["reward_cfg"]["tracking_sigma"]),
    )
    print(f"Number of environments: {env.num_envs}, obs dim: {env.num_obs}, action dim: {env.num_actions}")
    env.set_commands(lin_vel_x=lin_vel_x,
                     lin_vel_y=config["env_cfg"]["movement"]["lin_vel_y"],
                     ang_vel_yaw=config["env_cfg"]["movement"]["lin_vel_yaw"])
    policy = Network(
        num_outputs=env.num_actions,
        num_inputs=env.num_obs,
        epsilon=config["policy_cfg"]["epsilon"],
    ).to(device)
    buffer = Buffer(
        num_envs=args.num_envs,
        obs_dim=env.num_obs,
        act_dim=env.num_actions,
        max_length=config["training_cfg"]["steps_per_update"],
        device=device
    )

    optim = torch.optim.Adam(policy.parameters(), lr=config["training_cfg"]["learning_rate"])
    use_obs_normalization = False

    if args.start_update > 0:
        print(f"Loading checkpoint from update {args.start_update}")
        checkpoint = torch.load(f"{path}/checkpoints/{args.run_name}/go2_update_{args.start_update}.pt", map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        #lin_vel_x = checkpoint['lin_vel_x']
        
    # Do not like the configuration here TODO: fix later
    total_lin_reward = torch.zeros(20, device=device)
    desired_kl = 0.02
    print(f"Policy device: {next(policy.parameters()).device}")
    obs = env.reset()
    buffer.update_obs_stats(obs)
    for i in range(args.start_update, args.num_updates):
        with torch.no_grad():
            buffer.reset()
            obs_norm = buffer.normalize_obs(obs) if use_obs_normalization else obs
            buffer.init_obs(obs_norm, policy.get_value(obs_norm))
            for step in range(config["training_cfg"]["steps_per_update"]):
                obs = obs.to(device)
                obs_norm = buffer.normalize_obs(obs) if use_obs_normalization else obs
                actions, value = policy.get_actions(obs_norm)
                log_probs, _, _ = policy.compute_log_probs(obs_norm, actions)
                mu, sigma, _    = policy.forward(obs_norm) 
                next_obs, reward, done, time_outs = env.step(actions)
                reward, lin_vel_reward = reward

                next_obs_clean = torch.nan_to_num(next_obs.clone(), nan=0.0, posinf=0.0, neginf=0.0)
                actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
                log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=0.0)
                value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
                sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
                reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
                lin_vel_reward = torch.nan_to_num(lin_vel_reward, nan=0.0, posinf=0.0, neginf=0.0)

                finite_env_mask = torch.isfinite(next_obs).all(dim=-1)
                if reward.dim() == 1:
                    finite_env_mask &= torch.isfinite(reward)
                else:
                    finite_env_mask &= torch.isfinite(reward).all(dim=-1)
                finite_env_mask &= torch.isfinite(actions).all(dim=-1)
                if (~finite_env_mask).any():
                    done = done.clone()
                    done[~finite_env_mask] = True

                if use_obs_normalization:
                    buffer.update_obs_stats(next_obs_clean)
                buffer.add_step(next_obs_clean, actions, log_probs, reward, done, value, lin_vel_reward, mu, sigma, time_outs)
                
                obs = next_obs_clean
            last_obs_norm = buffer.normalize_obs(obs) if use_obs_normalization else obs
            buffer.values[buffer.steps].copy_(policy.get_value(last_obs_norm).detach())
            buffer.compute_returns_and_advantages(gamma=config["policy_cfg"]["gamma"],
                                                  lmbda=config["policy_cfg"]["lmbda"])
        rollout_lin_vel_reward = buffer.lin_vel_rewards[:buffer.steps].mean().item()
        total_lin_reward, lin_vel_x = utils.adjust_motion_command(total_lin_reward,
                                                                  rollout_lin_vel_reward,
                                                                  i,
                                                                  lin_vel_x,
                                                                  path,
                                                                  env,
                                                                  buffer,
                                                                  optim=optim,
                                                            policy=policy)
        lr = optim.param_groups[0]["lr"]
        stop_update = False
        last_epoch_avg_kl = 0.0
        for epoch in range(config["training_cfg"]["update_epochs"]):
            epoch_kl_sum = 0.0
            batch_count = 0 
            for batch in buffer.get_minibatches(args.num_batches):
                batch_is_finite = (
                    torch.isfinite(batch["obs"]).all()
                    and torch.isfinite(batch["actions"]).all()
                    and torch.isfinite(batch["advantages"]).all()
                    and torch.isfinite(batch["log_probs"]).all()
                    and torch.isfinite(batch["returns"]).all()
                    and torch.isfinite(batch["mu"]).all()
                    and torch.isfinite(batch["sigma"]).all()
                    and torch.isfinite(batch["values_old"]).all()
                )
                if not batch_is_finite:
                    continue

                critic_loss, actor_loss, entropy, kl = policy.compute_loss(
                    states=batch["obs"],
                    actions=batch["actions"],
                    advantages=batch["advantages"],
                    log_probs_old=batch["log_probs"],
                    returns=batch["returns"],
                    old_mu=batch["mu"],
                    old_sigma=batch["sigma"],
                    old_values=batch["values_old"],
                )

                entropy_loss = -config["entropy"] * entropy.mean()
                loss = actor_loss + critic_loss + entropy_loss

                if not torch.isfinite(loss):
                    continue

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()

                epoch_kl_sum += kl.item()
                batch_count += 1

                running_avg_kl = epoch_kl_sum / batch_count
                if running_avg_kl > desired_kl * 2.0:
                    #print(f"Early stop at epoch {epoch}: KL={running_avg_kl:.4f} > {desired_kl*2:.4f}")
                    stop_update = True
                    break
            epoch_avg_kl = epoch_kl_sum / max(batch_count, 1)
            last_epoch_avg_kl = epoch_avg_kl
            
            if stop_update:
                break
        if last_epoch_avg_kl > desired_kl * 2.0:
            lr = max(1e-4, lr / 1.5)
        elif last_epoch_avg_kl < desired_kl / 2.0 and last_epoch_avg_kl > 0.0:
            lr = min(3e-4, lr * 1.5)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        writer.add_scalar("learning_rate", lr, i)
        writer.add_scalar("kl_divergence", last_epoch_avg_kl, i)


        writer.add_scalar("loss", loss.item(), i)
        writer.add_scalar("actor_loss", actor_loss.item(), i)
        writer.add_scalar("critic_loss", critic_loss.item(), i)
        writer.add_scalar("entropy_loss", entropy_loss.item(), i)
        writer.add_scalar("avg_reward", buffer.rewards.mean().item(), i)
            

        if i % 100 == 0:
            utils.save_checkpoint(
                path=f"{path}/checkpoints/{args.run_name}/go2_update_{i}.pt",
                policy=policy,
                optim=optim,
                update=i,
                avg_rew=buffer.rewards.mean().item())
            print(utils.make_eval_video(
                env=env,
                policy=policy,
                filename=f"{path}/video/{args.run_name}/eval_update_{i}.mp4",
                eval_steps=600,
            ))
    writer.close()

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    main(config)