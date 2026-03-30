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
    )
    buffer = Buffer(
        num_envs=args.num_envs,
        obs_dim=env.num_obs,
        act_dim=env.num_actions,
        max_length=config["training_cfg"]["steps_per_update"],
        device=device
    )

    optim = torch.optim.Adam(policy.parameters(), lr=config["training_cfg"]["learning_rate"])

    if args.start_update > 0:
        print(f"Loading checkpoint from update {args.start_update}")
        checkpoint = torch.load(f"{path}/checkpoints/go2_update_{args.start_update}.pt", map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        #lin_vel_x = checkpoint['lin_vel_x']
        
    # Do not like the configuration here TODO: fix later
    total_lin_reward = torch.zeros((20, config["training_cfg"]["steps_per_update"], args.num_envs, 1), device=device)
    for i in range(args.start_update, args.num_updates):
        with torch.no_grad():
            buffer.reset()
            obs = env.reset()
            buffer.update_obs_stats(obs)
            obs_norm = buffer.normalize_obs(obs)
            buffer.init_obs(obs_norm, policy.get_value(obs_norm))
            for step in range(config["training_cfg"]["steps_per_update"]):
                obs = obs.to(device)
                obs_norm = buffer.normalize_obs(obs)
                actions, value = policy.get_actions(obs_norm)
                log_probs, _, _ = policy.compute_log_probs(obs_norm, actions)
                _, mu, sigma    = policy.forward(obs_norm) 
                next_obs, reward, done, info = env.step(actions)
                reward, lin_vel_reward = reward
                next_obs_clean = next_obs.clone()
                if torch.isnan(next_obs_clean).any():
                    next_obs_clean[torch.isnan(next_obs_clean).any(dim=-1)] = 0.0
                buffer.update_obs_stats(next_obs_clean) 
                buffer.add_step(next_obs_clean, actions, log_probs, reward, done, value, lin_vel_reward, mu, sigma)
                
                obs = next_obs

            buffer.compute_returns_and_advantages(gamma=config["policy_cfg"]["gamma"],
                                                  lmbda=config["policy_cfg"]["lmbda"])
        total_lin_reward, lin_vel_x = utils.adjust_motion_command(total_lin_reward,
                                                                  lin_vel_reward,
                                                                  i,
                                                                  lin_vel_x,
                                                                  path,
                                                                  env,
                                                                  buffer,
                                                                  optim=optim,
                                                            policy=policy)
        desired_kl = 0.01
        lr = optim.param_groups[0]['lr']
        stop_early = False
        for epoch in range(config["training_cfg"]["update_epochs"]):
            if stop_early:
                break
            for batch in buffer.get_minibatches(args.num_batches):

                critic_loss, actor_loss, entropy, kl = policy.compute_loss(
                                        states=batch['obs'], actions=batch['actions'],
                                        advantages=batch['advantages'], log_probs_old=batch['log_probs'],
                                        returns=batch['returns'],
                                        old_mu=batch['mu'], old_sigma=batch['sigma']     # ← übergeben
                                    )
                entropy_loss = -0.01 * entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # gradient clipping
                optim.step()
                with torch.no_grad():
                    lp_new, _, _ = policy.compute_log_probs(batch['obs'], batch['actions'])
                    kl = (batch['log_probs'].squeeze() - lp_new).mean().item()
                if kl > 2.0 * desired_kl:
                    stop_early = True
                    break
        
        if kl > 2.0 * desired_kl:
            lr = max(lr / 1.5, 1e-4)
        elif kl < 0.5 * desired_kl:
            lr = min(lr * 1.5, 1e-2)

        for param_group in optim.param_groups:
            param_group['lr'] = lr

        writer.add_scalar("learning_rate", lr, i)
        writer.add_scalar("kl_divergence", kl, i)


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