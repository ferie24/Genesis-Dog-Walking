import genesis as gs
import logging
import utils
import torch
import os
import argparse
import json
from torch.utils.tensorboard import SummaryWriter

gs.init(logging_level=logging.WARNING, backend=gs.gpu)

from buffer import Buffer
from network import Network
from make_environment import Go2WalkingEnv
from reward import Rewards


def main(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("-envs", "--num_envs", type=int, default=1)
    parser.add_argument("-updates", "--num_updates", type=int, default=1000)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-gamma", "--gamma", type=float, default=0.99)
    parser.add_argument("-resume", "--start_update", type=int, default=0)
    args = parser.parse_args()

    path = os.getcwd()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Running on: {device}")
    lin_vel_x = config["env_cfg"]["movement"]["lin_vel_x"]
    writer = SummaryWriter("runs/my_experiment")

    env = Go2WalkingEnv(
        num_envs=args.num_envs,
        device=device,
        show_viewer=False,
        use_terrain=False,
        episode_length_s=config["env_cfg"]["episode_length_s"],
        reward_fn=Rewards(scales=config["reward_cfg"]),
    )
    env.set_commands(lin_vel_x=lin_vel_x,
                     lin_vel_y=config["env_cfg"]["movement"]["lin_vel_y"],
                     ang_vel_yaw=config["env_cfg"]["movement"]["lin_vel_yaw"])
    policy = Network(
        num_outputs=env.num_actions,
        num_inputs=env.num_obs,
        #gamma=0.99,
        #lmbda=0.0,
        epsilon=0.1,
    )
    buffer = Buffer(
        num_envs=args.num_envs,
        obs_dim=env.num_obs,
        act_dim=env.num_actions,
        max_length=config["training_cfg"]["steps_per_update"],
        device=device
    )

    optim = torch.optim.Adam(policy.parameters(), lr=5e-4)

    if args.start_update > 0:
        print(f"Loading checkpoint from update {args.start_update}")
        checkpoint = torch.load(f"{path}/checkpoints/go2_update_{args.start_update}.pt", map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        #lin_vel_x = checkpoint['lin_vel_x']
    # Do not like the configuration here TODO: fix later
    total_lin_reward = torch.zeros((20, config["training_cfg"]["steps_per_update"], args.num_envs, 1), device=device)
    for i in range(args.start_update, args.num_updates):
        print(f"Running Sim: i: {i}")
        with torch.no_grad():
            buffer.reset()
            obs = env.reset()
            buffer.init_obs(obs, policy.get_value(obs))
            for step in range(config["training_cfg"]["steps_per_update"]):
                obs = obs.to(device)
                actions, value = policy.get_actions(obs)
                log_probs, log_probs_value, entropy = policy.compute_log_probs(obs, actions)
                next_obs, reward, done, info = env.step(actions)
                reward, lin_vel_reward = reward
                buffer.add_step(next_obs, actions, log_probs, reward, done, value, lin_vel_reward)

                obs = next_obs

                if done.any():
                    obs = env.reset()

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
        print(f"Running Epochs: i: {i}, Avg_reward: {buffer.rewards.mean():.3f}")
        for epoch in range(config["training_cfg"]["update_epochs"]):
            batch = buffer.get_batch(config["training_cfg"]["minibatch_size"])

            log_probs_new, values_new, entropy = policy.compute_log_probs(batch['obs'],
                                                                          batch['actions'])
            # print(batch)
            critic_loss, actor_loss = policy.compute_loss(states=batch['obs'],
                                                          actions=batch['actions'],
                                                          advantages=batch['advantages'],
                                                          critic_targets=batch['values'],
                                                          log_probs_old=batch['log_probs'],
                                                          returns=batch['returns'], )
            entropy_loss = -0.01 * entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)  # gradient clipping
            optim.step()
            writer.add_scalar("loss", loss.item(), i)
            writer.add_scalar("actor_loss", actor_loss.item(), i)
            writer.add_scalar("critic_loss", critic_loss.item(), i)
            writer.add_scalar("entropy_loss", entropy_loss.item(), i)

        if i % 10 == 0:
            print(utils.save_checkpoint(
                path=f"{path}/checkpoints/go2_update_{i}.pt",
                policy=policy,
                optim=optim,
                update=i,
                avg_rew=buffer.rewards.mean().item()))
            print(utils.make_eval_video(
                env=env,
                policy=policy,
                filename=f"{path}/video/eval_update_{i}.mp4",
                eval_steps=600,
            ))
    writer.close()

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    main(config)