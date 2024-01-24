import os
import wandb
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils import get_time, mkdirs
from models import SoftActorCritic
from buffer import EnvReplayBuffer
from evaluations import Evaluator
from configs.default import configs_dict
from environments.main_env import HedgeEnvironment

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_model():
    # create environment, buffer and models
    environment = HedgeEnvironment(configs_dict)
    state_space_size = environment.state_space_size()
    action_space_size = environment.action_space_size()
    buffer = EnvReplayBuffer(configs_dict=configs_dict, state_space_dim=state_space_size,
                             action_space_dim=environment.action_space_size())

    ## create SAC model
    SACModel = SoftActorCritic(model_dict=configs_dict, state_space_size=state_space_size,
                               action_space_size=environment.action_space_size()).to(configs_dict['device'])
    SACModel.switch2train()

    ## evaluation
    ep_evaluator = Evaluator(configs_dict=configs_dict, state_space_size=state_space_size,
                             action_space_size=action_space_size,
                             state_space_headers=environment.state_headers,
                             action_headers=environment.action_headers,
                             asset_value_columns=environment.hedgeAsset_value_columns)

    ### Training loop ####
    cum_steps, no_improve = 0, 0
    batch_size = configs_dict["batch_size"]
    pbar = tqdm(range(configs_dict['epochs']))
    for epoch in pbar:
        flag_done = False
        state, reward = environment.step() # 第一次会生成初始动作和0的reward

        # 先（随机生成动作）探索，然后再由模型生成
        while not flag_done:
            if cum_steps > configs_dict['random_action_steps']:
                action, log_probability = SACModel.Actor.choose_stochastic_action(
                    state_space=torch.tensor(np.array([state]), device='cuda:0').float(), training=True)
                action = action.detach().cpu().numpy()[0]
            else:
                action = np.array(np.random.uniform(-1, 1, action_space_size))  ## random_action_epochs之前随机动作

            ## 输出动作后更新position
            environment.update_positions(action)

            ## t时刻state + action -> reward, t+1时刻state
            next_state, reward = environment.step()
            flag_done = environment.flag_terminal

            ## 将transition加入到buffer中
            buffer.append_transition(state, action, reward, next_state, flag_done)

            if cum_steps > batch_size and cum_steps % configs_dict['update_interval'] == 0:
                SACModel.train_one_step(batch=buffer.sample_batch(sample_size=batch_size), n_steps=cum_steps)
            pbar.set_description(f"[Epoch] {epoch} [Step] {cum_steps}")
            cum_steps += 1

        environment.reset_env()

        # eval
        with torch.no_grad():
            if epoch > 1 and epoch % configs_dict["eval_interval"] == 0:
                average_eval_reward = 0
                SACModel.switch2eval()

                for idx in range(configs_dict["eval_epochs"]):
                    ep_evaluator.init_epoch_storage() # 要将一个epoch每个t的都存储起来再eval

                    #### Epoch start ####
                    epoch_reward = 0
                    flag_done = False
                    state, reward = environment.step()

                    while not flag_done:
                        current_t = state[0] * (configs_dict["T"] - 1)

                        action = SACModel.Actor.choose_deterministic_action(
                            torch.tensor(np.array([state]), device='cuda:0').float()).detach().cpu().numpy()[0]

                        environment.update_positions(action)

                        next_state, reward = environment.step()
                        epoch_reward += reward

                        # state, action, reward, timestep
                        ep_evaluator.append_state_action(state, action * configs_dict['action_scaler'], reward,
                                                         current_t)

                        flag_done = environment.flag_terminal
                        state = next_state

                    ep_evaluator.append_state_action(state, action, 0.0, state[0] * (configs_dict["T"] - 1))

                    average_eval_reward += epoch_reward / configs_dict["eval_epochs"]

                    environment.reset_env()

                wandb.termlog(f'Average eval reward: {average_eval_reward}')
                ep_evaluator.plot_episode_data(epoch, average_eval_reward)

                SACModel.switch2train()

                # early stop
                if average_eval_reward > -9999999:
                    no_improve = 0
                    best_model_path = os.path.join(
                        mkdirs(os.path.join(wandb.run.dir, 'ckpts')), f'SAC_model_episode_{epoch}.ckpt')
                    torch.save(SACModel, best_model_path)
                else:
                    no_improve += 1
        if no_improve > configs_dict["early_stop"]:
            break


def main(args):
    wandb.init(project='FIAHedging', name=f'train_{get_time()}', group=f'{args.exp_name}', config=configs_dict)
    train_model()
    print(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='system parameter')
    parser.add_argument('--exp_name', type=str, default='baseline', help='name for the experiment')
    args = parser.parse_args()
    main(args)