import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from utils import mkdirs
import wandb


#### class to store and graph evaluation scenarios ###
#### creates and stores graphs as it trains ###

class Evaluator:
    def __init__(self, configs_dict, state_space_size, action_space_size, state_space_headers, action_headers,
                 asset_value_columns):
        self.configs = configs_dict
        self.epoch_storage = []
        self.num_epochs = 0
        self.graph_dir = mkdirs(os.path.join(wandb.run.dir, 'eval_graphs'))
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.T = configs_dict['T']
        self.current_epoch = None
        self.plot_idx = 0

        if configs_dict["use_stock"]:
            self.action_columns = ['current_' + x for x in action_headers]
        else:
            self.action_columns = ['current_' + x for x in action_headers if 'stock' not in x]

        self.state_columns = state_space_headers
        self.columns = state_space_headers + self.action_columns + ['reward']

        self.asset_value_cols = asset_value_columns
        self.asset_delta_cols = [x for x in state_space_headers if 'delta' in x and 'fia' not in x]
        self.asset_delta_cols += ['stock_delta']

        mkdirs(self.graph_dir)

    def init_epoch_storage(self):
        self.epoch_storage.append(self.current_epoch.copy())
        self.current_epoch = np.zeros((self.T, self.state_space_size + self.action_space_size + 1))

    def append_state_action(self, state, action, reward, t):
        self.current_epoch[int(t), :] = np.concatenate((state, action, np.array([reward]))).copy()

    def calculate_agent_position(self, df):
        agent_position = np.zeros_like(df['s_t'])

        for each in zip(self.asset_value_cols, self.action_columns):
            val_np = df[each[0]].to_numpy()
            pos_np = df[each[1]].to_numpy()
            agent_position += pos_np * val_np

        return agent_position

    def calculate_agent_delta(self, df):
        agent_position = np.zeros_like(df['s_t'])
        df['stock_delta'] = np.ones_like(df['s_t'])

        for each in zip(self.asset_delta_cols, self.action_columns):
            delta_np = df[each[0]].to_numpy()
            pos_np = df[each[1]].to_numpy()
            agent_position += pos_np * delta_np

        return agent_position

    def plot_episode_data(self, time_steps, reward):
        output_dir = mkdirs(os.path.join(self.graph_dir, f'Plot_Epoch{time_steps}_{round(reward, 1)}_{self.plot_idx}'))
        time_col = np.arange(0, self.T) / (self.T - 1)

        num_row, num_col = 2, 2  # 决定取模拟结果中的多少个输出
        fig_sub1 = make_subplots(rows=num_row, cols=num_col)

        eval_num = 2
        ## select eval_num个 best and eval_num个 worst:
        temp_list = []
        for idx in range(0, len(self.epoch_storage)):
            current_df = pd.DataFrame(self.epoch_storage[idx], columns=self.columns)
            current_df['T'] = time_col
            reward = current_df['reward'].cumsum().to_numpy()[-1]
            temp_list.append((reward, current_df))
        temp_list.sort(key=lambda x: x[0])
        plot_list = temp_list[0:eval_num]
        plot_list.extend(temp_list[-eval_num:])
        plot_list = [x[1] for x in plot_list]

        idx = 0
        for row in range(1, num_row + 1):
            for col in range(1, num_col + 1):
                ep_df = plot_list[idx]
                ep_df['Agent_position'] = self.calculate_agent_position(ep_df)
                ep_df = ep_df[['T', 'fia_ov', 'Agent_position']]
                ep_df.columns = ['T', 'FIA OV', 'Agent Position OV']
                ep_df = ep_df[:-1]

                for column in list(ep_df.columns[1:]):
                    if 'Agent' in column:
                        fig_sub1.add_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                      name=column, line=dict(color='#832DEF')), row=row, col=col)
                    else:
                        fig_sub1.add_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                      name=column, line=dict(color='#FF4500')), row=row, col=col)
                idx += 1

        fig_sub1.update_layout(title_text='Agent position OV vs Liability OV (each figure is a sample)')
        fig_sub1.write_html(os.path.join(output_dir, f'OV_{self.plot_idx}.html'))
        fig_sub2 = make_subplots(rows=4, cols=4)

        idx = 0
        for row in range(1, num_row + 1):
            for col in range(1, num_col + 1):
                ep_df = plot_list[idx]
                ep_df['Agent_delta'] = self.calculate_agent_delta(ep_df)
                ep_df = ep_df[['T', 'fia_delta', 'Agent_delta']]
                ep_df.columns = ['T', 'FIA Delta', 'Agent Position Delta']
                ep_df = ep_df[:-1]

                for column in list(ep_df.columns[1:]):
                    if 'Agent' in column:
                        fig_sub2.add_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                      name=column, line=dict(color='#832DEF')), row=row, col=col)
                    else:
                        fig_sub2.add_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                      name=column, line=dict(color='#FF4500')), row=row, col=col)

                idx += 1

        fig_sub2.update_layout(title_text='Agent position Delta vs Liability Delta')
        fig_sub2.write_html(os.path.join(output_dir, f'Delta_{self.plot_idx}.html'))

        fig_sub3 = make_subplots(rows=4, cols=4)

        idx = 0
        for row in range(1, num_row + 1):
            for col in range(1, num_col + 1):
                ep_df = plot_list[idx]
                ep_df = ep_df[['T', 's_t', 'v_t', 'reward']]
                ep_df.columns = ['T', 'Stock Price', 'Volatility', 'reward']
                ep_df = ep_df[:-1]

                for column in list(ep_df.columns[1:]):
                    if 'Stock' in column:
                        fig_sub3.add_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                      line=dict(color='#2e8b57'),
                                                      name=column), row=row,
                                           col=col)
                    if 'Volatility' in column:
                        fig_sub3.add_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                      line=dict(color='#dc143c'),
                                                      name=column), row=row,
                                           col=col)
                    if 'reward' in column:
                        fig_sub3.add_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                      line=dict(color='#4b0082'),
                                                      name=column), row=row,
                                           col=col)
                idx += 1

        fig_sub3.update_layout(title_text='State Space and Agent Reward')
        fig_sub3.write_html(os.path.join(output_dir, f'State&Reward_{self.plot_idx}.html'))

        combined_figs = open(output_dir + '/' + 'combined_graphs_' + str(self.plot_idx) + ".html", 'w',
                             encoding='utf-8')
        combined_figs.write("<html><head></head><body>" + "\n")
        fig1_html = fig_sub1.to_html().split('<body>')[1].split('</body')[0]
        fig2_html = fig_sub2.to_html().split('<body>')[1].split('</body')[0]
        fig3_html = fig_sub3.to_html().split('<body>')[1].split('</body')[0]
        combined_figs.write(fig1_html)
        combined_figs.write(fig2_html)
        combined_figs.write(fig3_html)

        self.plot_idx += 1

        for idx, df in enumerate(plot_list):
            df.to_csv(os.path.join(output_dir, f'eval_res_sample{idx}.csv'), encoding="utf-8_sig")

        self.epoch_storage = []
