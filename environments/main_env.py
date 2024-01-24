from __future__ import print_function, absolute_import
from environments.sim_eng import SimEngine

from math import exp
import numpy as np


## hedge projection class ##
class HedgeEnvironment:
    def __init__(self, configs):
        self.configs = configs

        self.T = configs["T"]

        ### 多维存储时的维度设置
        self.tidx = 0  # 时间索引
        self.sidx = 1  # 股价维
        self.vidx = 2  # 波动率维
        self.fia_ov_idx = 3
        self.fia_delta_idx = 4

        # 衍生品设置
        self.fia_num = 1
        # call
        self.call_num = len(configs['call_strikes'])  # 用于对冲的call数量
        self.call_strikes = configs['call_strikes']
        # put
        self.put_num = len(configs['put_strikes'])
        self.put_strikes = configs['put_strikes']

        self.hedgeOption_notionals = np.zeros(self.call_num + self.put_num)
        # stock
        self.stock_notional = np.zeros(1)
        self.stock_notional_change = np.zeros(1)

        self.state_dim = self.fia_num * 2 + self.call_num * 2 + self.put_num + 3

        self.current_t = 0

        self.action_scaler = configs['action_scaler']

        self.episode_length = configs['episode_length']

        self.prob_exit_yearunit = configs['prob_exit_yearunit']
        self.prob_death_yearunit = configs['prob_death_yearunit']
        # 类似年利率转为日利率，大概0.98-0.99
        self.prob_exit_dayunit = 1 - (1 - self.prob_exit_yearunit) ** (1 / self.episode_length)
        self.prob_death_dayunit = 1 - (1 - self.prob_death_yearunit) ** (1 / self.episode_length)

        # states
        self.state_death = 0
        self.state_exit = 0
        self.flag_terminal = 0
        self.SimEng = None

        self.death_payoff_coef = 0.8

        self.reward_scale = False

        # fia_num ov/delta, call ov/delta, put ov/delta, s_t, v_t, T
        self.state_sim_unnormalized = np.zeros((self.T, (self.fia_num * 2 + self.call_num * 2 + self.put_num * 2 + 3)))

        self.use_calls = self.call_num > 0
        self.use_puts = self.put_num > 0
        self.use_stock = configs['use_stock']
        self.trans_cost = configs['trans_cost']

        ## state_headers for saving sims to file
        self.state_headers = ['T', 's_t', 'v_t']
        self.hedgeAsset_value_columns = []

        ### state state_headers ##
        self.state_headers.extend([f'fia_ov', f'fia_delta'])

        for i in range(self.call_num):
            call_strike = configs['call_strikes'][i]
            self.state_headers.extend([f'call_ov_{i}_strike{call_strike}', f'call_delta_{i}_strike{call_strike}'])
            self.hedgeAsset_value_columns += [f'call_ov_{i}_strike{call_strike}']

        for i in range(self.put_num):
            put_strike = configs['put_strikes'][i]
            self.state_headers.extend([f'put_ov_{i}_strike{put_strike}', f'put_delta_{i}_strike{put_strike}'])
            self.hedgeAsset_value_columns += [f'put_ov_{i}_strike{put_strike}']

        ### position state_headers ##
        position_headers = []
        for i in range(self.call_num):
            position_headers += [f'call_strike_{configs["call_strikes"][i]}_position']

        for i in range(self.put_num):
            position_headers += [f'put_strike_{configs["put_strikes"][i]}_position']

        if self.use_stock:
            self.hedgeAsset_value_columns.extend((['s_t']))

        position_headers += ['stock_notional']

        self.state_headers += position_headers
        self.action_headers = position_headers

    def action_space_size(self):
        return self.call_num + self.put_num + self.use_stock

    def state_space_size(self):
        return 3 + self.fia_num * 2 + self.call_num * 3 + self.put_num * 3 + self.use_stock  # 3是指(ov, delta, position)

    def reset_env(self):
        ## environments can be created on the fly (cuda projection) or using stored simulations
        # if self.Stored_Scen_location == None:
        self.SimEng = SimEngine(self.configs)
        # else:
        #     self.SimEng.step()

        ## Working Env Variables ##
        # fia_num ov/delta, call ov/delta, s_t, v_t, T
        self.state_sim_unnormalized = np.zeros((self.T, (self.fia_num * 2 + self.call_num * 2 + self.put_num * 2 + 3)))
        self.flag_terminal = False
        self.current_t = 0

        ## reset notionals ##
        self.call_notionals = np.zeros(self.call_num)  ## add for dynamic
        self.call_notionals_change = np.zeros(self.call_num)  ## add for dynamic

        self.put_notional = np.zeros(self.put_num)
        self.put_notional_change = np.zeros(self.put_num)

        self.stock_notional = np.zeros(1)
        self.stock_notional_change = np.zeros(1)
        ###

        # states
        self.state_death = 0
        self.state_exit = 0

        ### normalizer
        self.normalizer = 0

    def init_env(self):
        '''
        初始化环境，返回初始化的state和reward
        :return: state (shape: T,5+options_num), reward=0
        '''
        ## reset notionals ##
        ## 名义价值
        self.hedgeOption_notionals = np.zeros(self.call_num + self.put_num)
        # self.call_notionals_change = np.zeros(self.call_num)

        # self.put_notional = np.zeros(self.put_num)
        # self.put_notional_change = np.zeros(self.put_num)

        self.stock_notional = np.zeros(1)
        # self.stock_notional_change = np.zeros(1)

        # if self.Stored_Scen_location == None:
        self.SimEng = SimEngine(self.configs)
        # else:
        #     if self.SimEng == None:
        #         self.SimEng = CachedSimEngine(self.Stored_Scen_location)

        # (2,) (2,), (T,2)
        option_ovs, option_deltas, real_world_sim = self.SimEng.step()  # 模拟获取期权OV和delta，已经及对应的路径

        self.state_sim_unnormalized[:, self.tidx] = np.arange(0, self.T)  # 时间索引
        self.state_sim_unnormalized[:, self.sidx:(self.vidx + 1)] = real_world_sim  # 同时放入股价和波动率

        # 进行标准化，先找出S0
        self.normalizer = self.state_sim_unnormalized[self.SimEng.current_t, self.sidx]

        ## option ovs, deltas, real world path
        combined_output = np.zeros((option_ovs.shape[0] * 2))
        combined_output[::2] = option_ovs  # 在sidx后面的偶数索引为call和fia的value
        combined_output[1::2] = option_deltas  # 奇数索引为delta

        self.state_sim_unnormalized[self.SimEng.current_t, self.fia_ov_idx:] = combined_output

        # 对state进行标准化
        self.state_sim_normalized = self.state_normlization()

        return self.state_sim_normalized, 0  # state, reward=0

    def state_normlization(self):
        ### prepare state space observation ##
        state = self.state_sim_unnormalized[self.SimEng.current_t, :].copy()  # 注意copy
        state[1::2] = state[1::2] / self.normalizer
        state[0] = state[0] / self.T  # 时间标准化
        state = np.append(state, self.hedgeOption_notionals)
        state = np.append(state, self.stock_notional)
        return state

    def step(self):
        '''
        环境的核心功能
        :return: state, reward
        '''

        ## actions are processed, environment is projected/transitioned 1 step
        ## reward + new state observation are returned
        ## if it's the first t in the episode 0 reward and starting state observation returned
        ## rewards are always negative (objective to get to 0 reward)

        ### if it's the first time step, initialize and return first observation

        if self.current_t == 0:  # t=0时运行初始化
            state, reward = self.init_env()
            self.current_t += 1
            self.SimEng.current_t = self.current_t
            return state.astype(np.float_), float(reward)

        ## add the value of the stock if using stock
        stock_value = self.stock_notional * self.state_sim_unnormalized[
            self.SimEng.current_t, self.sidx] / self.normalizer

        ## 如果是最后一个时刻，将payoff返回为reward
        if self.SimEng.current_t == self.T - 1:
            self.flag_terminal = True
            state = self.state_normlization()
            ## asset payoff
            call_payoff, put_payoff = 0, 0
            for idx in range(self.call_num):
                # C += S_t - K
                call_payoff += (max(self.state_sim_unnormalized[self.current_t, self.sidx] - self.call_strikes[idx],
                                    0) / self.normalizer) * self.hedgeOption_notionals[idx]
            for idx in range(self.put_num):
                # C += K - S_t
                put_payoff += (max(self.put_strikes[idx] - self.state_sim_unnormalized[self.current_t, self.sidx],
                                   0) / self.normalizer) * self.hedgeOption_notionals[self.put_num + idx]

            ## liability payoff
            fia_payoff = max(self.state_sim_unnormalized[self.current_t, self.sidx] - self.normalizer,
                             0) / self.normalizer

            reward = -(abs(call_payoff + put_payoff + stock_value - fia_payoff))

            state[self.fia_ov_idx] = fia_payoff

            return state.astype(np.float_), float(reward)

        ## 计算当前时刻的期权OV和delta
        option_ov, option_deltas = self.SimEng.calculate_option_values()

        combined = np.zeros((option_ov.shape[0] * 2))
        combined[::2] = option_ov
        combined[1::2] = option_deltas

        self.state_sim_unnormalized[self.SimEng.current_t, self.fia_ov_idx:] = combined

        ## calculate payoff / reward
        fia_payoff, reward = 0, 0

        ## 生成随机的exit/death概率
        random_alive = np.random.uniform(0, 1)

        ## 计算期权价值
        hedgeOptions_value = 0
        for idx in range(1, len(option_ov)):
            hedgeOptions_value += (option_ov[idx] / self.normalizer) * self.hedgeOption_notionals[idx - 1]
            # 如果有交易成
            # if self.trans_cost > 0:
            #     reward += -abs(
            #         (self.call_notionals_change[idx - 1]) * (option_ov[idx] / self.normalizer) * self.trans_cost)

        # 如果退出，FIA payoff=0
        if random_alive < self.prob_exit_dayunit:
            self.state_exit = 1
            fia_payoff = 0
        # 若死亡，FIA payoff需要损失一部分
        elif random_alive < self.prob_death_dayunit + self.prob_exit_dayunit:
            self.state_death = 1
            fia_payoff = self.death_payoff_coef * max(
                self.state_sim_unnormalized[self.current_t, self.sidx] - self.normalizer, 0) / self.normalizer

        ## 退出或死亡时的reward
        if self.state_death == 1 or self.state_exit == 1:
            ## reward = - (期权价值 + 股票价值 - FIA payoff)
            reward += -abs(hedgeOptions_value + stock_value - fia_payoff)
            self.flag_terminal = True
        ## 存活时的reward
        else:
            if self.reward_scale:  # 指数放缩 exponentially scaled reward
                reward += -abs(exp(abs((hedgeOptions_value + stock_value - (
                        self.state_sim_unnormalized[self.SimEng.current_t, self.fia_ov_idx] / self.normalizer)))) - 1)
            else:  # normal reward
                reward += -abs((hedgeOptions_value + stock_value - (
                        self.state_sim_unnormalized[self.SimEng.current_t, self.fia_ov_idx] / self.normalizer)))

        state = self.state_normlization()
        self.current_t += 1
        self.SimEng.current_t += 1

        return state, float(reward)

    def update_positions(self, action_space):
        '''
        根据动作做出反应，即更新position（notional）
        :param action_space: action_space
        :return: None
        '''
        # 这里貌似有点问题，实际可以直接使用adjusted_action_space作为名义价值
        ### set action positions
        ## any action scaling is done in this function
        ## the change in asset position is tracked to calculate the transaction costs (if included)

        adjusted_action_space = action_space * self.action_scaler  # 缩放

        # # 计算出缩放后动作改变的大小
        # self.call_notionals_change = adjusted_action_space[0:self.call_num] - self.call_notionals
        # self.put_notional_change = adjusted_action_space[
        #                            self.call_num:(self.call_num + self.put_num)] - self.put_notional  # 剩余索引是put
        #
        # ## if using stock / notional
        # # 如果还使用了股票用于对冲
        # if self.use_stock:
        #     self.stock_notional_change = adjusted_action_space[(self.call_num + self.put_num):(
        #             self.call_num + self.put_num + 1)] - self.stock_notional
        #
        # # 计算更新后的名义价值
        # self.call_notionals = self.call_notionals_change + self.call_notionals
        # self.put_notional = self.put_notional_change + self.put_notional

        self.hedgeOption_notionals = adjusted_action_space[0:self.call_num + self.put_num]
        if self.use_stock:
            self.stock_notional = self.stock_notional_change + self.stock_notional


if __name__ == "__main__":
    from configs.default import configs_dict

    a = HedgeEnvironment(configs=configs_dict)
    a.init_env()
    b = a.step()
    b = a.step()
    b = a.step()
    print(b)
