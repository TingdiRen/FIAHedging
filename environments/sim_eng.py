import numpy as np
from configs.default import configs_dict
from environments.cuda_funcs import compute_heston_call, compute_heston_fia, create_xoroshiro128p_states


class SimEngine:
    def __init__(self, configs):
        self.configs = configs

        ## SDE参数
        self.mu = configs["mu"]
        self.kappa = configs["kappa"]
        self.theta = configs["theta"]
        self.xi = configs["xi"]
        self.mean = configs["mean"]
        self.cov_matrix = np.array([[1.0, configs["correlation"]], [configs["correlation"], 1.0]])
        self.correlation = configs["correlation"]
        self.shock_percent = configs["shock_percent"]
        self.rf = configs["rf"]
        self.x0 = configs["x0"]
        self.v0 = configs["v0"]

        ## 时间参数
        self.T = configs["T"]
        self.dt = 1 / self.T
        self.sqrt_dt = np.sqrt(self.dt)
        self.current_t = 0
        self.iteration_steps = configs["iteration_steps"]

        ## 期权参数
        self.call_strikes = configs["call_strikes"]
        self.put_strikes = configs["put_strikes"]

        ## CUDA加速参数
        self.cuda_threads = configs["cuda_threads"]
        self.cuda_blocks = configs["cuda_blocks"]

        ## 状态参数
        self.prob_exit_yearunit = configs["prob_exit_yearunit"]
        self.prob_death_yearunit = configs["prob_death_yearunit"]
        self.state_death = False
        self.state_exit = False
        self.state_alive = True

    def step(self):
        '''
        模拟器的核心功能，生成一次模拟路径，并计算出期权（包括用于对冲的欧式和FIA）的内在价值OV和delta
        :return: option_values (np.array, shape: 1+对冲期权个数n,)
        option_deltas (np.array, shape: 1+n,)
        RealWorldPath (np.array, shape: T,2)
        '''
        #
        # real world path generate
        self.RealWorldPath = self.generate_real_world_path()  # (252,2) dim 1: price，dim 2: volatility
        # calculate the option values
        option_values, option_deltas = self.calculate_option_values()

        return option_values, option_deltas, self.RealWorldPath

    def generate_real_world_path(self):
        '''
        模拟真实世界的价格路径
        :return: path_output (np.array, shape: T,2) 第1维是股价，第2维是波动率
        '''
        path_output = np.zeros((self.T, 2))
        path_output[0, 0] = self.x0
        path_output[0, 1] = self.v0

        # generate normals
        correlated_norms = np.random.multivariate_normal(self.mean, self.cov_matrix, self.T) * self.sqrt_dt

        ## real world drift parameters are off!
        ## should be N(0.05, 0.1) for a reasonable lognormal
        drift_adjustment = np.random.normal(0.05, 0.75, self.T)
        drift_adjustment = np.exp(drift_adjustment)  # scalar multiple on risk free rate for real world evolution

        # euler scheme for heston
        # heston - s_t+1 = st * exp(r_f - v(t)/2)dt + sqrt(v(t)) * dW_S_t)
        # vol - v(t+1) = v(t) + kappa (theta - v(t))*dt + xi * sqrt(v(t)) * dW_V_t
        x, v = self.x0, self.v0
        for i in range(1, self.T):
            sqrt_vol = np.sqrt(v)

            # stock price evolution
            x = x * (np.exp((self.rf * drift_adjustment[i] - 0.5 * v) * self.dt + correlated_norms[i, 0] * sqrt_vol))

            # volatility vol evolution
            v = max(v + self.kappa * (self.theta - v) * self.dt + self.xi * sqrt_vol * correlated_norms[i, 1], 1e-5)

            # write output
            path_output[i, 0], path_output[i, 1] = x, v

        return path_output

    def calculate_option_values(self):
        '''
        计算对冲欧式期权和FIA的内在价值和delta
        :return: option_values (np.array, shape: 2,)
        option_deltas (np.array, shape: 2,)
        '''
        ### call price and delta ###
        call_ovs, call_deltas = [], []
        # 计算call期权的内在价值OV和delta
        for idx in range(len(self.call_strikes)):  # need to update with index for strike
            call_ov, call_delta = self.calc_hedgeOption_ov_delta('call',
                                                                 self.T - self.current_t,
                                                                 self.RealWorldPath[self.current_t, 0],
                                                                 self.RealWorldPath[self.current_t, 1],
                                                                 self.call_strikes[idx])

            call_ovs.append(call_ov)
            call_deltas.append(np.clip(call_delta, 0, 1))  ## 确保落在0，1区间

        put_ovs, put_deltas = [], []
        for idx in range(len(self.put_strikes)):  # need to update with index for strike
            put_ov, put_delta = self.calc_hedgeOption_ov_delta('put',
                                                               self.T - self.current_t,
                                                               self.RealWorldPath[self.current_t, 0],
                                                               self.RealWorldPath[self.current_t, 1],
                                                               self.put_strikes[idx])

            put_ovs.append(put_ov)
            put_deltas.append(np.clip(put_delta, 0, 1))  ## 确保落在0，1区间

        ### fia_num ov and delta
        fia_ov, fia_delta = self.calc_fia_ov_delta(self.T - self.current_t,
                                                   self.RealWorldPath[self.current_t, 0],
                                                   self.RealWorldPath[self.current_t, 1])

        fia_delta = np.clip(fia_delta, 0, 1)

        option_values = [fia_ov] + call_ovs + put_ovs
        option_deltas = [fia_delta] + call_deltas + put_deltas

        return np.array(option_values), np.array(option_deltas)

    def calc_hedgeOption_ov_delta(self, option_type, current_day, x_t, v_t, strike):
        '''
        计算欧式期权的OV和delta，生成一个change，然后采用MC方法
        :param current_day: 当前时刻
        :param x_t: 当前时刻的股价
        :param v_t: 当前时刻的波动率
        :param strike: 执行价
        :return: ov_call (float32), delta_call (float32)
        '''
        ## 使用shock是为了计算delta，即期权价值变动/资产价值变动，使用shock来扰动
        shock_magnitude = self.shock_percent * x_t

        ## CUDA的并行设计
        split = int(self.cuda_threads * self.cuda_blocks / 3)
        cuda_seed = create_xoroshiro128p_states(self.cuda_threads * self.cuda_blocks,
                                                seed=np.random.randint(-50000, 250000))

        t_left2maturity = current_day / self.T  # num days left to maturity
        iterations = np.ceil(self.iteration_steps * t_left2maturity)  # reduce iterations when closer to maturity
        dt = t_left2maturity / iterations  # discretization steps

        ## MC计算贴现payoff
        out = np.zeros(self.cuda_threads * self.cuda_blocks, dtype=np.float32)  # output array
        compute_heston_call[self.cuda_blocks, self.cuda_threads](cuda_seed, x_t, v_t,
                                                                 self.theta, self.kappa, self.xi,
                                                                 self.rf, self.correlation,
                                                                 dt, shock_magnitude, out,
                                                                 split, iterations)
        out = np.maximum(0, out - strike) if option_type == 'call' else np.maximum(0, strike - out)
        hedgeOption_ov = out[0:split].mean()
        up = out[split:2 * split].mean()
        down = out[2 * split:3 * split].mean()
        hedgeOption_delta = (up - down) / (2 * shock_magnitude) if option_type == 'call' else (down - up) / (
                2 * shock_magnitude)
        return hedgeOption_ov, hedgeOption_delta

    def calc_fia_ov_delta(self, current_day, x_t, v_t):
        '''
        计算FIA的OV和delta
        :param current_day: 当前时刻
        :param x_t: 当前时刻的股价
        :param v_t: 当前时刻的波动率
        :return: ov_fia (float32), delta_fia (float32)
        '''
        # 用于计算期权delta时的微小变化
        # 5%是否合适？
        shock_magnitude = self.shock_percent * x_t

        ## cuda parameter setup
        split = int(self.cuda_threads * self.cuda_blocks / 3)
        cuda_seed = create_xoroshiro128p_states(self.cuda_threads * self.cuda_blocks,
                                                seed=np.random.randint(-50000, 250000))

        out = np.zeros(self.cuda_threads * self.cuda_blocks, dtype=np.float32)  # output array

        t_left2maturity = current_day / self.T  ## num days left
        iterations = np.ceil(self.iteration_steps * t_left2maturity)
        dt = t_left2maturity / iterations

        # 非退出的概率（日级）
        prob_nonexit_dayunit = (1 - self.prob_exit_yearunit) ** (1 / iterations)
        prob_nondeath_dayunit = (1 - self.prob_death_yearunit) ** (1 / iterations)
        compute_heston_fia[self.cuda_blocks, self.cuda_threads](cuda_seed, x_t, v_t,
                                                                self.theta, self.kappa, self.xi,
                                                                self.rf, self.correlation,
                                                                dt, self.RealWorldPath[0, 0],
                                                                shock_magnitude, out, split, iterations,
                                                                prob_nondeath_dayunit, prob_nonexit_dayunit,
                                                                self.state_alive, t_left2maturity)

        ov_fia = out[0:split].mean()
        fia_up = out[split:2 * split].mean()
        fia_down = out[2 * split:3 * split].mean()

        delta_fia = (fia_up - fia_down) / (2 * shock_magnitude)

        return ov_fia, delta_fia


if __name__ == "__main__":
    a = SimEngine(configs=configs_dict)
    b = a.step()
    print(b)
