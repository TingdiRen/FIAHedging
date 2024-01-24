from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
import math

@cuda.jit
def test(x):
    x += 1
    return

## cuda functions for option valuation of both FIA and call
## calculating the call option value
@cuda.jit
def compute_heston_call(seed, x_0, v_0, theta, kappa, xi, rf, rho, dt, shock, out, thread_split,
                        iterations):
    thread_id = cuda.grid(1)

    # shock会根据线程的thread_id值来确定何时被应用到资产价格x上
    if thread_id > thread_split - 1 and thread_id < 2 * thread_split - 1:
        x_0 = x_0 + shock
    elif thread_id > thread_split * 2 - 1:
        x_0 = x_0 - shock

    x, v = x_0, v_0
    sqrt_dt = math.sqrt(dt)
    for i in range(iterations):
        # euler scheme for heston
        # heston - x_t+1 = x_t * exp(r_f - v(t)/2)dt + sqrt(v(t)) * dW_x_t)
        # v - v(t+1) = v(t) + kappa (theta - v(t))*dt + xi * sqrt(v(t)) * dW_V_t

        # 标准正态随机数生成器
        w_x = xoroshiro128p_normal_float32(seed, thread_id)
        w_v = xoroshiro128p_normal_float32(seed, thread_id)

        corr_random = rho * w_v + math.sqrt(1 - rho * rho) * w_x
        x = x * math.exp((rf - .5 * v) * dt + math.sqrt(v) * sqrt_dt * corr_random)
        # v = v + kappa * dt * (theta - v) + xi * math.sqrt(v) * math.sqrt(dt) * w_v

        # if v < 0.01:
        #     v = 0.01

    out[thread_id] = x


### calculating the option value of the FIA
@cuda.jit
def compute_heston_fia(seed, x_0, v_0, theta, kappa, xi, rf, rho, dt, strike, shock, out, thread_split,
                       iterations, p_death, p_exit, alive, T):
    ##set s_0 outside of function with dim = #threads
    thread_id = cuda.grid(1)

    # shock会根据线程的thread_id值来确定何时被应用到资产价格x上
    if thread_id > thread_split - 1 and thread_id < 2 * thread_split - 1:
        x_0 = x_0 + shock
    elif thread_id > thread_split * 2 - 1:
        x_0 = x_0 - shock

    x, v = x_0, v_0
    sqrt_dt = math.sqrt(dt)
    for i in range(iterations):
        w_x = xoroshiro128p_normal_float32(seed, thread_id)
        w_v = xoroshiro128p_normal_float32(seed, thread_id)

        corr_random = rho * w_v + math.sqrt(1 - rho * rho) * w_x

        x = x * math.exp((rf - .5 * v) * dt + math.sqrt(v) * sqrt_dt * corr_random)
        v = v + kappa * dt * (theta - v) + xi * math.sqrt(v) * math.sqrt(dt) * w_v

        if v < 0.001:
            v = 0.001

        # random death
        if xoroshiro128p_uniform_float32(seed, thread_id) > p_death:
            # death - pay 80% of gain
            out[thread_id] = max(0, 0.8 * (x - strike) * math.exp(-rf * T * (i / iterations)))
            return
        # 锁定收益机制，如果达到特定条件，未来的收益或指数水平可能被“锁定”，保障一定程度的收益。
        if alive and x > 1.5 * strike:  # ph locks in index x
            if xoroshiro128p_uniform_float32(seed, thread_id) > (1 / math.exp((x / strike) * 0.005)):
                out[thread_id] = max(0, (x - strike) * math.exp(-rf * T * (i / iterations)))
                return
        # random exit
        if xoroshiro128p_uniform_float32(seed, thread_id) > p_exit:
            out[thread_id] = 0
            return

    out[thread_id] = max(0, (x - strike) * math.exp(-rf * T))
