import torch
from torch import nn as nn
from tqdm import tqdm
import wandb


## Q-Value model class ##
class QModel(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(QModel, self).__init__()
        dense = 256  # original model had 2 dense layers of 256 hidden nodes
        self.Layer1 = nn.Linear(state_space_size + action_space_size, dense)
        self.Layer2 = nn.Linear(dense, dense)
        self.Outlayer = nn.Linear(dense, 1)  # single Q - value output
        self.Activation = nn.ReLU()  # activation is relu

    def forward(self, state_space, action):
        ## forward function takes state & action and returns a state value action
        x = torch.cat([state_space, action], axis=1)
        x = self.Activation(self.Layer1(x))
        x = self.Activation(self.Layer2(x))
        x = self.Outlayer(x)

        return x


## critic model class ##
class Critic(nn.Module):
    ## double Q-learning - return the minimum of 2 Q networks
    ## used for critic and target critic
    ## placeholder class to make life easier, the implementation uses double clipped q-learning
    ## combine both Q models into a single critic class
    def __init__(self, state_space_size, action_space_size):
        super(Critic, self).__init__()
        ## 2 q networks for double clipped q learning
        self.Q1Network = QModel(state_space_size, action_space_size)
        self.Q2Network = QModel(state_space_size, action_space_size)

    def forward(self, state, action):
        ## base forward function ##
        q_1 = self.Q1Network.forward(state, action)
        q_2 = self.Q2Network.forward(state, action)
        return q_1, q_2

    def min_q(self, state, action):
        ## return min q to reduce the positive bias ##
        q_1, q_2 = self.forward(state, action)
        return torch.min(q_1, q_2)


## actor / policy model class
class Actor(nn.Module):
    ## Gaussian policy, 2 dense layers, a mean output layer and log-std-dev layer (for numerical stability)
    ## Uses the reparameterization trick
    def __init__(self, state_space_size, action_space_size):
        super(Actor, self).__init__()
        ## init function ##

        dense = 256
        self.Layer1 = nn.Linear(state_space_size, dense)
        self.Layer2 = nn.Linear(dense, dense)

        self.MeanLayer = nn.Linear(dense, action_space_size)
        self.LogSDLayer = nn.Linear(dense, action_space_size)

        self.Activation = nn.ReLU()

    def forward(self, state_space):
        ## forward through NN parameterizing the mean and std dev
        x = self.Activation(self.Layer1(state_space))
        x = self.Activation(self.Layer2(x))

        ## actor creates the mean and the log std deviation
        mu = self.MeanLayer(x)
        log_std = self.LogSDLayer(x)

        ## clamp the log std dev to between -20 and 2 (as in original paper)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mu, log_std

    def choose_stochastic_action(self, state_space, training):
        # used during training for exploration

        # squashed guassian sampling of action

        # calculate the mean and log standard deviation
        mean, log_std = self.forward(state_space)

        # create the norm class with mean
        norm_dist = torch.distributions.Normal(mean, log_std.exp())

        # randomly sample an action from the output distribution (reparam trick)
        random_output = norm_dist.rsample()

        # calculate log likelihood if in training mode for loss functions / updates - see project report
        log_likelihood = torch.tensor(0.0, device='cuda:0')  # convert to torch tensor zeros on gpu

        if training:
            log_likelihood = self.calculate_log_likelihood(random_output, norm_dist)
            log_likelihood = log_likelihood.sum(1, keepdim=True)

        ## apply tanh
        tanh_output = torch.tanh(random_output)

        return tanh_output, log_likelihood

    def calculate_log_likelihood(self, obs, norm_dist):
        ## calculate log_likelihood of observation given the squashed normal distribution
        log_probability = norm_dist.log_prob(obs).to('cuda:0')

        ## see paper for formula of squashed normal (tanh(N))
        power = torch.tanh(obs).pow(2)
        sum_log_tanh_sq = torch.log(torch.ones_like(power,
                                                    device='cuda:0') - power + 0.00000002)  # add some small number for numerical stability

        log_probability = log_probability - sum_log_tanh_sq

        return log_probability

    def choose_deterministic_action(self, state_space):
        # return tanh mean of the normal deterministically
        # picks the maximum a posteriori action (MAP)
        mean, log_std = self.forward(state_space)
        tanh_output = torch.tanh(mean)

        return tanh_output


class SoftActorCritic(nn.Module):
    def __init__(self, model_dict, state_space_size, action_space_size):
        super(SoftActorCritic, self).__init__()

        ## declaration of critics
        self.Critic = Critic(state_space_size, action_space_size)
        self.TargetCritic = Critic(state_space_size, action_space_size)
        self.polyak_average_params(self.Critic, self.TargetCritic,
                                   1)  # initialize the target and base critics to identical weights

        ## optimizer
        self.CriticOptimizer = torch.optim.Adam(self.Critic.parameters(), lr=model_dict['learning_rate'])

        ## single actor network for the policy
        self.Actor = Actor(state_space_size, action_space_size)
        self.ActorOptimizer = torch.optim.Adam(self.Actor.parameters(), lr=model_dict['learning_rate'])

        ## declare model specific parameters
        self.critic_loss = nn.MSELoss()
        self.batch_size = model_dict['batch_size']
        self.tau = model_dict['tau']
        self.gamma = model_dict['gamma']
        self.alpha = model_dict['alpha']
        self.TargetEntropy = -torch.prod(
            torch.Tensor(action_space_size).to('cuda:0')).item()  ## target alpha is state space dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda:0')  ## use log alpha for numerical stability
        self.AlphaOptimizer = torch.optim.Adam([self.log_alpha], lr=model_dict['learning_rate'])

    def polyak_average_params(self, model1, model2, tau):
        ## copies parameters from model1 to model2
        for model1params, model2params in zip(model1.parameters(), model2.parameters()):
            model2params.data.copy_((1 - tau) * model2params.data + tau * model1params.data)

    def train_one_step(self, batch, n_steps):
        state, action, reward, next_state, term = batch

        ## Train one step of the SAC model

        ## calculate the target values without contributing to the gradient
        with torch.no_grad():
            next_state_action, next_state_loglikelihood = self.Actor.choose_stochastic_action(next_state, True)
            # target = min(q1(s',a), q2(s',a)) - alpha * log likelihood (state|action)
            q_target = self.TargetCritic.min_q(next_state, next_state_action)  # min of the double q-models
            q_target -= self.alpha * next_state_loglikelihood
            # reward + gamma * q_target (if the next state is terminal q_target is 0)
            next_state_q_value = reward + (torch.ones_like(term, device='cuda:0') - term) * self.gamma * q_target

        ## calculate the double q-network critic network loss
        q_1_value, q_2_value = self.Critic.forward(state, action)
        q_1_loss = self.critic_loss(q_1_value, next_state_q_value)
        q_2_loss = self.critic_loss(q_2_value, next_state_q_value)  ## MSE loss function
        self.CriticOptimizer.zero_grad()
        (q_1_loss + q_2_loss).backward()  ## both q-models need to be updated (even though the min is used)
        self.CriticOptimizer.step()

        ## Actor Loss ##
        ## Determine stochastic action and calculate log likelihood
        stochastic_action, log_likelihood = self.Actor.choose_stochastic_action(state, True)
        min_q_value = self.Critic.min_q(state, stochastic_action)  # min q-value
        actor_loss = (self.alpha * log_likelihood - min_q_value).mean()  # loss is alpha * log likelihood - min(q_1,q_2)
        self.ActorOptimizer.zero_grad()
        actor_loss.backward()
        self.ActorOptimizer.step()

        ## Alpha Loss ##
        #### Tune the alpha parameter ####
        log_like_targ_entropy = (
                log_likelihood + self.TargetEntropy).detach()  ## detach so that we don't calc gradient - throws error
        alpha_loss = -(self.log_alpha * log_like_targ_entropy).mean()  ## loss is mean of the multiple
        self.AlphaOptimizer.zero_grad()
        alpha_loss.backward()
        self.AlphaOptimizer.step()
        self.alpha = self.log_alpha.exp()

        ## Add results to tensorboard for training graphs
        wandb.log({'Model Loss/Actor loss': actor_loss.item(),
                   'Model Loss/Critic 1 loss': q_1_loss.item(),
                   'Model Loss/Critic 2 loss': q_2_loss.item(),
                   'Model Loss/Alpha loss': alpha_loss.item(),
                   'Model Loss/Alpha': self.alpha.clone().item()
                   }, step=n_steps)

        ### Update the target critic network using Tau - tau chosen to be very small number
        self.polyak_average_params(self.Critic, self.TargetCritic, self.tau)

    def switch2train(self):
        self.train()
        self.Actor.train()
        self.Critic.train()
        self.TargetCritic.train()

    def switch2eval(self):
        self.eval()
        self.Actor.eval()
        self.Critic.eval()
        self.TargetCritic.eval()
