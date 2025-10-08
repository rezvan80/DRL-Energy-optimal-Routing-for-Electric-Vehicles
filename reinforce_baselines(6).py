import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
from nets.point_network import  Encoder
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import move_to
import torch
from torch import nn
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches
from nets.GraphEncoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # Use the probabilities of visiting each
        static, dynamic, distances, slope = x
        static = static.float().to(device)
        dynamic = dynamic.float().to(device)
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output.view(-1)


class StateCritic2(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 args,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 update_dynamic = None,
                 update_mask = None
                 ):
        super(StateCritic2, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.tanh_clipping = tanh_clipping
        self.feed_forward_hidden = 512
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.update_dynamic = update_dynamic
        self.update_mask = update_mask
        self.n_heads = n_heads

        self.start_soc = args.Start_SOC
        self.t_limit = args.t_limit
        self.custom_num = args.num_nodes
        self.charging_num = args.charging_num


        node_dim = 3
        self.init_embed_depot_and_station = nn.Linear(2, embedding_dim)
        self.init_embed_station = nn.Linear(2, embedding_dim)
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        self.critic = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        self.value_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )

        # routing context
        self.tour = nn.Linear(node_dim, self.embedding_dim, bias=False)
        self.FF_tour = nn.Sequential(
            nn. Linear(2*self.embedding_dim, self.feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(self.feed_forward_hidden, self.embedding_dim)
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)

        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input,):
        """
        :param input:
        :param return_pi:
        :return:
        """

        batch_size, _, num_node = input[0].shape
        lenth = len(input)
        if lenth == 4:
            static, dynamic, distances, slope = input
            static = static.float().to(device)
            dynamic = dynamic.float().to(device)
            distances = distances.float()
            slope = slope.float()
        else:
            static, dynamic, Elevations = input
            static = static.float().to(device)
            dynamic = dynamic.float().to(device)
            distances = torch.zeros(batch_size, num_node, num_node, device=device)
            for i in range(num_node):
                distances[:, i] = torch.sqrt(torch.sum(torch.pow(static[:, :, i:i + 1]- static[:, :, :], 2), dim=1))
            slope = torch.zeros(batch_size, num_node, num_node, device=device)
            for i in range(num_node):
                slope[:, i] = torch.clamp(torch.div((Elevations[:, i:i + 1] - Elevations[:, :]), distances[:, i] + 0.000001), min=-0.10,max=0.10)

        information = torch.cat((static, dynamic),dim=1).permute(0, 2, 1)
        #_log_p,  pi,  cost= self._inner(information, distances, slope, embeddings)

        #ll = self._calc_log_likelihood(_log_p, pi)
        h , _ = self.critic(self._init_embed(information[:, :, [0,1,3]]))  # [batch_size, N, embed_dim] -> [batch_size, N]
        value=self.value_head(h).mean(1)

                    
        return value
    def discounted_returns(self , cost, gamma):
       device = cost.device
       returns = torch.zeros_like(cost, dtype=torch.float32)

       G = torch.zeros(cost.size(0), device=device)  # batch-wise
       for t in reversed(range(cost.size(1))):  # اگر cost [batch, sequence] باشد
           G = cost[:, t] + gamma * G
           returns[:, t] = G

       return returns

    def _init_embed(self, input):

        return torch.cat(
            (
                self.init_embed_depot_and_station(input[:, 0:1, 0:2] / 100),
                self.init_embed_station(input[:, 1 : self.charging_num + 1, 0:2] / 100),
                self.init_embed(torch.cat((input[:, self.charging_num + 1:, 0:2] / 100, input[:, self.charging_num + 1:,2:3]), dim= 2)),
            ),
            dim=1)
def rollout(actor, dataset, args):
    # Put in greedy evaluation mode!

    actor.set_decode_type("greedy")
    actor.eval()

    def eval_model_bat(bat):
        # do not need backpropogation
        with torch.no_grad():
            _ ,ll ,R ,returns, v= actor(bat)
            
        return torch.stack([ll.data.cpu() , R.data.cpu() ,returns.data.cpu(), v.data.cpu()] , dim=1)

    # tqdm is a function to show the progress bar
    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=args.batch_size), disable=args.no_progress_bar)
    ], 0)

class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None 

    def eval(self, x, c):
        raise NotImplementedError("Override this method")  # 基线基类的eval，之后子类的都要对这个进行重写

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, ):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 1
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, a , b , c):

        if self.alpha == 1:
            return self.baseline.eval(x, a , b , c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, a , b , c)
        v, l = self.baseline.eval(x, a , b , c)
        vw, lw = self.warmup_baseline.eval(x, a , b , c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, a , b ,c): # x is data and c is cost in actor network

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return a.detach() , b.detach() ,c.detach() # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']



class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):

    def __init__(self, actor, valid_data, args, epoch=0):
        super(Baseline, self).__init__()

        self.dataset = valid_data
        self.args = args

        self._update_model(actor, epoch)

    def _update_model(self, actor, epoch):
        self.actor = actor
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.actor, self.dataset, self.args).cpu().numpy()[: , 1]
        
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print("Evaluating baseline on dataset...")
        
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        roll=rollout(self.actor, dataset, self.args)
        return BaselineDataset(dataset, roll[: , 0] , roll[: , 1] , roll[: , 2])

    def unwrap_batch(self, batch):
        
        return batch['data'], batch['ll'].view(-1), batch['R'].view(-1) , batch['v'].view(-1) # Flatten result to undo wrapping as 2D

    def eval(self, x, a , b , c):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            _, ll, R , v = self.actor(x)  # return baseline, cost

        # There is no loss
        return ll.detach(), R.detach() , v.detach()

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.args).cpu().numpy()[: , 1]
        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))

        # if candidate model have smaller cost than current baseline model
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.args.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.actor.state_dict(),
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = self.actor
        load_model.load_state_dict(state_dict['model'])  # 注意这里取消了 get_inner_model
        self._update_model(load_model, state_dict['epoch'])


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, ll=None , R=None , v=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        #self.baseline = baseline
        self.R=R
        self.v=v
        self.ll=ll
        assert (len(self.dataset) == len(self.R))

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'll': self.ll[item],
            'R': self.R[item],
            'v': self.v[item]
        }

    def __len__(self):
        return len(self.dataset)
