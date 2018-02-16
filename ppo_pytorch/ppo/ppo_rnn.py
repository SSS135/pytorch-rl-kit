import pprint
from collections import namedtuple
from functools import partial

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ..common import DecayLR, ValueDecay
from ..common.gae import calc_advantages, calc_returns
from ..common.multi_dataset import MultiDataset
from ..common.probability_distributions import DiagGaussianPd
from ..common.rl_base import RLBase
from ..models import MLPActorCritic
from .ppo import PPO, TrainingData


class PPO_QRNN(PPO):
    def __init__(self, observation_space, action_space,
                 *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs)

    def _reorder_data(self, data) -> (TrainingData, TrainingData):
        def reorder(input):
            x = input.view(-1, self.num_actors, *input.shape[1:])
            return x.transpose(0, 1).contiguous().view_as(input)

        data = data._asdict()
        data = [reorder(v) for v in data.values()]
        return TrainingData._make(data)

    def _ppo_update(self, data):
        self.model.train()
        # move model to cuda or cpu
        if next(self.model.parameters()).is_cuda != self.cuda_train:
            self.model = self.model.cuda() if self.cuda_train else self.model.cpu()

        data = self._reorder_data(data)

        actor_switch_flags = torch.zeros(self.horizon)
        actor_switch_flags[-1] = 1
        actor_switch_flags = actor_switch_flags.repeat(self.num_actors)

        # create dataloader
        dataset = MultiDataset(data.states, data.probs_old, data.values_old, data.actions, data.advantages,
                               data.returns, data.dones, actor_switch_flags)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for ppo_iter in range(self.ppo_iters):
            for loader_iter, batch in enumerate(dataloader):
                # prepare batch data
                st, po, vo, ac, adv, ret, done, ac_switch = [Variable(x) for x in batch]
                if self.cuda_train:
                    st, done, ac_switch = [Variable(x.data.cuda()) for x in (st, done, ac_switch)]

                if ppo_iter == self.ppo_iters - 1 and loader_iter == 0:
                    self.model.set_log(self.logger, self._do_log, self.step)
                actor_out = self.model(st, done, ac_switch)
                # get loss
                loss, kl = self._get_ppo_loss(actor_out.probs.cpu(), po, actor_out.state_values.cpu(), vo, ac, adv, ret)

                # optimize
                loss.backward()
                clip_grad_norm(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.model.set_log(self.logger, False, self.step)

            if self._do_log and ppo_iter == self.ppo_iters - 1:
                self.logger.add_scalar('learning rate', self.learning_rate, self.frame)
                self.logger.add_scalar('clip mult', self.clip_mult, self.frame)
                self.logger.add_scalar('total loss', loss, self.frame)
                self.logger.add_scalar('kl', kl, self.frame)

