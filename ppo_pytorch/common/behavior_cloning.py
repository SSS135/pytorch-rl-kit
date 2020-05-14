import pathlib

import PIL.Image
import gym
from torch.nn.utils import clip_grad_norm_
from optfn.virtual_adversarial_training import get_vat_inputs

from ..actors import create_ppo_cnn_actor
from .attr_dict import AttrDict
from .data_loader import DataLoader

from .gae import calc_value_targets
from mlagents.trainers.demo_loader import load_demonstration
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
import torchvision.transforms as tvt


def load_demo(path, visual_observations, discount):
    print(f'loading {path}')
    demo = load_demonstration(path)
    if visual_observations:
        states = torch.tensor([e.visual_observations for e in demo[1]]).mul_(255).squeeze_(1).transpose_(-1, -3).byte()
    else:
        states = torch.tensor([e.vector_observations for e in demo[1]], dtype=torch.float)
    actions = torch.tensor([e.previous_vector_actions for e in demo[1]]).long()
    rewards = torch.tensor([e.rewards for e in demo[1]], dtype=torch.float)
    dones = torch.tensor([e.local_done for e in demo[1]]).float()
    values = torch.zeros((rewards.shape[0] + 1, *rewards.shape[1:]), dtype=torch.float)
    value_targets = calc_value_targets(rewards, values, dones, discount)
    return [x.squeeze(1) for x in (states, actions, value_targets)]


def load_all_demos(name_mask, folder, visual_observations, discount):
    files = pathlib.Path(folder).resolve().glob(name_mask)
    data = [load_demo(fname, visual_observations, discount) for fname in files]
    return [torch.cat(x, 0) for x in zip(*data)]


def train(num_epochs, states, logits, net):
    with torch.no_grad():
        device = torch.device('cuda')
        batch_size = 256
        log_interval = 100

        net = net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
        data = AttrDict(states=states, logits=logits)

        total_batch_index = 0

        for epoch in range(num_epochs):
            rand_idx = torch.randperm(data.states.shape[0], device=device).split(batch_size)
            with DataLoader(data, rand_idx, device, 2, dim=0) as data_loader:
                for batch_index in range(len(rand_idx)):
                    batch = AttrDict(data_loader.get_next_batch())
                    do_log = total_batch_index % log_interval == 0
                    train_step(net, batch, optimizer, epoch, batch_index, do_log)
                    total_batch_index += 1
                    sched.step()

        return net


def train_step(net, batch, optimizer, epoch, batch_index, do_log):
    pd = net.heads.logits.pd

    with torch.enable_grad():
        ac_out = net(batch.states, evaluate_heads=['logits'])
        kl = pd.kl(batch.logits, ac_out.logits)
        loss = kl.mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if do_log:
        print(f'epoch {epoch}, batch {batch_index}, kl {loss:.4f}, lr {optimizer.param_groups[0]["lr"]}')