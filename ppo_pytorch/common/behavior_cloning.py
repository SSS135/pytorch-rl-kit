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


def train(num_epochs, states, actions, value_targets, model_factory, observation_space, action_space, visual):
    with torch.no_grad():
        cuda = torch.device('cuda')
        cpu = torch.device('cpu')
        batch_size = 32
        log_interval = 100

        value_targets = (value_targets - value_targets.mean()) / (value_targets.std() + 1e-6)

        net = model_factory(observation_space, action_space).to(cuda)
        net_target = model_factory(observation_space, action_space).to(cuda)
        # net_target.load_state_dict(torch.load('tensorboard/PPO_SimpleArenaDiscreteVisual__2019-11-18_12-15-20_zbrioi1h/model_0.pth'))
        optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [8, 15], 0.1)
        data = AttrDict(states=states, actions=actions, value_targets=value_targets)

        total_batch_index = 0

        transf = tvt.Compose([
            tvt.ToPILImage(),
            tvt.RandomAffine(5, (0.03, 0.03), (0.95, 1.05), (-5, 5, -5, 5), fillcolor=(77, 123, 144), resample=PIL.Image.BILINEAR),
            # tvt.ColorJitter(0.1, 0.1, 0.1, 0.05),
            tvt.ToTensor(),
            tvt.RandomErasing(),
        ])

        def process_chunk(name, x):
            if visual and name == 'states':
                x = torch.stack([transf(img) for img in x], 0)
            return x

        for epoch in range(num_epochs):
            rand_idx = torch.randperm(data.states.shape[0], device=cuda).split(batch_size)
            with DataLoader(data, rand_idx, cuda, 4, dim=0, chunk_fn=process_chunk) as data_loader:
                for batch_index in range(len(rand_idx)):
                    # prepare batch data
                    batch = AttrDict(data_loader.get_next_batch())
                    do_log = total_batch_index % log_interval == 0
                    train_step(net, net_target, batch, optimizer, epoch, batch_index, do_log)
                    total_batch_index += 1
            sched.step()

        return net


def train_step(net, net_target, batch, optimizer, epoch, batch_index, do_log):
    states = batch.states#.float().div_(255)
    pd = net.heads.logits.pd
    lr = optimizer.param_groups[0]['lr']
    ac_out_target = net_target(states)
    # target_actions = logits_to_actions(ac_out_target.logits)
    # onehot = actions_to_onehot(target_actions).float()
    # onehot = (onehot - onehot.mean()) * 5

    with torch.enable_grad():
        ac_out = net(states)
        # net_actions = logits_to_actions(ac_out.logits)
        # logp = pd.logp(target_actions, ac_out.logits).sum(-1)
        logp = pd.logp(batch.actions, ac_out.logits).sum(-1)
        entropy = pd.entropy(ac_out.logits).sum(-1)
        # policy_loss = F.mse_loss(ac_out.logits, onehot)
        policy_loss = -logp.mean()
        # policy_loss = F.mse_loss(ac_out.logits, ac_out_target.logits)
        value_loss = 0.0 * F.mse_loss(ac_out.state_values.squeeze(-1), batch.value_targets)
        loss = value_loss + policy_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if do_log:
        print(f'epoch {epoch}, batch {batch_index}, '
              # f'acc bc {(batch.actions == net_actions).float().mean():.4f}, '
              # f'acc target {(target_actions == net_actions).float().mean():.4f}, '
              f'nll {policy_loss:.4f}, entropy {entropy.mean():.4f}, '
              f'target diff {(ac_out.logits - ac_out_target.logits).pow(2).mean().sqrt():.4f}, '
              f'mse loss {F.mse_loss(ac_out.logits, ac_out_target.logits):.4f}, '
              f'value loss {value_loss:.4f}, lr {lr:.6f}')


# def logits_to_actions(logits):
#     return torch.stack([torch.max(x, -1)[1] for x in logits.chunk(2, -1)], -1)
#
#
# def actions_to_onehot(actions):
#     onehot = torch.zeros((actions.shape[0], 2, 5), device=actions.device, dtype=torch.uint8)
#     onehot = onehot.scatter_(-1, actions.unsqueeze(-1), 1)
#     return onehot.view(actions.shape[0], -1)