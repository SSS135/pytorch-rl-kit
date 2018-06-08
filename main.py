#!/usr/bin/env python3

import argparse
import torch
from ppo_pytorch import ppo
from ppo_pytorch.common.rl_alg_test import rl_alg_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO runner')
    parser.add_argument('--tensorboard-path', type=str, metavar='DIR', required=True,
                        help='tensorboard root output folder')
    parser.add_argument('--env-name', type=str, metavar='ENV', default='CartPole-v1',
                        help='gym env name')
    parser.add_argument('--steps', type=int, metavar='N',
                        help='number of executed environment steps across all actors; '
                             'one step is four frames for atari, one frame otherwise')
    parser.add_argument('--atari', action='store_true', default=False,
                        help='enable for atari envs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training for atari envs')
    parser.add_argument('--force-cuda', action='store_true', default=False,
                        help='enable CUDA training for non-atari envs')
    args = parser.parse_args()

    assert not args.atari or args.env_name.find('NoFrameskip') != -1, \
        'only NoFrameskip atari envs are supported, since library uses custom frameskip implementation.'
    assert not args.no_cuda or not args.force_cuda

    if args.force_cuda:
        args.cuda = torch.cuda.is_available()
    elif args.no_cuda:
        args.cuda = False
    else:
        # auto selection
        args.cuda = None

    # parameters for `PPO` class
    alg_params = ppo.create_cnn_kwargs() if args.atari else ppo.create_fc_kwargs()
    # parameters for `GymWrapper` class
    wrap_params = dict(
        atari_preprocessing=args.atari,
        log_time_interval=25 if args.atari else 5,
        log_path=args.tensorboard_path,
    )

    if args.steps is not None:
        alg_params['learning_decay_frames'] = args.steps
    if args.cuda is not None:
        alg_params.update(dict(cuda_eval=args.cuda, cuda_train=args.cuda))

    print('Training on {} for {} steps, CUDA {}'.format(
        args.env_name, int(alg_params['learning_decay_frames']),
        'enabled' if alg_params['cuda_train'] else 'disabled'))

    rl_alg_test(dict(), wrap_params, ppo.PPO, alg_params, args.env_name, use_worker_id=False,
                num_processes=1, iters=1, frames=alg_params['learning_decay_frames'])