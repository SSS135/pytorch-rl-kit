# Proximal Policy Optimization in PyTorch

## New gym environments

When library imported following gym environments are registered:

Continuous versions of Acrobot and CartPole `AcrobotContinuous-v1`, `CartPoleContinuous-v0`, `CartPoleContinuous-v1`

CartPole with 10000 steps limit `CartPoleContinuous-v2`, `CartPole-v2`

## Training

Training code does not print any information to console, only available output is TensorBoard logs.

#### Classic control
`CartPole-v1` for 500K steps without CUDA (`--force-cuda` to enable it, won't improve performance)

`python main.py --env-name CartPole-v1 --tensorboard-path /tensorboard/output/path`

#### Atari
`PongNoFrameskip-v4` for 3M steps (12M emulator frames)

`python main.py --atari --env-name PongNoFrameskip-v4 --steps 3000000 --tensorboard-path /tensorboard/output/path`


## Results

#### PongNoFrameskip-v4
<img src="images/pong.png" width="500">
Activations of first convolution layer
<img src="images/pong_activations.png" width="300">
Filters of first convolution layer
<img src="images/pong_filters.png" width="300">
Absolute value of gradients of state pixels (sort of pixel importance)
<img src="images/pong_attention.png" width="300">

#### BreakoutNoFrameskip-v4
<img src="images/breakout.png" width="500">

#### QbertNoFrameskip-v4
<img src="images/qbert.png" width="500">

#### SpaceInvadersNoFrameskip-v4
<img src="images/spaceinvaders.png" width="500">

#### SeaquestNoFrameskip-v4
<img src="images/seaquest.png" width="500">

#### CartPole-v1
<img src="images/cartpole-v1.png" width="500">

#### CartPoleContinuous-v2
<img src="images/cartpolecontinuous-v2.png" width="500">
