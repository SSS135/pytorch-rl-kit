import sys

from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(name='ppo_pytorch',
      packages=['ppo_pytorch'],
      install_requires=[
          'gym[mujoco,atari,classic_control]',
          'scipy',
          'tensorboardX',
          # 'Pillow',
          'torch',
          # 'opencv',
      ],
      description="Proximal Policy Optimization in PyTorch",
      author="Alexander Penkin",
      url='https://github.com/SSS135/ppo-pytorch',
      author_email="sss13594@gmail.com",
      version="0.1")
