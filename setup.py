import sys

from setuptools import setup

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(name='ppo_pytorch',
    install_requires=[
        'tensorboardX',
    ],
    description="Proximal Policy Optimization in PyTorch",
    author="Alexander Penkin",
    url='https://github.com/SSS135/pytorch-rl-kit',
    author_email="sss13594@gmail.com",
    version="0.1",
    packages=['ppo_pytorch'],
    zip_safe=False)
