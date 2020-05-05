import torch


class RewardWeightGenerator:
    def __init__(self, num_rewards):
        self.num_rewards = num_rewards

    @property
    def num_weights(self):
        return 4 * self.num_rewards

    def get_true_weights(self, reward_weights):
        return reward_weights[..., :self.num_rewards]

    def generate(self, num_actors, device=None):
        r = torch.rand((num_actors, self.num_rewards), device=device)
        return torch.cat([r, -r, 1 - r, r - 1], -1)


def test_RewardWeightGenerator():
    nr = 13
    rwg = RewardWeightGenerator(nr)
    assert rwg.num_rewards == nr
    gen = rwg.generate(1, torch.device('cuda'))
    assert gen.shape == (1, rwg.num_weights)
    assert gen.device.type == 'cuda'
    gen = rwg.generate(8)
    assert gen.shape == (8, rwg.num_weights)
    assert gen.device.type == 'cpu'
