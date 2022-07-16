import torch


class RewardWeightGenerator:
    def __init__(self, num_rewards):
        self.num_rewards = num_rewards

    @property
    def num_weights(self):
        return self.num_rewards

    def get_true_weights(self, reward_weights):
        return reward_weights#[..., :self.num_rewards]

    def generate(self, num_actors, device=None):
        # weights = torch.tensor([
        #     (1, 0, 0),
        #     (1, 0, 1),
        #     (1, -0.2, 1),
        #     (0, 1, 1),
        #     (0, 0, 1),
        #     (0, -0.2, 1),
        #     (-0.2, 0, 1),
        # ], device=device)
        # idx = torch.randint(0, len(weights), size=(num_actors,), device=device)
        # return weights[idx]

        # r = torch.rand((num_actors, 1), device=device)
        # return torch.cat([r * 0 + 1, r * 0, r * 0], -1)
        # return torch.randn((num_actors, self.num_rewards), device=device) + 0.5
        # return torch.cat([r, -r, 1 - r, r - 1], -1)

        randn = torch.zeros((num_actors, self.num_rewards), device=device)
        randn[:, 0] = 1
        # randn[:, 2] = 0.1
        # randn[:, 3] = 0
        return randn#.softmax(dim=1)


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
