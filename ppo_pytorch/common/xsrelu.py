import torch
import torch.nn as nn
import torch.nn.functional as F


class XSReLU(nn.Module):
    def forward(self, input):
        rms = input.view(input.shape[0], -1).clamp(min=0).pow(2).mean(-1).sqrt()
        return F.relu(input - rms.view(-1, *(len(input.shape) - 1) * [1]))


class XSReLU_noclamp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        rms = input.view(input.shape[0], -1).pow(2).mean(-1).sqrt()
        return F.relu(input - rms.view(-1, *(len(input.shape) - 1) * [1]))


class XSReLU_perc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        sort = input.view(input.shape[0], -1).sort(-1)[0]
        cutoff = sort[:, int(sort.shape[1] * 0.8)].contiguous().detach()
        return F.relu(input - cutoff.view(-1, *(len(input.shape) - 1) * [1]))


class XSReLU_cw(nn.Module):
    def forward(self, input):
        x = self.to_cw(input)
        x = x.clamp(min=0).pow(2).mean(-1).add(1e-8).sqrt()
        x = self.from_cw(input, x)
        return F.relu(input - x)

    def to_cw(self, x):
        if x.dim() > 2:
            return x.view(x.shape[0] * x.shape[1], -1)
        else:
            return x

    def from_cw(self, input, x):
        if input.dim() > 2:
            return x.view(*input.shape[:2], *(len(input.shape) - 2) * [1])
        else:
            return x


class XSReLU_cw_param(XSReLU_cw):
    def __init__(self, plogit=1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([plogit]))

    def forward(self, input):
        if self.plogit.numel() == 1:
            self.weight = nn.Parameter(self.weight.data.repeat(input.shape[1]))

        x = self.to_cw(input)
        x = x.clamp(min=0).pow(2).mean(-1).add(1e-8).sqrt()
        x = self.from_cw(input, x)
        x = input - x
        a_low = F.relu(x - 0.02)
        a_high = F.relu(x + 0.02)
        return a_low + (a_high - a_low) * self.weight


class XSReLU_cw_noclamp(XSReLU_cw):
    def forward(self, input):
        x = self.to_cw(input)
        x = x.pow(2).mean(-1).add(1e-8).sqrt()
        x = self.from_cw(input, x)
        return F.relu(input - x)


class XSReLU_cw_perc(XSReLU_cw):
    def forward(self, input):
        x = self.to_cw(input)
        sort = x.sort(-1)[0]
        x = sort[:, int(sort.shape[1] * 0.75)].contiguous().detach()
        x = self.from_cw(input, x)
        return F.relu(input - x)


class XSReLU_cw_perc_param(XSReLU_cw):
    def __init__(self, plogit=1.5):
        super().__init__()
        self.plogit = nn.Parameter(torch.Tensor([plogit]))

    def forward(self, input):
        if self.plogit.numel() == 1:
            self.plogit = nn.Parameter(self.plogit.data.repeat(input.shape[1]))

        x = self.to_cw(input)
        sort = x.sort(-1)[0]
        p = F.sigmoid(self.plogit).view(1, -1).repeat(input.shape[0], 1).view(-1)
        # print(p.shape, self.plogit.shape, input.shape, x.shape)
        x_low = sort[:, int(sort.shape[1] * p.sub(0.02).data[0])].contiguous().detach()
        x_high = sort[:, int(sort.shape[1] * p.add(0.02).data[0])].contiguous().detach()
        x = x_low + (x_high - x_low) * p
        x = self.from_cw(input, x)
        return F.relu(input - x)


class XSReLU_cw_perc_param_2(XSReLU_cw):
    def __init__(self, plogit=1.5, spread=0.01):
        super().__init__()
        self.spread = spread
        self.plogit = nn.Parameter(torch.Tensor([plogit]))

    def forward(self, input):
        x = input.view(*input.shape[:2], -1) if input.dim() > 2 else input.view(input.shape[0], 1, -1)

        sort = x.sort(-1)[0]
        p = F.sigmoid(self.plogit).view(1, -1)
        p_low = int(x.shape[-1] * p.sub(self.spread).data[0])
        p_high = int(x.shape[-1] * p.add(self.spread).data[0])
        x_low = sort[:, :, p_low].unsqueeze(-1).detach()
        x_high = sort[:, :, p_high].unsqueeze(-1).detach()

        r_low = F.relu(x - x_low)
        r_high = F.relu(x - x_high)
        res = r_low + (r_high - r_low) * p

        return res.view_as(input)


class XSReLU_cw_perc_param_3(XSReLU_cw):
    def __init__(self, plogit=1.5, spread=0.01):
        super().__init__()
        self.spread = spread
        self.plogit = nn.Parameter(torch.Tensor([plogit]))

    def forward(self, input):
        x = input.view(*input.shape[:2], -1) if input.dim() > 2 else input.view(input.shape[0], 1, -1)

        if self.plogit.shape[0] != x.shape[1]:
            self.plogit.data = self.plogit.data.repeat(x.shape[1])

        sort = x.sort(-1)[0]
        p = F.sigmoid(self.plogit)
        p_low = (x.shape[-1] * p.sub(self.spread).data).long()
        p_high = (x.shape[-1] * p.add(self.spread).data).long()
        range = torch.arange(sort.shape[1]).long()
        if input.is_cuda:
            range = range.cuda()
        x_low = sort[:, range, p_low].unsqueeze(-1).detach()
        x_high = sort[:, range, p_high].unsqueeze(-1).detach()

        r_low = F.relu(x - x_low)
        r_high = F.relu(x - x_high)
        res = r_low + (r_high - r_low) * p.view(-1, 1)

        return res.view_as(input)


class XSReLU_cw_perc_param_4(XSReLU_cw):
    def __init__(self, plogit=1.5, spread=0.01):
        super().__init__()
        self.spread = spread
        self.plogit = nn.Parameter(torch.Tensor([plogit / 10]))

    def forward(self, input):
        x = input.view(*input.shape[:2], -1) if input.dim() > 2 else input.view(input.shape[0], 1, -1)

        if self.plogit.shape[0] != x.shape[1]:
            self.plogit.data = self.plogit.data.repeat(x.shape[1])

        sort = x.sort(-1)[0]
        p = F.sigmoid(self.plogit * 10)
        p_low = (x.shape[-1] * p.sub(self.spread).data).long()
        p_high = (x.shape[-1] * p.add(self.spread).data).long()
        range = torch.arange(sort.shape[1]).long()
        if input.is_cuda:
            range = range.cuda()
        x_low = sort[:, range, p_low].unsqueeze(-1).detach()
        x_high = sort[:, range, p_high].unsqueeze(-1).detach()

        r_low = F.relu(x - x_low)
        r_high = F.relu(x - x_high)
        res = r_low + (r_high - r_low) * p.view(-1, 1)

        return res.view_as(input)
