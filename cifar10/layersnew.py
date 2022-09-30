import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

steps = 4
dt = 5
aa = 0.5 # pseudo derivative range
kappa = 0.5
tau = 0.25  # decay factor

class SpikeAct(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu


spikeAct = SpikeAct.apply


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n1, thre):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n1
    v_t1_n1 = u_t1_n1 - thre
    o_t1_n1 = spikeAct(v_t1_n1)
    return u_t1_n1, o_t1_n1


class tdConv(nn.Module):

    def __init__(self, layer):
        super(tdConv, self).__init__()
        self.layer = layer

    def forward(self, x):
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        return x_


class tdNorm(nn.Module):
    def __init__(self, bn):
        super(tdNorm, self).__init__()
        self.bn = bn

    def forward(self, x):
        x_ = torch.zeros(x.size(), device=x.device)
        x_ = self.bn(x)

        return x_


class LIFSpike(nn.Module):
    def __init__(self):
        super(LIFSpike, self).__init__()
        init_w = kappa
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, x):
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x[..., step], self.w.tanh())
        return out


class LIFOutSpike(nn.Module):
    def __init__(self):
        super(LIFOutSpike, self).__init__()
        init_w = kappa
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, x, lateral):
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step] = out_state_update(u, out[..., max(step - 1, 0)], x[..., step], self.w.tanh(), lateral[..., step])
        return out


def out_state_update(u_t_n1, o_t_n1, W_mul_o_t1_n1, thre, lateral):
    u_t1_n1 = tau * u_t_n1 * (1 - torch.gt(o_t_n1, thre).float()) + W_mul_o_t1_n1 + lateral
    o_t1_n1 = OutspikeAct(u_t1_n1, thre)
    return u_t1_n1, o_t1_n1


class OutSpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thre):
        ctx.save_for_backward(input)
        ctx.threshol = thre
        output = torch.gt(input, thre)
        output = output.float() * input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        thre = ctx.threshol
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < (thre-aa)] = 0
        return grad_input, None

OutspikeAct = OutSpikeAct.apply


class sigOut(nn.Module):
    def __init__(self):
        super(sigOut, self).__init__()

    def forward(self, x):
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step] = sig_state_update(u, out[..., max(step - 1, 0)], x[..., step])
        return out


def sig_state_update(u_t_n1, o_t_n1, W_mul_o_t1_n1):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n1
    o_t1_n1 = nn.Sigmoid()(u_t1_n1)
    return u_t1_n1, o_t1_n1


class newBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(newBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean([0, 2, 3, 4])
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean.copy_(exponential_average_factor * mean \
                                        + (1 - exponential_average_factor) * self.running_mean)
                self.running_var.copy_(exponential_average_factor * var * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_var)
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input

