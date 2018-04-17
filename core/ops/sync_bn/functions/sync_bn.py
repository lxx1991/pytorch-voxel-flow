import torch
from torch import cuda
from torch.autograd import Function
from .._ext import sync_bn_lib


class _sync_batch_norm(Function):
    def __init__(self, momentum, eps, queue):
        super(_sync_batch_norm, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.queue = queue
        self.allreduce_num = len(self.queue) + 1

    def all_reduce_thread(self, input):
        input_device = input.get_device()
        if input_device == 0:
            data_list = [input]
            for i in range(self.allreduce_num - 1):
                data_list.append(self.queue[i].get())

            cuda.synchronize()
            # total_sum = Synchronize.data_list[0].cpu().clone()
            # for i in range(1, Synchronize.device_num):
            #     total_sum = total_sum + Synchronize.data_list[i].cpu()

            # for i in range(0, Synchronize.device_num):
            #     with torch.cuda.device_of(Synchronize.data_list[i]):
            #         Synchronize.result_list[i] = total_sum.clone().cuda()

            cuda.nccl.all_reduce(data_list)
            cuda.synchronize()

            for i in range(self.allreduce_num - 1):
                self.queue[i].task_done()
        else:
            self.queue[input_device - 1].put(input)
            self.queue[input_device - 1].join()

        return input

    def forward(self, input, running_mean, running_var, weight, bias):

        with torch.cuda.device_of(input):
            mean = input.new().resize_(input.size(1)).zero_()
            var = input.new().resize_(input.size(1)).zero_()
            x_std = input.new().resize_(input.size(1)).zero_()
            x_norm = input.new().resize_as_(input)
            output = input.new().resize_as_(input)

        sync_bn_lib.bn_forward_mean_before_allreduce(input, mean,
                                                     self.allreduce_num)
        mean = self.all_reduce_thread(mean)
        sync_bn_lib.bn_forward_var_before_allreduce(input, mean, var, output,
                                                    self.allreduce_num)
        var = self.all_reduce_thread(var)

        sync_bn_lib.bn_forward_after_allreduce(
            mean, running_mean, var, running_var, x_norm, x_std, weight, bias,
            output, self.eps, 1.0 - self.momentum)

        self.save_for_backward(weight, bias)
        self.mean = mean
        self.x_norm = x_norm
        self.x_std = x_std

        return output

    def backward(self, grad_output):
        weight, bias = self.saved_tensors

        with torch.cuda.device_of(grad_output):
            grad_input = grad_output.new().resize_as_(grad_output).zero_()
            grad_weight = grad_output.new().resize_as_(weight).zero_()
            grad_bias = grad_output.new().resize_as_(bias).zero_()
            grad_local_weight = grad_output.new().resize_as_(weight).zero_()
            grad_local_bias = grad_output.new().resize_as_(bias).zero_()

        sync_bn_lib.bn_backward_before_allreduce(
            grad_output, self.x_norm, self.mean, self.x_std, grad_input,
            grad_local_weight, grad_local_bias, grad_weight, grad_bias)

        grad_local_weight = self.all_reduce_thread(grad_local_weight)
        grad_local_bias = self.all_reduce_thread(grad_local_bias)

        sync_bn_lib.bn_backward_after_allreduce(
            grad_output, self.x_norm, grad_local_weight, grad_local_bias,
            weight, self.x_std, grad_input, self.allreduce_num)

        return grad_input, None, None, grad_weight, grad_bias


def sync_batch_norm(input,
                    running_mean,
                    running_var,
                    weight=None,
                    bias=None,
                    momentum=0.1,
                    eps=1e-5,
                    queue=None):
    r"""Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _torch_ext.batchnormtrain:

    .. math::

        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    return _sync_batch_norm(momentum, eps, queue)(input, running_mean,
                                                  running_var, weight, bias)
