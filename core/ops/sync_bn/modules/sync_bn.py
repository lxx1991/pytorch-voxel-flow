import torch
from torch.autograd import Variable
from ..functions.sync_bn import Synchronize
from ..functions.sync_bn import sync_batch_norm


class SyncBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, parallel=False, **kwargs):
        self.parallel = parallel
        super(SyncBatchNorm2d, self).__init__(*args, **kwargs)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))

    def forward(self, input):
        if isinstance(input, Variable):
            self._check_input_dim(input)
            if self.training and self.parallel:
                B, C, H, W = input.size()
                rm = Variable(self.running_mean, requires_grad=False)
                rv = Variable(self.running_var, requires_grad=False)

                output = sync_batch_norm(
                    input.view(B, C, -1).contiguous(), rm, rv, self.weight,
                    self.bias, self.momentum, self.eps)

                self.running_mean = rm.data
                self.running_var = rv.data

                return output.view(B, C, H, W)
            else:
                return super(SyncBatchNorm2d, self).forward(input)
        else:
            raise RuntimeError('unknown input type')


def convert_bn(model, training, parallel=False):
    if isinstance(model, torch.nn.Module):
        if parallel:
            if not training:
                raise RuntimeError('unsupported parallel during testing')
            if not Synchronize.has_Listener:
                Synchronize.init(torch.cuda.device_count())

        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.train(training)
            if isinstance(m, SyncBatchNorm2d):
                m.parallel = parallel
    else:
        raise RuntimeError('unknown input type')
