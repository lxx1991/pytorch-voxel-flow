import numpy as np


class EvalPSNR(object):

    def __init__(self, max_level):
        self.max_level = max_level
        self.clear()

    def __call__(self, pred, gt, mask=None):
        assert (pred.shape == gt.shape)
        if mask is None:
            mask = np.ones((pred.shape[0], pred.shape[2], pred.shape[3]))
        for i in range(pred.shape[0]):
            temp = np.tile(mask[i, np.newaxis, :, :], (3, 1, 1))
            if np.sum(temp) == 0:
                continue
            delta = (pred[i, :, :, :] - gt[i, :, :, :]) * temp
            delta = np.sum(np.square(delta)) / np.sum(temp)
            self.psnr += 10 * np.log10(self.max_level * self.max_level / delta)
            self.count += 1

    def PSNR(self):
        return self.psnr / max(1, self.count)

    def clear(self):
        self.psnr = 0
        self.count = 0
