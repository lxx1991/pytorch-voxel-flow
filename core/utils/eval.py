import numpy as np


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.clear()

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = np.bitwise_and((gt != self.ignore_label),
                              (pred != self.ignore_label))
        sumim = gt + pred * self.num_class
        hs = np.bincount(
            sumim[locs], minlength=self.num_class**2).reshape(
                self.num_class, self.num_class)
        self.conf += hs

    def acc(self):
        return np.sum(np.diag(self.conf)) / float(np.sum(self.conf))

    def num(self):
        return np.sum(self.conf, axis=1)

    def IoU(self):
        return np.diag(self.conf) / (
            1e-20 + self.conf.sum(1) + self.conf.sum(0) - np.diag(self.conf))

    def mIoU(self):
        iou = self.IoU()
        return np.sum(iou) / len(iou)

    def clear(self):
        self.conf = np.zeros((self.num_class, self.num_class))


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
