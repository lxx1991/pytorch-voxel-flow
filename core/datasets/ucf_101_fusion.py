from torch.utils.data import Dataset
from core.datasets import UCF101, UCF101Test


class UCF101Fusion(Dataset):

    num_class = 21
    ignore_label = 255
    background_label = 0

    def __init__(self, config_train, config_test):
        super(UCF101Fusion, self).__init__()

        self.ds_train = UCF101(config_train)
        self.ds_test = UCF101Test(config_test)

    def __len__(self):
        return len(self.ds_train) + len(self.ds_test) * 100

    def __getitem__(self, idx):
        if idx < len(self.ds_train):
            return self.ds_train.__getitem__(idx) + (1, )
        else:
            idx = (idx - len(self.ds_train)) % len(self.ds_test)
            return self.ds_test.__getitem__(idx, False) + (0, )
