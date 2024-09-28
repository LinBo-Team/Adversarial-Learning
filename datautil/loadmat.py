import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, sample_path, Normalize=False):
        self.mat = sio.loadmat(sample_path)
        self.data = self.mat['sig'][:, :-1]
        self.label = self.mat['sig'][:, -1:]
        self.transform = Normalize

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        label = self.label[idx]
        if self.transform is True:
            data = (data-torch.mean(data))/torch.std(data)
        return data.float(), torch.LongTensor(label)

    def __len__(self):
        return len(self.data)


class TargetDataset(Dataset):
    def __init__(self, sample_path, transform=None):
        self.mat = sio.loadmat(sample_path)
        self.data = self.mat['sig'][:, :-1]

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        if self.transform is True:
            data = (data-torch.mean(data))/torch.std(data)
        return data.float()

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = TrainDataset('../GearData/condition1.mat', Normalize=True)
    dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=False)
    for i_batch, (batch_data, batch_label) in enumerate(dataloader):
        print(batch_data)
        if i_batch == 2:
            break
