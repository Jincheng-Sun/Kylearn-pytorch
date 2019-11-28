import numpy as np
from Dataloader.transformer import *
from torch.utils.data.sampler import SubsetRandomSampler
from Dataloader.samplers import BalanceSampler



class TextualDataset(Dataset):
    def __init__(self, file_path, max_length=None):
        super().__init__()
        # Load training data
        dataframe = pd.read_csv(file_path)

        self.input_sequence_index, self.position_index, max_length = process_indexes(dataframe['indexes'], max_length)

        self.targets = dataframe['is_vulnerable'].values.reshape(-1, 1)


        self.max_length = max_length

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (torch.from_numpy(self.input_sequence_index[index, :]),
                  torch.from_numpy(self.position_index[index, :]),
                  torch.from_numpy(self.targets[index, :]))

        return sample

class TextualDataloader():
    def __init__(self, train_path, test_path, batch_size, eval_portion, max_length=207, shuffle=True):
        train_set = TextualDataset(train_path, max_length=max_length)
        test_set = TextualDataset(test_path, max_length=max_length)

        train_size = len(train_set)
        train_indices = list(range(train_size))

        test_size = len(test_set)
        test_indices = list(range(test_size))
        split = int(np.floor(eval_portion * test_size))

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(test_indices)

        test_indices, eval_indices = test_indices[split:], test_indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        # train_sampler = BalanceSampler(train_set.targets, train_indices)
        valid_sampler = SubsetRandomSampler(eval_indices)

        self.train_loader = DataLoader(train_set, batch_size, sampler=train_sampler, num_workers=4)
        self.val_loader = DataLoader(train_set, batch_size, sampler=valid_sampler, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size, num_workers=4)


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader