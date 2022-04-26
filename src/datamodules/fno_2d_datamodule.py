import numpy as np
from typing import Optional, Tuple
import re
import os
import glob

import torch
from pytorch_lightning import LightningDataModule
from torch._utils import _accumulate
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.transforms import transforms

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


class FNO2dDataset(Dataset):
    def __init__(self, 
                 data_path: str = '/disk/student/dvoytan/FNO_acoustic_wave_propogation/data/layered_models/*',
                 n_models: int = 4000,
                 x_shape: Tuple = (0, 128, 0.01),
                 y_shape: Tuple = (0, 128, 0.01),
                 reshape_shape: Tuple[int, int] = (128, 128),
                 stage: str = 'train',
                 y_rescale: float = 10000,
                 train_freqs: Tuple = (0.5, 10.1, 0.5),  # Start stop step
                 test_freqs: Tuple = (0.25, 10.25, 0.5),
                 ):

        self.data_path = natural_sort(glob.glob(data_path))[0:n_models]
    
        # Use only a subset of frequencies depending on if we are training or testing
        self.stage = stage
        self.data = self.restrict_freqs(train_freqs, test_freqs)

        self.x_shape = x_shape
        self.y_shape = y_shape
        
        self.reshape_shape = reshape_shape
        self.y_rescale = y_rescale
        
    def __len__(self):
        return len(self.data)

    def restrict_freqs(self, train_freqs, test_freqs):
        if self.stage == 'train':
            freqs = np.arange(*train_freqs)
            # Convert to string
            freqs = ['{:.2f}'.format(item) for item in freqs]
            data = [f
                    for model in self.data_path
                    for freq in freqs
                    for f in [os.path.join(model, f'{freq}_Hz/')]
                   ]
        elif self.stage == 'test':
            freqs = np.arange(*test_freqs)
            # Convert to string
            freqs = ['{:.2f}'.format(item) for item in freqs]
            data = [f
                    for model in self.data_path
                    for freq in freqs
                    for f in [os.path.join(model, f'{freq}_Hz/*')]
                   ]
        else:
            raise ValueError(f'Argument {self.stage} is invalid. Valid args are `train`, or `test`')
        return data

    def to_tensor(self, array):
        # reshapes, converts to tensor, normalize to unit std
        array = torch.from_numpy(array).type(torch.FloatTensor)
        return array

    def get_grid(self):
        size_x, size_y = int(self.x_shape[1]), int(self.y_shape[1])

        gridx = torch.tensor(np.arange(size_x)*self.x_shape[-1], dtype=torch.float)
        gridx = gridx.reshape(size_x, 1, 1).repeat([1, size_y, 1])

        gridy = torch.tensor(np.arange(size_y)*self.y_shape[-1], dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1).repeat([size_x, 1, 1])

        return torch.cat((gridx, gridy), dim=-1)

    def load_x(self, idx): 
        path = re.sub('[0-9]+\.[0-9]+_Hz/', '', self.data[idx])
        path = glob.glob(os.path.join(path, 'layercake_*'))
        x = np.fromfile(path[0], dtype='float32').reshape(self.reshape_shape)
        return x

    def load_y(self, idx):
        y_real = np.fromfile(os.path.join(self.data[idx], 'real_srcx_64'), dtype='float32')
        y_real = y_real.reshape(self.reshape_shape)

        y_imag = np.fromfile(os.path.join(self.data[idx], 'imag_srcx_64'), dtype='float32')
        y_imag = y_imag.reshape(self.reshape_shape)

        y_real = y_real * self.y_rescale
        y_imag = y_imag * self.y_rescale
        return y_real, y_imag

    def __getitem__(self, idx):
        x = self.load_x(idx)
        x = self.to_tensor(x)
        frequency = float(re.search('[0-9]+\.[0-9]+', self.data[idx])[0])
        omega = 2 * torch.tensor(np.pi) * frequency
        x = omega / x
        grid = self.get_grid()
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1)
        y_real, y_imag = self.load_y(idx)
        y_real, y_imag = self.to_tensor(y_real).unsqueeze(-1), self.to_tensor(y_imag).unsqueeze(-1)
        y = torch.cat((y_real, y_imag), axis=-1)

        return (x, y)


class FNO2dDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset: Dataset,
        train_val_test_split: Tuple[int, int, int] = (3500, 250, 250),
        batch_size: int = 12,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.dataset = dataset

        self.corrected_split = self.convert_models_to_samples()

    def convert_models_to_samples(self):
        '''
        The train test split is done at the model level so we need to account for the number of
        frequencies.
        '''
        n_freqs = len(self.dataset) / sum(self.hparams.train_val_test_split)
        split = [int(n_freqs * item) for item in self.hparams.train_val_test_split]
        return split

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        indices = torch.arange(sum(self.corrected_split))
        lengths = self.corrected_split
        self.data_train, self.data_val, self.data_test = [Subset(
            self.dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
