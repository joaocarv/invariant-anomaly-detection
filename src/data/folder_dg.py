from __future__ import annotations
from anomalib.data.folder import FolderDataset, Folder
from src.utils.generate_ds_utils import *
from pathlib import Path
import os
import pandas as pd
import torch
import inspect

__all__ = ["FolderDg"]


class FolderDgDataset(FolderDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.normal_env_labels: torch.tensor = None
        self.abnormal_env_labels: torch.tensor = None

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        if self.split == 'train':
            item['env_label'] = self.normal_env_labels[index]
        else:
            item['env_label'] = 0
            """
            label = item["label"]
            if label == 0:
                item['env_label'] = self.normal_env_labels[index]
            else:
                item['env_label'] = self.abnormal_env_labels[index]
            """

        return item

    def set_env_labels(self):
        self.normal_env_labels = self._set_env_labels(self.normal_dir)
        self.abnormal_env_labels = self._set_env_labels(self.abnormal_dir)

    def _set_env_labels(self, dir_path):
        path = Path(self.root) / dir_path
        path_to_env_labels = path.parent / "env_label.csv"
        return torch.tensor(pd.read_csv(path_to_env_labels)['label'])

    @classmethod
    def from_folder_dataset(cls, folder: FolderDataset) -> FolderDgDataset:
        kwargs_keys = inspect.getfullargspec(FolderDataset.__init__).args[1:]
        folder_attributes = folder.__dict__
        kwargs = {key: folder_attributes[key] for key in kwargs_keys}
        return cls(**kwargs)

    @classmethod
    def get_init_args(cls) -> list:
        return inspect.getfullargspec(FolderDataset.__init__).args[1:]


class FolderDg(Folder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # convert train and test data to FolderDgDataset
        self.train_data = FolderDgDataset.from_folder_dataset(self.train_data)
        self.test_data = FolderDgDataset.from_folder_dataset(self.test_data)

    def setup(self, stage: str | None = None) -> None:
        root = self.train_data.root
        if not os.path.isdir(root):  # no data, need to generate first
            path = Path(root)
            yml_folder = path.parent
            if path.name != "dataset":
                raise NotADirectoryError("If the data doesn't exist, the root must end with 'dataset'")
            generate_env_and_labels(yml_folder)
        super().setup(stage)
        self.train_data.set_env_labels()
        self.test_data.set_env_labels()

    @classmethod
    def get_init_args(cls) -> set:
        # inspect.getfullargspec(Folder.__init__).defaults
        return inspect.getfullargspec(Folder.__init__).args[1:]
