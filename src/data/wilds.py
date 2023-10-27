import torch
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from anomalib.data.utils import get_transforms
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset
from wilds.common.data_loaders import get_train_loader, get_eval_loader


class album_transform_for_wilds():
    def __init__(self, album_transform):
        self.transform = album_transform

    def __call__(self, image):
        return self.transform(image=np.asarray(image))


class WildsAnomaly(WILDSSubset):
    def __init__(self, wilds_dataset: WILDSDataset, selected_ys: [int], split, grouper, subsample_ratio,
                 transform=None):
        if subsample_ratio > 1.0:
            raise ValueError("The subsample ratio can not be bigger than 1.0")
        ys = wilds_dataset.y_array
        mask_y = torch.tensor([True if y in selected_ys else False for y in ys])
        envs = grouper.metadata_to_group(wilds_dataset.metadata_array)
        divided_split = split.split("+")
        split = divided_split[0]
        if split != "all" and len(divided_split) == 1:
            mask_split = (wilds_dataset.split_array == wilds_dataset.split_dict[split])
        elif len(divided_split) > 1:  # we filter only samles with envs indicated in the split
            mask_split = wilds_dataset.split_array == wilds_dataset.split_dict[split]
            keep_env_labels = [int(idx) for idx in divided_split[1:]]
            env_split = torch.isin(envs, torch.tensor(keep_env_labels))
            mask_split = mask_split * np.array(env_split)
        else:
            mask_split = (wilds_dataset.split_array != wilds_dataset.split_dict["train"])
        idx = np.where(mask_y * mask_split)[0]
        if subsample_ratio < 1.0:
            subsample_size = int(subsample_ratio * len(idx))
            idx = np.random.choice(idx, size=subsample_size, replace=False)
        super().__init__(dataset=wilds_dataset, indices=idx, transform=transform)
        self.envs = envs[idx]
        # So far as I observed all wilds dataset has _input_array for storing image filepath
        self.input_array = wilds_dataset._input_array
        self.subsample_ratio = subsample_ratio

    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        # TODO Note that this only works for camelyon17, other datasets join their paths differently
        # image_path is added for visualization.
        image_path = os.path.join(
            self.data_dir,
            self.input_array[self.indices[idx]])
        return {"image": x['image'], "label": y, "metadata": metadata, "env_label": self.envs[idx],
                "image_path": image_path}


class WildsAnomalyLightning(LightningDataModule):
    def __init__(self, train_bach_size: int, eval_batch_size: int, num_workers: int,
                 wilds_dataset: WILDSDataset, normal_ys: [int], anomaly_ys: [int],
                 grouper, transform_config: dict, subsample_ratio: dict, splits: dict
                 ):
        super().__init__()
        self.wild_dataset = wilds_dataset
        train_transform = get_transforms(config=transform_config.train)
        val_transform = get_transforms(config=transform_config.eval)
        self.train_transform = album_transform_for_wilds(train_transform)
        self.val_transform = album_transform_for_wilds(val_transform)
        self.train_dataset = WildsAnomaly(wilds_dataset, normal_ys, split=splits["train"],
                                          transform=self.train_transform,
                                          grouper=grouper, subsample_ratio=subsample_ratio["train"])
        self.val_dataset = WildsAnomaly(wilds_dataset, anomaly_ys + normal_ys,
                                        split=splits["val"], transform=self.val_transform,
                                        grouper=grouper, subsample_ratio=subsample_ratio["val"])
        self.test_dataset = WildsAnomaly(wilds_dataset, anomaly_ys + normal_ys, split=splits["test"],
                                         transform=self.val_transform,
                                         grouper=grouper, subsample_ratio=subsample_ratio["test"])
        self.data_loader_configs = {"train_batch_size": train_bach_size, "eval_batch_size": eval_batch_size,
                                    "num_workers": num_workers}
        self.grouper = grouper

    def train_dataloader(self):
        return get_train_loader(dataset=self.train_dataset, batch_size=self.data_loader_configs['train_batch_size'],
                                num_workers=self.data_loader_configs['num_workers'], grouper=self.grouper,
                                loader="group", n_groups_per_batch=min(len(np.unique(self.train_dataset.envs)), 2))

    def val_dataloader(self):
        return get_eval_loader(loader="standard", dataset=self.val_dataset,
                               batch_size=self.data_loader_configs['eval_batch_size'],
                               num_workers=self.data_loader_configs['num_workers'], grouper=None)

    def test_dataloader(self):
        return get_eval_loader(loader="standard", dataset=self.test_dataset,
                               batch_size=self.data_loader_configs['eval_batch_size'],
                               num_workers=self.data_loader_configs['num_workers'], grouper=None)
