import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import os
import pytorch_lightning as pl

from pprint import pprint
from glob import glob
from functools import partial

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score, Accuracy
from torchsampler import ImbalancedDatasetSampler

from data_utils import create_features
from inception_time_ori import InceptionTime
from lit_model import LitModel

torch.multiprocessing.set_start_method('spawn', force=True)
torch.serialization.add_safe_globals([CrossEntropyLoss])


class XAU5M(Dataset):
    def __init__(self, window_h=24, mode="train"):
        super().__init__()

        self.window = window_h * 12
        self.normalize = False
        df = pd.read_csv("./XAU_5m_data_labels_24_6_3.csv")

        df["time"] = pd.to_datetime(df["time"])
        from_drop = datetime(2025, 9, 12)
        df = df.loc[df["time"] < from_drop]

        from_test = datetime(2024, 9, 11)
        if mode == "train":
            self.df = df[df["time"] < from_test].reset_index(drop=True)
            print(f"Train: {self.df['time'].iloc[0]} – {self.df['time'].iloc[-1]} | {len(self.df):,}")
        else:
            self.df = df[df["time"] >= from_test].reset_index(drop=True)
            print(f"Test:  {self.df['time'].iloc[0]} – {self.df['time'].iloc[-1]} | {len(self.df):,}")

        self.features = [
            'log_ret', 'vol_20',
            'close_over_ma', 'body_ratio',
            'rsi_14', 'rsi_7',
            'macd', 'macd_sig',
            'atr_14', 'bb_z',
            'donch_high_ratio', 'donch_low_ratio',
            # 'ma_15m_20'  # multi‑tf feature
        ]
        self.n_features = len(self.features)

        self.df = create_features(self.df)
        self.df.dropna(inplace=True)

        self.indices = list(range(self.window, len(self.df)))
        print(mode, " len of indices:", len(self.indices))

        self.label_map = {
            "buy": 0,
            "sell": 1,
            "hold": 2
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        current_idx = self.indices[idx]
        window_data = self.df.iloc[current_idx - self.window:current_idx]

        if self.normalize:
            window_data[["open", "close", "high", "low"]] -= window_data[["open", "close", "high", "low"]].iloc[0]

        # "open", "close", "high", "low", "volume", "rsi", "open", "close", "high", "low", "volume", "rsi", ...
        obs = window_data[self.features].values.flatten().astype(np.float32)
        obs = obs.reshape(self.window, self.n_features).transpose(1, 0)
        label = self.label_map[self.df.iloc[current_idx]["label"]]
        return obs, label

    def get_labels(self):
        return [self.label_map[self.df.iloc[i]["label"]] for i in self.indices]

    def print_com_stat_dict(self):
        sell = (self.df["label"] == "sell").sum()
        buy = (self.df["label"] == "buy").sum()
        hold = (self.df["label"] == "hold").sum()

        print("sell: {}, {:.2f}%".format(sell, sell / len(self.df) * 100))
        print("buy: {}, {:.2f}%".format(buy, buy / len(self.df) * 100))
        print("hold: {}, {:.2f}%".format(hold, hold / len(self.df) * 100))


def save_log(_params, _log_train, _log_val):
    with open(os.path.join(_params["model_base_path"], "log_f1_{:.2f}.json".format(_log_val[0]["val_f1"])), "w") as f:
        log = dict(_log_val[0])
        for key in _log_train[0]:
            log[key.replace("val", "train")] = _log_train[0][key]
        json.dump(log, f, indent=4, sort_keys=True)


def train(params: dict):
    pprint(params)

    train_dataset = XAU5M(window_h=params["window_h"], mode="train")
    val_dataset = XAU5M(window_h=params["window_h"], mode="val")

    print("\ntrain dataset stat:")
    train_dataset.print_com_stat_dict()
    print("\nval dataset stat:")
    val_dataset.print_com_stat_dict()

    train_loader = DataLoader(train_dataset,
                              sampler=ImbalancedDatasetSampler(train_dataset),
                              batch_size=params["train_batch_size"],
                              shuffle=False,  # sampler doas the shuffle
                              num_workers=params["num_workers"],
                              persistent_workers=params["num_workers"] > 0)
    val_loader = DataLoader(val_dataset,
                            batch_size=params["val_batch_size"],
                            shuffle=False,
                            num_workers=params["num_workers"],
                            persistent_workers=params["num_workers"] > 0)

    optimizer = partial(torch.optim.AdamW, lr=params["learning_rate"], weight_decay=params["wd"], amsgrad=True)
    metric_list = list()
    # Change to another metrics as it's not good for unbalanced
    acc = Accuracy(task="multiclass", num_classes=3).to(torch.device(params["device"]))
    acc.name = "acc"
    metric_list.append(acc)
    f1 = F1Score(task="multiclass", num_classes=3, average='macro').to(torch.device(params["device"]))
    f1.name = "f1"
    metric_list.append(f1)

    xe = CrossEntropyLoss()  # weight=normed_weights
    xe.name = "xe_loss"
    loss_list = [xe]

    early_stop_callback = EarlyStopping(monitor="val_f1", min_delta=0.0005, patience=params["patience"], mode="max")
    checkpoint_callback = ModelCheckpoint(dirpath=params["model_base_path"], save_top_k=1, monitor="val_f1", mode="max")
    model = InceptionTime(in_channels=train_dataset.n_features, out_size=3)

    if params["model_checkpoint_folder_path"] is not None:
        ckpt_path = sorted(glob(os.path.join(params["model_checkpoint_folder_path"], "*.ckpt")))[-1]
        print(ckpt_path)
        lit_model = LitModel.load_from_checkpoint(ckpt_path, model=model, weights_only=False)
    else:
        lit_model = LitModel(model=model, loss_list=loss_list, metric_list=metric_list, optimizer=optimizer)

    # inference_mode="predict"
    trainer = pl.Trainer(max_epochs=params["num_epoch"],
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=TensorBoardLogger(params["model_base_path"], default_hp_metric=False),
                         log_every_n_steps=10,
                         accelerator="cuda" if "cuda" in params["device"] else "cpu",
                         devices=int(params["device"].split(":")[1]) + 1 if "cuda" in params["device"] else 1)
    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=None)
    try:
        log_train = trainer.validate(model=lit_model, dataloaders=train_loader, ckpt_path="best", verbose=True,
                                     weights_only=False)
    except TypeError:
        log_train = trainer.validate(model=lit_model, dataloaders=train_loader, ckpt_path="best", verbose=True)
    try:
        log_val = trainer.validate(model=lit_model, dataloaders=val_loader, ckpt_path="best", verbose=True,
                                   weights_only=False)
    except TypeError:
        log_val = trainer.validate(model=lit_model, dataloaders=val_loader, ckpt_path="best", verbose=True)
    save_log(params, log_train, log_val)


if __name__ == "__main__":
    # PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python tensorboard --logdir ./models
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    param_dict = {
        "model_base_path": Path("./models") / datetime.now().strftime('%Y-%m-%d-%H-%M'),
        "model_checkpoint_folder_path": None,
        "num_epoch": 1000,
        "learning_rate": 0.0001,
        "train_batch_size": 64,
        "val_batch_size": 256,
        "wd": 0.001,
        "patience": 15,
        "num_workers": 8,
        "device": "cuda:0",

        "window_h": 24
    }
    train(param_dict)
