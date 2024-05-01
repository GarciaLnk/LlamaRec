import random

import numpy as np
import torch
import torch.utils.data as data_utils


def worker_init_fn(worker_id):
    random.seed(int(np.random.get_state()[1][0]) + worker_id)
    np.random.seed(int(np.random.get_state()[1][0]) + worker_id)


class RNNDataloader:
    def __init__(self, args, dataset):
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset["train"]
        self.val = dataset["val"]
        self.test = dataset["test"]
        self.umap = dataset["umap"]
        self.smap = dataset["smap"]
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_users = self.user_count
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len

    @classmethod
    def code(cls):
        return "rnn"

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.args.num_workers,
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    def _get_train_dataset(self):
        dataset = RNNTrainDataset(self.args, self.train, self.max_len)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode="val")

    def _get_test_loader(self):
        return self._get_eval_loader(mode="test")

    def _get_eval_loader(self, mode):
        batch_size = (
            self.args.val_batch_size if mode == "val" else self.args.test_batch_size
        )
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
        )
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == "val":
            dataset = RNNValidDataset(
                self.args, self.train, self.val, self.max_len, self.rng
            )
        elif mode == "test":
            dataset = RNNTestDataset(
                self.args, self.train, self.val, self.test, self.max_len, self.rng
            )
        return dataset


class RNNTrainDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len):
        self.args = args
        self.max_len = max_len
        self.all_seqs = []
        self.all_labels = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            for i in range(1, len(seq)):
                self.all_seqs += [seq[:-i]]
                self.all_labels += [seq[-i]]

        assert len(self.all_seqs) == len(self.all_labels)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index][-self.max_len :]
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)

        return (
            torch.LongTensor(tokens),
            torch.LongTensor([length]),
            torch.LongTensor([self.all_labels[index]]),
        )


class RNNValidDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, max_len, rng):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2answer[u]) > 0]
        self.max_len = max_len
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        tokens = self.u2seq[user][-self.max_len :]
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)

        answer = self.u2answer[user]

        return (
            torch.LongTensor(tokens),
            torch.LongTensor([length]),
            torch.LongTensor(answer),
        )


class RNNTestDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2val, u2answer, max_len, rng, subset_users=None):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2val[u]) > 0 and len(u2answer[u]) > 0]
        self.max_len = max_len
        self.rng = rng

        if subset_users is not None:
            self.users = subset_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        tokens = (self.u2seq[user] + self.u2val[user])[
            -self.max_len :
        ]  # append validation item after train seq
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)
        answer = self.u2answer[user]

        return (
            torch.LongTensor(tokens),
            torch.LongTensor([length]),
            torch.LongTensor(answer),
        )
