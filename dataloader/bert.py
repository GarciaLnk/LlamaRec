import random

import numpy as np
import torch
import torch.utils.data as data_utils


def worker_init_fn(worker_id):
    random.seed(int(np.random.get_state()[1][0]) + worker_id)
    np.random.seed(int(np.random.get_state()[1][0]) + worker_id)


class BERTDataloader:
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
        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.sliding_size = args.sliding_window_size
        self.CLOZE_MASK_TOKEN = self.item_count + 1

    @classmethod
    def code(cls):
        return "bert"

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
        dataset = BERTTrainDataset(
            self.args,
            self.train,
            self.max_len,
            self.mask_prob,
            self.max_predictions,
            self.sliding_size,
            self.CLOZE_MASK_TOKEN,
            self.rng,
        )
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
            dataset = BERTValidDataset(
                self.args,
                self.train,
                self.val,
                self.max_len,
                self.CLOZE_MASK_TOKEN,
                self.rng,
            )
        elif mode == "test":
            dataset = BERTTestDataset(
                self.args,
                self.train,
                self.val,
                self.test,
                self.max_len,
                self.CLOZE_MASK_TOKEN,
                self.rng,
            )
        return dataset


class BERTTrainDataset(data_utils.Dataset):
    def __init__(
        self,
        args,
        u2seq,
        max_len,
        mask_prob,
        max_predictions,
        sliding_size,
        mask_token,
        rng,
    ):
        self.args = args
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.max_predictions = max_predictions
        self.sliding_step = int(sliding_size * max_len)
        self.mask_token = mask_token
        self.num_items = args.num_items
        self.rng = rng

        assert self.sliding_step > 0
        self.all_seqs = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [
                    seq[i : i + max_len] for i in start_idx
                ]

    def __len__(self):
        return len(self.all_seqs)
        # return len(self.users)

    def __getitem__(self, index):
        # user = self.users[index]
        # seq = self._getseq(user)
        seq = self.all_seqs[index]

        tokens = []
        labels = []
        covered_items = set()
        for i in range(len(seq)):
            s = seq[i]
            if (len(covered_items) >= self.max_predictions) or (s in covered_items):
                tokens.append(s)
                labels.append(0)
                continue

            temp_mask_prob = self.mask_prob
            if i == (len(seq) - 1):
                temp_mask_prob += 0.1 * (1 - self.mask_prob)

            prob = self.rng.random()
            if prob < temp_mask_prob:
                covered_items.add(s)
                prob /= temp_mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]


class BERTValidDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, max_len, mask_token, rng):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2answer[u]) > 0]
        self.max_len = max_len
        self.mask_token = mask_token
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len :]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return (torch.LongTensor(seq), torch.LongTensor(answer))


class BERTTestDataset(data_utils.Dataset):
    def __init__(
        self, args, u2seq, u2val, u2answer, max_len, mask_token, rng, subset_users=None
    ):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2val[u]) > 0 and len(u2answer[u]) > 0]
        self.max_len = max_len
        self.mask_token = mask_token
        self.rng = rng

        if subset_users is not None:
            self.users = subset_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]
        answer = self.u2answer[user]

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len :]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return (torch.LongTensor(seq), torch.LongTensor(answer))
