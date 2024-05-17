import os
import pickle
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .base import AbstractDataset
from .utils import download, unzip

tqdm.pandas()


class MusicDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return "music"

    @classmethod
    def url(cls):
        return "https://archive.org/download/lastfm1k/lastfm1k.zip"

    @classmethod
    def zip_file_content_is_folder(cls):
        return False

    @classmethod
    def all_raw_file_names(cls):
        return [
            "lastfm1k.item",
            "lastfm1kspotify.inter",
        ]

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and all(
            folder_path.joinpath(filename).is_file()
            for filename in self.all_raw_file_names()
        ):
            print("Raw data already exists. Skip downloading")
            return

        print("Raw file doesn't exist. Downloading...")
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath("file.zip")
        tmpfolder = tmproot.joinpath("folder")
        download(self.url(), tmpzip)
        unzip(tmpzip, tmpfolder)
        if self.zip_file_content_is_folder():
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
        print()

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print("Already preprocessed. Skip preprocessing")
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        meta_raw = self.load_meta_dict()
        df = df[df["sid"].isin(meta_raw)]  # filter items without meta info
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}
        dataset = {
            "train": train,
            "val": val,
            "test": test,
            "meta": meta,
            "umap": umap,
            "smap": smap,
        }
        with dataset_path.open("wb") as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath("lastfm1kspotify.inter")
        df = pd.read_csv(file_path, sep="\t")
        df.columns = ["user", "timestamp", "sid", "uid"]
        return df

    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath("lastfm1k.item")
        df = pd.read_csv(file_path)
        meta_dict = {}
        for row in df.itertuples():
            track = str(row[1]).strip()
            artist = row[2]

            meta_dict[row[-1]] = track + ", " + artist
        return meta_dict
