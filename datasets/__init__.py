from .beauty import BeautyDataset
from .games import GamesDataset
from .ml_100k import ML100KDataset
from .music import MusicDataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    MusicDataset.code(): MusicDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
