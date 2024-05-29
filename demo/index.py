import argparse
import os.path
import pickle

from filelock import FileLock
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import FileIndex, create_in, open_dir


def load_index() -> FileIndex:
    lock = FileLock("indexdir.lock")
    try:
        with lock:
            if not os.path.exists("indexdir"):
                return create_index("dataset_meta.pkl", "indexdir")
        return open_dir("indexdir")
    finally:
        lock.release()
        os.remove("indexdir.lock")


def create_index(dataset_path: str, index_dir: str) -> FileIndex:
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    else:
        print("Index directory already exists.")
        return open_dir(index_dir)

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    print("Creating index...")
    schema = Schema(id=ID(stored=True), song=TEXT(stored=True))
    ix = create_in(index_dir, schema)

    writer = ix.writer()
    for key, song in dataset.items():
        writer.add_document(id=str(key), song=song)
    writer.commit()

    ix.close()
    print("Index created.")

    return ix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create index from dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset file",
        default="demo/dataset_meta.pkl",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        help="Path to the index directory",
        default="demo/indexdir",
    )
    args = parser.parse_args()

    create_index(args.dataset_path, args.index_dir)
