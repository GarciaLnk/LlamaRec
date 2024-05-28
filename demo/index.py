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
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    schema = Schema(id=ID(stored=True), song=TEXT(stored=True))

    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    else:
        print("Index directory already exists.")
        return open_dir(index_dir)

    print("Creating index...")
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
        "dataset_path", type=str, help="Path to the dataset file", nargs="?"
    )
    parser.add_argument(
        "index_dir", type=str, help="Path to the index directory", nargs="?"
    )
    args = parser.parse_args()

    if not args.dataset_path:
        args.dataset_path = "demo/dataset_meta.pkl"

    if not args.index_dir:
        args.index_dir = "demo/indexdir"

    create_index(args.dataset_path, args.index_dir)
