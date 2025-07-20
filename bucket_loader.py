import os
import glob
import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset

class LengthBucketedDataset(Dataset):
    def __init__(self, files_or_folder, bucket_size=5):
        if isinstance(files_or_folder, str):
            self.files = sorted(glob.glob(os.path.join(files_or_folder, '*.pt')))
        else:
            self.files = sorted(files_or_folder)
        if not self.files:
            raise ValueError("No .pt files found")

        self.bucket_size = bucket_size
        self.all_buckets = self._make_buckets()

    def _make_buckets(self):
        lengths = []
        for file in self.files:
            g = torch.load(file)
            lengths.append(g['seq'].shape[0])
        sorted_pairs = sorted(zip(lengths, self.files))

        buckets = defaultdict(list)
        for length, path in sorted_pairs:
            bucket_len = (length // self.bucket_size) * self.bucket_size
            buckets[bucket_len].append(path)
        return list(buckets.items())  # [(bucket_len, [paths...]), ...]
        
    def __len__(self):
        return len(self.buckets)

    def __getitem__(self, idx):
        bucket_files = self.buckets[idx]
        data_list = []
        for file in bucket_files:
            g = torch.load(file)
            g['z_t'] = torch.zeros_like(g['node_s'])
            data_list.append(g)
        return data_list



class BucketedDataLoader:
    def __init__(self, dataset, batch_size=16, shuffle=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        self._build_buckets()

    def _build_buckets(self):
        self.buckets = []
        for bucket_len, paths in self.dataset.all_buckets:
            if self.shuffle:
                random.shuffle(paths)
            for i in range(0, len(paths), self.batch_size):
                batch_paths = paths[i:i + self.batch_size]
                self.buckets.append((bucket_len, batch_paths))

    def __len__(self):
        return len(self.buckets)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.buckets)

        for bucket_len, batch_paths in self.buckets:
            batch_graphs = []
            lengths = []
            for path in batch_paths:
                g = torch.load(path)
                g['z_t'] = torch.zeros_like(g['node_s'])
                batch_graphs.append(g)
                lengths.append(g['seq'].shape[0])

            batch = self.collate_fn(batch_graphs)
            min_len, max_len = min(lengths), max(lengths)
            yield batch


def plot_bucket_distribution(bucket_dict, bucket_size=5):
    import matplotlib.pyplot as plt

    bucket_keys = sorted(bucket_dict.keys())
    counts = [len(bucket_dict[k]) for k in bucket_keys]
    labels = [f"{k}-{k + bucket_size - 1}" for k in bucket_keys]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts, color='steelblue')
    plt.title(f"Sequence Count by Length Buckets (bucket size = {bucket_size})")
    plt.xlabel("Sequence Length Ranges")
    plt.ylabel("Number of Sequences")
    plt.xticks(rotation=45)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()
