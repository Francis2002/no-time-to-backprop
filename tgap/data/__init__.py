import pandas as pd
import numpy as np
from collections import namedtuple
from functools import cached_property
import math
from os import path

DEFAULT_DATA_FOLDER = "data"
base_header = ['source', 'destination', 'timestamp', 'label']
feature_header_template = 'feature_{}'

def load_dataframe(dataset="wikipedia", data_folder=DEFAULT_DATA_FOLDER, features=True, preview=False):
    df = pd.read_csv(
        path.join(data_folder, f"{dataset}.csv"), 
        skiprows=1, 
        header=None, 
        names=None if features else base_header, 
        usecols=None if features else range(4),
        nrows=100 if preview else None,
        ) 
    if features:
        df.columns = base_header + [feature_header_template.format(i) for i in range(df.shape[1] - 4)]

    if np.any(np.diff(df.timestamp.values) < 0): # out of temporal order
        df.sort_values('timestamp', inplace=True).reset_index(drop=True, inplace=True)
    return df


GraphData = namedtuple("GraphData", ["sources", "destinations", "timestamps", "labels", "features"])

class Datasets:
    def __init__(
            self,
            dataset,
            data_folder=DEFAULT_DATA_FOLDER,
            features=True, 
            val_fraction=0.15, 
            test_fraction=0.15, 
            reserve_fraction=0.1, 
            bipartite=True, 
            seed=None
        ):
        """Graph dataset and split information

        Args:
            dataset (str, optional): The dataset to load. 
            features (bool, optional): Whether to load the dataset features. Defaults to True.
            val_fraction (float, optional): The fraction of the dataset reserved for validation. Defaults to 0.15.
            test_fraction (float, optional): The fraction of the dataset reserved for testing. Defaults to 0.15.
            reserve_fraction (float, optional): The fraction of total nodes to be reserved and removed from the training set. Defaults to 0.1.
            bipartite (bool, optional): Whether the graph is bipartite. Defaults to True.
            seed (int | Generator, optional): Seed or random number generator. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            Dataset: The constructed dataset instance
        """
        df = load_dataframe(dataset=dataset, data_folder=data_folder, features=features)
        val_index, test_index = split_indices(len(df), val_fraction, test_fraction)

        self.full_sources = df.iloc[:,0].values
        self.full_destinations = df.iloc[:,1].values

        if reserve_fraction > 0:
            train_mask, self.reserved_mask = reserve_test_nodes(
                self.full_sources, 
                self.full_destinations, 
                val_index,
                fraction=reserve_fraction, bipartite=bipartite, seed=seed)
            df = df.iloc[train_mask]
            
            num_removed_edges = train_mask.size - self.reserved_mask.size
            val_index -= num_removed_edges
            test_index -= num_removed_edges
        else:
            self.reserved_mask = slice(0)

        self.sources = df.iloc[:,0].values
        self.destinations = df.iloc[:,1].values
        self.timestamps = df.iloc[:,2].values
        self.labels = df.iloc[:,3].values
        self.features = df.iloc[:,4:].values

        self.val_index = val_index
        self.test_index = test_index

        self.num_sources = self.full_sources.max() + 1
        self.num_destinations = self.full_destinations.max() + 1

    @cached_property
    def data(self):
        return GraphData(self.sources, self.destinations, self.timestamps, self.labels, self.features)
    
    @cached_property
    def train(self):
        mask = slice(self.val_index)
        return GraphData(
            self.sources[mask], 
            self.destinations[mask], 
            self.timestamps[mask], 
            self.labels[mask], 
            self.features[mask]
            )
    
    @cached_property
    def val(self):
        mask = slice(self.val_index, self.test_index)
        return GraphData(
            self.sources[mask], 
            self.destinations[mask], 
            self.timestamps[mask], 
            self.labels[mask], 
            self.features[mask]
            )
    
    @cached_property
    def test(self):
        mask = slice(self.test_index, None)
        return GraphData(
            self.sources[mask], 
            self.destinations[mask], 
            self.timestamps[mask], 
            self.labels[mask], 
            self.features[mask]
            )
    

def split_indices(size, val_fraction=0.15, test_fraction=0.15):
    train_fraction = 1 - val_fraction - test_fraction
    trainval_fraction = 1 - test_fraction
    return math.ceil(train_fraction*size), math.ceil(trainval_fraction*size)


def reserve_test_nodes(sources, destinations, val_index, fraction=0.1, bipartite=True, seed=None):
    """Reserves a random sample of nodes for transductive evaluation.

    Args:
        sources (array): An array of sources nodes.
        destinations (array): An array of destination nodes.
        val_index (int): The index where the validation set starts.
        test_index (int): The index where the training set starts
        fraction (float, optional): The fraction of nodes to reserve. Defaults to 0.1.
        bipartite (bool, optional): Whether the graph is bipartite. Defaults to True.
        seed (int | Generator, optional): Seed or random number generator. Defaults to None.

    Raises:
        NotImplementedError: Non bipartite node reservation not implemented yet

    Returns:
        train_mask: A mask that removes reserved instances from the training set.
        reserved_mask: A mask that applied to the reduced dataset selects the edges containing at least 1 reserved node.
    """
    prng = np.random.default_rng(seed)
    if bipartite:
        num_sources = sources.max() + 1
        num_destinations = destinations.max() + 1
        num_nodes = num_sources + num_destinations

        test_sources = np.unique(sources[val_index:])
        test_destinations = np.unique(destinations[val_index:])
        test_pool = test_sources.size + test_destinations.size

        num_reserved_nodes = int(fraction * num_nodes)
        reserved_nodes = np.sort(prng.choice(test_pool, num_reserved_nodes, replace=False))
        src_split = np.searchsorted(reserved_nodes, test_sources.size)

        reserved_sources = test_sources[reserved_nodes[:src_split]]
        reserved_destinations = test_destinations[reserved_nodes[src_split:] - test_sources.size]

        is_reserved = np.logical_or(np.isin(sources, reserved_sources), np.isin(destinations, reserved_destinations))
    else:
        raise NotImplementedError

    train_mask = ~is_reserved
    train_mask[val_index:] = True

    return train_mask, is_reserved[train_mask]

