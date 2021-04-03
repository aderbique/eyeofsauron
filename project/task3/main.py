import argparse
from collections import Counter, defaultdict
import math
import pickle
from typing import Any, Dict, Tuple

from minisom import MiniSom
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

# minisom repo, including helpful examples: https://github.com/JustGlowing/minisom

# type alias to make things a bit prettier
Coordinate = Tuple[int, int]

class Model:
    def __init__(self, som: MiniSom, labels_map: Dict[Coordinate, Counter]):
        """ Create a MiniSom model with label data

        :param som: MiniSom instance
        :param labels_map: Label map derived from MiniSom instance
        """
        self.som = som
        self.map = labels_map
        self.labels = {}

        for coord, ctr in self.map.items():
            num_benign = ctr["BENIGN"]
            num_scan = ctr["PortScan"]
            if num_benign > num_scan:
                self.labels[coord] = "BENIGN"
            elif num_scan > num_benign:
                self.labels[coord] = "PortScan"
            else:
                self.labels[coord] = None

    def winner(self, data_point: Any) -> Coordinate:
        """ Determine the coordinate that this data point maps to in the SOM

        :param data_point: Data point to map
        :return: Coordinate data point maps to
        """
        return self.som.winner(data_point)

def load_data(path: str, train_size: float) -> Tuple[DataFrame, DataFrame]:
    """ Process the dataset into training and test data.

    :param path: Path to dataset file
    :param train_size: Amount of dataset used for training data, as a float in the range [0, 1]
    :return: A tuple containing, in order, the training dataset and the test dataset
    """
    names = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
             "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
             "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
             "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
             "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
             "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
             "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
             "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
             "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
             "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
             "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
             "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length 2",
             "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
             "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
             "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
             "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
             "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"]
    data = pd.read_csv(path, header=0, names=names)
    # split dataset between training and testing data,
    # then drop the Label column from the training data (as SOM is unsupervised)
    train, test = train_test_split(data, train_size=train_size)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    return train, test

def train_som(dataset: DataFrame, x_dim: int, y_dim: int, *, verbose: bool = False, **kwargs) -> Model:
    """ Train the SOM on the given dataset.

    :param dataset: Labelled training dataset
    :param x_dim: X dimension of the output SOM
    :param y_dim: Y dimension of the output SOM
    :param verbose: Display verbose output if True
    :param kwargs: Additional parameters passed to MiniSom to control SOM behavior, e.g. sigma or learning_rate
    :return: The trained model
    """
    # remove the label from the dataset to have proper unsupervised learning
    labels = dataset.pop("Label")
    model = MiniSom(x_dim, y_dim, dataset.shape[1], **kwargs)
    model.train(dataset.to_numpy(), dataset.shape[0], verbose=verbose)

    return Model(model, model.labels_map(dataset.to_numpy(), labels))

def test_som(model: Model, dataset: DataFrame) -> dict:
    """ Test the SOM model on the given dataset.

    :param model: The trained model to test
    :param dataset: Labelled dataset of test data
    :return: Dict containing test results
    """
    labels = dataset.pop("Label")

    # positive = detected a port scan, negative = detected benign traffic
    label_map = {
        "BENIGN": "Negative",
        "PortScan": "Positive"
    }

    stats = {
        "TruePositive": 0,
        "FalsePositive": 0,
        "TrueNegative": 0,
        "FalseNegative": 0,
        "Map": defaultdict(Counter)
    }

    for i, row in enumerate(dataset):
        coord = model.winner(row)
        expected_label = model.labels[coord]
        actual_label = labels[i]
        if actual_label == expected_label:
            prefix = "True"
        else:
            prefix = "False"
        key = prefix + label_map[actual_label]
        stats[key] += 1
        stats["Map"][coord][actual_label] += 1
        stats["Map"][coord][key] += 1

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOM to detect port scans")
    parser.add_argument("-d", "--dataset",
                        help="Dataset containing training and testing data",
                        default="datasets/ids/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    parser.add_argument("-s", "--split",
                        help="Percentage of the dataset we will use to train the SOM in range [0, 1]",
                        default=0.5,
                        type=float)
    parser.add_argument("-m", "--model",
                        help="Path to an already-trained SOM to load; no training will be performed")
    parser.add_argument("-o", "--output",
                        help="Output the trained SOM model to the given file for later re-use")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Display verbose output")
    parser.add_argument("-x",
                        type=int,
                        help="X dimension of SOM; automatically chooses an appropriate default if unset")
    parser.add_argument("-y",
                        type=int,
                        help="Y dimension of SOM; automatically chooses an appropriate default if unset")
    args = parser.parse_args()

    # load the dataset
    train_data, test_data = load_data(args.dataset, train_size=args.split)

    # generate our SOM
    if args.model:
        som = pickle.load(args.model)
    else:
        # SOM grid recommended to be approximately 5 * sqrt(N) neurons where N is the number of data rows
        auto_dim = int(math.ceil(math.sqrt(5 * math.sqrt(train_data.shape[0]))))
        x = args.x if args.x else auto_dim
        y = args.y if args.y else auto_dim
        som = train_som(train_data, x, y, verbose=args.verbose)
        if args.output:
            pickle.dump(som, args.output)

    # test our SOM
    results = test_som(som, test_data)

    # output results
    for key, value in results.items():
        if key == "Map":
            continue
        print(key, value)
