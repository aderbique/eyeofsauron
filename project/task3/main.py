import argparse
import math
import pickle
import sys
from typing import Tuple

from minisom import MiniSom
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

# minisom repo, including helpful examples: https://github.com/JustGlowing/minisom

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
             "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length",
             "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
             "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
             "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
             "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
             "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"]
    data = pd.read_csv(path, header=0, names=names)
    # split dataset between training and testing data,
    # then drop the Label column from the training data (as SOM is unsupervised)
    train, test = train_test_split(data, train_size=train_size)
    del train["Label"]

    return train, test

def train_som(dataset: DataFrame, x_dim: int, y_dim: int, **kwargs) -> MiniSom:
    """ Train the SOM on the given dataset.

    :param dataset: Training dataset
    :param x_dim: X dimension of the output SOM
    :param y_dim: Y dimension of the output SOM
    :param kwargs: Additional parameters passed to MiniSom to control SOM behavior, e.g. sigma or learning_rate
    :return: The trained SOM
    """
    model = MiniSom(x_dim, y_dim, dataset.shape[1], **kwargs)
    model.train(dataset, dataset.shape[0])
    return model

def test_som(model: MiniSom, dataset: DataFrame) -> dict:
    """ Test the SOM model on the given dataset.

    :param dataset: Labelled dataset of test data
    :return: Dict containing test results
    """
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOM to detect port scans")
    parser.add_argument("-d", "--dataset",
                        help="Dataset containing training and testing data",
                        default="datasets/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    parser.add_argument("-s", "--split",
                        help="Percentage of the dataset we will use to train the SOM in range [0, 1]",
                        default=0.5,
                        type=float)
    parser.add_argument("-m", "--model",
                        help="Path to an already-trained SOM to load; no training will be performed")
    parser.add_argument("-o", "--output",
                        help="Output the trained SOM model to the given file for later re-use")
    parser.add_argument("-x",
                        type=int,
                        help="X dimension of SOM; automatically chooses an appropriate default if unset")
    parser.add_argument("-y",
                        type=int,
                        help="Y dimension of SOM; automatically chooses an appropriate default if unset")
    args = parser.parse_args(sys.argv)

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
        som = train_som(train_data, x, y)
