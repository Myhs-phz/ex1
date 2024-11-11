import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="the config of this Training Frame")

    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()
    return args

