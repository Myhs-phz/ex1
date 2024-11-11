import deepspeed
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from data import *
from options import parse_args

params = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "cpu_offload": False
    }
}


def main():
    args = parse_args()
    dataset = build_datasets(args)
    breakpoint()


if __name__ == '__main__':
    main()


