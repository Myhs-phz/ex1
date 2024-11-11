import datasets


def build_datasets(args):

    if args.dataset == 'glue_cola':
        dataset = datasets.load_dataset('glue','cola')

    return dataset