import json
import logging
import os
from pprint import pformat
from importlib import import_module
from vocab import Vocab
from util.dataset import Dataset, Ontology
from util.preprocess_data import dann


def load_dataset(splits=('train', 'dev', 'test'), domains='all', strict=False,
                 base_path=None):
    """

    :param splits:
    :param domains: filter for domains (if 'all', use all available)
    :param strict: if True, select only dialogs that contain only a single domain
    :return:
    """
    path = base_path if base_path else dann
    # TODO implement filtering with `domains` and `strict`
    with open(os.path.join(path, 'ontology.json')) as f:
        ontology = Ontology.from_dict(json.load(f))
    with open(os.path.join(path, 'vocab.json')) as f:
        vocab = Vocab.from_dict(json.load(f))
    with open(os.path.join(path, 'emb.json')) as f:
        E = json.load(f)

    w2v = {w: E[i] for i, w in enumerate(vocab.to_dict()['index2word'])}

    dataset = {}
    for split in splits:
        with open(os.path.join(path, '{}.json'.format(split))) as f:
            logging.warn('loading split {}'.format(split))
            dataset[split] = Dataset.from_dict(json.load(f))
            # dataset[split] = Dataset.from_dict(json.load(f), domains, strict)

    logging.info('dataset sizes: {}'.format(pformat({k: len(v) for k, v in dataset.items()})))
    return dataset, ontology, vocab, w2v


def get_models():
    return [m.replace('.py', '') for m in os.listdir('models') if not m.startswith('_') and m != 'model']


def load_model(model, *args, **kwargs):
    Model = import_module('models.{}'.format(model)).Model
    model = Model(*args, **kwargs)
    logging.info('loaded model {}'.format(Model))
    return model


def split_on_uppercase(s, keep_contiguous=False):
    """
    From https://stackoverflow.com/questions/2277352/

    Args:
        s (str): string
        keep_contiguous (bool): flag to indicate we want to
                                keep contiguous uppercase chars together

    Returns:

    """

    string_length = len(s)
    is_lower_around = (lambda: s[i-1].islower() or
                       string_length > (i + 1) and s[i + 1].islower())

    start = 0
    parts = []
    for i in range(1, string_length):
        if s[i].isupper() and (not keep_contiguous or is_lower_around()):
            parts.append(s[start: i])
            start = i
    parts.append(s[start:])

    return parts