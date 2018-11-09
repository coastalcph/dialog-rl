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

    logging.info('dataset sizes: {}'.format(pformat({k: len(v) for k, v in dataset.items()})))
    return dataset, ontology, vocab, w2v


def get_models():
    return [m.replace('.py', '') for m in os.listdir('models') if not m.startswith('_') and m != 'model']


def load_model(model, *args, **kwargs):
    Model = import_module('models.{}'.format(model)).Model
    model = Model(*args, **kwargs)
    logging.info('loaded model {}'.format(Model))
    return model


