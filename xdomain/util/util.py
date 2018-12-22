import json
import logging
import os
import torch
from pprint import pformat, pprint
from importlib import import_module
from vocab import Vocab
from util.dataset import Dataset, Ontology
from util.preprocess_data import dann
from tqdm import tqdm
from models.statenet import *
import argparse


def load_dataset(splits=('train', 'dev', 'test'), domains='all', strict=False,
                 base_path=None, elmo=False):
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


def generate_dataset_elmo(elmo, splits=('train', 'dev', 'test'), domains='all', strict=False,
                      base_path=None):
    """
    """
    path = base_path if base_path else ''
    with open(os.path.join(path, 'ontology.json')) as f:
        ontology = Ontology.from_dict(json.load(f))

    dataset = {}
    for split in splits:
        with open(os.path.join(path, '{}.json'.format(split))) as f:
            logging.warn('loading split {}'.format(split))
            dataset[split] = Dataset.from_dict(json.load(f))
            dataset[split] = dataset[split].to_elmo(elmo)

    logging.info('dataset sizes: {}'.format(pformat({k: len(v) for k, v in dataset.items()})))
    return dataset, ontology


def get_models():
    return [m.replace('.py', '') for m in os.listdir('models') if not m.startswith('_') and m != 'model']


def load_model(*args, **kwargs):
    StateNet = import_module("models.statenet").StateNet
    model = StateNet(*args, **kwargs)
    logging.info('loaded model.')
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


def delexicalize(s2v):
    allowed_slots = [
        "attraction-area",
        "attraction-name",
        "attraction-type",
        "hotel-area",
        "hotel-day",
        "hotel-internet",
        "hotel-name",
        "hotel-parking",
        "hotel-people",
        "hotel-pricerange",
        "hotel-stars",
        "hotel-stay",
        "hotel-type",
        "restaurant-area",
        "restaurant-day",
        "restaurant-food",
        "restaurant-name",
        "restaurant-people",
        "restaurant-pricerange",
        "restaurant-time",
        "taxi-arriveBy",
        "taxi-leaveAt",
        "taxi-type",
        "train-arriveBy",
        "train-day",
        "train-leaveAt",
        "train-people"]
    out = {}
    for s, v in s2v.items():
        if s in allowed_slots:
            out[s] = v
        else:
            out[s] = ["<true>"]
    return out


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fix_s2v(_s2v, dialogs, splits=('train', 'dev', 'test')):
    all_slots = set()
    s2v_new = {}
    for s in splits:
        for d in dialogs[s]:
            for t in d['turns']:
                for s, v in t['turn_label']:
                    all_slots.add(s)

    for s in all_slots:
        s2v_new[s] = _s2v[s]

    return s2v_new


def featurize_s2v(s2v_dict, slot_featurizer, value_featurizer):
    out = {}
    print("Featurizing slots and values...")
    for s, vs in tqdm(s2v_dict.items()):
        # remove domain prefix ('restaurant-priceRange' -> 'priceRange')
        domain, slot = s.split("-", 1)
        # split at uppercase to get vectors ('priceRange' -> ['price', 'range'])
        words = split_on_uppercase(slot, keep_contiguous=True)
        slot_emb = slot_featurizer.featurize_turn(words)
        v_embs = value_featurizer.featurize_batch([v.split() for v in vs])
        vs_out = [Value(v, v_embs[idx], idx)
                  for idx, v in enumerate(vs)]
        out[s] = Slot(domain, slot_emb, vs_out)
    return out


def s2v_to_device(s2v, device):
    out = {}
    for s, vs in s2v.items():
        dom, slot_emb, vs_out = s2v[s]
        vs_out = [Value(v, v_emb.to(device), idx) for v, v_emb, idx in vs_out]
        out[s] = Slot(dom, slot_emb.to(device), vs_out)
    return out


def filter_dialogs(data, domains, strict, max_dialogs, max_turns_per_dialog):
    out = []
    for dg in data:
        if len(dg['turns']) > max_turns_per_dialog > 0:
            continue

        # # # Check domain constraints # # #
        # if 'all' in domains, don't worry about anything, else
        # check how allowed domains and dialog domain intersect
        if 'all' not in domains:
            dialog_domains = set(dg['domain'])
            allowed_domains = set(domains)

            # strictly restricted to some domain(s), check that
            # dialog has no other domains
            if strict:
                if not dialog_domains.issubset(allowed_domains):
                    continue
            # else, check there's at least one valid domain in the dialog
            else:
                if not allowed_domains.intersection(dialog_domains):
                    continue
        out.append(dg)

    if max_dialogs > 0:
        out = out[:max_dialogs]
    return out


def get_device(device_id):
    if device_id is not None and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu = device_id % num_gpus
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')


def make_batches(dialogs, batch_size):
    dialogs = list(dialogs)
    slices = [(i*batch_size, (i+1)*batch_size)
              for i in range(len(dialogs)//batch_size + 1)]
    for beg, end in slices:
        if beg < len(dialogs)-1:
            yield dialogs[beg:end]


