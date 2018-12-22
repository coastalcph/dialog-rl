import json
from tqdm import tqdm
from util import util
from pprint import pprint
from allennlp.commands.elmo import ElmoEmbedder
from util.featurize import ElmoFeaturizer
from collections import namedtuple

Elmo = namedtuple('Elmo', ['utterance_feat', 'sys_act_feat'])
DELEX = True

def main():
    # Init ELMO model
    elmo_emb = ElmoEmbedder(weight_file='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',
                            options_file='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json')

    # "Warm up" ELMo embedder (https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)
    warmup_data, _, _, _ = util.load_dataset(splits=['train'], base_path='../data/multiwoz/ann/')
    warmup_data = [dg.to_dict() for dg in warmup_data['train'].iter_dialogs()][:500]

    print('Warming up ELMo embedder on train dialogs')
    for d in tqdm(warmup_data):
        utts = []
        for t in d['turns']:
            utts.append(t['transcript'])
        _ = elmo_emb.batch_to_embeddings(utts)

    base_path = '../data/multiwoz/ann/'
    #splits = ['train', 'test', 'dev']
    splits = ['dev']

    # Load dialogs
    print('Creating elmo embeddings for annotated data')
    utterance_featurizer = ElmoFeaturizer(elmo_emb, 'utterance')
    sys_act_featurizer = ElmoFeaturizer(elmo_emb, 'act')

    elmo = Elmo(utterance_featurizer, sys_act_featurizer)

    dia_data, ontology = util.generate_dataset_elmo(elmo, splits=splits, base_path=base_path)

    # Save dataset
    for split in splits:
        json.dump(dia_data[split], open('{}_elmo.json'.format(base_path + split)))

    ## Create s2v embedding
    s2v = ontology.values
    if DELEX:
        s2v = util.delexicalize(s2v)
    s2v = util.fix_s2v(s2v, dia_data, splits=splits)

    slot_featurizer = ElmoFeaturizer(elmo_emb, "slot")
    value_featurizer = ElmoFeaturizer(elmo_emb, "value")

    s2v = util.featurize_s2v(s2v, slot_featurizer, value_featurizer)

    # Save s2v
    json.dump(s2v, open('{}s2v_elmo.json'.format(base_path)))

if __name__ == '__main__':
    main()
