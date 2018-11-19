from models.statenet import *
import numpy as np
import torch
from util import util
from util.featurize import *
from tqdm import tqdm
from pprint import pprint
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

DIM_INPUT = 400
M = 3
DIM_HIDDEN_LSTM = 256
DIM_HIDDEN_ENC = 128
N_RECEPTORS = 4

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', help='absolute path to dialog-rl project', default='/home/joachim/projects/')
args = parser.parse_args()

base = args.path

data, ontology, vocab, w2v = util.load_dataset(splits=['dev'],
    base_path=base+'dialog-rl/data/multiwoz/ann')

#utt_ftz = UserInputFeaturizer(w2v, n=M)
#sys_ftz = UserInputFeaturizer(w2v, n=M)
utt_ftz = UserInputNgramFeaturizer(w2v, n=M)
sys_ftz = UserInputNgramFeaturizer(w2v, n=M)
act_ftz = ActionFeaturizer(w2v)
# slt_ftz = SlotFeaturizer(w2v)
# val_ftz = SlotFeaturizer(w2v)

data_dv = data['dev']
data_tr = data_dv
s2v = ontology.values

_data_tr = []
for i, d in enumerate(data_tr.iter_dialogs()):
    if i == 20:
        break
    _data_tr.append(d)

data_tr = _data_tr


def fix_s2v(_s2v, dialogs):
    all_slots = set()
    s2v_new = {}
    for d in dialogs:
        d = d.to_dict()
        for t in d['turns']:
            for s, v in t['turn_label']:
                all_slots.add(s)

    for s in all_slots:
        s2v_new[s] = _s2v[s.lower()]

    return s2v_new


s2v = fix_s2v(s2v, data_tr)


def featurize_dialogs(data):
    featurized_dialogs = []
    for dg in tqdm(data):
        featurized_turns = []
        dg = dg.to_dict()
        for t in dg['turns']:
            utt = t['transcript']
            sys = t['system_transcript']
            act = t['system_acts']
            lbl = [(s, v.lower()) for s, v in t['turn_label']]
            dom = [slot.split("-")[0] for slot, _ in lbl]
            x_utt = utt_ftz.featurize_turn(utt)
            x_act = act_ftz.featurize_turn(act)
            x_sys = sys_ftz.featurize_turn(sys)
            if len(x_utt) == 800:
                print(utt)
            ys = {}
            for slot, val in lbl:
                # x_slt = slt_ftz.featurize_slot(slot)
                # x_val = val_ftz.featurize_slot(val)
                ys[slot] = torch.zeros(len(s2v[slot]))
                idx = s2v[slot].index(val)
                ys[slot][idx] = 1
            featurized_turns.append((x_utt, x_act, x_sys, ys))
        featurized_dialogs.append(featurized_turns)
    return featurized_dialogs


def train(model, data):
    for dialog in tqdm(data):
        predictions, loss = model.forward(dialog, s2v)
        for slot, argmax in predictions.items():
            # print(slot, "-->", s2v[slot][argmax])
            pass



print("Featurizing...")
data_f_tr = featurize_dialogs(data_tr)
# print(data_tr[0].to_dict()['turns'][0]['system_acts'])

print("Initializing network.")
sn = StateNet(DIM_INPUT * M, DIM_INPUT, DIM_HIDDEN_ENC, N_RECEPTORS, w2v)

print("Training...")
train(sn, data_f_tr)


# hidden = torch.randn(1, 1, DIM_HIDDEN_LSTM)
# sn.forward_turn(u, a, s2v, hidden)

