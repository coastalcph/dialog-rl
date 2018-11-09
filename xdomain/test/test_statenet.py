from models.statenet import *
import numpy as np
import torch
from util import util
from util.featurize import *

DIM = 100
DIM_HIDDEN_LSTM = 256
DIM_HIDDEN_ENC = 128
N_RECEPTORS = 4


fs = {'user': Featurizer(), 'actions': Featurizer(), 'slots': Featurizer()}

s2v = {'tel': ['1', '2', '3'], 'loc': ['a', 'b']}

data, ontology, vocab, w2v = util.load_dataset(splits=['dev'],
    base_path="/home/joachim/projects/dialog-rl/data/multiwoz/ann")

utt_ftz = UserInputFeaturizer(w2v)
sys_ftz = UserInputFeaturizer(w2v)
act_ftz = ActionFeaturizer(w2v)

data_dv = data['dev']
data_tr = data_dv


featurized_dialogs = []
for dg in data_tr.iter_dialogs():
    featurized_turns = []
    dg = dg.to_dict()
    for t in dg['turns']:
        utt = t['transcript']
        sys = t['system_transcript']
        act = t['system_acts']
        lbl = t['turn_label']
        dom = [slot.split("-")[0] for slot, value in lbl]
        x_utt = utt_ftz.featurize_turn(utt)
        x_act = act_ftz.featurize_turn(act)
        s2v = ontology.values
        ys = {}
        for slot, val in lbl:
            ys[slot] = torch.zeros(len(s2v[slot]))
            idx = s2v[slot].index(val)
            ys[slot][idx] = 1
        featurized_turns.append((x_utt, x_act, ys))
    featurized_dialogs.append(featurized_turns)

print(featurized_dialogs[0])


u = [torch.randn(1, DIM)] * 5
a = torch.randn(1, DIM)
s = torch.randn(1, DIM)

sn = StateNet(featurizers, DIM_HIDDEN_ENC, N_RECEPTORS)

hidden = torch.randn(1, 1, DIM_HIDDEN_LSTM)
sn.forward_turn(u, a, s2v, hidden)

