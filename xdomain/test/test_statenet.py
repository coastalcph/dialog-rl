from models.statenet import *
import numpy as np
import torch
from util import util
from util.featurize import *
from tqdm import tqdm
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import random
from collections import namedtuple
from pprint import pprint
import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

DIM_INPUT = 400
M = 2
DIM_HIDDEN_LSTM = 128
DIM_HIDDEN_ENC = 128
N_RECEPTORS = 2

Turn = namedtuple("Turn", ["user_utt", "system_act", "system_utt",
                           "x_utt", "x_act", "x_sys", "labels", "belief_state"])
Dialog = namedtuple("Dialog", ["turns"])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fix_s2v(_s2v, dialogs):
    all_slots = set()
    s2v_new = {}
    for d in dialogs:
        d = d.to_dict()
        for t in d['turns']:
            for s, v in t['turn_label']:
                all_slots.add(s)

    for s in all_slots:
        s2v_new[s] = _s2v[s]

    return s2v_new


def train(model, data_tr, data_dv, s2v, args):
    model.run_train(data_tr, data_dv, s2v, args)


def run(args):

    domains = args.pretrain_domains
    strict = args.pretrain_single_domain
    print('Training on domains: ',  domains)
    print('Single-domain dialogues only?', strict)

    data, ontology, vocab, w2v = util.load_dataset(splits=['train','dev'],
                                                   base_path=args.path)

    utt_ftz = UserInputNgramFeaturizer(w2v, n=M)
    sys_ftz = UserInputNgramFeaturizer(w2v, n=M)
    act_ftz = ActionFeaturizer(w2v)

    def featurize_dialogs(_data, _domains, _strict):
        featurized_dialogs = []

        for dg in tqdm(_data):
            featurized_turns = []
            dg = dg.to_dict()

            if len(dg['turns']) > args.max_dialog_length > 0:
                continue

            # # # Check domain constraints # # #
            # if 'all' in domains, don't worry about anything, else
            # check how allowed domains and dialog domain intersect
            if 'all' not in _domains:
                dialog_domains = set(dg['domain'])
                allowed_domains = set(_domains)

                # strictly restricted to some domain(s), check that
                # dialog has no other domains
                if _strict:
                    if not allowed_domains.issuperset(dialog_domains):
                        continue
                # else, check there's at least one valid domain in the dialog
                else:
                    if allowed_domains.intersection(dialog_domains):
                        continue

            for t in dg['turns']:
                utt = t['transcript']
                sys = t['system_transcript']
                act = t['system_acts']
                bst = t['belief_state']
                lbl = [(s, v.lower()) for s, v in t['turn_label']]
                dom = [slot.split("-")[0] for slot, _ in lbl]
                x_utt = utt_ftz.featurize_turn(utt)
                x_act = act_ftz.featurize_turn(act)
                x_sys = sys_ftz.featurize_turn(sys)

                ys = {}
                for slot, val in lbl:
                    ys[slot] = torch.zeros(len(s2v[slot]))
                    idx = s2v[slot].index(val)
                    ys[slot][idx] = 1
                featurized_turns.append(Turn(utt, act, sys,
                                             x_utt, x_act, x_sys,
                                             ys, bst))
            featurized_dialogs.append(Dialog(featurized_turns))

        print('length of featurized dialogs: ', len(featurized_dialogs))
        return featurized_dialogs

    data_dv = data['dev']
    data_tr = data['train']
    s2v = ontology.values

    _data_tr = []
    _data_dv = []

    data_tr = list(data_tr.iter_dialogs())
    data_dv = list(data_dv.iter_dialogs())
    random.shuffle(data_tr)

    for i, d in enumerate(data_tr):
        if i == args.debug_data_amount:
            break
        _data_tr.append(d)

    for i, d in enumerate(data_dv):
        if i == args.debug_data_amount:
            break
        _data_dv.append(d)

    data_dv = _data_dv
    data_tr = _data_tr

    s2v = fix_s2v(s2v, data_tr + data_dv)

    print("Featurizing...")
    data_f_tr = featurize_dialogs(data_tr, domains, strict)
    data_f_dv = featurize_dialogs(data_dv, domains, strict)
    # print(data_tr[0].to_dict()['turns'][0]['system_acts'])

    model = util.load_model(DIM_INPUT * M, DIM_INPUT, DIM_HIDDEN_ENC,
                            N_RECEPTORS, w2v, args)
    if args.resume:
        model.load_best_save(directory=args.resume)

    model = model.to(model.device)
    for name, param in model.named_parameters():
        print(name, param.device)

    print("Training...")
    train(model, data_f_tr, data_f_dv, s2v, args)


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dexp', help='root experiment folder', default='exp')
    parser.add_argument('--model', help='which model to use', default='statenet')
    parser.add_argument('--epochs', help='max epochs to run for', default=50, type=int)
    parser.add_argument('--demb', help='word embedding size', default=400, type=int)
    parser.add_argument('--dhid', help='hidden state size', default=200, type=int)
    parser.add_argument('--batch_size', help='batch size', default=50, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--stop', help='slot to early stop on', default='joint_goal')
    parser.add_argument('--resume', help='save directory to resume from')
    parser.add_argument('-n', '--nick', help='nickname for model', default='default')
    parser.add_argument('--pretrain_reinforce', action='store_true', help='train with RL')
    parser.add_argument('--gamma', help='RL discount', default=0.99, type=float)
    parser.add_argument('--seed', default=42, help='random seed', type=int)
    parser.add_argument('--test', action='store_true', help='run in evaluation only mode')
    parser.add_argument('--gpu', type=int, help='which GPU to use')
    parser.add_argument('--dropout', nargs='*', help='dropout rates', default=['emb=0.2', 'local=0.2', 'global=0.2'])
    parser.add_argument('--pretrain_domains', nargs='+', help='Domains on which to pretrain', default='all')
    parser.add_argument('--finetune_domain', nargs=1, help='Domain on which to finetune')
    parser.add_argument('--pretrain_single_domain', action='store_true', help='Restrict pretraining to single-domain dialogs')
    parser.add_argument('--finetune_single_domain', action='store_true', help='Restrict finetuning to single-domain dialogs')
    parser.add_argument('--eta', help='factor for loss for binary slot filling prediction', default=0.5, type=float)
    parser.add_argument('--path', help='path to data files',
                        default='../data/multiwoz/ann/')
    parser.add_argument('--debug_data_amount', default=-1, type=int)
    parser.add_argument('--max_dialog_length', default=-1, type=int)
    parser.add_argument('--elmo', action='store_true', help="If True, use ELMo for embedding")
    parser.add_argument('--elmo_weights', default='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
    parser.add_argument('--elmo_options',
                        default='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json')

    args = parser.parse_args()
    args.dout = os.path.join(args.dexp, args.model, args.nick)
    args.dropout = {d.split('=')[0]: float(d.split('=')[1]) for d in args.dropout}
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    return args


if __name__ == '__main__':
    args = get_args()
    run(args)

# hidden = torch.randn(1, 1, DIM_HIDDEN_LSTM)
# sn.forward_turn(u, a, s2v, hidden)
