from models.statenet import *
from util import util
from util.featurize import *
from tqdm import tqdm
import random
import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from allennlp.commands.elmo import ElmoEmbedder


# DIM_INPUT = 400
DIM_HIDDEN_LSTM = 128
DIM_HIDDEN_ENC = 128


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


def fix_s2v(_s2v, dialogs):
    all_slots = set()
    s2v_new = {}
    for d in dialogs:
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
        words = util.split_on_uppercase(slot, keep_contiguous=True)
        slot_emb = slot_featurizer.featurize_turn(words)
        v_embs = value_featurizer.featurize_batch([v.split() for v in vs])
        vs_out = [Value(v, v_embs[idx], idx)
                  for idx, v in enumerate(vs)]
        out[s] = Slot(domain, slot_emb, vs_out)
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
                if not allowed_domains.issuperset(dialog_domains):
                    continue
            # else, check there's at least one valid domain in the dialog
            else:
                if not allowed_domains.intersection(dialog_domains):
                    continue
        out.append(dg)

    if max_dialogs > 0:
        out = out[:max_dialogs]
    return out


def featurize_dialogs(_data, _domains, _strict, s2v, w2v, args):
    featurized_dialogs = []

    def get_value_index(_values, _val):
        for _idx, candidate in enumerate(_values):
            if candidate.value == _val:
                return _idx
        return -1

    if args.elmo:
        elmo = ElmoEmbedder(weight_file=args.elmo_weights,
                            options_file=args.elmo_options)
        utt_ftz = ElmoFeaturizer(elmo, "utterance")
        sys_ftz = ElmoFeaturizer(elmo, "utterance")
        act_ftz = ElmoFeaturizer(elmo, "act")
    else:
        utt_ftz = UserInputNgramFeaturizer(w2v, n=args.M)
        sys_ftz = UserInputNgramFeaturizer(w2v, n=args.M)
        act_ftz = ActionFeaturizer(w2v)

    for dg in tqdm(_data):
        featurized_turns = []

        all_user_utts = []
        all_system_acts = []
        all_system_utts = []
        all_lbls = []
        all_ys = []
        all_bsts = []

        for t in dg['turns']:
            utt = t['transcript']
            sys = t['system_transcript']
            act = t['system_acts']
            bst = t['belief_state']
            lbls = {}
            for s, v in t['turn_label']:
                v = v.lower()
                if v in [_v.value for _v in s2v[s].values]:
                    lbls[s] = v
                else:
                    lbls[s] = "<true>"

            ys = {}
            for slot, val in lbls.items():
                values = s2v[slot].values
                ys[slot] = torch.zeros(len(values))
                idx = get_value_index(values, val)
                ys[slot][idx] = 1

            all_user_utts.append(utt)
            all_system_acts.append(act)
            all_system_utts.append(sys.split())
            all_ys.append(ys)
            all_lbls.append(lbls)
            all_bsts.append(bst)

        all_x_utt = utt_ftz.featurize_batch(all_user_utts)
        all_x_act = act_ftz.featurize_batch(all_system_acts)
        all_x_sys = sys_ftz.featurize_batch(all_system_utts)

        for i in range(len(dg['turns'])):
            featurized_turns.append(Turn(
                all_user_utts[i], all_system_acts[i], all_system_utts[i],
                all_x_utt[i], all_x_act[i], all_x_sys[i],
                all_ys[i], all_lbls[i], all_bsts[i]))

        featurized_dialogs.append(Dialog(featurized_turns))

    return featurized_dialogs


def run(args):
    print(args)
    domains = args.train_domains
    strict = args.train_strict
    print('Training on domains: ',  domains)
    print('Single-domain dialogues only?', strict)

    data, ontology, vocab, w2v = util.load_dataset(splits=['train', 'dev'],
                                                   base_path=args.path)

    data_dv = data['dev']
    data_tr = data['train']
    s2v = ontology.values
    if args.delexicalize_labels:
        s2v = delexicalize(s2v)

    data_tr = [dg.to_dict() for dg in data_tr.iter_dialogs()]
    data_dv = [dg.to_dict() for dg in data_dv.iter_dialogs()]
    random.shuffle(data_tr)

    data_tr = filter_dialogs(data_tr, domains, strict, args.max_train_dialogs,
                             args.max_dialog_length)
    data_dv = filter_dialogs(data_dv, domains, strict, args.max_dev_dialogs,
                             args.max_dialog_length)

    print(len(s2v))
    s2v = fix_s2v(s2v, data_tr + data_dv)
    print(s2v, len(s2v))

    if args.elmo:
        if args.gpu and torch.cuda.is_available():
            elmo = ElmoEmbedder(weight_file=args.elmo_weights,
                                options_file=args.elmo_options,
                                cuda_device=args.gpu)
        else:
            elmo = ElmoEmbedder(weight_file=args.elmo_weights,
                                options_file=args.elmo_options)
        slot_featurizer = ElmoFeaturizer(elmo, "slot")
        value_featurizer = ElmoFeaturizer(elmo, "value")
    else:
        slot_featurizer = SlotFeaturizer(w2v)
        value_featurizer = ValueFeaturizer(w2v)
    s2v = featurize_s2v(s2v, slot_featurizer, value_featurizer)

    print("Featurizing...")
    data_f_tr = featurize_dialogs(data_tr, domains, strict, s2v, w2v, args)
    data_f_dv = featurize_dialogs(data_dv, domains, strict, s2v, w2v, args)
    # print(data_tr[0].to_dict()['turns'][0]['system_acts'])

    DIM_INPUT = len(data_f_tr[0].turns[0].x_act)
    model = util.load_model(DIM_INPUT, DIM_INPUT, DIM_INPUT, DIM_INPUT,
                            DIM_HIDDEN_ENC, args.receptors, args)
    if args.resume:
        model.load_best_save(directory=args.resume)

    model = model.to(model.device)
    for name, param in model.named_parameters():
        print(name, param.device)

    print("Training...")
    if args.reinforce:
        model.run_train_reinforce(data_f_tr, data_f_dv, s2v, args)
    else:
        model.run_train(data_f_tr, data_f_dv, s2v, args)


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dexp', help='root experiment folder', default='exp')
    parser.add_argument('--model', help='which model to use',
                        default='statenet')
    parser.add_argument('--epochs', help='max epochs to run for', default=50,
                        type=int)
    parser.add_argument('--demb', help='word embedding size', default=400,
                        type=int)
    parser.add_argument('--dhid', help='hidden state size', default=200,
                        type=int)
    parser.add_argument('--batch_size', help='batch size', default=50, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--stop', help='slot to early stop on',
                        default='joint_goal')
    parser.add_argument('--resume', help='save directory to resume from')
    parser.add_argument('-n', '--nick', help='nickname for model',
                        default='default')
    parser.add_argument('--reinforce', action='store_true',
                        help='train with RL')
    parser.add_argument('--gamma', help='RL discount', default=0.99, type=float)
    parser.add_argument('--seed', default=42, help='random seed', type=int)
    parser.add_argument('--test', action='store_true',
                        help='run in evaluation only mode')
    parser.add_argument('--gpu', type=int, help='which GPU to use')
    parser.add_argument('--dropout', nargs='*', help='dropout rates',
                        default=['emb=0.2', 'local=0.2', 'global=0.2'])
    parser.add_argument('--train_domains', nargs='+',
                        help='Domains on which to train, If finetune_domains is'
                             ' also set, these will be used for pretraining.',
                        default='all')
    parser.add_argument('--finetune_domains', nargs='+',
                        help='Domains on which to finetune')
    parser.add_argument('--eval_domains', nargs='+',
                        help='Domains on which to evaluate', default='all')
    parser.add_argument('--train_strict', action='store_true',
                        help='Restrict pretraining to dialogs with '
                             'train_domains only')
    parser.add_argument('--finetune_strict', action='store_true',
                        help='Restrict finetuning dialogs with '
                             'finetune_domains only')
    parser.add_argument('--eta', help='factor for loss for binary slot filling '
                                      'prediction', default=0.5, type=float)
    parser.add_argument('--path', help='path to data files',
                        default='../data/multiwoz/ann/')
    parser.add_argument('--max_dialog_length', default=-1, type=int)
    parser.add_argument('--max_train_dialogs', default=-1, type=int)
    parser.add_argument('--max_dev_dialogs', default=-1, type=int)
    parser.add_argument('--elmo', action='store_true',
                        help="If set, use ELMo for encoding inputs")
    parser.add_argument('--elmo_weights',
                        default='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_'
                                'weights.hdf5')
    parser.add_argument('--elmo_options',
                        default='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_'
                                'options.json')
    parser.add_argument('--delexicalize_labels', action='store_true',
                        help="If set, replaces labels with dummy for select "
                             "slots")
    parser.add_argument('--encode_sys_utt', action='store_true',
                        help="If set, uses system utterance too, instead of "
                             "just system act representation")
    parser.add_argument('--receptors', default=3,
                        help='number of receptors per n-gram', type=int)
    parser.add_argument('--M', default=3, help='max n-gram size', type=int)

    _args = parser.parse_args()
    _args.dout = os.path.join(_args.dexp, _args.model, _args.nick)
    _args.dropout = {d.split('=')[0]: float(d.split('=')[1])
                     for d in _args.dropout}
    if not os.path.isdir(_args.dout):
        os.makedirs(_args.dout)
    return _args


if __name__ == '__main__':
    run(get_args())
