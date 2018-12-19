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


def featurize_dialogs(_data, _domains, _strict, s2v, w2v, device, args):
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
            all_system_utts.append(sys)
            all_ys.append(ys)
            all_lbls.append(lbls)
            all_bsts.append(bst)

        all_x_utt = utt_ftz.featurize_batch(all_user_utts)
        all_x_act = act_ftz.featurize_batch(all_system_acts)
        all_x_sys = sys_ftz.featurize_batch(all_system_utts)

        for i in range(len(dg['turns'])):

            # Encode user and action representations
            if args.elmo:
                x_utt = all_x_utt[i].to(device)
                x_sys = all_x_sys[i].to(device)
            else:
                x_utt = [t.to(device) for t in
                         all_x_utt[i]]  # one vector per n
                x_sys = [t.to(device) for t in
                         all_x_sys[i]]  # one vector per n
            x_act = all_x_act[i].to(device)

            featurized_turns.append(Turn(
                all_user_utts[i], all_system_acts[i], all_system_utts[i],
                x_utt, x_act, x_sys,
                all_ys[i], all_lbls[i], all_bsts[i]))

        featurized_dialogs.append(Dialog(featurized_turns))

    return featurized_dialogs


def run(args):
    print(args)
    device = util.get_device(args.gpu)
    domains = args.train_domains
    strict = args.train_strict
    print('Training on domains: ',  domains)
    print('Single-domain dialogues only?', strict)

    splits = ["train", "dev"]
    if args.test or args.pred:
        splits = ["test"]
    data, ontology, vocab, w2v = util.load_dataset(splits=splits,
                                                   base_path=args.path)
    all_data = []
    data_filtered = {}
    data_featurized = {}

    s2v = ontology.values
    if args.delexicalize_labels:
        s2v = util.delexicalize(s2v)

    # filter data for domains
    for split in splits:
        _data = [dg.to_dict() for dg in data[split].iter_dialogs()]
        max_dialogs = {"train": args.max_train_dialogs,
                       "dev": args.max_dev_dialogs}.get(split, -1)
        data_filtered[split] = filter_dialogs(_data, domains, strict,
                                              max_dialogs,
                                              args.max_dialog_length)
        if split == "train":
            random.shuffle(data_filtered[split])
        all_data.extend(data_filtered[split])

    print(len(s2v))
    s2v = util.fix_s2v(s2v, all_data)
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
    s2v = util.featurize_s2v(s2v, slot_featurizer, value_featurizer, device)

    print("Featurizing...")
    for split in splits:
        data_featurized[split] = featurize_dialogs(data_filtered[split],
                                                   domains, strict, s2v,
                                                   w2v, device, args)

    key = list(data_featurized.keys())[0]
    DIM_INPUT = len(data_featurized[key][0].turns[0].x_act)
    model = util.load_model(DIM_INPUT, DIM_INPUT, DIM_INPUT, DIM_INPUT,
                            DIM_HIDDEN_ENC, args.receptors, args)
    if args.resume:
        model.load_best_save(directory=args.resume)

    model = model.to(device)

    if args.test:
        results = model.run_eval(data_featurized["test"], s2v,
                                 args.eval_domains, args.outfile)
        print(results)
    elif args.pred:
        raise NotImplementedError
        # model.run_predict(data_featurized["test"], s2v, args)
    else:
        print("Training...")
        if args.reinforce:
            model.run_train_reinforce(data_featurized["train"],
                                      data_featurized["dev"], s2v, args)
        else:
            model.run_train(data_featurized["train"], data_featurized["dev"],
                            s2v, args)


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
    parser.add_argument('--pred', action='store_true',
                        help='run in prediction only mode')
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
    parser.add_argument('--outfile', help='output file for test')
    parser.add_argument('--log_level', help='log level. default is info',
                        default="info")

    _args = parser.parse_args()
    _args.dout = os.path.join(_args.dexp, _args.model, _args.nick)
    _args.dropout = {d.split('=')[0]: float(d.split('=')[1])
                     for d in _args.dropout}
    if not os.path.isdir(_args.dout):
        os.makedirs(_args.dout)
    return _args


if __name__ == '__main__':
    run(get_args())
