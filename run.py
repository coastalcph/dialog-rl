import os
from util import util
from util.featurize import *
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def main(args):
    print(args)
    device = util.get_device(args.gpu)
    domains = args.train_domains
    strict = args.train_strict
    print('Training on domains: ',  domains)
    print('Single-domain dialogues only?', strict)
    #random.seed(args.seed)
    splits = ["train", "dev"]
    if args.test or args.pred:
        splits = ["test"]

    if args.elmo:
        data, s2v = util.load_dataset_elmo(splits=splits, base_path=args.path)
    else:
        data, ontology, vocab, embeddings = util.load_dataset(
            splits=splits, base_path=args.path)
        s2v = ontology.values

    data_filtered = {}
    data_featurized = {}

    # filter data for domains
    for split in splits:
        _data = [dg.to_dict() for dg in data[split].iter_dialogs()]
        max_dialogs = {"train": args.max_train_dialogs,
                       "dev": args.max_dev_dialogs}.get(split, -1)
        data_filtered[split] = util.filter_dialogs(_data, domains, strict,
                                                   max_dialogs,
                                                   args.max_dialog_length)

        if split == "train":
            random.shuffle(data_filtered[split])

    # If not using ELMo featurized dataset, create slot-to-value featurization
    if not args.elmo:
        # Retrieve and clean slot-value pairs
        if args.delexicalize_labels:
            s2v = util.delexicalize(s2v)
        s2v = util.fix_s2v(s2v, data_filtered, splits=splits)

        # Featurize slots and values
        slot_featurizer = SlotFeaturizer(embeddings)
        value_featurizer = ValueFeaturizer(embeddings)
        s2v = util.featurize_s2v(s2v, slot_featurizer, value_featurizer)

    print("device : ", device)
    s2v = util.s2v_to_device(s2v, device)

    print("Featurizing...")
    for split in splits:
        if args.elmo:
            data_featurized[split] = featurize_dialogs_elmo(
                data_filtered[split], s2v, device, args)
        else:
            data_featurized[split] = featurize_dialogs(
                data_filtered[split], s2v, device, args, w2v=embeddings)

    _key = list(data_featurized.keys())[0]
    DIM_INPUT = len(data_featurized[_key][0].turns[0].x_act)
    model = util.load_model(DIM_INPUT, DIM_INPUT, DIM_INPUT, DIM_INPUT,
                            args.dhid, args.receptors, args)
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
            if args.baseline:
                print('Loading baseline model')
                baseline = util.load_model(DIM_INPUT, DIM_INPUT, DIM_INPUT, DIM_INPUT,
                                           args.dhid, args.receptors, args)
                baseline.load_best_save(directory=args.resume)
                #baseline.trainable = False
                #baseline = copy.deepcopy(model)
                for param in baseline.parameters():
                    param.requires_grad = False
            else:
                baseline = None
            model.run_train_reinforce(data_featurized["train"],
                                      data_featurized["dev"], s2v, args,
                                      baseline=baseline)
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
    parser.add_argument('--baseline', help='use baseline for variance reduction when fine tuning',
                        action='store_true')
    parser.add_argument('--patience', help='Patience for early stopping',
                        default=20, type=int)
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
                        default='data/multiwoz/ann/')
    parser.add_argument('--max_dialog_length', default=-1, type=int)
    parser.add_argument('--max_train_dialogs', default=-1, type=int)
    parser.add_argument('--max_dev_dialogs', default=-1, type=int)
    parser.add_argument('--elmo', action='store_true',
                        help="If set, use ELMo for encoding inputs")
    parser.add_argument('--pooled', action='store_true',
                        help="If set, use max pooled ELMo embeddings")
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
    main(get_args())
