import os
import re
import logging
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.autograd import Variable as V
import numpy as np
from tqdm import tqdm
from util import util
from collections import defaultdict
from pprint import pformat
from eval import evaluate_preds
from collections import namedtuple

# TODO refactor such that encoder classes are declared within StateNet, allows
# for better modularization and sharing of instances/variables such as
# embeddings

Turn = namedtuple("Turn", ["user_utt", "system_act", "system_utt",
                           "x_utt", "x_act", "x_sys", "labels", "labels_str",
                           "belief_state"])
Dialog = namedtuple("Dialog", ["turns"])
Slot = namedtuple("Slot", ["domain", "embedding", "values"])
Value = namedtuple("Value", ["value", "embedding", "idx"])


class MultiScaleReceptors(nn.Module):
    """

    """
    def __init__(self, in_dim, out_dim, receptors):
        super().__init__()
        # Number of linear networks
        self.receptors = receptors
        # Init linear networks
        for i in range(self.receptors):
            setattr(self, 'linear_out_{}'.format(i), nn.Linear(in_dim, out_dim))

    def forward(self, utt_ngram_rep):
        out = []
        # Get output for every linear network
        for i in range(self.receptors):
            lin_layer = getattr(self, 'linear_out_{}'.format(i))
            out.append(lin_layer(utt_ngram_rep))
        # Return concatenated ngram representation
        return torch.cat(out)


class MultiScaleReceptorsModule(nn.Module):
    """

    """
    def __init__(self, in_dim, out_dim, receptors, n):
        super().__init__()
        self.receptors = receptors
        self.n = n
        #self.layer_norm = LayerNorm(in_dim)
        self.layer_norm = LayerNorm(receptors * out_dim)
        #self.linear_out = nn.Linear(in_dim, out_dim)
        self.linear_out = nn.Linear(receptors * out_dim, out_dim)

        # Initialize the c linear networks for each k-gram utt rep for 1 >= k >= n
        for i in range(n):
            msr_in_dim = in_dim * (i + 1)
            setattr(self, 'linear_out_r{}'.format(i),
                    MultiScaleReceptors(msr_in_dim, out_dim, self.receptors))

    def forward(self, user_ngram_utterances):
        """
        :param user_ngram_utterances:
        :return:
        """
        rets = []
        # For each k-gram utterance representation, get output from MSR networks
        for i in range(self.n):
            msr = getattr(self, 'linear_out_r{}'.format(i))
            msr_out = msr(user_ngram_utterances[i])
            rets.append(msr_out)

        rets = torch.stack(rets)
        out = torch.sum(rets, 0)
        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.linear_out(out)

        return out


class UtteranceEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim, receptors):
        super().__init__()
        self.layer_norm = LayerNorm(in_dim)
        self.linear_out = nn.Linear(in_dim, out_dim)

    def forward(self, user_utterance):
        """

        :param user_utterance:
        :return:
        """
        try:
            out = self.layer_norm(user_utterance)
        except RuntimeError:
            print(user_utterance, user_utterance.shape)
            raise RuntimeError

        out = F.relu(out)
        out = self.linear_out(out)
        return out


class ActionEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, action):
        """

        :param action:
        :return:
        """
        return F.relu(self.linear(action))


class SlotEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim, device):
        """

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.device = device

    def forward(self, slot_embedding):
        """

        :param slot_embedding: Vector representation of slot
        :return:
        """
        return F.relu(self.linear(slot_embedding)).view(-1).to(self.device)


class PredictionEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        """

        """
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, inputs, hidden):
        """
        Runs the RNN to compute outputs based on history from previous
        slots and turns. We maintain the hidden state across calls to this
        function.
        :param inputs: shape (batch_size, embeddings)
        :param hidden:
        :return: shape (batch_size, self.out_dim)
        """
        batch_size, embedding_length = inputs.view(1, -1).shape
        # reshape input to length 1 sequence (RNN expects input shape
        # [sequence_length, batch_size, embedding_length])
        inputs = inputs.view(1, batch_size, embedding_length)
        # compute output and new hidden state
        rnn_out, hidden = self.rnn(inputs, hidden)
        # reshape to [batch_size,
        rnn_out = rnn_out.view(batch_size, -1)
        o = F.relu(self.linear(rnn_out))
        # print("prediction vector:", o.shape)
        return o, hidden


class ValueEncoder(nn.Module):
    """

    """
    def __init__(self, in_dim, out_dim, device):
        """

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.device = device

    def forward(self, value_embedding):
        """

        :param value_embedding:
        :return:
        """
        return F.relu(self.linear(value_embedding)).to(self.device)


class StateNet(nn.Module):
    """
    Implementation based on Ren et al. (2018): Towards Universal Dialogue
    State Tracking. EMNLP 2018. http://aclweb.org/anthology/D18-1299.

    The paper remains unclear regarding a number of points, for which we
    make decisions based on our intuition. These are, for example:

    (1) How are predictions across turns aggregated on the dialogue level?
        Is the probability for a slot-value pair maxed across turns?
        - We assume yes.
    (2) The paper says that parameters are optimized based on cross-entropy
        between slot-value predictions and gold labels. How does this integrate
        the LSTM that is located outside the turn loop?
        - Not really sure how to handle this yet...
    (3) Is the LSTM updated after every turn AND every slot representation
        computation?
        - We assume yes.

    """

    def __init__(self, input_user_dim, input_action_dim, input_slot_dim,
                 input_value_dim, hidden_dim, receptors,
                 args):
        """

        :param input_user_dim: dimensionality of user input embeddings
        :param input_action_dim: dimensionality of action embeddings
        :param hidden_dim:
        :param receptors:
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.args = args
        self.device = self.get_device()

        if args.encode_sys_utt:
            slot_hidden_dim = 3 * hidden_dim
        else:
            slot_hidden_dim = 2 * hidden_dim

        if not args.elmo:
            input_user_dim = input_user_dim * args.M
        u_in_dim = input_user_dim
        a_in_dim = input_action_dim
        s_in_dim = input_user_dim
        # self.utterance_encoder = UtteranceEncoder(u_in_dim, hidden_dim,
        #                                           receptors)
        n = int(u_in_dim / a_in_dim)
        if args.elmo:
            self.utt_enc = UtteranceEncoder(u_in_dim, hidden_dim, receptors)
        else:
            self.utt_enc = MultiScaleReceptorsModule(a_in_dim, hidden_dim,
                                                     receptors, n)
        self.action_encoder = ActionEncoder(a_in_dim, hidden_dim)
        self.slot_encoder = SlotEncoder(input_slot_dim, slot_hidden_dim,
                                        self.device)
        self.value_encoder = ValueEncoder(input_value_dim, hidden_dim,
                                              self.device)
        self.prediction_encoder = PredictionEncoder(slot_hidden_dim,
                                                    hidden_dim, hidden_dim)
        self.slot_fill_indicator = nn.Linear(hidden_dim, 1)
        self.optimizer = None
        self.epochs_trained = 0
        self.logger = self.get_train_logger()
        self.logger.setLevel(logging.INFO)
        self.logger.info(args)

    def set_epochs_trained(self, e):
        self.epochs_trained = e

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

    def get_train_logger(self):
        logger = logging.getLogger(
            'train-{}'.format(self.__class__.__name__))
        formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] '
                                      '[%(levelname)-5.5s]  %(message)s')
        file_handler = logging.FileHandler(
            os.path.join(self.args.dout, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    # @property
    def get_device(self):
        if self.args.gpu is not None and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu = self.args.gpu % num_gpus
            return torch.device('cuda:{}'.format(gpu))
        else:
            return torch.device('cpu')

    def embed(self, w, numpy=False):
        e = self.embeddings.get(w)
        if not e:
            e = np.zeros(self.embeddings_len)
        if numpy:
            return np.array(e)
        else:
            return torch.Tensor(e)

    def embed_batch(self, b):
        e = [self.embed(w, numpy=True) for w in b]
        return torch.Tensor(e)

    def forward_turn(self, turn, slots2values, hidden):
        """

        # :param x_user: shape (batch_size, user_embeddings_dim)
        # :param x_action: shape (batch_size, action_embeddings_dim)
        # :param x_sys: shape (batch_size, sys_embeddings_dim)
        :param turn:
        :param hidden: shape (batch_size, 1, hidden_dim)
        :param slots2values: dict mapping slots to values to be tested
        # :param labels: dict mapping slots to one-hot ground truth value
        representations
        :return: tuple (loss, probs, hidden), with `loss' being the overall
        loss across slots, `probs' a dict mapping slots to probability
        distributions over values, `hidden' the new hidden state
        """
        probs = {}
        binary_filling_probs = {}

        # Encode user and action representations
        if self.args.elmo:
            utt = turn.x_utt.to(self.device)
            sys = turn.x_sys.to(self.device)
        else:
            utt = [t.to(self.device) for t in turn.x_utt]  # one vector per n
            sys = [t.to(self.device) for t in turn.x_sys]  # one vector per n
        act = turn.x_act.to(self.device)

        fu = self.utt_enc(utt)  # user input encoding
        fa = self.action_encoder(act)  # action input encodingdebug
        fy = None
        if self.args.encode_sys_utt:
            fy = self.utt_enc(sys)

        loss_updates = torch.Tensor([0]).to(self.device)

        # iterate over slots and values, compute probabilities
        for slot_id, slot in slots2values.items():
            # compute encoding of inputs as described in StateNet paper, Sec. 2
            fs = self.slot_encoder(slot.embedding)
            if self.args.encode_sys_utt:
                i = F.mul(fs, torch.cat((fu, fa, fy), 0))  # inputs encoding
            else:
                i = F.mul(fs, torch.cat((fu, fa), 0))  # inputs encoding
            o, hidden = self.prediction_encoder(i, hidden)

            # get binary prediction for slot presence
            binary_filling_probs[slot_id] = torch.sigmoid(self.slot_fill_indicator(o))

            # get probability distribution over values...
            values = slot.values

            if binary_filling_probs[slot_id] > 0.5:
                probs[slot_id] = torch.zeros(len(values))
                for v, value in enumerate(values):
                    venc = self.value_encoder(value.embedding)
                    # ... by computing 2-Norm distance following paper, Sec. 2.6
                    probs[slot_id][v] = -torch.dist(o, venc)
                probs[slot_id] = F.softmax(probs[slot_id], 0)  # softmax it!

        loss = torch.Tensor([0]).to(self.device)
        if self.training:
            for slot_id in slots2values.keys():
                # 1 if slot in turn.labels (meaning it's filled), 0 else
                gold_slot_filling = torch.Tensor([float(slot_id in turn.labels)]).to(self.device)
                loss += self.args.eta * F.binary_cross_entropy(
                    binary_filling_probs[slot_id],
                    gold_slot_filling).to(self.device)
                loss_updates += 1
                if slot_id in turn.labels and binary_filling_probs[slot_id] > 0.5:
                    loss += F.binary_cross_entropy(
                        probs[slot_id],
                        turn.labels[slot_id]
                    ).to(self.device)
                    loss_updates += 1

        loss = loss / loss_updates
        mean_slots_filled = len(probs) / len(slots2values)
        return loss, probs, hidden, mean_slots_filled

    def forward(self, dialog, slots2values):
        """

        :param dialog:
        :param slots2values:
        :return:
        """
        hidden = torch.zeros(1, 1, self.hidden_dim).to(self.device)
        global_probs = {}
        global_loss = torch.Tensor([0]).to(self.device)
        per_turn_mean_slots_filled = []
        ys_turn = []

        for turn in dialog.turns:
            loss, turn_probs, hidden, mean_slots_filled = \
                self.forward_turn(turn, slots2values, hidden)
            per_turn_mean_slots_filled.append(mean_slots_filled)
            global_loss += loss
            turn_preds = {}
            for slot_id, slot in slots2values.items():
                if slot_id in turn_probs:
                    global_probs[slot_id] = torch.zeros(len(slot.values))
                    argmax = np.argmax(turn_probs[slot_id].detach().numpy(), 0)
                    turn_preds[slot_id] = slots2values[slot_id].values[
                        int(argmax)].value
                    for v, value in enumerate(slot.values):
                        global_probs[slot_id][v] = max(global_probs[slot_id][v],
                                                       turn_probs[slot_id][v])

            ys_turn.append(turn_preds)

        # get final predictions
        ys = {}
        for slot, probs in global_probs.items():
            score, argmax = probs.max(0)
            ys[slot] = slots2values[slot].values[int(argmax)].value

        global_loss = global_loss / len(dialog.turns)
        dialog_mean_slots_filled = np.mean(per_turn_mean_slots_filled)
        return ys, ys_turn, global_loss, dialog_mean_slots_filled

    def run_train(self, dialogs_train, dialogs_dev, s2v, args):
        track = defaultdict(list)
        if self.optimizer is None:
            self.set_optimizer()
        self.logger.info("Starting training...")
        s2v = self.s2v_to_device(s2v)
        best = {}
        iteration = 0
        for epoch in range(1, args.epochs+1):
            global_mean_slots_filled = []
            # logger.info('starting epoch {}'.format(epoch))

            if not hasattr(self, "epochs_trained"):
                self.set_epochs_trained(0)
            self.epochs_trained += 1

            # train and update parameters
            self.train()
            train_predictions = []
            for dialog in tqdm(dialogs_train):
                iteration += 1
                self.zero_grad()
                predictions, turn_predictions, loss, mean_slots_filled = \
                    self.forward(dialog, s2v)
                train_predictions.append((predictions, turn_predictions))
                # print(turn_predictions)
                global_mean_slots_filled.append(mean_slots_filled)
                loss.backward()
                self.optimizer.step()
                track['loss'].append(loss.item())

            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': self.epochs_trained}
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            self.logger.info("Evaluating...")
            predictions, turn_predictions = zip(*train_predictions)
            summary.update({'eval_train_{}'.format(k):v for k, v in
                            evaluate_preds(dialogs_train, predictions,
                                           turn_predictions).items()})
            summary.update({'eval_dev_{}'.format(k):v for k, v in
                            self.run_eval(dialogs_dev, s2v).items()})

            global_mean_slots_filled = np.mean(global_mean_slots_filled)
            self.logger.info("Predicted {}% slots as present".format(global_mean_slots_filled*100))
            self.logger.info("Epoch summary: " + str(summary))

            # do early stopping saves
            stop_key = 'eval_dev_{}'.format(args.stop)
            train_key = 'eval_train_{}'.format(args.stop)
            if best.get(stop_key, 0) <= summary[stop_key]:
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                self.save(best,
                          identifier='epoch={epoch},iter={iteration},'
                                     'train_{key}={train},dev_{key}={dev}'
                                     ''.format(
                                        epoch=self.epochs_trained,
                                        iteration=iteration,
                                        train=best_train, dev=best_dev,
                                        key=args.stop)
                          )
                self.prune_saves()
                # dialogs_dev.record_preds(  #TODO self.run_pred returns list of tuples (predictions_dialog, predictions_turn)
                #     preds=self.run_pred(dialogs_dev, s2v, self.args),
                #     to_file=os.path.join(self.args.dout, 'dev.pred.json'),
                # )
            summary.update({'best_{}'.format(k): v for k, v in best.items()})
            self.logger.info(pformat(summary))
            track.clear()

    def run_pred(self, dialogs, s2v):
        self.eval()
        predictions = []
        for d in tqdm(dialogs):
            predictions_d, turn_predictions, _, _ = self.forward(d, s2v)
            predictions.append((predictions_d, turn_predictions))
        return predictions

    def run_eval(self, dialogs, s2v):
        predictions, turn_predictions = zip(*self.run_pred(dialogs, s2v))
        return evaluate_preds(dialogs, predictions, turn_predictions,
                              self.args.dout+"/prediction.json")

    def save(self, summary, identifier):
        fname = '{}/{}.t7'.format(self.args.dout, identifier)
        logging.info('saving model to {}'.format(fname))
        state = {
            'args': vars(self.args),
            'model': self.state_dict(),
            'summary': summary,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epochs_trained
        }
        torch.save(state, fname)

    def load(self, path):
        logging.info('loading model from {}'.format(path))
        state = torch.load(path)
        self.load_state_dict(state['model'])
        self.set_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])
        resume_from_epoch = state.get('epoch', 0)
        self.set_epochs_trained(resume_from_epoch)
        self.logger.info("Resuming from epoch {}".format(resume_from_epoch))

    def prune_saves(self, n_keep=5):
            scores_and_files = self.get_saves()
            if len(scores_and_files) > n_keep:
                for score, fname in scores_and_files[n_keep:]:
                    os.remove(fname)

    def load_best_save(self, directory):
        if directory is None:
            directory = self.args.dout

        scores_and_files = self.get_saves(directory=directory)
        if scores_and_files:
            assert scores_and_files, 'no saves exist at {}'.format(directory)
            score, fname = scores_and_files[0]
            self.load(fname)

    def get_saves(self, directory=None):
        if directory is None:
            directory = self.args.dout
        files = [f for f in os.listdir(directory) if f.endswith('.t7')]
        scores = []
        for fname in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop)
            dev_acc = re.findall(re_str, fname)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(directory, fname)))
        if not scores:
            raise Exception('No files found!')
        scores.sort(key=lambda tup:tup[0], reverse=True)
        return scores

    def s2v_to_device(self, s2v):
        s2v_new = {}
        for slot_name, slot in s2v.items():
            slot_emb = torch.Tensor(slot.embedding).to(self.device)
            vals_new = []
            for val in slot.values:
                val_emb = torch.Tensor(val.embedding).to(self.device)
                vals_new.append(Value(val.value, val_emb, val.idx))
            s2v_new[slot_name] = Slot(slot.domain, slot_emb, vals_new)
        return s2v_new

