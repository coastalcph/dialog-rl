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

# TODO refactor such that encoder classes are declared within StateNet, allows
# for better modularization and sharing of instances/variables such as
# embeddings

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
        self.receptors = receptors
        # TODO multiple receptors
        # self.layers = [nn.Linear(in_dim, out_dim) for _ in range(receptors)]
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

    def __init__(self, in_dim, out_dim, embeddings):
        """

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.embeddings = embeddings
        self.embeddings_len = len(embeddings.get("i"))
        self.linear = nn.Linear(self.embeddings_len, out_dim)

    def forward(self, slot):
        """

        :param slot:
        :return:
        """
        # remove domain prefix ('restaurant-priceRange' -> 'priceRange')
        domain, slot = slot.split("-", 1)
        # split at uppercase to get vectors ('priceRange' -> ['price', 'range'])
        words = util.split_on_uppercase(slot, keep_contiguous=True)
        vecs = [self.embeddings.get(w.lower()) for w in words]
        if not vecs:
            vecs = [[0 for _ in range(len(self.embeddings.get("i")))]]

        sv = torch.Tensor(vecs)
        sv, _ = torch.max(sv, 0)
        # print(domain, slot, words, len(vecs), sv.shape)

        return F.relu(self.linear(sv))


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
    def __init__(self, out_dim, embeddings):
        """

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.embeddings = embeddings
        self.embeddings_len = len(embeddings.get("i"))
        self.linear = nn.Linear(self.embeddings_len, out_dim)

    def forward(self, slot):
        """

        :param slot:
        :return:
        """
        v = self.embeddings.get(slot)
        if not v:
            v = torch.zeros(len(self.embeddings.get("i")))
        v = torch.Tensor(v)
        return F.relu(self.linear(v))


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

    def __init__(self, input_user_dim, input_action_dim, hidden_dim, receptors,
                 embeddings, args):
        """

        :param input_user_dim: dimensionality of user input embeddings
        :param input_action_dim: dimensionality of action embeddings
        :param hidden_dim:
        :param receptors:
        :param embeddings:
        """
        super().__init__()
        u_in_dim = input_user_dim
        a_in_dim = input_action_dim
        s_in_dim = input_user_dim
        self.hidden_dim = hidden_dim
        n = int(u_in_dim / a_in_dim)
        self.utt_enc = MultiScaleReceptorsModule(a_in_dim, hidden_dim, receptors, n)
        #self.utterance_encoder = UtteranceEncoder(u_in_dim, hidden_dim,
        #                                          receptors)
        self.action_encoder = ActionEncoder(a_in_dim, hidden_dim)
        self.slot_encoder = SlotEncoder(s_in_dim, 3*hidden_dim, embeddings)
        self.prediction_encoder = PredictionEncoder(3*hidden_dim, hidden_dim, hidden_dim)
        self.slot_fill_indicator = nn.Linear(hidden_dim, 1)

        self.value_encoder = ValueEncoder(hidden_dim, embeddings)
        self.embeddings = embeddings
        self.embeddings_len = len(embeddings.get("i"))
        self.args = args
        self.device = self.device()
        self.optimizer = None


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
    def device(self):
        if self.args.gpu is not None and torch.cuda.is_available():
            return torch.device('cuda')
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

        # Encode user and action representations offline
        #fu = self.utterance_encoder(turn.user_utt)  # user input encoding
        fu = self.utt_enc(turn.user_utt)  # user input encoding
        fa = self.action_encoder(turn.system_act)  # action input encoding
        #fy = self.utterance_encoder(turn.system_utt)
        fy = self.utt_enc(turn.system_utt)

        # iterate over slots and values, compute probabilities
        for slot in slots2values.keys():
            # compute encoding of inputs as described in StateNet paper, Sec. 2
            fs = self.slot_encoder(slot).view(-1)  # slot encoding
            # i = torch.cat((fu, fa), 0)
            # i = F.mul(fs, i)
            i = F.mul(fs, torch.cat((fu, fa, fy), 0))  # inputs encoding
            o, hidden = self.prediction_encoder(i, hidden)

            # get binary prediction for slot presence
            binary_filling_probs[slot] = F.sigmoid(self.slot_fill_indicator(o))

            # get probability distribution over values...
            values = slots2values[slot]
            probs[slot] = torch.zeros(len(values))

            if binary_filling_probs[slot] > 0.5:
                for v, value in enumerate(values):
                    venc = self.value_encoder(value)
                    # ... by computing 2-Norm distance following paper, Sec. 2.6
                    probs[slot][v] = -torch.dist(o, venc)
                probs[slot] = F.softmax(probs[slot], 0)  # softmax it!

        if self.training:
            loss = 0
            for slot in slots2values.keys():
                # 1 if slot in turn.labels (meaning it's filled), 0 else
                gold_slot_filling = torch.tensor(float(slot in turn.labels))
                loss += self.args.eta * F.binary_cross_entropy(
                    binary_filling_probs[slot],
                    gold_slot_filling)
                if slot in turn.labels and binary_filling_probs[slot] > 0.5:
                    loss += F.binary_cross_entropy(probs[slot], turn.labels[slot])
        else:
            loss = torch.Tensor([0]).to(self.device)

        return loss, probs, hidden

    def forward(self, dialog, slots2values):
        """

        :param dialog:
        :param slots2values:
        :return:
        """
        hidden = torch.zeros(1, 1, self.hidden_dim)
        global_probs = {}
        global_loss = torch.Tensor([0]).to(self.device)

        ys_turn = []
        
      

        for turn in dialog.turns:
            loss, turn_probs, hidden = self.forward_turn(turn, slots2values,
                                                         hidden)

            global_loss += loss
            turn_preds = {}
            for slot, values in slots2values.items():
                global_probs[slot] = torch.zeros(len(values))
                turn_preds[slot] = np.argmax(turn_probs[slot].detach().numpy()
                                             , 0)

                for v, value in enumerate(values):
                    global_probs[slot][v] = max(global_probs[slot][v],
                                                turn_probs[slot][v])

            ys_turn.append(turn_preds)

        # get final predictions
        ys = {}
        for slot, probs in global_probs.items():
            score, argmax = probs.max(0)
            ys[slot] = int(argmax)

        return ys, ys_turn, global_loss

    def run_train(self, dialogs_train, dialogs_dev, s2v, args):
        track = defaultdict(list)
        logger = self.get_train_logger()
        if self.optimizer is None:
            self.set_optimizer()
        best = {}
        iteration = 0
        for epoch in range(1, args.epochs+1):
            # logger.info('starting epoch {}'.format(epoch))

            if not hasattr(self, "epochs_trained"):
                self.set_epochs_trained(0)
            self.epochs_trained += 1

            # train and update parameters
            self.train()
            for dialog in tqdm(dialogs_train):
                iteration += 1
                self.zero_grad()
                predictions, turn_predictions, loss = self.forward(dialog, s2v)
                loss.backward()
                self.optimizer.step()
                track['loss'].append(loss.item())

            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': self.epochs_trained}
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            summary.update({'eval_train_{}'.format(k):v for k, v in
                            self.run_eval(dialogs_train, s2v, args).items()})
            summary.update({'eval_dev_{}'.format(k):v for k, v in
                            self.run_eval(dialogs_dev, s2v, args).items()})

            print(summary)

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
                # self.prune_saves()
                # dialogs_dev.record_preds(  #TODO self.run_pred returns list of tuples (predictions_dialog, predictions_turn)
                #     preds=self.run_pred(dialogs_dev, s2v, self.args),
                #     to_file=os.path.join(self.args.dout, 'dev.pred.json'),
                # )
            summary.update({'best_{}'.format(k):v for k, v in best.items()})
            logger.info(pformat(summary))
            track.clear()

    def run_pred(self, dialogs, s2v, args):
        self.eval()
        predictions = []
        for d in dialogs:
            predictions_d, turn_predictions, _ = self.forward(d, s2v)
            predictions.append((predictions_d, turn_predictions))
        return predictions

    def run_eval(self, dialogs, s2v, args):
        predictions, turn_predictions = zip(*self.run_pred(dialogs, s2v, args))
        return evaluate_preds(dialogs, predictions, turn_predictions)

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
        print("Resuming from epoch {}".format(resume_from_epoch))

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

