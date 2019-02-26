import numpy as np
import json
from pprint import pprint

def zero_if_nan(n):
    return 0 if np.isnan(n) else n

def filter_labels(gold, eval_doms):
    filtered = {}
    for s, v in gold.items():
        if s.split('-')[0] in eval_doms:
            filtered[s] = v
    return filtered

def delex_labels(gold):
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
    for s, v in gold.items():
        if s in allowed_slots:
            out[s] = v
        else:
            out[s] = "<true>"
    return out

def evaluate_preds(dialogs, preds, turn_predictions, eval_domains=None,
                   write_out=None, delex=True):
    inform = []
    joint_goal = []
    belief_state = []
    dialog_reward = []
    final_binary_slot_precision = []
    final_binary_slot_recall = []
    binary_slot_precision = []
    binary_slot_recall = []
    turn_joint_goal = []
    f = None
    if write_out:
        f = open(write_out, "w")

    dialogs_out = []

    for di, d in enumerate(dialogs[:-1]):

        dialog_out = {"turns": []}
        for ti, turn in enumerate(d.turns):

            turn_out = {"user_utt": ' '.join(turn.user_utt),
                        "system_act": turn.system_act,
                        "system_utt": turn.system_utt}
            turn_gold = delex_labels(turn.labels_str) if delex else turn.labels_str

            gold_inform = turn_gold
            pred_inform = turn_predictions[di][ti]

            turn_out["gold"] = turn_gold
            turn_out["pred"] = pred_inform

            golds = filter_labels(turn_gold, eval_domains)

            for s, v in gold_inform.items():
                s_domain = s.split("-")[0]
                #if eval_domains and s_domain not in eval_domains:
                if s_domain not in eval_domains:
                    continue
                s_in_pred_inform = s in pred_inform
                binary_slot_recall.append(s_in_pred_inform)
                if s_in_pred_inform:
                    inform.append(v == pred_inform[s])
                else:
                    inform.append(False)

            ## Turn level inform accuracy
            golds = set([(s, v) for s, v in golds.items()])
            #if len(golds) == 0 and not ti == len(d.turns) - 1 and len(pred_inform) > 0:
            #    print("Dialog", di)
            #    print("turn", ti, "out of", len(d.turns))
            #    pprint(turn_out)
            predictions = set([(s, v) for s, v in pred_inform.items()])

            turn_joint_goal.append(golds == predictions)

            #if di == 5:
            #if golds == predictions and len(golds) > 0:
            #    print(turn_out)
            #    print("{} Gold:\t{}".format(ti, golds))
            #    print("{} Pred:\t{}".format(ti, predictions))
            #    print("")


            for s in pred_inform:
                s_domain = s.split("-")[0]
                if s_domain not in eval_domains:
                    continue
                binary_slot_precision.append(s in gold_inform)

            dialog_out["turns"].append(turn_out)

        # evaluate final dialog-level performance
        gold_final_belief = {b['slots'][0]: b['slots'][1]
                             for b in d.turns[-1].belief_state}

        gold_final_belief = delex_labels(gold_final_belief) if delex else gold_final_belief

        for s, v in gold_final_belief.items():
            s_domain = s.split("-")[0]
            if s_domain not in eval_domains:
                continue
            #if s in preds[di]:
            #    belief_state.append(v == preds[di][s])
            final_binary_slot_recall.append(s in preds[di])

        for s in preds[di]:
            s_domain = s.split("-")[0]
            if s_domain not in eval_domains:
                continue
            final_binary_slot_precision.append(s in gold_final_belief)

        gold_final_belief = set([(s, v) for s, v in gold_final_belief.items()])
        dialog_preds = set([(s,v) for s, v in preds[di].items()])

        # How well did we predict the final belief state
        common_bs = dialog_preds.intersection(gold_final_belief)
        #if len(gold_final_belief) == len(common_bs) == 0:
        if len(gold_final_belief) == len(dialog_preds) == 0:
            dia_rew = 1
        else:
            dia_rew = len(common_bs) / (len(gold_final_belief) + len(dialog_preds - common_bs))

        #if dia_rew > 0.5:
        #    print(dia_rew)
        #    print(dialog_preds)
        #    print(gold_final_belief)

        belief_state.append(gold_final_belief == dialog_preds)
        dialog_reward.append(dia_rew)

        dialog_out["gold_final_belief"] = gold_final_belief
        dialog_out["pred_final_belief"] = {s: v for s, v in preds[di].items()}
        dialogs_out.append(dialog_out)

    if f:
        # print(dialogs_out)
        try:
            json.dump(dialogs_out, f)
        except:
            pass
        f.close()

    final_R = np.mean(final_binary_slot_recall)
    final_P = np.mean(final_binary_slot_precision)
    final_binary_slot_F1 = 2 * final_R * final_P / (final_R + final_P)
    if np.isnan(final_binary_slot_F1):
        final_binary_slot_F1 = 0

    R = np.mean(binary_slot_recall)
    P = np.mean(binary_slot_precision)
    binary_slot_F1 = 2*R*P / (R+P)
    joint_goal = inform
    out_dict = {
                # 'turn_inform': np.mean(inform),
                # 'turn_request': np.mean(request),
                'joint_goal': np.mean(joint_goal),
                'turn_joint_goal': np.mean(turn_joint_goal),
                'belief_state': np.mean(belief_state),
                'dialog_reward': np.mean(dialog_reward),
                'final_binary_slot_f1': final_binary_slot_F1,
                'binary_slot_p': P,
                'binary_slot_r': R,
                'binary_slot_f1': binary_slot_F1
    }

    # fix NaNs
    return {k: zero_if_nan(v) for k, v in out_dict.items()}


def shape_reward(reward, scale_in=(0, 1), scale_out=(-2, 2), continuous=False):
    # Linear interpolation
    scaled = (reward - scale_in[0]) / (scale_in[1] - scale_in[0]) * (scale_out[1] - scale_out[0]) + scale_out[0]
    if not continuous:
        scaled = round(scaled)
    return scaled

def get_reward(e_scores):
    #return e_scores['joint_goal'] * w[0] + e_scores['belief_state'] * w[1]
    return e_scores['dialog_reward']

