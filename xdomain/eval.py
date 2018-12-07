import numpy as np
import json


def zero_if_nan(n):
    return 0 if np.isnan(n) else n


def evaluate_preds(dialogs, preds, turn_predictions, eval_domains=None,
                   write_out=None):
    inform = []
    joint_goal = []
    belief_state = []
    final_binary_slot_precision = []
    final_binary_slot_recall = []
    binary_slot_precision = []
    binary_slot_recall = []
    f = None
    if write_out:
        f = open(write_out, "w")

    dialogs_out = []

    for di, d in enumerate(dialogs):

        dialog_out = {"turns": []}
        for ti, turn in enumerate(d.turns):

            turn_out = {"user_utt": turn.user_utt,
                        "system_act": turn.system_act,
                        "system_utt": turn.system_utt}
            turn_gold = turn.labels_str

            # gold_inform = {s: int(np.argmax(v)) for s, v in turn_gold.items()}
            # pred_inform = {s: int(v) for s, v in turn_predictions[di][ti].items()}

            gold_inform = turn_gold
            pred_inform = turn_predictions[di][ti]

            turn_out["gold"] = turn_gold
            turn_out["pred"] = pred_inform

            # print("GOLD INFORM", gold_inform)
            # print("PRED INFORM", pred_inform)
            # print("=======================")
            for s, v in gold_inform.items():
                s_domain = s.split("-")[0]
                if eval_domains and s_domain not in eval_domains:
                    continue
                s_in_pred_inform = s in pred_inform
                binary_slot_recall.append(s_in_pred_inform)
                if s_in_pred_inform:
                    inform.append(v == pred_inform[s])
                else:
                    inform.append(False)

            for s in pred_inform:
                s_domain = s.split("-")[0]
                if eval_domains and s_domain not in eval_domains:
                    continue
                binary_slot_precision.append(s in gold_inform)

            dialog_out["turns"].append(turn_out)

        # evaluate final dialog-level performance
        gold_final_belief = {b['slots'][0]: b['slots'][1]
                             for b in d.turns[-1].belief_state}

        for s, v in gold_final_belief.items():
            s_domain = s.split("-")[0]
            if eval_domains and s_domain not in eval_domains:
                continue
            if s in preds[di]:
                belief_state.append(v == preds[di][s])
            final_binary_slot_recall.append(s in preds[di])

        for s in preds[di]:
            s_domain = s.split("-")[0]
            if eval_domains and s_domain not in eval_domains:
                continue
            final_binary_slot_precision.append(s in gold_final_belief)

        dialog_out["gold_final_belief"] = gold_final_belief
        dialog_out["pred_final_belief"] = {s: v for s, v in preds[di].items()}
        dialogs_out.append(dialog_out)

    if f:
        # print(dialogs_out)
        json.dump(dialogs_out, f)
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
                'belief_state': np.mean(belief_state),
                'final_binary_slot_f1': final_binary_slot_F1,
                'binary_slot_p': P,
                'binary_slot_r': R,
                'binary_slot_f1': binary_slot_F1
            }

    # fix NaNs
    return {k: zero_if_nan(v) for k, v in out_dict.items()}
