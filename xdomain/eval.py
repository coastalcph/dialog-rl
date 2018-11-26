import numpy as np


def evaluate_preds(dialogs, preds, turn_predictions):
    inform = []
    joint_goal = []
    belief_state = []
    binary_slot_precision = []
    binary_slot_recall = []
    for di, d in enumerate(dialogs):

        for ti, turn in enumerate(d.turns):
            turn_gold = turn.labels

            gold_inform = {s: int(np.argmax(v)) for s, v in turn_gold.items()}
            pred_inform = {s: v for s, v in turn_predictions[di][ti].items()}

            for s, v in gold_inform.items():
                s_in_pred_inform = s in pred_inform
                binary_slot_recall.append(s_in_pred_inform)
                if s_in_pred_inform:
                    inform.append(v == pred_inform[s])
                else:
                    inform.append(False)

            for s in pred_inform:
                binary_slot_precision.append(s in gold_inform)

        # evaluate final dialog-level performance
        gold_final_belief = {b['slots'][0]: b['slots'][1]
                             for b in d.turns[-1].belief_state}
        for s, v in gold_final_belief.items():
            if s in preds[di]:
                belief_state.append(v == preds[di][s])

    R = np.mean(binary_slot_recall)
    P = np.mean(binary_slot_precision)
    binary_slot_F1 = 2*R*P / (R+P)
    joint_goal = inform
    return {
                # 'turn_inform': np.mean(inform),
                # 'turn_request': np.mean(request),
                'joint_goal': np.mean(joint_goal),
                'dialog_inform': np.mean(belief_state),
                'binary_slot_p': P,
                'binary_slot_r': R,
                'binary_slot_f1': binary_slot_F1
            }
