import json
import os
import argparse
import operator

path1 = 'statenet/'


parser = argparse.ArgumentParser(description='Input domain specification')
parser.add_argument('domain', 
                    help='The domain you input should correspond to a folder in statenet_preds/')
             

args = parser.parse_args()
print(args.domain)
cur_path = path1 +'/'+ args.domain


def write_output_file(accuracies, path):
    highest = accuracies[max(accuracies.items(), key=operator.itemgetter(1))[0]]
    best_epoch = max(accuracies.items(), key=operator.itemgetter(1))[0]

    output = open(path +"/"+'revised_accuracy.txt', 'w')


    for key, value in accuracies.items():
        output.write('\n')
        output.write(key + " : " + str(value))


    output.write('\n')
    output.write('\n')
    output.write("Best epoch: " + best_epoch + " : " + str(highest))

    output.close()
    
    return print("Done outputting file to: %s" %path)

def compute_acc(path):
    accuracies = {}
    for item in os.listdir(path):
        if "prediction_dv" in item:
            
            f = open(path+ "/" +item, 'r')
            preds = json.load(f)
            count = 0
            correct = 0
            for i in range(len(preds)):
                for turn in preds[i]['turns']:
                    ref = list(turn['gold'].items())
                    pred = list(turn['pred'].items())

                    if ref == [] :
                        if pred == []:
                            count = count + 1
                            correct = correct + 1
                        else:
                            count = count + 1
                    else:
                        
                        intersection = list(set(ref).intersection(pred))

                        count = count + len(ref)
                        correct = correct + len(intersection)
            accuracies[item]= correct/count
            
    write_output_file(accuracies, path)
    return accuracies

#compute
print("Computing accuracies for " , args.domain)
compute_acc(cur_path)