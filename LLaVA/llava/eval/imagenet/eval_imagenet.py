import os
import argparse
import json



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default=None)
    return parser.parse_args()



def eval_single(result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    # annotations = json.load(open(annotation_file))
    results = [json.loads(line) for line in open(result_file)]
    total = len(results)
    right = 0
    for result in results:
        ground_truth = result['reference']

        if result['text'].lower() == ground_truth.lower():
            right += 1


    acc = 100. * right / total
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(results), acc))
    

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.result_file)