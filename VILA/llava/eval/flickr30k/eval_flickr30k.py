import argparse
import json
import os
import re

from llava.eval.m4c_evaluator import Flickr30kCiderEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    pred_list = []
    # load annotations (jsonline file)
    with open(args.result_file, 'r') as f:
        for line in f:
            ann = json.loads(line)
            pred_list.append({
                "gt_answers": ann['reference'],
                "pred_answer": ann['text'],
            })

    # # load results
    # results = json.load(open(args.result_file))
    # pred_list = [{
    #     "gt_answers": result['reference'],
    #     "pred_answer": result['pred'],
    # } for result in results]

    # evaluate
    evaluator = Flickr30kCiderEvaluator()
    score = evaluator.eval_pred_list(pred_list)
    print(f"Flickr30k Cider score: {score:.4}")