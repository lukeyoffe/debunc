import json
import re

import numpy as np

from debate.eval_utils import (
    get_uncertainties,
    get_uncertainties_round,
    mean_and_95ci,
    most_frequent,
)
from lm_polygraph import WhiteboxModel

np.set_printoptions(precision=3)


def parse_answer(input_string):
    pattern = r"\d+(?:\.\d+)?"
    matches = re.findall(pattern, input_string)
    if not matches:
        return None

    return round(float(matches[-1]))


def compute_accuracy(gt, pred_solutions):
    pred_answers = [
        parse_answer(pred_solution)
        for pred_solution in pred_solutions
        if pred_solution is not None
    ]
    if len(pred_answers) == 0:
        return 0

    pred_answer = most_frequent(pred_answers)
    if pred_answer is None:
        return 0
    return 1 if gt == float(pred_answer) else 0


def eval(filename):
    trials = json.load(open(filename, "r"))
    accuracies = []
    for response_dict in trials:
        trial_accuracies = []
        for question, (responses, gt) in response_dict.items():
            gt = float(gt)
            pred_solutions = [response[-1]["content"] for response in responses]
            accuracy = compute_accuracy(gt, pred_solutions)
            trial_accuracies.append(accuracy)
        trial_accuracy = np.mean(trial_accuracies)
        accuracies.append(trial_accuracy)

    mean, ci = mean_and_95ci(accuracies)

    print(
        f"{mean:.3f}",
        f"{ci[0]:.3f}",
        f"{ci[1]:.3f}",
    )


def uncertainty_stats(filename):
    np.set_printoptions(precision=3)
    trials = json.load(open(filename, "r"))
    print(len(trials[-1]))
    correct_uncertainties = []
    incorrect_uncertainties = []
    fail_uncertainties = []
    for response_dict in trials:
        for question, (responses, gt) in response_dict.items():
            if "standard" not in filename:
                cu, iu, fu = get_uncertainties(responses, gt, parse_answer)
                correct_uncertainties.extend(cu)
                incorrect_uncertainties.extend(iu)
                fail_uncertainties.extend(fu)

    print(
        np.mean(correct_uncertainties),
        np.mean(incorrect_uncertainties + fail_uncertainties),
        np.mean(incorrect_uncertainties + fail_uncertainties)
        - np.mean(correct_uncertainties),
    )
    return correct_uncertainties, incorrect_uncertainties + fail_uncertainties


def get_stats(filename):
    trials = json.load(open(filename, "r"))
    correct_uncertainties = [[], [], []]
    incorrect_uncertainties = [[], [], []]
    fail_uncertainties = [[], [], []]
    accuracies = []
    for response_dict in trials:
        trial_accuracies = []
        for question, (responses, gt) in response_dict.items():
            pred_solutions = [response[-1]["content"] for response in responses]
            if "standard" not in filename and "perfect" not in filename:
                cu, iu, fu = get_uncertainties_round(responses, gt, parse_answer)
                for i in range(3):
                    correct_uncertainties[i].extend(cu[i])
                    incorrect_uncertainties[i].extend(iu[i])
                    fail_uncertainties[i].extend(fu[i])

            accuracy = compute_accuracy(gt, pred_solutions)
            trial_accuracies.append(accuracy)
        trial_accuracy = np.mean(trial_accuracies)
        accuracies.append(trial_accuracy)
    mean, ci = mean_and_95ci(accuracies)

    return {
        "mean_accuracy": mean,
        "correct_uncertainties": correct_uncertainties,
        "incorrect_uncertainties": incorrect_uncertainties,
        "fail_uncertainties": fail_uncertainties,
    }
