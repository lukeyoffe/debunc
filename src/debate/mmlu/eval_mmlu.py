from collections import Counter
import json
import numpy as np
import time
import re
import numpy as np
import scipy.stats as stats

np.set_printoptions(precision=3)


def mean_and_95ci(data):
    # Calculate the mean
    mean = np.mean(data)

    # Calculate the standard error of the mean
    sem = stats.sem(data)

    # Get the t-statistic for 95% confidence interval
    confidence_level = 0.95
    degrees_freedom = len(data) - 1
    t_statistic = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

    # Calculate the margin of error
    margin_of_error = t_statistic * sem

    # Calculate the confidence interval
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    return mean, margin_of_error


def parse_answer(input_str):
    pattern = r"(?i)Answer\s*:\s*([A-D])"
    match = re.search(pattern, input_str)
    if match:
        return match.group(1).upper()

    return None


def compute_accuracy(gt, pred_solutions):
    pred_answers = [
        parse_answer(pred_solution)
        for pred_solution in pred_solutions
        if pred_solution is not None
    ]
    if len(pred_answers) == 0:
        return 0
    pred_answer = most_frequent(pred_answers)

    return 1 if gt == pred_answer else 0


def most_frequent(ls):
    count = Counter(ls)
    most_common_element = count.most_common(1)[0][0]
    return most_common_element


def get_uncertainties(responses, gt):
    correct_uncertainties = []
    incorrect_uncertainties = []
    fail_uncertainties = []
    for agent_responses in responses:
        for agent_response in agent_responses[1::2]:
            parsed = parse_answer(agent_response["content"])
            uncertainty = agent_response["uncertainty"]
            if parsed is None:
                fail_uncertainties.append(uncertainty)
            elif parsed == gt:
                correct_uncertainties.append(uncertainty)
            else:
                incorrect_uncertainties.append(uncertainty)
    return correct_uncertainties, incorrect_uncertainties, fail_uncertainties


def eval(filename, print_uncertainty_stats=True):
    trials = json.load(open(filename, "r"))
    print(len(trials), len(trials[-1]))
    correct_uncertainties = []
    incorrect_uncertainties = []
    fail_uncertainties = []
    accuracies = []
    for response_dict in trials:
        trial_accuracies = []
        for question, (responses, gt) in response_dict.items():
            pred_solutions = [response[-1]["content"] for response in responses]
            if "standard" not in filename and "oracle" not in filename:
                cu, iu, fu = get_uncertainties(responses, gt)
                correct_uncertainties.extend(cu)
                incorrect_uncertainties.extend(iu)
                fail_uncertainties.extend(fu)
            accuracy = compute_accuracy(gt, pred_solutions)
            trial_accuracies.append(accuracy)
        trial_accuracy = np.mean(trial_accuracies)
        accuracies.append(trial_accuracy)

    mean, margin = mean_and_95ci(accuracies)

    print(
        f"{mean:.3f}",
        np.mean(correct_uncertainties),
        np.mean(incorrect_uncertainties),
        np.mean(fail_uncertainties),
    )


def uncertainty_stats(filename):
    trials = json.load(open(filename, "r"))
    correct_uncertainties = []
    incorrect_uncertainties = []
    fail_uncertainties = []
    accuracies = []
    for response_dict in trials:
        trial_accuracies = []
        for question, (responses, gt) in response_dict.items():
            pred_solutions = [response[-1]["content"] for response in responses]
            if "standard" not in filename and "oracle" not in filename:
                cu, iu, fu = get_uncertainties(responses, gt)
                correct_uncertainties.extend(cu)
                incorrect_uncertainties.extend(iu)
                fail_uncertainties.extend(fu)
            accuracy = compute_accuracy(gt, pred_solutions)
            trial_accuracies.append(accuracy)
        trial_accuracy = np.mean(trial_accuracies)
        accuracies.append(trial_accuracy)

    print(
        np.mean(correct_uncertainties),
        np.mean(incorrect_uncertainties),
        np.mean(fail_uncertainties),
    )


if __name__ == "__main__":
    filename = "final/mmlu_3_3_5_prompt_MeanTokenEntropy_0shot.json"
    eval(filename)
