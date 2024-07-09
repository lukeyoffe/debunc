from collections import Counter

import numpy as np
from scipy import stats


def mean_and_95ci(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    confidence_level = 0.95
    degrees_freedom = len(data) - 1
    t_statistic = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_of_error = t_statistic * sem
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    return mean, (ci_lower, ci_upper)


def most_frequent(ls):
    count = Counter(ls)
    most_common_element = count.most_common(1)[0][0]
    return most_common_element


def get_uncertainties(responses, gt, parse_answer):
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


def get_uncertainties_round(responses, gt, parse_answer):
    correct_uncertainties = [[], [], []]
    incorrect_uncertainties = [[], [], []]
    fail_uncertainties = [[], [], []]
    for agent_responses in responses:
        for i, agent_response in enumerate(agent_responses[1::2]):
            parsed = parse_answer(agent_response["content"])
            uncertainty = agent_response["uncertainty"]
            if parsed is None:
                fail_uncertainties[i].append(uncertainty)
            elif parsed == gt:
                correct_uncertainties[i].append(uncertainty)
            else:
                incorrect_uncertainties[i].append(uncertainty)
    return correct_uncertainties, incorrect_uncertainties, fail_uncertainties
