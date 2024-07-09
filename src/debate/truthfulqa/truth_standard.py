import os
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json

import numpy as np
import torch
from common import (
    construct_message_standard,
    gen_question,
)
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen_utils import (
    Debate,
    Debates,
    construct_assistant_message,
    generate_answer_standard,
)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    )
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
        torch_dtype=torch.bfloat16,
)

if __name__ == "__main__":
    agents = 3
    rounds = 3
    trials = 5
    questions = 100

    np.random.seed(0)
    question_data = json.load(open("data/prepared.json"))
    all_trial_data: List[Debates] = []
    for trial in trange(trials):
        response_dict: Debates = {}
        all_trial_data.append(response_dict)
        for i, q_data in enumerate(tqdm(question_data)):
            question, answer = gen_question(q_data)
            agent_contexts: Debate = [
                [{"role": "user", "content": question}] for agent in range(agents)
            ]

            for round in range(rounds):
                torch.cuda.empty_cache()
                for i, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        agent_contexts_other = (
                            agent_contexts[:i] + agent_contexts[i + 1 :]
                        )
                        message = construct_message_standard(
                            other_agents=agent_contexts_other,
                            conv_idx=2 * round - 1,
                        )
                        agent_context.append(message)

                    completion = generate_answer_standard(
                        agent_context, model, tokenizer
                    )

                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)

                response_dict[question] = (agent_contexts, answer)
                all_trial_data[-1] = response_dict
                json.dump(
                    all_trial_data,
                    open(f"truth_{agents}_{rounds}_{trials}_standard.json", "w"),
                )
