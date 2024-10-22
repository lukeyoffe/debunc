import json
import random
from typing import List

import torch
from debate.gen_utils import generate_answer_standard
from debate.gen_utils import (
    Debate,
    Debates,
    RWJSONEncoder,
    construct_assistant_message,
    generate_answer_standard,
)
from debate.gsm8k.common import (
    construct_message_standard,
    format_question,
    read_jsonl,
)
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


if __name__ == "__main__":
    agents = 3
    rounds = 3
    trials = 2

    random.seed(0)
    questions = read_jsonl("./data/test.jsonl")
    random.shuffle(questions)

    all_trial_data: List[Debates] = []
    for trial in trange(trials):
        response_dict: Debates = {}
        all_trial_data.append(response_dict)
        for q_i, data in enumerate(tqdm(questions[:100])):
            if trial == 0 and q_i < 12:
                continue
            question = data["question"]
            answer = data["answer"]
            formatted_question = format_question(question)
            agent_contexts: Debate = [
                [{"role": "user", "content": formatted_question}]
                for agent in range(agents)
            ]

            for round in range(rounds):
                torch.cuda.empty_cache()
                for i, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        agent_contexts_other = (
                            agent_contexts[:i] + agent_contexts[i + 1 :]
                        )
                        message = construct_message_standard(
                            question,
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
                    open(
                        f"gsm_{agents}_{rounds}_{trials}_standard_part2.json",
                        "w",
                    ),
                )
