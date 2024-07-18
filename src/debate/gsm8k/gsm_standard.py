import json
import random
from typing import List

import torch
from debate.gen_utils import generate_answer_standard
from debate.gsm8k.common import (
    Conversation,
    Debate,
    Debates,
    Message,
    format_question,
    read_jsonl,
)
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


def construct_message(
    question: str,
    other_agents: List[Conversation],
    conv_idx: int,
) -> Message:
    prefix_string = "These are solutions to the problem from other agents: "

    for agent in other_agents:
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

    prefix_string += f"""

Based off the opinion of other agents, can you provide an updated response? The original problem is:

{question}

The last line of your response should be of the following format: 'Answer: $INTEGER' (without quotes) where INTEGER is the integer answer."""

    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion) -> Message:
    return {"role": "assistant", "content": completion}


if __name__ == "__main__":
    agents = 3
    rounds = 3
    trials = 5

    random.seed(0)
    questions = read_jsonl("./data/test.jsonl")
    random.shuffle(questions)

    all_trial_data: List[Debates] = []
    for trial in trange(trials):
        response_dict: Debates = {}
        all_trial_data.append(response_dict)
        for q_i, data in enumerate(questions[:100]):
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
                        message = construct_message(
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
                        f"gsm_{agents}_{rounds}_{trials}_standard.json",
                        "w",
                    ),
                )
