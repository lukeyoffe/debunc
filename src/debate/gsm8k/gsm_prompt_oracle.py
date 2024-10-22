import json
import random
from typing import List

import torch
from debate.gen_utils import (
    Debate,
    Debates,
    RWJSONEncoder,
    construct_assistant_message,
    generate_answer_uncertainty,
)
from debate.gsm8k.common import (
    construct_message_prompt,
    format_question,
    read_jsonl,
)
from debate.gsm8k.eval_gsm import parse_answer
from lm_polygraph.estimators import MeanTokenEntropy, TokenSAR
from lm_polygraph.utils.manager import estimate_uncertainty
from models.model import WhiteboxModel
from tqdm import trange
from transformers import AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = WhiteboxModel.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

ue_method = MeanTokenEntropy()


if __name__ == "__main__":
    agents = 3
    rounds = 3
    trials = 5

    random.seed(0)
    questions = read_jsonl("./data/test.jsonl")
    random.shuffle(questions)
    questions = questions[:100]

    filename = f"gsm_{agents}_{rounds}_{trials}_prompt_oracle.json"
    all_trial_data: List[Debates] = []

    current_trial = 0
    for trial in trange(trials):
        current_question = 0
        response_dict = {}
        all_trial_data.append(response_dict)

        for q_i in trange(
            current_question,
            len(questions),
            initial=current_question,
            total=len(questions),
        ):
            data = questions[q_i]
            question = data["question"]
            answer = data["answer"]
            formatted_question = format_question(question)
            agent_contexts: Debate = [
                [{"role": "user", "content": formatted_question}]
                for agent in range(agents)
            ]

            for round in range(rounds):
                torch.cuda.empty_cache()
                confidences = None
                if round != 0:
                    confidences = []
                    for agent in agent_contexts:
                        confidences.append(agent[-1]["confidence"])
                for i, agent_context in enumerate(agent_contexts):
                    if confidences is not None:
                        agent_contexts_other = (
                            agent_contexts[:i] + agent_contexts[i + 1 :]
                        )
                        other_confidences = confidences[:i] + confidences[i + 1 :]
                        message = construct_message_prompt(
                            question,
                            other_agents=agent_contexts_other,
                            other_confidences=other_confidences,
                            conv_idx=2 * round - 1,
                        )
                        agent_context.append(message)

                    completion, _ = generate_answer_uncertainty(
                        agent_context, model, tokenizer, ue_method
                    )

                    assistant_message = construct_assistant_message(completion)
                    parsed = parse_answer(completion)

                    gt = float(answer.replace(",", "").split("#### ")[1])
                    assistant_message["confidence"] = 10 if parsed == gt else 1

                    agent_context.append(assistant_message)

                response_dict[question] = (agent_contexts, answer)
                all_trial_data[-1] = response_dict
                json.dump(
                    all_trial_data,
                    open(filename, "w"),
                )
