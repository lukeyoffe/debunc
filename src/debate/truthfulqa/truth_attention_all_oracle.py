import json
from typing import List

import numpy as np
import torch
from debate.gen_utils import (
    Debate,
    Debates,
    RWJSONEncoder,
    construct_assistant_message,
    generate_answer_uncertainty,
)
from debate.truthfulqa.common import (
    construct_message_attention_all,
    gen_question,
)
from debate.truthfulqa.eval_truth import parse_answer
from lm_polygraph.estimators import MeanTokenEntropy, TokenSAR
from models.model import WhiteboxModel
from tqdm import tqdm, trange
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = WhiteboxModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

ue_method = MeanTokenEntropy()


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
                        message = construct_message_attention_all(
                            this_agent=agent_context,
                            this_confidence=confidences[i],
                            other_agents=agent_contexts_other,
                            other_confidences=other_confidences,
                            conv_idx=2 * round - 1,
                            tokenizer=tokenizer,
                        )
                        agent_context.append(message)

                    completion, _ = generate_answer_uncertainty(
                        agent_context, model, tokenizer, ue_method
                    )

                    assistant_message = construct_assistant_message(completion)
                    parsed = parse_answer(completion)
                    assistant_message["confidence"] = 1 if parsed == answer else 1e-5
                    agent_context.append(assistant_message)

                response_dict[question] = (agent_contexts, answer)
                all_trial_data[-1] = response_dict
                json.dump(
                    all_trial_data,
                    open(
                        f"truth_{agents}_{rounds}_{trials}_attention_all_oracle.json",
                        "w",
                    ),
                    cls=RWJSONEncoder,
                )
