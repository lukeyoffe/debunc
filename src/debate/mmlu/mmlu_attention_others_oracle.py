import json
import os

import torch
from debate.gen_utils import (
    Debate,
    RWJSONEncoder,
    construct_assistant_message,
    generate_answer_uncertainty,
)
from debate.mmlu.common import (
    construct_message_attention_others,
)
from debate.mmlu.eval_mmlu import parse_answer
from lm_polygraph.estimators import MeanTokenEntropy, TokenSAR
from models.model import WhiteboxModel
from tqdm import trange
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

    for num_shots in [0, 5]:
        questions = json.load(open(f"data/qas_{num_shots}_shot.json"))
        filename = f"results/{os.path.basename(__file__)[:-3]}_{agents}_{rounds}_{trials}_{num_shots}_oracle.json"

        all_trial_data = []
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
                q_data = questions[q_i]
                question = q_data["question"]
                answer = q_data["answer"]
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
                            message = construct_message_attention_others(
                                this_agent=agent_context,
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
                        assistant_message["confidence"] = (
                            1 if parsed == answer else 1e-5
                        )
                        agent_context.append(assistant_message)

                    response_dict[question] = (agent_contexts, answer)
                    all_trial_data[-1] = response_dict
                    json.dump(
                        all_trial_data,
                        open(filename, "w"),
                        cls=RWJSONEncoder,
                    )
