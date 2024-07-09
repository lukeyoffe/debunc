import json
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np
from lm_polygraph.utils.manager import estimate_uncertainty
from models.common import RangeWeight
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import NotRequired


class Message(TypedDict):
    role: str
    content: str
    uncertainty: NotRequired[float]
    range_weights: NotRequired[List[RangeWeight]]
    confidence: NotRequired[float]


Conversation = List[Message]

Debate = List[Conversation]

Debates = Dict[str, Tuple[Debate, Any]]


def unc_to_confidence(uncertainties: np.ndarray):
    confidences = 1 / uncertainties
    confidences = confidences * 14 / np.sum(confidences) + 1 / uncertainties.shape[0]
    confidences = np.clip(confidences, 1, 10)
    confidences = np.round(confidences).astype(int)
    return confidences


def get_len(
    context: Conversation, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
):
    return (
        len(
            tokenizer.apply_chat_template(
                context, tokenize=True, add_generation_prompt=True
            )
        )
        - 4
    )


def construct_assistant_message(completion) -> Message:
    return {"role": "assistant", "content": completion}


def generate_answer_standard(answer_context, model, tokenizer) -> str:
    prompt = tokenizer.apply_chat_template(
        answer_context, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs.to("cuda")
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=1024,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    del input_ids
    output = tokenizer.decode(outputs[0, inputs.shape[1] : -1])
    return output


def generate_answer_uncertainty(
    answer_context, model, tokenizer, ue_method
) -> Tuple[str, float]:
    prompt = tokenizer.apply_chat_template(
        answer_context, tokenize=False, add_generation_prompt=True
    )
    extra = {}
    if "range_weights" in answer_context[-1]:
        extra["range_weights"] = answer_context[-1]["range_weights"]

    out = estimate_uncertainty(model, ue_method, input_text=prompt, **extra)
    return out.generation_text, out.uncertainty


class RWJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, RangeWeight):
            return obj.__dict__
        return super().default(obj)
