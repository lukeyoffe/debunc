import json
from typing import Dict, List, Tuple, TypedDict

import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import NotRequired


class RangeWeight(TypedDict):
    start: int
    end: int
    weight: float


class Message(TypedDict):
    role: str
    content: str
    uncertainty: NotRequired[float]
    range_weights: NotRequired[List[RangeWeight]]
    confidence: NotRequired[float]


Conversation = List[Message]

Debate = List[Conversation]

Debates = Dict[str, Tuple[Debate, str]]


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


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


START_PREFIX_STD = "These are solutions to the problem from other agents: "
START_PREFIX_PROMPT = "These are solutions and confidence values from 1 to 10 (higher means more confident) to the problem from other agents: "
LAST_LINE_INST = "The last line of your response should be of the following format: 'Answer: $INTEGER' (without quotes) where INTEGER is the integer answer."


def get_end_prefix_std(question: str) -> str:
    return f"""

Based off the opinion of other agents, can you provide an updated response? The original problem is:

{question}

{LAST_LINE_INST}"""


def get_end_prefix_no_conf(question: str) -> str:
    return f"""

Based off the opinion of other agents, can you provide an updated response? The original problem is:

{question}

Do not mention your confidence. {LAST_LINE_INST}"""


def construct_message_standard(
    question: str,
    other_agents: List[Conversation],
    conv_idx: int,
) -> Message:
    prefix_string = START_PREFIX_STD

    for agent in other_agents:
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

    prefix_string += get_end_prefix_std(question)
    return {"role": "user", "content": prefix_string}


def construct_message_prompt_no_conf(
    question: str,
    other_agents: List[Conversation],
    other_confidences: np.ndarray,
    conv_idx: int,
) -> Message:
    prefix_string = START_PREFIX_PROMPT

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution (confidence level is {confidence}): ```{agent_response}```"

    prefix_string += get_end_prefix_no_conf(question)
    return {"role": "user", "content": prefix_string}


def construct_message_attention_all(
    question: str,
    this_agent: Conversation,
    this_confidence: float,
    other_agents: List[Conversation],
    other_confidences: List[float],
    conv_idx: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Message:
    range_weights = []
    this_agent = this_agent.copy()
    len_before_prev = get_len(this_agent[:-1], tokenizer) + 4
    current_len = get_len(this_agent, tokenizer) + 3
    range_weights.append(RangeWeight(len_before_prev, current_len, this_confidence))

    prefix_string = START_PREFIX_STD

    this_agent.append({"role": "user", "content": prefix_string})
    current_len = get_len(this_agent, tokenizer)

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

        this_agent[-1] = {"role": "user", "content": prefix_string}
        new_len = get_len(this_agent, tokenizer)
        range_weights.append(RangeWeight(current_len, new_len, confidence))
        current_len = new_len

    prefix_string += get_end_prefix_std(question)
    return {"role": "user", "content": prefix_string, "range_weights": range_weights}


def construct_message_attention_others(
    question: str,
    this_agent: Conversation,
    other_agents: List[Conversation],
    other_confidences: np.ndarray,
    conv_idx: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Message:
    this_agent = this_agent.copy()
    this_agent.append({})

    prefix_string = START_PREFIX_STD

    this_agent[-1] = {"role": "user", "content": prefix_string}
    current_len = get_len(this_agent, tokenizer)

    range_weights = []
    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\n One agent solution: ```{agent_response}```"

        this_agent[-1] = {"role": "user", "content": prefix_string}
        new_len = get_len(this_agent, tokenizer)
        range_weights.append(RangeWeight(current_len, new_len, confidence))
        current_len = new_len

    prefix_string += get_end_prefix_std(question)
    return {"role": "user", "content": prefix_string, "range_weights": range_weights}


def format_question(question, eight_shot=False) -> str:
    if eight_shot:
        return format_question_few_shot(question)
    formatted_question = f"""
Answer the following math problem. The last line of your response should be of the following format: 'Answer: $INTEGER' (without quotes) where INTEGER is the integer answer. Think step by step before answering.

{question}
""".strip()

    return formatted_question


def format_question_few_shot(question) -> str:
    formatted_question = f"""
I am going to give you a series of demonstrations of math problems and solutions. When you respond, respond only with the solution of the final problem.  The last line of your response should be of the following format: 'Answer: $INTEGER' (without quotes) where INTEGER is the integer answer. Think step by step before answering.

Examples:

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Answer: 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Answer: 5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Answer: 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Answer: 8

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Answer: 9

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Answer: 29

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Answer: 33

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
Answer: 8

Your task:

{question}
""".strip()

    return formatted_question


class RWJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, RangeWeight):
            return obj.__dict__
        return super().default(obj)
