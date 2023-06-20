import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import LLaMA, ModelArgs, Tokenizer, Transformer

from lz_77 import lz77_encode
from scienceqa.base_prompt import *


def load_data(args):
    problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    qids = pid_splits['%s' % (args.test_split)]
    qids = qids[:args.test_number] if args.test_number > 0 else qids
    print(f"number of test problems: {len(qids)}\n")

    # pick up shot examples from the training set
    shot_qids = args.shot_qids
    train_qids = pid_splits['train']
    if shot_qids == None:
        assert args.shot_number >= 0 and args.shot_number <= 32
        shot_qids = random.sample(train_qids, args.shot_number)  # random sample
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            assert qid in train_qids  # check shot_qids
    print("training question ids for prompting: ", shot_qids, "\n")

    return problems, qids, shot_qids


def get_model_result(prompts, args, generator):
    output = generator.generate(prompts, max_gen_len=512, temperature=0.1, top_p=0.75)[0]

    # extract the answer
    pattern = re.compile(r'The answer is ([A-Z]).')
    res = pattern.findall(output)
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"

    return answer, output


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


def get_result_file(args):
    result_file = "{}/{}/{}_{}_{}_{}_seed_{}.json".format(args.output_root, args.model, args.label, args.test_split,
                                                          args.prompt_format, args.shot_number, args.seed)

    return result_file


def save_results(result_file, acc, correct, count, shot_qids, args, results, outputs):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['shot_qids'] = shot_qids
    data['args'] = vars(args)
    data['results'] = results
    data['outputs'] = outputs

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))


def parse_args():
    class Arguments:
        def __init__(self):
            self.data_root = '../ScienceQA/data/scienceqa'
            self.output_root = '../ScienceQA/results'
            self.caption_file = '../ScienceQA/data/captions.json'
            self.model = 'llama7b'
            self.options = ["A", "B", "C", "D", "E"]
            # user options
            self.label = 'exp0'
            self.test_split = 'val'
            self.test_number = 120
            self.use_caption = False
            self.save_every = 10
            self.debug = True
            self.prompt_format = 'QCM-A'
            self.shot_number = 0
            self.shot_qids = None
            self.seed = 10
            # GPT-3 settings
            self.engine = 'text-davinci-002'
            self.temperature = 0.0
            self.max_tokens = 512
            self.top_p = 1.0
            self.frequency_penalty = 0.0
            self.presence_penalty = 0.0

    args = Arguments()
    return args


def science_qa_main(generator):
    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    problems, qids, shot_qids = load_data(args)  # probelms, test question ids, shot example ids

    result_file = get_result_file(args)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(result_file))
        acc = check_point['acc']
        correct = check_point['correct']
        results = check_point['results']
        outputs = check_point['outputs']
        print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%")
    else:
        correct = 0
        results = {}
        outputs = {}

    # for qid in tqdm(qids):
    for i, qid in enumerate(qids):
        if qid in results:
            continue

        choices = problems[qid]["choices"]
        answer = problems[qid]["answer"]  # 0, 1, ..., 4
        label = args.options[answer]  # 'A', ..., 'E'

        # generate prompt
        prompt = build_prompt(problems, shot_qids, qid, args)

        # prompts = [PROMPT_DICT["prompt_no_input"].format_map({"instruction": x, "input": ""}) for x in [prompt]]
        # prompts = [lz77_encode(x) for x in prompts]
        prompts = [lz77_encode(prompt)]

        # skip prompts bigger than 510 tokens
        if len(prompts[0]) > 510:
            print("skipped", len(prompts[0]))
            continue
        else:
            print("all good", len(prompts[0]))

        # generate prediction
        prediction, output = get_model_result(prompts, args, generator)  # 'A', ..., 'E'
        pred_idx = get_pred_idx(prediction, choices, args.options)  # 0, 1, ..., 4

        results[qid] = pred_idx
        outputs[qid] = output
        if pred_idx == answer:
            correct += 1

        acc = correct / len(results) * 100

        if args.debug or i < 3:
            print("##################################")
            print(prompt, "\n")
            print("# labeled answer:", label)
            print("# predicted answer:", prediction)
            print("# predicted index:", pred_idx)
            print("# predicted output:", output)

        if (i + 1) % args.save_every == 0 or (i + 1) == len(qids):
            print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
            save_results(result_file, acc, correct, i + 1, shot_qids, args, results, outputs)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(adapter_checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    temperature: float = 0.1,
    top_p: float = 0.75,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size)
    science_qa_main(generator)


if __name__ == "__main__":
    fire.Fire(main)