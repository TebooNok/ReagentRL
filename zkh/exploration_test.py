from datasets import load_from_disk, Dataset
import random
import torch
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
import os
from transformers import AutoTokenizer
import argparse

from utils.utils import set_random_seed, calculate_action_coverage, calculate_self_bleu, calculate_entropy, calculate_avg_pair_distance, get_time, set_timer, end_timer


def sample_questions(dataset_path='dataset/questions', n_questions=20):
    data = load_from_disk(dataset_path)['question']

    idxs = list(range(len(data)))
    random.shuffle(idxs)
    idxs = idxs[:n_questions]

    questions = [data[idx] for idx in idxs]

    return questions


def test(llm,
         tokenizer,
         sampling_params,
         prompts,
         embedding_model,
         log_fn,
         dataset_path='dataset/questions',
         n_questions=20,
         with_cot=True,
         verbose=1):

    questions = sample_questions(dataset_path, n_questions)
    print(f'************* Sampled {n_questions} questions *************')
    if n_questions > 50:
        for i in range(10):
            print(f'{i + 1}: {questions[i]}')
        print('......')
        for i in range(n_questions - 10, n_questions):
            print(f'{i + 1}: {questions[i]}')
    else:
        for i, q in enumerate(questions):
            print(f'{i + 1}: {q}')
    print('************************************************************\n')

    ### generate all responses
    inputs = [prompts[1 - with_cot] + q for q in questions]
    outputs = llm.generate(inputs, sampling_params)

    qa_cot, qa = {}, {}
    q_groups = []
    log_data = []
    pbar = tqdm(total=n_questions)
    for i, output in enumerate(outputs):

        ### primary indicators
        thinks, actions, raw_actions, action_count = [], [], [], []
        record = []
        for j, info in enumerate(output.outputs):
            response = info.text
            # print(f"Response {j + 1}: {response}")
            # responses.append(response)

            loc = response.find("<search>")  # 在第一个查询之前，都是 think 过程
            if loc != -1:
                thinks.append(response[:loc])
            else:
                loc = response.find(
                    "<answer>")  # llm 没有做查询，直接输出答案，在输出答案之前，都属于 think 过程
                if loc != -1:
                    thinks.append(response[:loc])
                else:
                    thinks.append(response)

            action = re.findall(r'<search>(.*?)</search>',
                                response)  # 可能有多个 search 行为
            if len(action) == 0:
                raw_actions.append("-")
            else:
                raw_actions.append(
                    action[0].strip())  # 不考虑多轮 reasoning，只取第一个 action

            if raw_actions[-1] not in actions and raw_actions[-1] != "-":
                actions.append(raw_actions[-1])
                record.append(j)

            action_count.append(len(actions))

        print(
            f"\033[92mThere are {len(actions)} different queries for question {i + 1}.\033[0m"
        )
        if verbose > 0:
            print(f"[Question {i + 1}] {questions[i]}")
            for a in actions:
                print(f"<{a}>", end='')
            print('')
            print('-' * 60)
            for t in thinks[:2]:
                print(f"|{t}|", end='')
            print('......')
            print('-' * 60)
        
        ### action coverage
        n_docs = [0 for _ in range(len(output.outputs))]
        docs_detail = calculate_action_coverage(actions)
        for j, num_docs in zip(record, docs_detail):
            n_docs[j] = num_docs

        group_len = 0
        for idx, a in zip(record, actions):
            key = f"{prompts[0]}{questions[i]}{thinks[idx]}"
            # if key in qa_cot.keys():
            #     qa_cot[key].append(a)
            # else:
            #     qa_cot[key] = [a]
            if key not in qa_cot.keys():
                group_len += 1
            qa_cot[key] = [f"<search> {a} </search>"]
        q_groups.append(group_len)
        
        if verbose > 0:
            print(f"[action count] {action_count[-1]} ==> [group len] {group_len}")

        qa[f"{prompts[1]}{questions[i]} "] = [
            f"<search> {a} </search>" for a in actions
        ]  # without cot

        self_bleu = calculate_self_bleu(actions)

        log_entry = {
            'question': questions[i],
            'thinks': thinks,
            'raw_actions': raw_actions,
            'actions': actions,
            'action_count': action_count,
            'action_coverage': n_docs,
            'self-BLEU': self_bleu,

            # 中间变量 TODO: 中间结果写入临时文件
            # 'group_len': group_len,
            # 'qa_cot': qa_cot,
            # 'qa': qa
        }

        log_data.append(log_entry)

        print(
            '************************************************************\n\n')
        pbar.update(1)

    pbar.close()

    print("Calculating entrophy...")
    st_t = set_timer()
    entropys_cot = calculate_entropy(qa_cot, llm, tokenizer,
                                     q_groups)  # with cot
    entropys = calculate_entropy(qa, llm, tokenizer)  # without cot
    print(f"Used {end_timer(st_t)}.")

    # TODO: del llm

    print("Calculating average paired distance...")
    st_t = set_timer()
    all_actions = [a for vals in qa.values() for a in vals]
    a_groups = [len(vals) for vals in qa.values()]
    embedding_dists = calculate_avg_pair_distance(all_actions, a_groups, embedding_model)
    print(f"Calculating avg paired distance used {end_timer(st_t)}.")
    
    # with open(log_fn, "a", encoding="utf-8") as fo:
    for log_entry, entropy_cot, entropy, embedding_dist in zip(log_data, entropys_cot,
                                                   entropys, embedding_dists):
        log_entry['entropy_cot'] = entropy_cot
        log_entry['entropy'] = entropy
        log_entry['avg_paired_distance'] = embedding_dist
            # fo.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    data = Dataset.from_list(log_data)
    data.save_to_disk(log_fn)

if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description='Test exploration ability of primitive LLM.')
    parse.add_argument('--n_questions', type=int, default=20)
    parse.add_argument('--n_responses', type=int, default=10)

    parse.add_argument('--dataset', type=str, default='dataset/questions')
    parse.add_argument('--model',
                       type=str,
                       default='/home/hwai/weights/Qwen2.5-3B-Instruct')
    parse.add_argument('--embedding_model',
                       type=str,
                       default='/home/hwai/weights/e5')

    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--no_cot', action='store_true')
    parse.add_argument('--verbose', type=int, default=1)
    parse.add_argument('--gpu_utilization', type=float, default=0.4)
    args = parse.parse_args()

    st_time = set_timer()

    n_questions = args.n_questions
    n_responses = args.n_responses
    model_path = args.model
    seed = args.seed

    set_random_seed(seed)

    device = torch.device('cuda:0')

    prompts = [
        "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: ",  # w cot
        "Answer the given question. If you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: ",  # w/o cot
    ]

    llm = LLM(model=model_path,
              device=device,
              dtype='bfloat16',
              seed=args.seed,
              gpu_memory_utilization=args.gpu_utilization)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(
        max_tokens=256,  # 最大生成的token数,
        temperature=0.8,
        top_p=0.9,
        # logprobs=1,  # 请求返回 logprobs
        # prompt_logprobs=1,  # 请求返回 prompt 的 logprobs
        seed=args.seed,
        n=args.n_responses  # 对每一个 prompt，生成 n 个 response
    )
    embedding_model = SentenceTransformer(args.embedding_model, device=device)

    log_fn = f"./log/{args.model.split('/')[-1]}-{n_questions}q-{n_responses}r-{'withoutcot' if args.no_cot else 'withcot'}-{seed}-{get_time()}"
    # print(log_fn)
    test(llm=llm,
         tokenizer=tokenizer,
         sampling_params=sampling_params,
         prompts=prompts,
         embedding_model=embedding_model,
         log_fn=log_fn,
         dataset_path=args.dataset,
         n_questions=n_questions,
         with_cot=not args.no_cot,
         verbose=args.verbose)

    del llm, embedding_model
    with torch.cuda.device('cuda:0'):  # 指定要操作的设备
        torch.cuda.empty_cache()

    used_time = end_timer(st_time)
    print(f"\033[92m{n_questions} questions * {n_responses} responses ==> {used_time}\033[0m")
    print('\nTo plot:')
    print(f"cd utils\npython plot.py --fn=.{log_fn}")
