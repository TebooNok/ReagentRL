import pyarrow.parquet as pq
import random
import torch
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
import os

import math
import argparse

from utils.utils import set_random_seed, calculate_self_bleu, calculate_avg_pair_distance, get_time

def sample_questions(data_path='nq_questions/train.parquet'):
    global n_questions

    pf = pq.ParquetFile(data_path)
    df = pf.read().to_pandas()

    idxs = []
    for _ in range(n_questions):
        idx = random.randint(0, len(df) - 1)
        while idx in idxs:
            idx = random.randint(0, len(df) - 1)
        idxs.append(idx)

    questions = [df['question'][idx] for idx in idxs]

    return questions

def calculate_action_coverage(actions):
    pass

def test(llm, sampling_params, prompts, log_fn, n_questions=20, n_responses=10, with_cot=True):

    questions = sample_questions()
    print(f'************* Sampled {n_questions} questions *************')
    for i, q in enumerate(questions):
        print(f'{i + 1}: {q}')
    print('************************************************************\n')

    # pbar = tqdm(total=n_questions)
    for i, q in enumerate(questions):
        prompt = prompts[1 - with_cot] + q
        print(f'Prompt {i + 1}: {prompt}')
        
        responses, actions, raw_actions, action_count = [], [], [], []
        # outputs = llm.generate([prompt] * n_responses, sampling_params)
        for j in range(n_responses):
            output = llm.generate(prompt, sampling_params)
            # output = output[0]

            completion_output = output[0].outputs[0]
            response = completion_output.text

            print(f"Response {j + 1}: {response}")
            responses.append(response)

            action_ = re.findall(r'<search>(.*?)</search>', response) # 可能有多个 search 行为
            raw_actions.append(action_)
            if len(action_) == 0: # 没有 search 行为也是一种动作，人为定义成"-"
                action = ["-"]
            action = []
            print('Action: ', end='')
            for a in action_:
                print(a)
                if a not in actions:
                    action.append(a)
            
            actions.extend(action)
            action_count.append(len(actions))

        # 计算条件熵
        conditional_entrophy = 0.
        # reshaped_prompts = [prompts[1] + '<search>' + a + '</search>' for a in actions] # TODO: prompt 是否需要加上问题 q？
        # casual_sampling_params = SamplingParams(
        #         max_tokens=16,
        #         logprobs=1,
        #         prompt_logprobs=1
        #     )
        # outputs = llm.generate(reshaped_prompts, casual_sampling_params)

        for a in actions:
            reshaped_prompt = prompts[1] + '<search>' + a + '</search>'
            casual_sampling_params = SamplingParams(
                max_tokens=16,
                logprobs=1,
                prompt_logprobs=1
            )
            output = llm.generate(reshaped_prompt, casual_sampling_params)

            prompt_logprobs = output[0].prompt_logprobs
            prompt_token_ids = output[0].prompt_token_ids
            # 初始化字典保存 logprob
            prompt_logprob_dict = []
            prompt_logprob_sum = 0
            # 如果 prompt_logprobs 存在，则从中 pop 元素并根据 prompt_token_ids 顺序加入字典
            if prompt_logprobs is not None:
                for token_id in prompt_token_ids[1:]:  # 从第一个 token 后开始
                    # 查找 prompt_logprobs 中是否有这个 token_id 的 logprob
                    found_logprob = None
                    for logprob_dict in prompt_logprobs[1:]:
                        # print(logprob_dict)
                        if token_id in logprob_dict:
                            found_logprob = logprob_dict[token_id]
                            prompt_logprobs.remove(logprob_dict)  # 找到后移除该项
                            break
                        else:
                            prompt_logprobs.remove(logprob_dict)  # 找到后移除该项

                    if found_logprob:
                        prompt_logprob_dict.append({found_logprob.decoded_token:found_logprob.logprob})

                        # 计算 logprob 总和
                        prompt_logprob_sum += found_logprob.logprob
            else:
                prompt_logprob_sum = 0
            
            last_token = ' '
            logprob = 0.
            for p in reversed(prompt_logprob_dict):
                token, logprob_ = list(p.items())[0]
                logprob += logprob_
                if token == '<' and last_token == 'search':
                    break
                last_token = token
            
            conditional_entrophy += - math.exp(logprob) * logprob # TODO: whether regularization?

        self_bleu = calculate_self_bleu(actions)
        # avg_par_distance = calculate_avg_pair_distance(actions, embedding_model)
        # action_coverage = 1
        log_entry = {
            'question': q,
            'responses': responses,
            'raw_actions': raw_actions,
            'actions': actions,
            'action_count': action_count,
            'entrophy': conditional_entrophy,
            'self-BLEU': self_bleu
            # 'avg_paired_distance': avg_par_distance
        }

        with open(log_fn, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print('************************************************************\n\n')
    #     pbar.update(1)
    # pbar.close()

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Test exploration ability of primitive LLM.')
    parse.add_argument('--n_questions', type=int, default=20)
    parse.add_argument('--n_responses', type=int, default=10)
    parse.add_argument('--model', type=str, default='/mnt/hd/huggingface/Qwen/Qwen2.5-3B-Instruct')
    # parse.add_argument('--model', type=str, default='/home/zkh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987')
    
    parse.add_argument('--seed', type=int, default=111)
    parse.add_argument('--no_cot', action='store_true')
    args = parse.parse_args()


    n_questions = args.n_questions
    n_responses = args.n_responses
    model_path = args.model
    seed = args.seed

    set_random_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    prompts = [
        "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: ", # w cot
        "Answer the given question. If you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: ", # w/o cot
    ]
    llm = LLM(model=model_path, device=device, dtype='float16')
    sampling_params = SamplingParams(
        max_tokens=500,  # 最大生成的token数,
        logprobs=1,  # 请求返回 logprobs
        prompt_logprobs=1  # 请求返回 prompt 的 logprobs
    )
    
    log_fn = f"./log/raw_{args.model.split('/')[-1]}-{n_questions}q-{n_responses}r-{'withoutcot' if args.no_cot else 'withcot'}-{seed}-{get_time()}.jsonl"
    # print(log_fn)
    test(llm, sampling_params, prompts, log_fn, n_questions, n_responses, not args.no_cot)

    del llm

    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', trust_remote_code=True, cache_folder='/mnt/hd/multilingual-e5-large-instruct')

    with open(log_fn, 'r', encoding='utf-8') as infile, \
        open(log_fn.split('raw_')[0]  + log_fn.split('raw_')[1], 'w', encoding='utf-8') as outfile:

        for line in infile:
            entry = json.loads(line)

            avg_paired_distance = calculate_avg_pair_distance(entry['actions'], embedding_model)
            entry['avg_paired_distance'] = avg_paired_distance

            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

    os.remove(log_fn)
