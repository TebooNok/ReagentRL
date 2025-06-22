import random
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from time import strftime, localtime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import requests
# from vllm import SamplingParams
import time
from datasets import load_from_disk

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_time():
    return strftime('%Y-%m-%d-%H-%M', localtime())

def set_timer():
    return time.time()

def end_timer(st_time):
    cur_time = time.time()
    total_seconds = int(cur_time - st_time)
    
    S = total_seconds % 60
    M = total_seconds // 60
    if M < 60:
        return f"{M} min {S} s"
    else:
        H = M // 60
        M = M % 60
        return f"{H} hour {M} min {S} s"

### basic util
def retrieve_documents(queries, topk=3):
    url = "http://124.220.175.42:8000/retrieve"
    headers = {"Content-Type": "application/json"}
    
    # 构造请求数据
    payload = {
        "queries": queries,  # 可以是单个字符串或字符串列表
        "topk": topk,
        "return_scores": True
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # 检查请求是否成功
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

### indicator
def calculate_self_bleu(texts, n_gram=3):
    if len(texts) <= 1:
        return 1
    
    scores = []
    for i in range(len(texts)):
        hypothesis = texts[i].split()
        references = [text.split() for j, text in enumerate(texts) if j != i]
        # 使用平滑函数避免0分（如SmoothingFunction.method1）
        smooth = SmoothingFunction().method1
        score = sentence_bleu(references, hypothesis, weights=(1./n_gram,) * n_gram,
                              smoothing_function=smooth)
        scores.append(score)
    return sum(scores) / len(scores)

def calculate_avg_pair_distance(texts, group_list, embedding_model):
    all_embeddings = embedding_model.encode(texts, batch_size=1024, show_progress_bar=True)
    
    st = 0
    distances = []
    for group_len in group_list:
        ed = st + group_len
        
        embeddings = all_embeddings[st: ed, :]
        
        cos_sim = cosine_similarity(embeddings)
        upper_triangle = np.triu(cos_sim, k = 1)
        if np.prod(upper_triangle.shape) == 1:
            distances.append(1.0)
        else:
            distances.append(float(np.sum(upper_triangle)) * 2 / (group_len * (group_len - 1)))
    
        st = ed
    
    return distances

def calculate_action_coverage(texts, topk=3):
    results = retrieve_documents(texts, topk)
    
    ids = []
    for i, r in enumerate(results['result']):
        # print(f"Response for query {i + 1}:")
        # print(f"{d['document']['id']}: {d['document']['contents']}")
        ids.append([d['document']['id'] for d in r])
        # print('*' * 60)
    ### ids = [[1, 2, 3], [2, 3, 4],...]
    record = set()
    for i in range(0, len(ids)):
        dedup_docs = [doc for doc in ids[i] if doc not in record]
        
        ids[i] = dedup_docs
        for doc in dedup_docs:
            record.add(doc)

    return [len(docs) for docs in ids]

def compute_sentence_token_len(xs, tokenizer):
    mm = tokenizer(xs, padding=True, return_tensors='pt').to('cuda:0')
    xl = torch.sum(mm['attention_mask'], dim=1, keepdim=False)
    return xl.tolist()

def calculate_entropy(qa, model, tokenizer, group_list=None, seed=42):
    from vllm import SamplingParams

    num_response_list = [len(responses) for responses in qa.values()]
    all_responses = [response for responses in qa.values() for response in responses]
    len_list = compute_sentence_token_len(all_responses, tokenizer)

    prompts = [x + y for x in qa.keys() for y in qa[x]]

    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1, seed=seed)
    outputs = model.generate(prompts, sampling_params)

    logprob_list = [output.prompt_logprobs for output in outputs]
    token_id_list = [output.prompt_token_ids for output in outputs]

    entropys, logprob4all = [], []
    st = 0
    for batch_size in num_response_list:
        ed = st + batch_size
        logprobs = logprob_list[st: ed]
        token_ids = token_id_list[st: ed]
        suffix_len_list = len_list[st: ed]

        lps = []
        for id_entry, logprob_entry, suffix_len in zip(token_ids, logprobs, suffix_len_list):
            id = id_entry[-suffix_len:]
            logprob = logprob_entry[-suffix_len:]

            lp = 0.0
            for a, b in zip(id, logprob):
                lp += b[a].logprob
                # print(b[a].decoded_token)

            lps.append(lp / float(suffix_len)) # normalized logprob

        if group_list is None:
            entropy = 0.0
            for lp in lps:
                entropy += - math.exp(lp) * lp
            
            entropys.append(entropy)
        else:
            logprob4all.append(lps[0])
        
        st = ed
        
    if group_list is None:
        return entropys
    else:
        st = 0
        for group_len in group_list:
            ed = st + group_len
            lps = logprob4all[st: ed]
            
            entropy = 0.0
            for lp in lps:
                entropy += - math.exp(lp) * lp
            
            entropys.append(entropy)
            st = ed
        return entropys

### plot
def plot(fn, action_detail=True):
    # timestamp = get_time()
    name = fn.split('/')[-1]
    img_path = f"../img/{name}"
    os.makedirs(img_path, exist_ok=True)
    
    ds = load_from_disk(fn)
    entropy_cot = [math.log(entry['entropy_cot']) for entry in ds]
    entropy = [math.log(entry['entropy']) for entry in ds]
    bleu = [entry['self-BLEU'] for entry in ds]
    embedding_dist = [min(entry['avg_paired_distance'], 1.0) for entry in ds]
    action_coverage = [np.cumsum(entry['action_coverage']).tolist() for entry in ds]
    
    
    data = [entropy_cot, entropy, bleu, embedding_dist]
    label = ['entropy_cot', 'entropy', 'self-BLEU', 'avg paired distance']

    # 创建箱线图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.boxplot(data, labels=label)  # 设置标签

    # 添加标题和坐标轴标签
    plt.title(f"Boxplot of Four Indicators", fontsize=15)
    plt.xlabel('Indicators', fontsize=12)
    plt.ylabel('Values', fontsize=12)

    # 显示图形
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线（可选）
    plt.savefig(f"{img_path}/indicators.svg")
    # plt.show()

    if action_detail:
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 6))
        for i, line in enumerate(action_coverage):
            sns.lineplot(x=range(len(line)), y=line, label=f'Line {i+1}')

        plt.title('Action Coverage', fontsize=14)
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Count of Action', fontsize=12)
        plt.legend()
        plt.savefig(f"{img_path}/raw_action_coverage.svg")

    fig, ax = plt.subplots(figsize=(10, 6))
    action_coverage = np.array(action_coverage)
    mean = np.mean(action_coverage, axis=0)
    std = np.std(action_coverage, axis=0)
    ci = 1.96 * std / np.sqrt(action_coverage.shape[0])
    steps = np.arange(action_coverage.shape[1])

    ax.plot(steps, mean, color='red', label='Action Coverage', linewidth=2)
    ax.fill_between(steps, mean - ci, mean + ci, alpha=0.2)

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Count of Action', fontsize=12)
    ax.set_title('Action Coverage (Mean ± 95% CI)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # plt.savefig('gan_loss_ci.png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{img_path}/action_coverage.svg")
    

if __name__ == '__main__':
    
    plot('../log/Qwen2.5-3B-Instruct-20q-100r-withcot-42-2025-06-20-16-30')
