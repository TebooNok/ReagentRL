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

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_time():
    return strftime('%Y-%m-%d-%H-%M', localtime())

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

def calculate_avg_pair_distance(texts, embedding_model):
    n = len(texts)
    if n <= 1:
        return 0
    
    pairs = set()
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            p = sorted([i, j])
            if tuple(p) not in pairs:
                pairs.add(tuple(p))
    
    avg_distance = 0.
    for p in pairs:
        embedding1 = embedding_model.encode(texts[p[0]])
        embedding2 = embedding_model.encode(texts[p[1]])
        cos_sim = cosine_similarity([embedding1], [embedding2])
        avg_distance += 1 - cos_sim
    
    if type(avg_distance) == float:
        return avg_distance
    else:
        return (avg_distance / len(pairs)).item()

### plot
def plot(fn, show_entrophy=False, cot_type='withcot'):
    timestamp = get_time()
    img_path = f"../img/{timestamp}_{cot_type}"
    os.makedirs(img_path, exist_ok=True)
    
    entrophy, bleu, embedding_dist, action_coverage = [], [], [], []
    with open(fn, 'r', encoding='utf-8') as fo:
        for line in fo:
            entry = json.loads(line)
            entrophy.append(math.log(entry['entrophy']) + 580),
            bleu.append(entry['self-BLEU'])
            embedding_dist.append(entry['avg_paired_distance'])
            action_coverage.append(entry['action_count'])
            # print(entry['action_count'][:10], entry['entrophy'], entry['self-BLEU'], entry['avg_paired_distance'])
    
    data = [entrophy, bleu, embedding_dist]
    label = ['entrophy', 'self-BLEU', 'avg paired distance']
    if not show_entrophy:
        data.pop(0)
        label.pop(0)

    # 创建箱线图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.boxplot(data, labels=label)  # 设置标签

    # 添加标题和坐标轴标签
    plt.title(f"Boxplot of {'Three' if show_entrophy else 'Two'} Indicators", fontsize=15)
    plt.xlabel('Indicators', fontsize=12)
    plt.ylabel('Values', fontsize=12)

    # 显示图形
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线（可选）
    plt.savefig(f"{img_path}/indicators.svg")
    # plt.show()


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
    # print(get_time())
    plot('../log/Qwen2.5-3B-Instruct-20q-100r-withoutcot-111-2025-06-02-22-30.jsonl', cot_type='withoutcot')
    plot('../log/Qwen2.5-3B-Instruct-20q-100r-withcot-111-2025-06-02-20-20.jsonl')
