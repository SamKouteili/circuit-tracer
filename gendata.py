
from pathlib import Path
import torch
import gc

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("xxz224/prompt-injection-attack-dataset")

# How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter
MAX_N_LOGITS = 10
# Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)
DESIRED_LOGIT_PROB = 0.95
# Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.
MAX_FEATURE_NODES = 4096
BATCH_SIZE = 256  # Batch size when attributing
# Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)
OFFLOAD = 'cpu'
VERBOSE = True  # Whether to display a tqdm progress bar and timing report

NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98

model_name = 'google/gemma-2-2b'
transcoder_name = "gemma"
model = ReplacementModel.from_pretrained(
    model_name, transcoder_name, dtype=torch.bfloat16)


def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def save_graph(prompt, slug, dir):

    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=MAX_N_LOGITS,
        desired_logit_prob=DESIRED_LOGIT_PROB,
        batch_size=BATCH_SIZE,
        max_feature_nodes=MAX_FEATURE_NODES,
        offload=OFFLOAD,
        verbose=VERBOSE
    )

    create_graph_files(
        graph_or_path=graph,
        slug=slug,
        output_path=dir,
        node_threshold=NODE_THRESHOLD,
        edge_threshold=EDGE_THRESHOLD
    )

    del graph
    clear_gpu_memory()


def generate_data(ds):
    for row in ds['train']:
        # print(row)
        id_ = str(row['id'])
        print(f'* {id_}')
        prompt = row['target_text']
        prompt_nve_atk = 'nve', row['naive_attack']
        prompt_esc_atk = 'esc', row['escape_attack']
        prompt_ign_atk = 'ign', row['ignore_attack']
        prompt_cmp_atk = 'cmp', row['fake_comp_attack']
        save_graph(prompt, id_, './data/benign')
        for p in [prompt_nve_atk, prompt_esc_atk, prompt_ign_atk, prompt_cmp_atk]:
            save_graph(p[1], f'{id_}-{p[0]}', './data/injected')


if __name__ == '__main__':
    generate_data(ds)
