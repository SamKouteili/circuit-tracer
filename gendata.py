
from pathlib import Path
import torch
import gc
import os
import psutil
import time

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

# Initialize model as None - we'll load it lazily
model = None


def get_memory_info():
    """Get current memory usage info"""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_cached = torch.cuda.memory_reserved() / 1e9
        print(
            f"GPU Memory: {gpu_mem:.2f}GB allocated, {gpu_cached:.2f}GB cached")

    ram_used = psutil.virtual_memory().used / 1e9
    ram_total = psutil.virtual_memory().total / 1e9
    print(f"RAM: {ram_used:.2f}GB / {ram_total:.2f}GB used")


def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
    else:
        # CPU only - still do aggressive garbage collection
        for _ in range(3):
            gc.collect()


def load_model():
    """Lazy load model only when needed"""
    global model
    if model is None:
        print("Loading model...")
        model = ReplacementModel.from_pretrained(
            model_name, transcoder_name, dtype=torch.bfloat16)
        clear_gpu_memory()
    return model


def unload_model():
    """Completely unload model to free memory"""
    global model
    if model is not None:
        print("Unloading model...")
        del model
        model = None
        clear_gpu_memory()


def save_graph(prompt, slug, dir):
    """Save graph with aggressive memory management"""
    get_memory_info()

    # Load model only when needed
    current_model = load_model()

    try:
        graph = attribute(
            prompt=prompt,
            model=current_model,
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

        # Aggressively clean up
        del graph
        clear_gpu_memory()

    except Exception as e:
        print(f"Error processing {slug}: {e}")
        # Clean up on error
        clear_gpu_memory()
        raise


def generate_data(ds):
    processed_count = 0

    for i, row in enumerate(ds['train']):
        print(f'\n=== Processing {i} (#{i + 1}) ===')

        if i < 1101:
            continue
        try:
            prompt = row['target_text']
            prompt_nve_atk = 'nve', row['naive_attack']
            prompt_esc_atk = 'esc', row['escape_attack']
            prompt_ign_atk = 'ign', row['ignore_attack']
            prompt_cmp_atk = 'cmp', row['fake_comp_attack']

            # Process benign first
            print(f"Processing benign: {i}")
            save_graph(prompt, str(i), './data/benign')

            # Process each attack type
            for p in [prompt_nve_atk, prompt_esc_atk, prompt_ign_atk, prompt_cmp_atk]:
                print(f"Processing {p[0]} attack: {i}-{p[0]}")
                save_graph(p[1], f'{i}-{p[0]}', './data/injected')

            processed_count += 1

            # Aggressive cleanup every few samples
            if processed_count % 3 == 0:
                print(
                    f"\n--- Periodic cleanup after {processed_count} samples ---")
                unload_model()  # Completely unload model
                time.sleep(2)   # Give system time to clean up
                get_memory_info()

        except Exception as e:
            print(f"Failed to process {id_}: {e}")
            # Clean up on error and continue
            unload_model()
            time.sleep(2)
            continue

    print(f"\nCompleted processing {processed_count} samples")


if __name__ == '__main__':
    generate_data(ds)
