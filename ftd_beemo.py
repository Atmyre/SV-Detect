# !pip install --no-deps bitsandbytes accelerate xformers peft trl triton cut_cross_entropy unsloth_zoo
# !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
# !pip install --no-deps unsloth
# !pip install --no-deps trl==0.22.2
# !pip install -Uq steering-vectors


import unsloth
from unsloth import FastLanguageModel
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from steering_vectors import record_activations
from tqdm import tqdm
from datasets import load_dataset
import os


LLM_name = "unsloth/phi-2"
# LLM_name = "meta-llama/Llama-3.1-8B"
# LLM_name = "Qwen/Qwen3-4B"
# LLM_name = "meta-llama/Llama-3.2-3B"
# LLM_name = "meta-llama/Llama-3.2-1B"
# LLM_name = "Qwen/Qwen3-1.7B"
# LLM_name = "google/gemma-2-2b"
layer_type = 'decoder_block'

dataset_name="/beemo"
dir_path = f'./data/{LLM_name.split("/")[-1]}_unsloth/{dataset_name.split("/")[-1]}/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

max_seq_length = 4096
print('Loading LLM')
LLM, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LLM_name,
    max_seq_length=max_seq_length,
    device_map='cuda'
)
FastLanguageModel.for_inference(LLM)

# tokenizer = AutoTokenizer.from_pretrained(LLM_name, token='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')
# LLM = AutoModelForCausalLM.from_pretrained(LLM_name, token='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')
# LLM.to('cuda')

num_layers = LLM.config.num_hidden_layers
hidden_size = LLM.config.hidden_size

print(num_layers, hidden_size, max_seq_length)


print('Loading datasets')
# dataset = load_dataset(dataset_name)
df = pd.read_parquet("hf://datasets/toloka/beemo/data/train-00000-of-00001.parquet")

real = df['human_output'].tolist()#.filter(lambda example: example["label"]==0)
fake = df['model_output'].tolist()#.filter(lambda example: example["label"]==1)
fake_refined = df['human_edits'].tolist()

def refine(data):
    real_refined= []
    for item in data:
        x = item.split('}, {')
        y = x[0].split('\'P1\': ')[1][1:]
        if y.endswith('\'') or y.endswith('\"'):
            y = y[:-1]
        real_refined.append(y)

        y = x[1].split('\'P2\': ')[1][1:-1]
        if y.endswith('\'') or y.endswith('\"'):
            y = y[:-1]
        real_refined.append(y)

        y = x[2].split('\'P3\': ')[1][1:-2]
        if y.endswith('\'') or y.endswith('\"'):
            y = y[:-1]
        real_refined.append(y)
    return real_refined

real_refined_llama = refine(df['llama-3.1-70b_edits'].tolist())
real_refined_gpt = refine(df['gpt-4o_edits'].tolist())

print(len(real), len(fake), len(fake_refined), len(real_refined_llama), len(real_refined_gpt))

print('Сalculation of activations...')
def get_sample_activations(sample, layer_type=layer_type):
    with record_activations(LLM, layer_type=layer_type) as recorded_activations:
        inputs = tokenizer(sample,
                           return_tensors='pt',
                           max_length=max_seq_length,
                           truncation=True,
                           padding=False
                          ).to('cuda')
        with torch.no_grad():
            LLM.forward(**inputs)
        activations = torch.stack([layer_activations[0].mean(dim=1).flatten().to(torch.float32) for layer_activations in recorded_activations.values()]).cpu().numpy()
    return activations

def get_dataset_activations(dataset, prefix):
    samples = []
    num_batch = 0
    means = []

    while os.path.exists(f'{dir_path}{prefix}_activations_{num_batch*20000}-{(num_batch+1)*20000}.npy'):
        batch = np.load(f'{dir_path}{prefix}_activations_{num_batch*20000}-{(num_batch+1)*20000}.npy')
        means.append(batch.mean(axis=0))
        print(f'Loaded activations from {prefix}, {num_batch*20000}-{(num_batch+1)*20000}.npy')
        num_batch += 1

    if num_batch*20000 > len(dataset):
        return np.array(means).mean(axis=0)
        
    for sample in tqdm(dataset[num_batch*20000:]):
        samples.append(get_sample_activations(sample))
        if len(samples) % 20000 == 0:
            print('Saving activations...')
            np.save(f'{dir_path}{prefix}_activations_{num_batch*20000}-{(num_batch+1)*20000}.npy', np.array(samples))
            # for layer in range(num_layers):
            #     np.save(f'{dir_path}layers/{prefix}_activations_layer_{layer}_{num_batch*20000}-{(num_batch+1)*20000}.npy', np.array(samples)[:, layer])
            means.append(np.array(samples).mean(axis=0))
            num_batch += 1
            samples = []
    print('Saving activations...')
    np.save(f'{dir_path}{prefix}_activations_{num_batch*20000}-{(num_batch+1)*20000}.npy', np.array(samples))
    # for layer in range(num_layers):
    #     np.save(f'{dir_path}layers/{prefix}_activations_layer_{layer}_{num_batch*20000}-{(num_batch+1)*20000}.npy', np.array(samples)[:, layer])
    means.append(np.array(samples).mean(axis=0))
    num_batch += 1
    samples = []
    return np.array(means).mean(axis=0)



real_means = get_dataset_activations(real, prefix='real')
np.save(f'{dir_path}real_means.npy', real_means)
# real_train_means = np.load(f'{dir_path}real_train_means.npy')

fake_means = get_dataset_activations(fake, prefix='fake')
np.save(f'{dir_path}fake_means.npy', fake_means)
# fake_train_means = np.load(f'{dir_path}fake_train_means.npy')

fake_refined_means = get_dataset_activations(fake_refined, prefix='fake_refined')
np.save(f'{dir_path}fake_refined_means.npy', fake_refined_means)
# real_val_means = np.load(f'{dir_path}real_val_means.npy')

real_refined_llama_means = get_dataset_activations(real_refined_llama, prefix='real_refined_llama')
np.save(f'{dir_path}real_refined_llama_means.npy', real_refined_llama_means)

real_refined_gpt_means = get_dataset_activations(real_refined_gpt, prefix='real_refined_gpt')
np.save(f'{dir_path}real_refined_gpt_means.npy', real_refined_gpt_means)

    
# print('Сalculating mean steering vectors...')
# steering_vectors_mean = fake_train_means - real_train_means
# steering_vectors_mean = steering_vectors_mean / np.linalg.norm(steering_vectors_mean, axis=-1, keepdims=True)
# np.save(f'{dir_path}steering_vectors_mean.npy', steering_vectors_mean)
