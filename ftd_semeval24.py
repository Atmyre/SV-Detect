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


# LLM_name = "microsoft/phi-2"
LLM_name = "meta-llama/Llama-3.1-8B"
# LLM_name = "Qwen/Qwen3-4B"
# LLM_name = "meta-llama/Llama-3.2-3B"
# LLM_name = "meta-llama/Llama-3.2-1B"
# LLM_name = "Qwen/Qwen3-1.7B"
# LLM_name = "google/gemma-2-2b"
layer_type = 'decoder_block'

dataset_name="/semeval24"
dir_path = f'./data/{LLM_name.split("/")[-1]}/{dataset_name.split("/")[-1]}/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


# print('Loading LLM')
# LLM, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=LLM_name,
#     max_seq_length=max_seq_length,
#     device_map='cuda'
# )
# FastLanguageModel.for_inference(LLM)

tokenizer = AutoTokenizer.from_pretrained(LLM_name, token='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')
LLM = AutoModelForCausalLM.from_pretrained(LLM_name, token='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')
LLM.to('cuda')

num_layers = LLM.config.num_hidden_layers
hidden_size = LLM.config.hidden_size
max_seq_length = 4096

print(num_layers, hidden_size, max_seq_length)


print('Loading datasets')
# dataset = load_dataset(dataset_name)
train_df = pd.read_json('data/dataset/semeval24/subtaskA_train_monolingual.jsonl', lines=True)
dev_df = pd.read_json('data/dataset/semeval24/subtaskA_dev_monolingual.jsonl', lines=True)
test_df = pd.read_json('data/dataset/semeval24/subtaskA_dev_monolingual.jsonl', lines=True)

real_train = train_df[train_df['label']==0]#.filter(lambda example: example["label"]==0)
fake_train = train_df[train_df['label']==1]#.filter(lambda example: example["label"]==1)

print('Train', len(real_train), len(fake_train))

real_dev = dev_df[dev_df['label']==0]#.filter(lambda example: example["label"]==0)
fake_dev = dev_df#.filter(lambda example: example["label"]==1)

print('Val', len(real_dev), len(fake_dev))


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
    if prefix != 'test':
        dataset = dataset['text']

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



real_train_means = get_dataset_activations(real_train['text'].tolist(), prefix='real_train')
np.save(f'{dir_path}real_train_means.npy', real_train_means)
# real_train_means = np.load(f'{dir_path}real_train_means.npy')

fake_train_means = get_dataset_activations(fake_train['text'].tolist(), prefix='fake_train')
np.save(f'{dir_path}fake_train_means.npy', fake_train_means)
# fake_train_means = np.load(f'{dir_path}fake_train_means.npy')

real_val_means = get_dataset_activations(real_dev['text'].tolist(), prefix='real_val')
np.save(f'{dir_path}real_val_means.npy', real_val_means)
# real_val_means = np.load(f'{dir_path}real_val_means.npy')

fake_val_means = get_dataset_activations(fake_dev['text'].tolist(), prefix='fake_val')
np.save(f'{dir_path}fake_val_means.npy', fake_val_means)

test_means = get_dataset_activations(test_df['text'].tolist(), prefix='test')

    
print('Сalculating mean steering vectors...')
steering_vectors_mean = fake_train_means - real_train_means
steering_vectors_mean = steering_vectors_mean / np.linalg.norm(steering_vectors_mean, axis=-1, keepdims=True)
np.save(f'{dir_path}steering_vectors_mean.npy', steering_vectors_mean)

# print('Сalculating differences...')
# num_vecs = min(len(fake_train_activations), len(real_train_activations))
# differences = fake_train_activations[:num_vecs] - real_train_activations[:num_vecs]

# def get_steering_vectors_with_pca(differences, layer):
#     pca = PCA(n_components=5, random_state=42)
#     layer_pca = pca.fit_transform(differences[layer])
#     return layer_pca

# print('Сalculating PCA steering vectors...')
# steering_vectors_pca = []
# for layer in tqdm(range(num_layers)):
#     layer_pca = get_steering_vectors_with_pca(differences.transpose((1, 2, 0)), layer)
#     layer_pca = layer_pca / np.linalg.norm(layer_pca, axis=0, keepdims=True)
#     steering_vectors_pca.append(layer_pca)
# np.save(f'{dir_path}steering_vectors_pca.npy', steering_vectors_pca)


# def get_steering_vectors_with_logreg(fake_activations, real_activations, layer):
#     fake_acts = fake_activations[layer]
#     real_acts = real_activations[layer]
#     mean_acts = np.stack([fake_acts, real_acts]).mean(axis=0)
#     fake_acts_centered = fake_acts - mean_acts
#     real_acts_centered = real_acts - mean_acts
#     X = np.concatenate([fake_acts_centered, real_acts_centered], axis=0)
#     y = np.concatenate([np.ones(fake_acts_centered.shape[0], dtype=np.int8),
#                         np.zeros(real_acts_centered.shape[0], dtype=np.int8)])
#     logreg = LogisticRegression(solver='liblinear', fit_intercept=False, random_state=42)
#     layer_logreg = logreg.fit(X, y)
#     logreg_coef = logreg.coef_
#     normalized_logreg_coef = logreg_coef / np.linalg.norm(logreg_coef, axis=-1, keepdims=True)
#     return normalized_logreg_coef.squeeze(0)

# print('Сalculating LogReg steering vectors...')
# steering_vectors_logreg = np.array(
#     [get_steering_vectors_with_logreg(fake_train_activations.transpose((1, 0, 2)),
#                                       real_train_activations.transpose((1, 0, 2)),
#                                       layer
#                                       ) for layer in tqdm(range(num_layers))])

# np.save(f'{dir_path}steering_vectors_logreg.npy', steering_vectors_logreg)

# def get_dot_products(activations, steering_vectors):
#     dot_products = np.sum(activations / np.linalg.norm(activations, axis=-1, keepdims=True) * steering_vectors, axis=2)
#     return dot_products

# print('Calculating dot products (PCA, train)...')
# real_train_dot_products_pca = get_dot_products(real_train_activations, steering_vectors_pca[:, :, 0])
# fake_train_dot_products_pca = get_dot_products(fake_train_activations, steering_vectors_pca[:, :, 0])
# X_train_pca = np.concatenate([real_train_dot_products_pca, fake_train_dot_products_pca], axis=0)
# np.save(f'{dir_path}X_train_pca.npy', X_train_pca)

# print('Calculating dot products (PCA, val)...')
# real_val_dot_products_pca = get_dot_products(real_val_activations, steering_vectors_pca[:, :, 0])
# fake_val_dot_products_pca = get_dot_products(fake_val_activations, steering_vectors_pca[:, :, 0])
# X_val_pca = np.concatenate([real_val_dot_products_pca, fake_val_dot_products_pca], axis=0)
# np.save(f'{dir_path}X_val_pca.npy', X_val_pca)

# print('Calculating dot products (mean, train)...')
# real_train_dot_products_mean = get_dot_products(real_train_activations, steering_vectors_mean)
# fake_train_dot_products_mean = get_dot_products(fake_train_activations, steering_vectors_mean)
# X_train_mean = np.concatenate([real_train_dot_products_mean, fake_train_dot_products_mean], axis=0)
# np.save(f'{dir_path}X_train_mean.npy', X_train_mean)

# print('Calculating dot products (mean, val)...')
# real_val_dot_products_mean = get_dot_products(real_val_activations, steering_vectors_mean)
# fake_val_dot_products_mean = get_dot_products(fake_val_activations, steering_vectors_mean)
# X_val_mean = np.concatenate([real_val_dot_products_mean, fake_val_dot_products_mean], axis=0)
# np.save(f'{dir_path}X_val_mean.npy', X_val_mean)

# print('Calculating dot products (logreg, train)...')
# real_train_dot_products_logreg = get_dot_products(real_train_activations, steering_vectors_logreg)
# fake_train_dot_products_logreg = get_dot_products(fake_train_activations, steering_vectors_logreg)
# X_train_logreg = np.concatenate([real_train_dot_products_logreg, fake_train_dot_products_logreg], axis=0)
# np.save(f'{dir_path}X_train_logreg.npy', X_train_logreg)

# print('Calculating dot products (logreg, val)...')
# real_val_dot_products_logreg = get_dot_products(real_val_activations, steering_vectors_logreg)
# fake_val_dot_products_logreg = get_dot_products(fake_val_activations, steering_vectors_logreg)
# X_val_logreg = np.concatenate([real_val_dot_products_logreg, fake_val_dot_products_logreg], axis=0)
# np.save(f'{dir_path}X_val_logreg.npy', X_val_logreg)