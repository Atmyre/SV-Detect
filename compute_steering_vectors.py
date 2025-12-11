# !pip install --no-deps bitsandbytes accelerate xformers peft trl triton cut_cross_entropy unsloth_zoo
# !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
# !pip install --no-deps unsloth
# !pip install --no-deps trl==0.22.2
# !pip install -Uq steering-vectors


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import torch
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from datasets import load_from_disk
from steering_vectors import record_activations
from tqdm import tqdm
from datasets import load_dataset
import os
import pandas as pd

# LLM_name = 'microsoft/phi-2' 
# LLM_name = "meta-llama/Llama-3.1-8B"
# LLM_name = "Qwen/Qwen3-4B"
# LLM_name = "meta-llama/Llama-3.2-3B"
# LLM_name = "meta-llama/Llama-3.2-1B"
# LLM_name = "Qwen/Qwen3-1.7B"
# LLM_name = "google/gemma-2-2b"
# LLM_name = "Qwen/Qwen2-7B"
LLM_name = "EleutherAI/gpt-neo-2.7B"
dataset_name="Jinyan1/COLING_2025_MGT_en"
# dataset_name="/beemo"
dir_path = f'./data/{LLM_name.split("/")[-1]}/{dataset_name.split("/")[-1]}/'

tokenizer = AutoTokenizer.from_pretrained(LLM_name, token='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')
LLM = AutoModelForCausalLM.from_pretrained(LLM_name, token='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')
LLM.to('cuda')

num_layers = LLM.config.num_hidden_layers
hidden_size = LLM.config.hidden_size

def get_dataset_activations_layer(prefix, layer):
    samples = []
    for file in os.listdir(f'{dir_path}layers/'):
        if file.startswith(f'{prefix}_activations_layer_{layer}'):
            samples.append(np.load(f'{dir_path}layers/{file}'))

    samples = np.concatenate(samples, axis=0)
    return np.array(samples)

def get_steering_vectors_with_pca(differences):
    pca = IncrementalPCA(n_components=5, batch_size=10000)
    layer_pca = pca.fit_transform(differences)
    layer_pca = layer_pca / np.linalg.norm(layer_pca, axis=0, keepdims=True)
    return layer_pca

def get_steering_vectors_with_logreg(fake_activations, real_activations):
    mean_acts = np.concatenate([fake_activations, real_activations], axis=0).mean(axis=0)
    fake_acts_centered = fake_activations - mean_acts
    real_acts_centered = real_activations - mean_acts
    X = np.concatenate([fake_acts_centered, real_acts_centered], axis=0)
    y = np.concatenate([np.ones(fake_acts_centered.shape[0], dtype=np.int8),
                        np.zeros(real_acts_centered.shape[0], dtype=np.int8)])
    logreg = LogisticRegression(solver='liblinear', C=0.01, fit_intercept=False, random_state=42)
    layer_logreg = logreg.fit(X, y)
    logreg_coef = logreg.coef_
    normalized_logreg_coef = logreg_coef / np.linalg.norm(logreg_coef, axis=-1, keepdims=True)
    return normalized_logreg_coef.squeeze(0)

print('Calculating steering vectors Mean')
needed_models = ['13B', '30B', '65B', '7B', 'GLM130B', 
        'cohere', 'davinci', 
        'gemma-7b-it', 'gemma2-9b-it', 'gpt4',
        'gpt4o', 'gpt_j', 'gpt_neox', 'human', 'llama3-70b', 'llama3-8b',
        'mixtral-8x7b', 'text-davinci-002',
        'text-davinci-003']

train_pd = pd.read_json('./en_train.jsonl', lines=True)
train_pd_fake = train_pd[train_pd['label'] == 1].reset_index()
train_pd_real = train_pd[train_pd['label'] == 0].reset_index()
train_pd_fake = train_pd_fake[train_pd_fake['model'].isin(needed_models)]
train_pd_real = train_pd_real[train_pd_real['model'].isin(needed_models)]


steering_vectors = []

for layer in tqdm(range(num_layers)):
    print('Layer', layer)
    real_train_vectors = get_dataset_activations_layer(prefix='real_train', layer=layer)
    fake_train_vectors = get_dataset_activations_layer(prefix='fake_train', layer=layer)

    real_train_vectors = real_train_vectors[list(train_pd_real.index)]
    fake_train_vectors = fake_train_vectors[list(train_pd_fake.index)]
    steering_vector = np.mean(fake_train_vectors, axis=0) - np.mean(real_train_vectors, axis=0)
    steering_vectors.append(steering_vector)

steering_vectors = np.array(steering_vectors)
steering_vectors = steering_vectors / np.linalg.norm(steering_vectors, axis=-1, keepdims=True)
print(steering_vectors.shape)
np.save(f'{dir_path}mean_vectors_woTGPT35weak.npy', steering_vectors)

pca_vectors = np.array(pca_vectors)
print(pca_vectors.shape)
np.save(f'{dir_path}pca_vectors.npy', pca_vectors)

logreg_vectors = np.array(logreg_vectors)
print(logreg_vectors.shape)
np.save(f'{dir_path}logreg_vectors.npy', pca_vectors)



# steering_vectors = np.load(f'{dir_path}steering_vectors_mean.npy')
# steering_vectors = np.load(f'/home/t50045037/ftd_steering/data/Llama-3.1-8B/COLING_2025_MGT_en/steering_vectors_mean.npy')


def get_dot_products(activations, steering_vectors):
    dot_products = np.sum(activations / np.linalg.norm(activations, axis=-1, keepdims=True) * steering_vectors, axis=2)
    return dot_products

# def get_dataset_activations(prefix):
#     samples = []
#     for file in tqdm(os.listdir(f'{dir_path}/')):
#         if file.startswith(f'{prefix}_activations_'):
#             samples.append(np.load(f'{dir_path}layers/{file}'))

#     return np.array(samples)


print('Calculating dot products Mean')
real_train_dot_products = []
fake_train_dot_products = []
real_val_dot_products = []
fake_val_dot_products = []
test_dot_products = []
for file in tqdm(os.listdir(f'{dir_path}/')):
    if file.startswith('real_train_activations_'):
        vectors = np.load(f'{dir_path}/{file}')
        real_train_dot_products.append(get_dot_products(vectors, steering_vectors))

    if file.startswith('fake_train_activations_'):
        vectors = np.load(f'{dir_path}/{file}')
        fake_train_dot_products.append(get_dot_products(vectors, steering_vectors))

    if file.startswith('real_val_activations_'):
        vectors = np.load(f'{dir_path}/{file}')
        real_val_dot_products.append(get_dot_products(vectors, steering_vectors))

    if file.startswith('fake_val_activations_'):
        vectors = np.load(f'{dir_path}/{file}')
        fake_val_dot_products.append(get_dot_products(vectors, steering_vectors))

    if file.startswith('test_activations_'):
        vectors = np.load(f'{dir_path}/{file}')
        test_dot_products.append(get_dot_products(vectors, steering_vectors))

real_train_dot_products = np.concatenate(real_train_dot_products, axis=0)
np.save(f'{dir_path}real_train_dot_products_mean.npy', real_train_dot_products)
fake_train_dot_products = np.concatenate(fake_train_dot_products, axis=0)
np.save(f'{dir_path}fake_train_dot_products_mean.npy', fake_train_dot_products)
real_val_dot_products = np.concatenate(real_val_dot_products, axis=0)
np.save(f'{dir_path}real_val_dot_products_mean.npy', real_val_dot_products)
fake_val_dot_products = np.concatenate(fake_val_dot_products, axis=0)
np.save(f'{dir_path}fake_val_dot_products_mean.npy', fake_val_dot_products)
test_dot_products = np.concatenate(test_dot_products, axis=0)
np.save(f'{dir_path}test_dot_products_mean.npy', test_dot_products)

# beemo
# print('Calculating dot products Mean')
# real_dot_products = []
# fake_dot_products = []
# real_refined_llama_dot_products = []
# real_refined_gpt_dot_products = []
# fake_refined_dot_products = []
# for file in tqdm(os.listdir(f'{dir_path}/')):
#     if file.startswith('real_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         real_dot_products.append(get_dot_products(vectors, steering_vectors))

#     if file.startswith('fake_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         fake_dot_products.append(get_dot_products(vectors, steering_vectors))

#     if file.startswith('fake_refined_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         fake_refined_dot_products.append(get_dot_products(vectors, steering_vectors))

#     if file.startswith('real_refined_llama_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         real_refined_llama_dot_products.append(get_dot_products(vectors, steering_vectors))

#     if file.startswith('real_refined_gpt_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         real_refined_gpt_dot_products.append(get_dot_products(vectors, steering_vectors))

# real_dot_products = np.concatenate(real_dot_products, axis=0)
# np.save(f'{dir_path}real_dot_products_mean.npy', real_dot_products)
# fake_dot_products = np.concatenate(fake_dot_products, axis=0)
# np.save(f'{dir_path}fake_dot_products_mean.npy', fake_dot_products)
# real_refined_llama_dot_products = np.concatenate(real_refined_llama_dot_products, axis=0)
# np.save(f'{dir_path}real_refined_llama_dot_products_mean.npy', real_refined_llama_dot_products)
# real_refined_gpt_dot_products = np.concatenate(real_refined_gpt_dot_products, axis=0)
# np.save(f'{dir_path}real_refined_gpt_dot_products_mean.npy', real_refined_gpt_dot_products)
# fake_refined_dot_products = np.concatenate(fake_refined_dot_products, axis=0)
# np.save(f'{dir_path}fake_refined_dot_products_mean.npy', fake_refined_dot_products)

# needed_models = ['13B', '30B', '65B', '7B', 'GLM130B', 
#             'cohere', 'davinci', 
#             'gemma-7b-it', 'gemma2-9b-it', 'gpt4', 'gpt-3.5-turbo', 'gpt-35',
#             'gpt4o', 'gpt_j', 'gpt_neox', 'human', 'llama3-70b', 'llama3-8b',
#             'mixtral-8x7b', 'text-davinci-002',
#             'text-davinci-003']
# train_pd = pd.read_json('./data/dataset/coling25_en_train.jsonl', lines=True)
# train_pd_fake = train_pd[train_pd['label'] == 1].reset_index()
# train_pd_real = train_pd[train_pd['label'] == 0].reset_index()
# train_pd_fake = train_pd_fake[train_pd_fake['model'].isin(needed_models)]
# train_pd_real = train_pd_real[train_pd_real['model'].isin(needed_models)]

# fake_index = list(train_pd_fake.index)
# real_index = list(train_pd_real.index)
# del train_pd_fake, train_pd_real

# steering_vectors_mean_woweak = []
# for layer in range(num_layers):
#     print('Layer', layer)
#     real_train_activations = []
#     fake_train_activations = []

#     for file in tqdm(os.listdir(f'{dir_path}/')):
#         if file.startswith('real_train_activations_'):
#             vectors = np.load(f'{dir_path}/{file}')
#             real_train_activations.append(vectors[:, layer, :])
#             del vectors

#     real_train_activations = np.concatenate(real_train_activations, axis=0)
#     real_train_activations = real_train_activations[real_index].mean(axis=0)

#     for file in tqdm(os.listdir(f'{dir_path}/')):
#         if file.startswith('fake_train_activations_'):
#             vectors = np.load(f'{dir_path}/{file}')
#             fake_train_activations.append(vectors[:, layer, :])
#             del vectors
    
#     fake_train_activations = np.concatenate(fake_train_activations, axis=0)
#     fake_train_activations = fake_train_activations[fake_index].mean(axis=0)

#     steering_vectors = fake_train_activations - real_train_activations
#     steering_vectors_mean_woweak.append(steering_vectors / np.linalg.norm(steering_vectors, axis=-1, keepdims=True))

# print(steering_vectors_mean_woweak.shape)
# np.save(f'{dir_path}steering_vectors_mean_woweak.npy', steering_vectors_mean_woweak)

# del real_train_activations, fake_train_activations

# print('Calculating dot products Mean')
# real_train_dot_products = []
# fake_train_dot_products = []
# real_val_dot_products = []
# fake_val_dot_products = []
# test_dot_products = []
# for file in tqdm(os.listdir(f'{dir_path}/')):
#     if file.startswith('real_train_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         real_train_dot_products.append(get_dot_products(vectors, steering_vectors_mean_woweak))

#     if file.startswith('fake_train_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         fake_train_dot_products.append(get_dot_products(vectors, steering_vectors_mean_woweak))

#     if file.startswith('real_val_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         real_val_dot_products.append(get_dot_products(vectors, steering_vectors_mean_woweak))

#     if file.startswith('fake_val_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         fake_val_dot_products.append(get_dot_products(vectors, steering_vectors_mean_woweak))

#     if file.startswith('test_activations_'):
#         vectors = np.load(f'{dir_path}/{file}')
#         test_dot_products.append(get_dot_products(vectors, steering_vectors_mean_woweak))

# real_train_dot_products = np.concatenate(real_train_dot_products, axis=0)
# np.save(f'{dir_path}real_train_dot_products_mean_woweak.npy', real_train_dot_products)
# fake_train_dot_products = np.concatenate(fake_train_dot_products, axis=0)
# np.save(f'{dir_path}fake_train_dot_products_mean_woweak.npy', fake_train_dot_products)
# real_val_dot_products = np.concatenate(real_val_dot_products, axis=0)
# np.save(f'{dir_path}real_val_dot_products_mean_woweak.npy', real_val_dot_products)
# fake_val_dot_products = np.concatenate(fake_val_dot_products, axis=0)
# np.save(f'{dir_path}fake_val_dot_products_mean_woweak.npy', fake_val_dot_products)
# test_dot_products = np.concatenate(test_dot_products, axis=0)
# np.save(f'{dir_path}test_dot_products_mean_woweak.npy', test_dot_products)

