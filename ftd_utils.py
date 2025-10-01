# !pip install --no-deps bitsandbytes accelerate xformers peft trl triton cut_cross_entropy unsloth_zoo
# !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
# !pip install --no-deps unsloth
# !pip install --no-deps trl==0.22.2
# !pip install -Uq steering-vectors


import unsloth
from unsloth import FastLanguageModel
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from datasets import load_from_disk
from steering_vectors import record_activations
from tqdm.notebook import tqdm


LLM_name = 'unsloth/mistral-7b-bnb-4bit'
layer_type = 'decoder_block'
dir_path = './'


num_layers = 32
hidden_size = 4096
max_seq_length = 4096

print('Loading LLM')
LLM, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LLM_name,
    max_seq_length=max_seq_length,
    device_map='cuda'
)
FastLanguageModel.for_inference(LLM)

print('Loading datasets')
fake_train_val = load_from_disk(f'{dir_path}fake_train_val')
real_train_val = load_from_disk(f'{dir_path}real_train_val')


print('Сalculation of activations')
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
        activations = torch.stack([layer_activations[0].mean(dim=1).flatten() for layer_activations in recorded_activations.values()]).cpu().numpy()
    return activations

def get_dataset_activations(dataset):
    return np.array([get_sample_activations(sample['generation']) for sample in tqdm(dataset)])


real_train_activations = get_dataset_activations(real_train_val['train'])
np.save(f'{dir_path}real_train_activations.npy', real_train_activations)

fake_train_activations = get_dataset_activations(fake_train_val['train'])
np.save(f'{dir_path}fake_train_activations.npy', fake_train_activations)

real_val_activations = get_dataset_activations(real_train_val['val'])
np.save(f'{dir_path}real_val_activations.npy', real_val_activations)

fake_val_activations = get_dataset_activations(fake_train_val['val'])
np.save(f'{dir_path}fake_val_activations.npy', fake_val_activations)


print('Сalculation of steering vectors')
differences = fake_train_activations - np.concatenate([real_train_activations,
                                                       real_train_activations,
                                                       real_train_activations,
                                                       real_train_activations])

def get_steering_vectors_with_pca(differences, layer):
    pca = PCA(n_components=1, random_state=42)
    layer_pca = pca.fit_transform(differences[layer])
    return layer_pca

steering_vectors_pca = np.array(
    [get_steering_vectors_with_pca(differences.transpose((1, 2, 0)), layer) for layer in tqdm(range(num_layers))]
    ).squeeze(-1)
np.save(f'{dir_path}steering_vectors_pca.npy', steering_vectors_pca)

steering_vectors_mean = np.mean(fake_train_activations, axis=0) - np.mean(real_train_activations, axis=0)
steering_vectors_mean = steering_vectors_mean / np.linalg.norm(steering_vectors_mean, axis=-1, keepdims=True)
np.save(f'{dir_path}steering_vectors_mean.npy', steering_vectors_mean)

def get_steering_vectors_with_logreg(fake_activations, real_activations, layer):
    fake_acts = fake_activations[layer]
    real_acts = real_activations[layer]
    mean_acts = np.stack([fake_acts, real_acts]).mean(axis=0)
    fake_acts_centered = fake_acts - mean_acts
    real_acts_centered = real_acts - mean_acts
    X = np.concatenate([fake_acts_centered, real_acts_centered], axis=0)
    y = np.concatenate([np.ones(fake_acts_centered.shape[0], dtype=np.int8),
                        np.zeros(real_acts_centered.shape[0], dtype=np.int8)])
    logreg = LogisticRegression(solver='liblinear', fit_intercept=False, random_state=42)
    layer_logreg = logreg.fit(X, y)
    logreg_coef = logreg.coef_
    normalized_logreg_coef = logreg_coef / np.linalg.norm(logreg_coef, axis=-1, keepdims=True)
    return normalized_logreg_coef.squeeze(0)


steering_vectors_logreg = np.array(
    [get_steering_vectors_with_logreg(fake_train_activations.transpose((1, 0, 2)),
                                      np.concatenate([real_train_activations,
                                                      real_train_activations,
                                                      real_train_activations,
                                                      real_train_activations]).transpose((1, 0, 2)),
                                      layer
                                      ) for layer in tqdm(range(num_layers))])

np.save(f'{dir_path}steering_vectors_logreg.npy', steering_vectors_logreg)

print('Calculation of dot products')
def get_dot_products(activations, steering_vectors):
    dot_products = np.sum(activations / np.linalg.norm(activations, axis=-1, keepdims=True) * steering_vectors, axis=2)
    return dot_products

real_train_dot_products_pca = get_dot_products(real_train_activations, steering_vectors_pca)
fake_train_dot_products_pca = get_dot_products(fake_train_activations, steering_vectors_pca)
X_train_pca = np.concatenate([real_train_dot_products_pca, fake_train_dot_products_pca], axis=0)
np.save(f'{dir_path}X_train_pca.npy', X_train_pca)

real_train_dot_products_mean = get_dot_products(real_train_activations, steering_vectors_mean)
fake_train_dot_products_mean = get_dot_products(fake_train_activations, steering_vectors_mean)
X_train_mean = np.concatenate([real_train_dot_products_mean, fake_train_dot_products_mean], axis=0)
np.save(f'{dir_path}X_train_mean.npy', X_train_mean)

real_train_dot_products_logreg = get_dot_products(real_train_activations, steering_vectors_logreg)
fake_train_dot_products_logreg = get_dot_products(fake_train_activations, steering_vectors_logreg)
X_train_logreg = np.concatenate([real_train_dot_products_logreg, fake_train_dot_products_logreg], axis=0)
np.save(f'{dir_path}X_train_logreg.npy', X_train_logreg)

real_val_dot_products_pca = get_dot_products(real_val_activations, steering_vectors_pca)
fake_val_dot_products_pca = get_dot_products(fake_val_activations, steering_vectors_pca)
X_val_pca = np.concatenate([real_val_dot_products_pca, fake_val_dot_products_pca], axis=0)
np.save(f'{dir_path}X_val_pca.npy', X_val_pca)

real_val_dot_products_mean = get_dot_products(real_val_activations, steering_vectors_mean)
fake_val_dot_products_mean = get_dot_products(fake_val_activations, steering_vectors_mean)
X_val_mean = np.concatenate([real_val_dot_products_mean, fake_val_dot_products_mean], axis=0)
np.save(f'{dir_path}X_val_mean.npy', X_val_mean)

real_val_dot_products_logreg = get_dot_products(real_val_activations, steering_vectors_logreg)
fake_val_dot_products_logreg = get_dot_products(fake_val_activations, steering_vectors_logreg)
X_val_logreg = np.concatenate([real_val_dot_products_logreg, fake_val_dot_products_logreg], axis=0)
np.save(f'{dir_path}X_val_logreg.npy', X_val_logreg)