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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# LLM_name = 'microsoft/phi-2' 
# LLM_name = "meta-llama/Llama-3.1-8B"
# LLM_name = "Qwen/Qwen3-4B"
# LLM_name = "meta-llama/Llama-3.2-3B"
# LLM_name = "meta-llama/Llama-3.2-1B"
# LLM_name = "Qwen/Qwen3-1.7B"
# LLM_name = "google/gemma-2-2b"
LLM_name = "Qwen/Qwen2-7B"
dataset_name_steering_vecs="Jinyan1/COLING_2025_MGT_en"
# dataset_name_test="/beemo"
dataset_name_test = "Jinyan1/COLING_2025_MGT_en"
dir_path_steering_vecs = f'./data/{LLM_name.split("/")[-1]}/{dataset_name_steering_vecs.split("/")[-1]}/'
dir_path_test = f'./data/{LLM_name.split("/")[-1]}/{dataset_name_test.split("/")[-1]}/'

LLM = AutoModelForCausalLM.from_pretrained(LLM_name, token='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')
num_layers = LLM.config.num_hidden_layers

print(LLM_name, dataset_name_steering_vecs, dataset_name_test)

X_train_real = np.load('{}/real_train_dot_products_mean.npy'.format(dir_path_steering_vecs))
X_train_fake = np.load('{}/fake_train_dot_products_mean.npy'.format(dir_path_steering_vecs))
X_val_real = np.load('{}/real_val_dot_products_mean.npy'.format(dir_path_steering_vecs))
X_val_fake = np.load('{}/fake_val_dot_products_mean.npy'.format(dir_path_steering_vecs))
X_test = np.load('{}/test_dot_products_mean.npy'.format(dir_path_test))

# X_test_real = np.load('{}/real_dot_products_mean.npy'.format(dir_path_test))
# X_test_fake = np.load('{}/fake_dot_products_mean.npy'.format(dir_path_test))

# needed_models = ['13B', '30B', '65B', '7B', 'GLM130B', 
#         'cohere', 'davinci', 
#         'gemma-7b-it', 'gemma2-9b-it', 'gpt4', #'gpt-3.5-turbo', 'gpt-35',
#         'gpt4o', 'gpt_j', 'gpt_neox', 'human', 'llama3-70b', 'llama3-8b',
#         'mixtral-8x7b', 'text-davinci-002',
#         'text-davinci-003']
# train_pd = pd.read_json('./data/dataset/coling25_en_train.jsonl', lines=True)
# train_pd_fake = train_pd[train_pd['label'] == 1].reset_index()
# train_pd_real = train_pd[train_pd['label'] == 0].reset_index()
# train_pd_fake = train_pd_fake[train_pd_fake['model'].isin(needed_models)]
# train_pd_real = train_pd_real[train_pd_real['model'].isin(needed_models)]

# X_train_real = X_train_real[list(train_pd_real.index)]
# X_train_fake = X_train_fake[list(train_pd_fake.index)]

# dev_pd = pd.read_json('./data/dataset/coling25_en_dev.jsonl', lines=True)
# dev_pd_fake = dev_pd[dev_pd['label'] == 1].reset_index()
# dev_pd_real = dev_pd[dev_pd['label'] == 0].reset_index()
# dev_pd_fake = dev_pd_fake[dev_pd_fake['model'].isin(needed_models)]
# dev_pd_real = dev_pd_real[dev_pd_real['model'].isin(needed_models)]

# X_val_real = X_val_real[list(dev_pd_real.index)]
# X_val_fake = X_val_fake[list(dev_pd_fake.index)]

print(X_train_real.shape[0], X_train_fake.shape[0], X_val_real.shape[0], X_val_fake.shape[0])



# fake_df = train_pd[train_pd['label'] == 1].reset_index()
# X_train_fake_new = []
# for model in np.unique(train_pd['model']):
#     df_tmp = fake_df[fake_df['model']==model]
#     # idx = list(df_tmp.index)[:int(0.25*len(df_tmp))]
#     idx = list(df_tmp.index)
#     if len(idx) > 0:
#         X_train_fake_new.append(X_train_fake[idx])
# X_train_fake_new = np.concatenate(X_train_fake_new, axis=0)
# np.random.shuffle(X_train_real)
# X_train_real_new = X_train_real[:X_train_fake_new.shape[0]]

# val_pd = pd.read_json('./en_dev.jsonl', lines=True)
# fake_df_val = val_pd[val_pd['label'] == 1].reset_index()
# X_val_fake_new = []
# for model in np.unique(val_pd['model']):
#     df_tmp = fake_df_val[fake_df_val['model']==model]
#     # idx = list(df_tmp.index)[:int(0.25*len(df_tmp))]
#     idx = list(df_tmp.index)
#     if len(idx) > 0:
#         X_val_fake_new.append(X_val_fake[idx])
# X_val_fake_new = np.concatenate(X_val_fake_new, axis=0)
# X_val_fake_new.shape
# np.random.shuffle(X_val_real)
# X_val_real_new = X_val_real[:X_val_fake_new.shape[0]]

y_train_real = np.zeros(X_train_real.shape[0])
y_train_fake = np.ones(X_train_fake.shape[0])
y_val_real = np.zeros(X_val_real.shape[0])
y_val_fake = np.ones(X_val_fake.shape[0])

X_train = np.concatenate([X_train_real, X_train_fake], axis=0)
y_train = np.concatenate([y_train_real, y_train_fake], axis=0)
X_val = np.concatenate([X_val_real, X_val_fake], axis=0)
y_val = np.concatenate([y_val_real, y_val_fake], axis=0)

y_test = np.array(pd.read_json('./data/dataset/coling25_en_test.jsonl', lines=True)['label'])
# X_test = np.concatenate([X_test_real, X_test_fake], axis=0)
# y_test = [0]*len(X_test_real) + [1]*len(X_test_fake)


def train_test_logreg(X_train, y_train, X_val, y_val, X_test, y_test, layer=list(range(32))):
    best_score = 0
    best_threshold = 0
    best_model = None
    for penalty in ['l1']:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            solver = 'saga' if penalty == 'l1' else 'lbfgs'
            if C is None:
                model = LogisticRegression(max_iter=1000,
                                            random_state=42
                                            )
            else:
                model = LogisticRegression(solver=solver,
                                            max_iter=1000,
                                            penalty=penalty,
                                            C=C,
                                            random_state=42
                                            )
            model.fit(X_train[:, layer], y_train)

            # y_proba_valid = model.predict_proba(X_val[:, layer])[:, 1]
            # thresholds = np.linspace(0, 1, num=100, endpoint=False) #создается массив с значениями вероятностей
            # # для каждого порога вычисляется метрика f1
            # f1_scores = np.array([f1_score(y_val, y_proba_valid>threshold, average='macro') for threshold in thresholds])
            # # best_f1_score = np.max(f1_scores)
            # threshold = thresholds[np.argmax(f1_scores)] #определяется порог, соответствующий максимальному значению f1
            threshold = 0.5

            y_proba_val = model.predict_proba(X_val[:, layer])[:, 1]
            f1_score_val = f1_score(y_val, y_proba_val>threshold, average='macro')
            roc_auc_score_val = roc_auc_score(y_val, y_proba_val)
            accuracy_score_val = accuracy_score(y_val, y_proba_val>threshold)

            y_proba_test = model.predict_proba(X_test[:, layer])[:, 1]
            f1_score_test = f1_score(y_test, y_proba_test>threshold, average='macro')
            roc_auc_score_test = roc_auc_score(y_test, y_proba_test)
            accuracy_score_test = accuracy_score(y_test, y_proba_test>threshold)

            # if best_f1_score > best_score:
            #     best_score = best_f1_score
            #     best_threshold = threshold
            #     best_model = model

            print(layer, threshold)
            print('Accuracy', accuracy_score_val, accuracy_score_test)
            print('F1', f1_score_val, f1_score_test)
            print('ROC AUC', roc_auc_score_val, roc_auc_score_test)

        # print(best_model.coef_)

# for layer in range(num_layers):
#     train_test_logreg(X_train, y_train, X_val, y_val, X_test, y_test, layer=[layer])

train_test_logreg(X_train, y_train, X_val, y_val, X_test, y_test, layer=list(range(num_layers)))