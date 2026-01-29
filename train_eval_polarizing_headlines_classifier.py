import os
import csv
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import random
import string
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import softmax


# %%
class HyperpartisanNewsTitlesDataLoader:
    def __init__(self, tokenizer, lower):
        self.tokenizer = tokenizer
        self.seed = 0
        self.lower = lower
        self._read_data()
        self._create_datasets()

    def _read_data(self):
        train_data = pd.read_csv('external_data/hyperpartisan_news_titles/training_set.csv', sep=',', quoting=csv.QUOTE_ALL, dtype=str, usecols=["title", "label"]) # put file from Lyu et al. https://github.com/VIStA-H/Hyperpartisan-News-Titles
        train_data.columns = ["text", "labels"]
        train_data = train_data[train_data["text"] != ""]
        train_data = train_data[~train_data["text"].isna()]
        if self.lower:
            train_data["text"] = train_data["text"].str.lower()

        train_data["labels"] = train_data["labels"].astype(int)
        self.train_data = train_data.sample(frac=1, random_state=self.seed)

        test_data = pd.read_csv('external_data/hyperpartisan_news_titles/testing_set.csv', sep=',', quoting=csv.QUOTE_ALL, dtype=str, usecols=["title", "label"]) # put file from Lyu et al. https://github.com/VIStA-H/Hyperpartisan-News-Titles
        test_data.columns = ["text", "labels"]
        test_data = test_data[test_data["text"] != ""]
        test_data = test_data[~test_data["text"].isna()]
        if self.lower:
            test_data["text"] = test_data["text"].str.lower()

        test_data["labels"] = test_data["labels"].astype(int)
        self.test_data = test_data.sample(frac=1, random_state=self.seed)

    def _create_datasets(self):
        def tokenize(examples):
            res = self.tokenizer(examples["text"], max_length=128, truncation=True, padding='max_length', return_tensors='pt')
            return res

        train_dataset = Dataset.from_pandas(self.train_data[['text', 'labels']])
        self.train_dataset = train_dataset.map(tokenize, batched=True, batch_size=32, remove_columns=['text'])
        print(self.train_dataset)

        test_dataset = Dataset.from_pandas(self.test_data[['text', 'labels']])
        self.test_dataset = test_dataset.map(tokenize, batched=True, batch_size=32, remove_columns=['text'])


def compute_metrics(pred_output):
    #print(pred_output.predictions)
    #print(pred_output.label_ids)
    metrics = {}

    labels = pred_output.label_ids

    predictions = np.argmax(pred_output.predictions, 1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    metrics.update({
        f'acc': acc, f'prec': prec, f'rec': rec, f'f1': f1})

    return metrics


# %%
# List of models
# "google-bert/bert-base-uncased"
# "microsoft/deberta-v3-large"
# "microsoft/deberta-v3-base"

path_or_id = "microsoft/deberta-v3-large"
model_id_or_path = path_or_id
tokenizer_id_or_path = path_or_id

epochs = 5
learning_rate = 2e-5
tokenizer_max_len = 128

dataloader_config = {'per_device_train_batch_size': 32,
                     'per_device_eval_batch_size': 256}

config = {
    "model_id_or_path": model_id_or_path,
    "tokenizer_id_or_path": tokenizer_id_or_path,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "tokenizer_max_len": tokenizer_max_len,
    "batch_size": dataloader_config["per_device_train_batch_size"],
    "lower": False
}


def train():
    # create model directory
    output_dir = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
    while output_dir in os.listdir("hp_models"):
        output_dir = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))

    os.mkdir(f"hp_models/{output_dir}")
    print(output_dir)

    with open(f"hp_models/{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print('load model and tokenizer')
    tokenizer_config = {'pretrained_model_name_or_path': tokenizer_id_or_path,
                        'max_len': tokenizer_max_len}

    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)

    model_config = {'pretrained_model_name_or_path': model_id_or_path,
                    'num_labels': 2,
                    'problem_type': "single_label_classification",
    }

    model = AutoModelForSequenceClassification.from_pretrained(**model_config).cuda()

    print(f'load data')
    dl = HyperpartisanNewsTitlesDataLoader(tokenizer, config["lower"])
    train_ds, train_df = dl.train_dataset, dl.train_data
    test_ds, test_df = dl.test_dataset, dl.test_data

    input_example = train_df["text"].tolist()[0]
    print(f"input_example: {input_example}\n")

    training_args = TrainingArguments(
        output_dir=f"hp_models/{output_dir}",
        num_train_epochs=epochs,
        **dataloader_config,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_strategy='no',
        save_strategy='epoch',
        eval_strategy="epoch",
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    metrics = trainer.state.log_history
    with open(f"hp_models/{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    output = trainer.evaluate()

    with open(f"hp_models/{output_dir}/best_model_eval.json", "w") as f:
        json.dump(output, f, indent=2)

    print(output)
    print(f"saved to {output_dir}")

train()

# %% load best model checkpoint (with highest test f1)
model_dir = "j8fpy" # specify model directory

with open(f"hp_models/{model_dir}/metrics.json", "r") as f:
    metrics = json.load(f)

step2f1 = {m["step"]: m["eval_f1"] for m in metrics if "eval_f1" in m}
step_highest_f1 = max(step2f1, key=step2f1.get)

best_step = step_highest_f1
best_metric = step2f1[best_step]

best_ckpt_dir = f"checkpoint-{best_step}"

print(f"load best model from hp_models/{model_dir}/{best_ckpt_dir} with F1 of {best_metric}")

best_model = AutoModelForSequenceClassification.from_pretrained(f"hp_models/{model_dir}/{best_ckpt_dir}").cuda()

with open(f"hp_models/{model_dir}/config.json", "r") as f:
    best_model_config = json.load(f)

tokenizer_config = {'pretrained_model_name_or_path': best_model_config['model_id_or_path'],
                    'max_len': best_model_config['tokenizer_max_len']}

tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)

lowercase = best_model_config["lower"]

#%% or load model used in this study from Huggingface
best_model = AutoModelForSequenceClassification.from_pretrained("ICWSM/polarizing_headline_classifier").cuda()

tokenizer_config = {'pretrained_model_name_or_path': "microsoft/deberta-v3-large",
                    'max_len': 128}

tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)

lowercase = False

# %% check results on test set by Lyu et al. (file from Lyu et al. https://github.com/VIStA-H/Hyperpartisan-News-Titles)
print(f'load data')
dl = HyperpartisanNewsTitlesDataLoader(tokenizer, lowercase)
test_ds, test_df = dl.test_dataset, dl.test_data

per_device_eval_batch_size = 128

training_args = TrainingArguments(
    output_dir=f"results",
    per_device_eval_batch_size=per_device_eval_batch_size,
    no_cuda=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=10,
    dataloader_prefetch_factor=2,
    report_to='none'
)

trainer = Trainer(
    model=best_model,
    args=training_args,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

output = trainer.evaluate()
print(output)


# %% predict science news article headlines
def predict_science_news_article_titles():
    news = pd.Dataframe() # ... provide file with text column and newsid (ID) column

    ds = Dataset.from_pandas(news[['text', "newsid"]], preserve_index=False).with_format("torch")

    def tokenize_with_padding(examples):
        return tokenizer(examples["text"], max_length=128, truncation=True, padding="longest", return_tensors="pt")

    ds = ds.map(tokenize_with_padding, batched=True, batch_size=per_device_eval_batch_size, remove_columns=['text'])

    dataloader = DataLoader(ds, batch_size=per_device_eval_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    newsids = []
    scores = []

    best_model.eval()

    for step, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        newsids.extend(batch.pop("newsid"))
        batch = {k: v.to(best_model.device, non_blocking=True) for k, v in batch.items()}

        with torch.inference_mode():
            logits = best_model(**batch).logits
            batch_scores = softmax(logits, dim=1)[:, 1]
            scores.append(batch_scores.detach().clone())

    scores = torch.cat(scores).cpu().numpy()
    newsid2score = {newsid: s for newsid, s in zip(newsids, scores)}

    news["score"] = news["newsid"].apply(lambda x: newsid2score.get(x, None))

    return news

news_with_scores = predict_science_news_article_titles()
