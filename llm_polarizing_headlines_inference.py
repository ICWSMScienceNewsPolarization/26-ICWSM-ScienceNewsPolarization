import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import pandas as pd
from tqdm import tqdm
import re
import datasets
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import wandb

tqdm.pandas()

# %%
# Two settings used:
# Setting #1: direct prompting (reasoning = "no", stage=1)
# Setting #2: Two-step approach (reasoning ="before", stage=1) + (reasoning="before", stage=2, prev_conversation="file path of stage 1 model output")


# Models used:
# Qwen/Qwen2.5-32B-Instruct
# meta-llama/Meta-Llama-3.1-8B-Instruct
# meta-llama/Llama-3.3-70B-Instruct
# openai/gpt-oss-20b

config = {
    "reasoning": "no",
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "sample": True,
    "stage": 1,
    "prev_conversation": "filename.pkl"
}


def llm_prompt(news_article_title, reasoning="no", system_prompt=True):
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    news_article_title = _RE_COMBINE_WHITESPACE.sub(" ", news_article_title).strip()

    # create intro strings
    title_intro = f"""Headline of news article: "{news_article_title}"."""

    definition_intro = f"""A news headline is considered polarizing if it meets any of these two conditions:\n(1) headline expresses a one-sided opinion (e.g., denouncement, criticism) against a policy, a political party, or a politician in a biased and aggressive tone\n(2) headline describes a confrontation or a conflict between opposing parties indicating the underlying political climate is polarized.\nA headline is not polarizing if it only mentions a political figure, party or reports about them in neutral language."""

    # create message(s) to LLM
    if reasoning == "no":
        first_message = {"role": "user",
                         "content": f"""Given the headline of a news article, decide whether the headline is polarizing or not. {definition_intro} \nFollow the output format "Answer: [yes/no]."\n{title_intro}"""}
        last_message = {'role': 'assistant',
                        'content': f"""Answer: """}
        messages = [first_message, last_message]

    elif reasoning == "before":
        first_message = {"role": "user",
                         "content": f"""Given the headline of a news article, decide whether the headline is polarizing or not. {definition_intro} \nFollow the output format "Answer: [yes/no]."\n{title_intro}"""}
        last_message = {'role': 'assistant',
                        'content': f"""First, I will provide a rationale for both options (polarizing headline vs. not polarizing).\n\nRationale for "Yes" (polarizing headline):"""}
        messages = [first_message, last_message]

    if system_prompt:
        messages = [{"role": "system", "content": """You need to annotate the headlines of news articles."""}] + messages

    return messages


def main():
    run = wandb.init(project="W&B project name", config=config)  # Insert "W&B project name here
    print(run.name)

    model_name = wandb.config["model_name"]

    # derive device_map (cuda:0 if working on a single GPU else balanced across all GPUs)
    device_map = "cuda:0" if torch.cuda.device_count() == 1 else "balanced"
    wandb.config.update({"device_map": device_map})

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        dtype="auto",
        device_map=device_map
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # set up and log generation config
    max_new_tokens = 1024 if (wandb.config["reasoning"] == "before" and wandb.config["stage"] == 1) else 200
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.do_sample = wandb.config["sample"]

    generation_config = model.generation_config
    generation_config_dict = generation_config.to_dict()
    print(generation_config_dict)
    wandb.config.update(generation_config_dict)

    system_prompt = True

    # load data
    if config["stage"] == 1:
        test_data = pd.read_csv('testing_set.csv', sep=',', quoting=csv.QUOTE_ALL, dtype=str, usecols=["title", "label"]) # put file from Lyu et al. https://github.com/VIStA-H/Hyperpartisan-News-Titles
        test_data.columns = ["text", "labels"]
        test_data = test_data[test_data["text"] != ""]
        test_data = test_data[~test_data["text"].isna()]
        test_data["labels"] = test_data["labels"].astype(int)
        test_data = test_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=False)
        test_data["messages"] = test_data.apply(lambda x: llm_prompt(news_article_title=x["text"], reasoning=wandb.config["reasoning"], system_prompt=system_prompt), axis=1)

    elif config["stage"] == 2:  # for CoT prompt approach
        prev_conv_data = pd.read_pickle(config["prev_conversation"])
        prev_conv_data = prev_conv_data[["index", "text", "labels", "messages", "llm_response"]]
        prev_conv_data = prev_conv_data.rename(columns={"llm_response": "prev_llm_response"})
        prev_messages = prev_conv_data["messages"].copy().tolist()
        prev_llm_responses = prev_conv_data["prev_llm_response"].copy().tolist()
        new_messages = []
        for pm, pr in zip(prev_messages, prev_llm_responses):
            pm[-1]["content"] = pm[-1]["content"] + pr + '. I need to answer in the following format "Answer: [yes/no]". Answer: '
            new_messages.append(pm)

        prev_conv_data["messages"] = new_messages
        test_data = prev_conv_data.copy()

    message_example = test_data["messages"].tolist()[0]
    wandb.config.update({"message_example": message_example})
    print(f"message_example: {message_example}\n")

    test_data["input_text"] = test_data["messages"].apply(lambda x: tokenizer.apply_chat_template(x,
                                                                                                  tokenize=False,
                                                                                                  add_generation_prompt=False,
                                                                                                  continue_final_message=False,
                                                                                                  truncation=False,
                                                                                                  max_length=tokenizer.model_max_length))

    # remove last eos token to continue message
    eos_token = tokenizer.special_tokens_map["eos_token"]
    test_data["input_text"] = test_data["input_text"].apply(lambda x: x[:x.rindex(eos_token)] if eos_token in x else x)

    # print input example
    input_example = test_data["input_text"].tolist()[0]
    wandb.config.update({"input_example": input_example})
    print(f"input_example: {input_example}\n")

    ds = datasets.Dataset.from_pandas(test_data[["index", "input_text"]], preserve_index=False).with_format("torch")

    def tokenize(examples):
        return tokenizer(examples["input_text"], return_tensors="pt", padding=True, truncation=False)

    ds = ds.map(tokenize, batched=True, batch_size=1)
    ds = ds.map(lambda x: {"input_length": len(x["input_ids"]), "max_length": min(tokenizer.model_max_length, 1000000)})
    ds = ds.sort("input_length", reverse=True)

    batch_size = 8
    wandb.config.update({"batch_size": batch_size})

    ds = ds.map(tokenize, batched=True, batch_size=batch_size, remove_columns=['input_text'])
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def get_responses(model, dataloader):
        ids = []
        input_lens = []
        max_lens = []
        responses = []

        for batch in tqdm(dataloader, total=len(dataloader)):
            ids.extend(batch.pop("index"))
            input_lens.extend(batch.pop("input_length"))
            max_lens.extend(batch.pop("max_length"))

            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                generated_ids = model.generate(
                    **batch,
                    generation_config=generation_config
                )

            batch_response_ids = [output_ids[l:] for l, output_ids in zip([len(x) for x in batch["input_ids"]], generated_ids)]
            batch_responses = tokenizer.batch_decode(batch_response_ids, skip_special_tokens=True)

            print(f"response: {batch_responses[0]}\n")

            responses.extend(batch_responses)

        return ids, responses, input_lens, max_lens

    ids, responses, input_lens, max_lens = get_responses(model, dataloader)

    try:
        responses = {i.item(): r for i, r in zip(ids, responses)}
        id2input_len = {i.item(): r for i, r in zip(ids, input_lens)}
        id2max_len = {i.item(): r for i, r in zip(ids, max_lens)}
    except:
        responses = {i: r for i, r in zip(ids, responses)}
        id2input_len = {i: r for i, r in zip(ids, input_lens)}
        id2max_len = {i: r for i, r in zip(ids, max_lens)}

    test_data["input_len"] = test_data["index"].apply(lambda x: id2input_len[x])
    test_data["max_len"] = test_data["index"].apply(lambda x: id2max_len[x])

    response_example = list(responses.values())[0]
    wandb.config.update({"response_example": response_example})
    print("response_example", response_example, "\n\n")

    test_data["llm_response"] = test_data["index"].apply(lambda x: responses[x])

    test_data.to_pickle(f"llm_output/test_{run.name}.pkl")

    def response2pred(response):
        pred = "UNK"

        response = response.strip().lower()

        if response.startswith("yes"):
            pred = 1
        elif response.startswith("no"):
            pred = 0

        return pred

    test_data["llm_pred"] = test_data["llm_response"].apply(lambda x: response2pred(x))
    print(classification_report(test_data["labels"].tolist(), test_data["llm_pred"].tolist()))

    wandb.finish()


if __name__ == '__main__':
    main()