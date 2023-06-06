import csv
import os
import random
from itertools import chain
from string import punctuation

import datasets
import nltk
import numpy as np
import pandas as pd
import torch
import time
from datasets import Dataset as dDataset
from datasets import load_metric
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pickle as pkl
from tqdm import tqdm
from icecream import ic
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5Tokenizer)
from jiwer import wer
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_name = 't5-base'
nltk.download('punkt')

os.environ["WANDB_DISABLED"] = "true"
pd.set_option('display.max_colwidth', None)

if not os.path.exists("dataframePandasGrammar.pkl"):
    pklData = open('./datasetCW.pkl', 'rb')
    pklData = pkl.load(pklData)
    pklData = pklData[:1000000]
    dataDict = {"input": [], "output": []}
    for data in tqdm(pklData, total=len(pklData), desc="Loading data"):
        dataDict["input"].append(data[0])
        dataDict["output"].append(data[1])

    del pklData
    df = pd.DataFrame(dataDict)
    df.dropna(inplace=True)
    del dataDict
    df.to_pickle("./dataframePandasGrammar.pkl")
else:
    df = pd.read_pickle("./dataframePandasGrammar.pkl")

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)


def calc_token_len(example):
    return len(tokenizer(example).input_ids)


train_df, val_df = train_test_split(df, test_size=0.20, shuffle=True)
train_df, test_df = train_test_split(train_df, test_size=0.20, shuffle=True)
ic(train_df.shape, val_df.shape, test_df.shape)

val_df['input_token_len'] = val_df['input'].apply(calc_token_len)


train_dataset = dDataset.from_pandas(train_df)
test_dataset = dDataset.from_pandas(val_df)


class LangDataset(Dataset):
    def __init__(self, dataset, tokenizer, print_text=False):
        self.dataset = dataset
        self.maxPad = False
        self.tokenizer = tokenizer
        self.max_len = 64

    def __len__(self):
        return len(self.dataset)

    def tokenize_data(self, example):
        input_, target_ = example['input'], example['output']

        # tokenize inputs
        tokenized_inputs = tokenizer(input_, pad_to_max_length=self.maxPad,
                                     max_length=self.max_len,
                                     return_attention_mask=True)

        tokenized_targets = tokenizer(target_, pad_to_max_length=self.maxPad,
                                      max_length=self.max_len,
                                      return_attention_mask=True)

        inputs = {"input_ids": tokenized_inputs['input_ids'],
                  "attention_mask": tokenized_inputs['attention_mask'],
                  "labels": tokenized_targets['input_ids']
                  }

        return inputs

    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset[index])

        return inputs


dataset = LangDataset(test_dataset, tokenizer, True)


rouge_metric = load_metric("rouge")


data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding='longest', return_tensors='pt')


# defining training related arguments
batch_size = 10
args = Seq2SeqTrainingArguments(output_dir="./weights",
                                evaluation_strategy="steps",
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                learning_rate=2e-5,
                                num_train_epochs=2,
                                weight_decay=0.01,
                                save_total_limit=10,
                                predict_with_generate=True,
                                fp16=True,
                                gradient_accumulation_steps=5,
                                eval_steps=9000,
                                save_steps=2000
                                )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    result = rouge_metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}


trainer = Seq2SeqTrainer(model=model,
                         args=args,
                         train_dataset=LangDataset(
                             train_dataset, tokenizer),
                         eval_dataset=LangDataset(test_dataset, tokenizer),
                         tokenizer=tokenizer,
                         data_collator=data_collator,
                         compute_metrics=compute_metrics)

if not os.path.exists('t5'):
    trainer.train("./weights/checkpoint-3000")
    trainer.save_model('t5')

print('Loading model')
model = T5ForConditionalGeneration.from_pretrained('./t5/')
model.to(device)


def correct_grammar(input_text, num_return_sequences):
    batch = tokenizer([input_text], truncation=True, padding='max_length',
                      max_length=64, return_tensors="pt")
    batch = batch.to(device)
    translated = model.generate(**batch, max_length=64, num_beams=4,
                                num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


if not os.path.exists('./grammarOutput.csv'):
    test_df = test_df[:1000]
    grammarOutputFile = open('./grammarOutput.csv', 'w')
    grammarOutputWriter = csv.DictWriter(
        grammarOutputFile, fieldnames=['input', 'output', 'truth'])
    grammarOutputWriter.writeheader()

    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
        inp = row['input']
        prediction = correct_grammar(inp, num_return_sequences=1)[0]
        grammarOutputWriter.writerow(
            {'input': inp, 'output': prediction, 'truth': row['output']})
        # truth = row['output']
        # total_wer += wer(truth, prediction)
        # total_fake_wer += wer(truth, inp)

    grammarOutputFile.close()
# print('WER: ', total_wer/len(test_df))
# print('Fake WER: ', total_fake_wer/len(test_df))

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
total_rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
total_rouge_scoresFake = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
grammarFile = open('./grammarOutput.csv', 'r')
grammarReader = csv.DictReader(grammarFile)
total_wer = 0
total_fake_wer = 0
correct = []
predicted = []
Inps = []
data = list(grammarReader)
for row in tqdm(data, desc="Calculating Stats"):
    truth = row['truth']
    prediction = row['output']
    inp = row['input']

    correct.append(truth)
    predicted.append(prediction)
    Inps.append(inp)

    rougeScore = scorer.score(truth, prediction)
    fakeRougeScore = scorer.score(truth, inp)

    for key in rougeScore:
        total_rouge_scores[key] += rougeScore[key].fmeasure
        total_rouge_scoresFake[key] += fakeRougeScore[key].fmeasure

    total_wer += wer(truth, prediction)
    total_fake_wer += wer(truth, inp)


print('WER: ', total_wer/len(data))
print('Fake WER: ', total_fake_wer/len(data))
percentChange = (total_wer-total_fake_wer)/total_fake_wer
print(f"Percent change: {percentChange*100}%")
avgRouge = {}
avgRougeFake = {}
for key in total_rouge_scores:
    avgRouge[key] = total_rouge_scores[key]/len(data)
    avgRougeFake[key] = total_rouge_scoresFake[key]/len(data)
print('Avg ROUGE: ', avgRouge)
print('Avg Fake ROUGE: ', avgRougeFake)

percChangeRouge = {}
for key in avgRouge:
    percChangeRouge[key] = 100 * \
        (avgRouge[key]-avgRougeFake[key])/avgRougeFake[key]
print(f'Percent change ROUGE: {percChangeRouge}')
bleuCorrect = [[text.split()] for text in correct]
bleuPredicted = [text.split() for text in predicted]
blueInps = [text.split() for text in Inps]
actualBleu = corpus_bleu(bleuCorrect, bleuPredicted)
print('BLEU: ', actualBleu)

fakeBleu = corpus_bleu(bleuCorrect, blueInps)
print("Fake BLEU: ", fakeBleu)

print(f"Percent change: {((actualBleu-fakeBleu)/fakeBleu)*100}%")
