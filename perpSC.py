from icecream import ic
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from datasets import load_dataset
import csv
import numpy as np
import torch
import transformers
import os
import pickle as pkl
import random
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from matplotlib import pyplot as plt
random.seed(time.time())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2)
model.to(device)
os.environ["WANDB_DISABLED"] = "true"
if not os.path.exists("SCTransformersDataTest.csv") or not os.path.exists("SCTransformersDataTrain.csv"):
    dataSetLocal = {"train": [], "validation": [],  "test": []}
    dataPkl = pkl.load(open("./datasetCW.pkl", "rb"))
    correctSentences = [data[1] for data in dataPkl[:100000]]
    incorrectSentences = [data[0] for data in dataPkl[:100000]]
    errors = 0
    for sentence in tqdm(correctSentences, total=len(correctSentences)):
        try:
            if len(sentence) < 3:
                continue
            randomNo = random.random()
            if randomNo < 0.6:
                dataSetLocal["train"].append({"text": sentence, "labels": 1})
            elif randomNo < 0.8:
                dataSetLocal["validation"].append(
                    {"text": sentence, "labels": 1})
            else:
                dataSetLocal["test"].append({"text": sentence, "labels": 1})
        except:
            errors += 1

    for sentence in tqdm(incorrectSentences, total=len(incorrectSentences)):
        try:
            if len(sentence) < 3:
                continue
            randomNo = random.random()
            if randomNo < 0.6:
                dataSetLocal["train"].append({"text": sentence, "labels": 0})
            elif randomNo < 0.8:
                dataSetLocal["validation"].append(
                    {"text": sentence, "labels": 0})
            else:
                dataSetLocal["test"].append({"text": sentence, "labels": 0})
        except:
            errors += 1

    print("Errors: ", errors)

    # pkl.dump(dataSetLocal, open("SCTransformersData.pkl", "wb"))
    csvOutputFile = open("SCTransformersDataTrain.csv", "w")
    csvWriter = csv.DictWriter(csvOutputFile, fieldnames=["text", "labels"])
    csvWriter.writeheader()
    for data in tqdm(dataSetLocal["train"], total=len(dataSetLocal["train"]), desc="Writing Train Data"):
        csvWriter.writerow(data)
    csvOutputFile.close()

    csvOutputFile = open("SCTransformersDataValidation.csv", "w")
    csvWriter = csv.DictWriter(csvOutputFile, fieldnames=["text", "labels"])
    csvWriter.writeheader()
    for data in tqdm(dataSetLocal["validation"], total=len(dataSetLocal["validation"]), desc="Writing Validation Data"):
        csvWriter.writerow(data)
    csvOutputFile.close()

    csvOutputFile = open("SCTransformersDataTest.csv", "w")
    csvWriter = csv.DictWriter(csvOutputFile, fieldnames=["text", "labels"])
    csvWriter.writeheader()
    for data in tqdm(dataSetLocal["test"], total=len(dataSetLocal["test"]), desc="Writing Test Data"):
        csvWriter.writerow(data)
    csvOutputFile.close()


def tokenize_function(collection):
    return tokenizer(collection["text"], truncation=True, padding="max_length")


if not os.path.exists("SCTransformersDatasetComplete.pkl"):
    raw_datasets = load_dataset("csv", data_files={
                                "train": "SCTransformersDataTrain.csv",
                                "validation": "SCTransformersDataValidation.csv",
                                "test": "SCTransformersDataTest.csv"})

    print(raw_datasets)

    complete_dataset = raw_datasets.map(tokenize_function, batched=True)

    pkl.dump(complete_dataset, open("SCTransformersDatasetComplete.pkl", "wb"))
else:
    complete_dataset = pkl.load(
        open("SCTransformersDatasetComplete.pkl", "rb"))


# # remove text
# complete_dataset["train"] = complete_dataset["train"].remove_columns(["text"])
# complete_dataset["test"] = complete_dataset["test"].remove_columns(["text"])

print(complete_dataset)
print(complete_dataset["train"])
print(complete_dataset["validation"])
print(complete_dataset["test"])


if not os.path.exists('SCSaved'):

    BATCH_SIZE = 10
    trainingArgs = TrainingArguments("SCTransformersModel", evaluation_strategy="epoch", learning_rate=1e-5, per_device_train_batch_size=BATCH_SIZE, save_strategy="epoch",
                                     per_device_eval_batch_size=BATCH_SIZE, num_train_epochs=5, weight_decay=0.01, load_best_model_at_end=True)

    trainer = Trainer(model=model, args=trainingArgs,
                      train_dataset=complete_dataset["train"], eval_dataset=complete_dataset["validation"])
    trainer.train('./SCTransformersModel/checkpoint-5992')
    trainer.save_model("SCSaved")

print("Loading model")
model = AutoModelForSequenceClassification.from_pretrained("./SCSaved/")
model.to(device)


def classify_sentence(sentence, model, tokenizer):
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_tensors="pt",
        truncation=True
    )
    # put inputs to device
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    category = torch.argmax(logits).item()
    return category, logits


predVals = []
trueVals = []
predValProbs = []

pbar = tqdm(complete_dataset["test"], total=len(
    complete_dataset["test"]), desc="Classifying")
correct = 0
total = 0
for tup in pbar:
    result, logits = classify_sentence(tup["text"], model, tokenizer)
    predVals.append(result)
    trueVals.append(tup["labels"])

    if predVals[-1] == trueVals[-1]:
        correct += 1
    predValProbs.append(logits[0][1].item())

    total += 1
    pbar.set_description(f"Classifying | Accuracy: {100 * correct/total}%")

print(classification_report(trueVals, predVals))
confusion = confusion_matrix(trueVals, predVals)
cm = ConfusionMatrixDisplay(confusion)
cm.plot()
plt.savefig("confusionTransSCClassifier.png")
# clear plt
plt.clf()


# ic(predValProbs.shape, trueVals.shape)
roc = roc_curve(trueVals, predValProbs, pos_label=1)
# axis names
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(roc[0], roc[1])

plt.savefig("rocTransSCClassifer.png")
auc = roc_auc_score(trueVals, predValProbs)
print("AUC: ", auc)
