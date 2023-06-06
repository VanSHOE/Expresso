from icecream import ic
import torchtext
from torchtext.vocab import GloVe
from tqdm import tqdm
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import csv
from pathlib import Path
import random
import time
import pickle as pkl

random.seed(time.time())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BATCH_SIZE = 32
GLOVE_DIM = 200
LEARNING_RATE = 0.001
HIDDEN_SIZE = 200
EPOCHS = 50
PATIENCE = 5

MODELNAME = "KN_CLASSIFER"

if not os.path.exists("dataKNClassifier.pkl"):
    datasetLocal = {"train": [], "validation": [], "test": []}

    testFile = "./knPerps.csv"

    with open(testFile, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            copyRow = {}
            copyRow["correct"] = 1 if row["label"] == "correct" else 0
            copyRow["Perplexity"] = row["perplexity"]
            randomNo = random.random()
            if randomNo < 0.6:
                datasetLocal["train"].append(copyRow)
            elif randomNo < 0.8:
                datasetLocal["validation"].append(copyRow)
            else:
                datasetLocal["test"].append(copyRow)

    pkl.dump(datasetLocal, open("dataKNClassifier.pkl", "wb"))
else:
    print("Loading from pickle")
    datasetLocal = pkl.load(open("dataKNClassifier.pkl", "rb"))

ic(len(datasetLocal["train"]), len(datasetLocal["validation"]),
    len(datasetLocal["test"]))
ic(datasetLocal["train"][0:2])
ic(datasetLocal["validation"][0:2])
ic(datasetLocal["test"][0:2])


class Perp_Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor([float(self.dataset[idx]["Perplexity"])], device=device), torch.tensor([int(self.dataset[idx]["correct"])], device=device)


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.classifier(x)


def train(model, trainData, valData):
    totalLoss = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    prevLoss = 999999999
    prevValLoss = 999999999
    curPat = PATIENCE
    for epoch in range(EPOCHS):
        model.train()

        ELoss = 0
        print("Epoch: ", epoch + 1)
        pbar = tqdm(
            dataLoader, desc=f"Pre-Training")
        cur = 0
        for (perplexity, label) in pbar:
            optimizer.zero_grad()
            output = model(perplexity)
            # ic(output.shape, label.shape)
            # ic(label.shape)
            label = label.squeeze(dim=-1)
            # ic(label.shape)

            loss = criterion(output, label)

            # ic(output.shape, label.shape)
            ELoss += loss.item()
            cur += 1

            pbar.set_description(
                f"Pre-Training | Loss: {ELoss / cur : .10f}")
            loss.backward()
            optimizer.step()

        prevLoss = ELoss

        with torch.no_grad():
            model.eval()
            ELoss_V = 0
            dataLoaderV = DataLoader(
                valData, batch_size=BATCH_SIZE, shuffle=True)
            pbar = tqdm(
                dataLoaderV, desc=f"Validation")
            cur = 0
            for (perplexity, label) in pbar:
                output = model(perplexity)
                loss = criterion(output, label.squeeze(dim=1))

                ELoss_V += loss.item()
                cur += 1

                pbar.set_description(
                    f"Validation | Loss: {ELoss_V / cur : .10f}")

            if prevValLoss > ELoss_V:
                torch.save(model, f"{MODELNAME}.pt")
                curPat = PATIENCE
            else:
                curPat -= 1
                if curPat == 0:
                    break

            prevValLoss = ELoss_V


def testModel(model, testDataset, test=True):
    model.eval()
    dataLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)
    trueVals = np.array([])
    predVals = np.array([])
    predValProbs = np.array([])
    with torch.no_grad():
        for (perplexity, label) in tqdm(dataLoader, desc="Testing"):
            score = model(perplexity)
            # ic(perplexity, score)
            # exit(0)
            probs = torch.softmax(score, dim=1)[:, 1]

            pred = torch.argmax(score, dim=1)
            trueVals = np.append(trueVals, label.cpu().numpy())
            predVals = np.append(predVals, pred.cpu().numpy())
            predValProbs = np.append(predValProbs, probs.cpu().numpy())

    print(classification_report(trueVals, predVals))
    if test:
        confusion = confusion_matrix(trueVals, predVals)
        cm = ConfusionMatrixDisplay(confusion)
        cm.plot()
        # save
        plt.savefig("confusionKNClassifier.png")
        # clear plt
        plt.clf()
        # ic(predValProbs.shape, trueVals.shape)
        roc = roc_curve(trueVals, predValProbs, pos_label=1)
        # axis names
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot(roc[0], roc[1])

        plt.savefig("rocKNClassifer.png")
        auc = roc_auc_score(trueVals, predValProbs)
        print("AUC: ", auc)


modelGlobal = Classifier()
modelGlobal.to(device)
if os.path.exists(f"{MODELNAME}.pt"):
    modelGlobal = torch.load(f"{MODELNAME}.pt")
else:
    train(modelGlobal, Perp_Dataset(datasetLocal["train"]),
          Perp_Dataset(datasetLocal["validation"]))

testModel(modelGlobal, Perp_Dataset(datasetLocal["test"]))
