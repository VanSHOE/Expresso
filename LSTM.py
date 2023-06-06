import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import nltk  # for tokenizing sentences
from smoothing import get_token_list, sentence_tokenizer, rem_low_freq
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import numpy as np
import random
import time
import sys
import csv
from tqdm import tqdm
import pickle as pkl
import random
import time
from icecream import ic
import os
import gc
TESTLIMIT = 473892
random.seed(time.time())
MODEL = "LM6"

# nltk.download('punkt')

sentenceLens = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

# device = "cpu"

CUT_OFF = 40


class Data(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        self.device = device
        print(len(self.sentences))

        self.cutoff = CUT_OFF

        self.Ssentences = [
            sentence for sentence in self.sentences if self.cutoff > len(sentence) > 0]
        self.Lsentences = [
            sentence for sentence in self.sentences if len(sentence) >= self.cutoff]
        # split long sentences into 2
        self.sentences = []
        for sentence in self.Lsentences:
            half = len(sentence) // 2
            self.sentences.append(sentence[:half])
            self.sentences.append(sentence[half:])
        self.sentences += self.Ssentences

        self.sentences = [sentence + ["<eos>"]
                          for sentence in self.sentences if len(sentence) <= self.cutoff]

        # print(self.sentences)
        print("Sentences: ", len(self.sentences))
        ic(self.sentences[:3])
        self.vocab = set()
        self.mxSentSize = 0
        # self.mxSentSize = 20
        for sentence in self.sentences:
            if len(sentence) not in sentenceLens:
                sentenceLens[len(sentence)] = 1
            else:
                sentenceLens[len(sentence)] += 1
            for token in sentence:
                self.vocab.add(token)

            if len(sentence) > self.mxSentSize:
                self.mxSentSize = len(sentence)

        self.vocab = list(self.vocab)

        # add padding token
        self.vocab.append("<pad>")
        # add Unknown
        if "<unk>" not in self.vocab:
            self.vocab.append("<unk>")

        if "<eos>" not in self.vocab:
            self.vocab.append("<eos>")

        self.vocabSet = set(self.vocab)
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        # pad each sentence to 40
        for i in range(len(self.sentences)):
            # print(type(self.sentences[i]))
            self.sentences[i] = self.sentences[i] + ["<pad>"] * \
                (self.mxSentSize - len(self.sentences[i]))
        self.sentencesIdx = torch.tensor([[self.w2idx[token] for token in sentence] for sentence in self.sentences],
                                         device=self.device)
        self.padIdx = self.w2idx["<pad>"]

    def handle_unknowns(self, vocab_set, vocab):
        wrd2rem = set()
        for i in tqdm(range(len(self.sentences)), desc="Handling unknowns"):
            for j in range(len(self.sentences[i])):
                if self.sentences[i][j] not in vocab_set:
                    # remove from vocab and vocab set
                    # if self.sentences[i][j] in self.vocab:
                    #     self.vocab.remove(self.sentences[i][j])
                    wrd2rem.add(self.sentences[i][j])
                    if self.sentences[i][j] in self.vocabSet:
                        self.vocabSet.remove(self.sentences[i][j])
                    self.sentences[i][j] = "<unk>"

        self.vocab = [w for w in self.vocab if w not in wrd2rem]
        self.w2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2w = {i: w for i, w in enumerate(vocab)}
        self.sentencesIdx = torch.tensor([[self.w2idx[token] for token in sentence] for sentence in self.sentences],
                                         device=self.device)

    def __len__(self):
        return len(self.sentencesIdx)

    def __getitem__(self, idx):
        # sentence, last word
        return self.sentencesIdx[idx][:-1], self.sentencesIdx[idx][1:]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, vocab_size):
        # call the init function of the parent class
        super(LSTM, self).__init__()
        self.device = device
        self.num_layers = num_layers  # number of LSTM layers
        self.hidden_size = hidden_size  # size of LSTM hidden state
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)  # LSTM layer
        # linear layer to map the hidden state to output classes
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.train_data = None

        self.elayer = nn.Embedding(vocab_size, input_size)

        self.to(self.device)

    def forward(self, x, state=None):
        # Set initial states for the LSTM layer or use the states passed from the previous time step
        embeddings = self.elayer(x)

        # Forward propagate through the LSTM layer
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(embeddings)
        return self.decoder(out)


def train(model, data, optimizer, criterion, valDat, maxPat=5):
    epoch_loss = 0

    dataL = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    lossDec = True
    prevLoss = 10000000
    prevValLoss = 10000000
    epoch = 0
    es_patience = maxPat
    model.train_data = data

    if os.path.exists(f"{MODEL}.pt"):
        print("Loading partially trained model")
        model.load_state_dict(torch.load(f"{MODEL}.pt"))

    while lossDec:
        model.train()
        epoch_loss = 0
        pbar = tqdm(enumerate(dataL),
                    desc=f"Epoch {epoch + 1}", total=len(dataL))
        cur = 0
        for i, (x, y) in pbar:
            optimizer.zero_grad()
            x = x.to(model.device)

            y = y.to(model.device)

            output = model(x)

            y = y.view(-1)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            cur += 1

            pbar.set_description(
                f"Epoch {epoch + 1} | Loss: {epoch_loss / cur:.10f}")
            # print loss every 100 batches
            # if i % 100 == 0:
            #     print(f"Epoch {epoch + 1} Batch {i} loss: {loss.item()}")

        validationLoss = getLossDataset(valDat, model)
        print(f"Validation loss: {validationLoss}")
        if validationLoss - epoch_loss / len(dataL) > 1:
            print("Validation loss increased")
            if es_patience > 0:
                es_patience -= 1

            else:  # early stopping
                print("Early stopping")
                # model = torch.load(f"{MODEL}.pt")
                model.load_state_dict(torch.load(f"{MODEL}.pt"))
                lossDec = False
        else:
            # torch.save(model, f"{MODEL}.pt")
            torch.save(model.state_dict(), f"{MODEL}.pt")
            es_patience = maxPat
        prevValLoss = validationLoss
        model.train()
        if epoch_loss / len(dataL) > prevLoss:
            lossDec = False
        prevLoss = epoch_loss / len(dataL)

        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataL)}")
        epoch += 1


def perplexity(model, sentence):
    sentence = get_token_list(sentence)
    if model.train_data is None:
        print("No training data")
        return
    for tokenIdx in range(len(sentence)):
        if sentence[tokenIdx] not in model.train_data.vocabSet:
            sentence[tokenIdx] = "<unk>"

    # print(sentence)
    sentence = torch.tensor([model.train_data.w2idx[token]
                            for token in sentence], device=model.device)
    y = model(sentence[:-1])
    # print(y.shape)
    probs = torch.nn.functional.softmax(y, dim=-1).cpu().detach().numpy()
    target = sentence[1:]
    perp = 0
    for i in range(len(target)):
        perp += -np.log(probs[i][target[i]])
    return np.exp(perp / len(target.cpu().numpy()))


def getPerpDataset(model, data: Data, filename: str):
    model.eval()

    # check perplexity for each sentence in data
    perp = 0
    perps = {}

    toWrite = []
    csvDicts = []
    print("Calculating perplexity for ", filename)
    sentDone = 0
    with alive_bar(min(TESTLIMIT, len(data.sentences))) as bar:
        for sentence in data.sentences:
            newPerp = perplexity(model, ' '.join(sentence))
            # sentence<tab>perp
            toWrite.append(f"{' '.join(sentence)}\t{newPerp}")
            csvDict = {}  # sentence: perp
            csvDict["Sentence"] = ' '.join(sentence)
            csvDict["Perplexity"] = newPerp
            csvDicts.append(csvDict)

            perp += newPerp
            if newPerp in perps:
                perps[newPerp] += 1
            else:
                perps[newPerp] = 1
            bar()
            sentDone += 1
            if sentDone >= TESTLIMIT:
                break
    # csv Write
    csvOutputFile = open(f"{filename}.csv", "w", encoding="utf-8")
    csvWriter = csv.DictWriter(csvOutputFile, fieldnames=[
                               "Sentence", "Perplexity"])
    csvWriter.writeheader()
    csvWriter.writerows(csvDicts)
    csvOutputFile.close()
    output = open(filename, "w", encoding="utf-8")
    output.write(f"{perp / (min(TESTLIMIT, len(data.sentences)))}\n")
    # write sentences
    output.write('\n'.join(toWrite))
    output.close()
    # print(perps)
    # histogram binned
    plt.hist(perps.keys(), bins=100)
    plt.show()
    print(f"Mean: {perp / (min(TESTLIMIT, len(data.sentences)))}")
    # print median taking into account the freq
    perpList = []
    for perp in perps:
        perpList += [perp] * perps[perp]
    # histogram log scale and name it with filename
    plt.hist(perpList, bins=100, log=True)
    plt.title(filename)
    plt.savefig(f"{filename}.png")
    print(f"Median: {np.median(perpList)}")


def rem_low_freq_sentences(sentences, freq):
    dist = {}
    for sentence in sentences:
        for token in sentence:
            if token in dist:
                dist[token] += 1
            else:
                dist[token] = 1

    # replace with unk
    for sentence in sentences:
        for tokenIdx in range(len(sentence)):
            if dist[sentence[tokenIdx]] < freq:
                sentence[tokenIdx] = "<unk>"

    return sentences


def getLossDataset(data: Data, model):
    model.eval()

    dataL = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    loss = 0

    for i, (x, y) in tqdm(enumerate(dataL), desc=f"Finding Loss on Dataset", total=len(dataL)):
        x = x.to(model.device)
        y = y.to(model.device)

        output = model(x)

        y = y.view(-1)
        output = output.view(-1, output.shape[-1])

        loss += criterion(output, y).item()

    return loss / len(dataL)


def runPerpAnalysis(path: str):
    # seed random with time

    sentences = []
    # csvReader = csv.DictReader(open(path, encoding="utf-8"))

    if not os.path.exists("LSTMdata.pkl"):
        dataSet = pkl.load(open(path, "rb"))
        # ic(dataSet[:5])
        # ic(len(dataSet))
        correctSentences = [data[1] for data in dataSet[:5000000]]
        incorrectSentences = [data[0] for data in dataSet[:5000000]]

        del dataSet
        gc.collect()

        ic(correctSentences[:2])
        ic(len(correctSentences))

        ic(incorrectSentences[:2])
        ic(len(incorrectSentences))

        # split train test validation using random
        trainText = []
        testText = []
        valText = []
        testIncorrect = [get_token_list(sentence)
                         for sentence in incorrectSentences]
        errors = 0
        trainCall = 0
        testCall = 0
        valCall = 0
        for sentence in tqdm(correctSentences, desc="Splitting"):
            try:
                if len(sentence) > 100:
                    continue
                randomPick = random.random()
                if randomPick <= 0.6:
                    trainText.append(get_token_list(sentence))
                    trainCall += 1
                elif randomPick <= 0.8:
                    testText.append(get_token_list(sentence))
                    testCall += 1
                else:
                    valText.append(get_token_list(sentence))
                    valCall += 1
            except:
                errors += 1

        print(f"Errors: {errors}")
        ic(trainCall / (trainCall + testCall + valCall), testCall / (trainCall +
           testCall + valCall), valCall / (trainCall + testCall + valCall))

        trainText = rem_low_freq_sentences(trainText, 3)

        train_data = Data(trainText)
        test_data = Data(testText)
        val_data = Data(valText)
        incorrect_data = Data(testIncorrect)
        test_data.handle_unknowns(train_data.vocabSet, train_data.vocab)
        val_data.handle_unknowns(train_data.vocabSet, train_data.vocab)
        incorrect_data.handle_unknowns(train_data.vocabSet, train_data.vocab)

        pkl.dump((train_data, test_data, val_data, incorrect_data),
                 open("LSTMdata.pkl", "wb"))

    else:
        train_data, test_data, val_data, incorrect_data = pkl.load(
            open("LSTMdata.pkl", "rb"))

    print("Data Loaded")
    model = LSTM(300, 300, 1, len(train_data.vocab), len(train_data.vocab))
    model.train_data = train_data

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    if not os.path.exists(f"2020115006_{MODEL}.pth") and False:
        train(model, train_data, optimizer, criterion, val_data, 4)
        torch.save(model, f"2020115006_{MODEL}.pth")
    elif os.path.exists(f"./LSTM_Final.pt"):
        model.load_state_dict(torch.load(f"./LSTM_Final.pt"))

    # model = torch.load(f"2020115006_{MODEL}.pth")
    # getPerpDataset(model, val_data, "val.log")
    # getPerpDataset(model, test_data, f"2020115006_{MODEL}_test-perplexity.txt")

    # getPerpDataset(model, train_data,
    #                f"2020115006_{MODEL}_train-perplexity.txt")

    getPerpDataset(model, incorrect_data,
                   f"2020115006_{MODEL}_incorrect-perplexity.txt")


if __name__ == '__main__':
    ic(len(sys.argv))
    if len(sys.argv) == 2:
        path = sys.argv[1]
        sent = input("input sentence: ")

        model = torch.load(path)
        print(perplexity(model, sent))
        exit(0)

    MODEL = "LSTM_Final"
    runPerpAnalysis("./datasetCW.pkl")
    #
    # MODEL = "LM6"
    # runPerpAnalysis("./corpus/Ulysses - James Joyce.txt")
