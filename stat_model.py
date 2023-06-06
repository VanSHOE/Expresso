import re
import numpy as np
from pprint import pprint
import random
import time
import sys
from alive_progress import alive_bar
import nltk
from nltk import word_tokenize
import pickle
import csv

NGRAM_SIZE = 4
ngramDicts = {}

MODEL = 1


def get_token_list(in_text: str) -> list:
    """
    Tokenizes the input text file
    :param in_text: input text
    :return: list of tokens
    """

    # lower case it
    in_text = in_text.lower()
    # tokenize hashtags
    in_text = re.sub(r"#(\w+)", r"<HASHTAG> ", in_text)
    in_text = re.sub(r'\d+(,(\d+))*(\.(\d+))?%?\s', '<NUMBER> ', in_text)
    # tokenize mentions
    in_text = re.sub(r"@(\w+)", r"<MENTION> ", in_text)
    # tokenize urls
    in_text = re.sub(r"http\S+", r"<URL> ", in_text)
    # starting with www
    in_text = re.sub(r"www\S+", r"<URL> ", in_text)

    special_chars = [' ', '*', '!', '?', '.', ',', ';', ':', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '-', '_', '—'
                                                                                                                     '=',
                     '+', '`', '~', '@', '#', '$', '%', '^', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                     '9']

    # pad the special characters with spaces
    for char in special_chars:
        in_text = in_text.replace(char, ' ')

    # pad < and > with spaces
    in_text = in_text.replace('<', ' <')
    in_text = in_text.replace('>', '> ')

    return in_text.split()


def sentence_tokenizer(fullText: str, thresh: int) -> list:
    """
    Tokenizes the input text file into sentences
    :param fullText: input text
    :return: list of sentences
    """
    # lower case it
    fullText = fullText.lower()
    # tokenize hashtags
    fullText = re.sub(r"#(\w+)", r"<HASHTAG> ", fullText)
    # tokenize mentions
    fullText = re.sub(r"@(\w+)", r"<MENTION> ", fullText)
    # tokenize urls
    fullText = re.sub(r"http\S+", r"<URL> ", fullText)
    # starting with www
    fullText = re.sub(r"www\S+", r"<URL> ", fullText)

    sentenceEnders = ['.', '!', '?']

    # split on sentence enders handling cases such as Mr. etc

    fullText = fullText.replace('mr.', 'mr')
    fullText = fullText.replace('mrs.', 'mrs')
    fullText = fullText.replace('dr.', 'dr')
    fullText = fullText.replace('st.', 'st')
    fullText = fullText.replace('co.', 'co')
    fullText = fullText.replace('inc.', 'inc')
    fullText = fullText.replace('e.g.', 'eg')
    fullText = fullText.replace('i.e.', 'ie')
    fullText = fullText.replace('etc.', 'etc')
    fullText = fullText.replace('vs.', 'vs')
    fullText = fullText.replace('u.s.', 'us')

    # fullText = rem_low_freq(fullText.split(), 1)
    # # join list with space
    # fullText = ' '.join(fullText)

    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', fullText)

    sentences = [s.replace('\n', ' ') for s in sentences]
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s != '']
    sentences = [get_token_list(s) for s in sentences]

    tokenDict = {}
    for sentence in sentences:
        for token in sentence:
            if token in tokenDict:
                tokenDict[token] += 1
            else:
                tokenDict[token] = 1

    for sentence in sentences:
        for i in range(len(sentence)):
            if tokenDict[sentence[i]] <= thresh:
                sentence[i] = '<unk>'

    return sentences


def rem_low_freq(tokens: list, threshold: int) -> list:
    """
    Removes tokens from the input list that occur less than the threshold and replace them with <unk>
    :param tokens: list of tokens
    :param threshold: threshold
    :return: list of tokens with low frequency tokens removed
    """
    # get the frequency of each token
    freq = {}
    for token in tokens:
        if token in freq:
            freq[token] += 1
        else:
            freq[token] = 1

    # remove tokens with frequency less than threshold
    for token in list(freq.keys()):
        if freq[token] <= threshold:
            del freq[token]

    # replace all tokens not in freq with <unk>
    for i in range(len(tokens)):
        if tokens[i] not in freq:
            tokens[i] = '<unk>'

    return tokens


def construct_ngram(n: int, token_list: list) -> dict:
    """
    Constructs an n-gram dictionary from the input token list
    :param n: n-gram size
    :param token_list: list of tokens
    :return: n-gram dictionary
    """
    ngram_dict = {}

    for i in range(len(token_list) - n + 1):
        ngram_to_check = token_list[i:i + n]
        cur_dict = ngram_dict
        for j in range(n):
            if ngram_to_check[j] not in cur_dict:
                if j == n - 1:
                    cur_dict[ngram_to_check[j]] = 1
                else:
                    cur_dict[ngram_to_check[j]] = {}
            else:
                if j == n - 1:
                    cur_dict[ngram_to_check[j]] += 1
            cur_dict = cur_dict[ngram_to_check[j]]

    # remove all entities in dictionary tree with count 1 and add <unk> instead

    return ngram_dict


def dfs_count(ngram_dict: dict) -> int:
    """
    Performs a depth first search on the input n-gram dictionary to count the number of n-grams
    :param ngram_dict: n-gram dictionary
    :return: number of n-grams
    """
    count = 0
    for key, value in ngram_dict.items():
        if isinstance(value, dict):
            count += dfs_count(value)
        else:
            count += 1
    return count


def ngram_count(ngram_dict: dict, ngram: list) -> int:
    """
    Returns the count of the input n-gram
    :param ngram_dict: n-gram dictionary
    :param ngram: n-gram to be counted
    :return: count of the n-gram
    """
    cur_dict = ngram_dict[len(ngram)]
    if len(ngram) == 1:
        if ngram[0] in cur_dict:
            return cur_dict[ngram[0]]
        else:
            return cur_dict['<unk>']
    for i in range(len(ngram)):
        if ngram[i] in cur_dict:
            cur_dict = cur_dict[ngram[i]]
        else:
            return 0
    return cur_dict


dfs_countD = {}


def kneser_ney_smoothing(ngram_dict: dict, d: float, ngram: list) -> float:
    """
    Performs Kneser-Ney smoothing on the input n-gram dictionary
    :param ngram_dict: n-gram dictionary
    :param d: discounting factor
    :param ngram: n-gram to be smoothed
    :return: smoothed probability
    """
    # replace unknown in ngram with <unk>
    for i in range(len(ngram)):
        ngram[i] = ngram[i].lower()
        if ngram[i] not in ngram_dict[1]:
            ngram[i] = '<unk>'

    # print(f'Final ngram: {ngram}')
    if len(ngram) == 1:
        if 2 not in dfs_countD:
            denom = dfs_count(ngram_dict[2])
            dfs_countD[2] = denom
        else:
            denom = dfs_countD[2]
        # count all bigrams ending with ngram[-1]
        count = 0

        for key, value in ngram_dict[2].items():
            if ngram[-1] in value:
                count += 1

        # print(f'Count: {count}, Denom: {denom}')
        return count / denom

    try:
        first = max(ngram_count(ngram_dict, ngram) - d, 0) / ngram_count(ngram_dict, ngram[:-1])
    except ZeroDivisionError:
        return 0

    try:
        cur_dict = ngram_dict[len(ngram)]
        # len of ngram - 1
        for i in range(len(ngram) - 1):
            cur_dict = cur_dict[ngram[i]]
        second_rhs = len(cur_dict)
    except KeyError:
        second_rhs = 0
    second = d * second_rhs / ngram_count(ngram_dict, ngram[:-1])

    return first + second * kneser_ney_smoothing(ngram_dict, d, ngram[1:])


def witten_bell_smoothing(ngram_dict: dict, ngram: list) -> float:
    """
    Performs Witten-Bell smoothing on the input n-gram dictionary
    :param ngram_dict: n-gram dictionary
    :param ngram: n-gram to be smoothed
    :return: smoothed probability
    """
    # replace unknown in ngram with <unk>
    for i in range(len(ngram)):
        ngram[i] = ngram[i].lower()
        if ngram[i] not in ngram_dict[1]:
            ngram[i] = '<unk>'

    if len(ngram) == 1:
        return ngram_count(ngram_dict, ngram) / len(ngram_dict[1])
    try:
        cur_dict = ngram_dict[len(ngram)]
        # len of ngram - 1
        for i in range(len(ngram) - 1):
            cur_dict = cur_dict[ngram[i]]
        lambda_inv_num = len(cur_dict)
    except KeyError:
        lambda_inv_num = 0

    try:
        lambda_inv_num = lambda_inv_num / (lambda_inv_num + ngram_count(ngram_dict, ngram[:-1]))
    except ZeroDivisionError:
        return 0
    lambd = 1 - lambda_inv_num

    first_term = lambd * ngram_count(ngram_dict, ngram) / ngram_count(ngram_dict, ngram[:-1])
    second_term = lambda_inv_num * witten_bell_smoothing(ngram_dict, ngram[1:])

    return first_term + second_term


def sentence_likelihood(ngram_dict: dict, sentence: list, smoothing: str, kneserd=0.75) -> float:
    """
    Calculates the likelihood of the input sentence
    :param ngram_dict: n-gram dictionary
    :param sentence: input sentence
    :param smoothing: smoothing method
    :param kneserd: discounting factor for Kneser-Ney smoothing
    :return: likelihood of the sentence
    """
    # print(sentence)
    tokens = sentence
    if smoothing == 'w' or smoothing == 'wb':
        likelihood = 1
        for i in range(len(tokens) - NGRAM_SIZE + 1):
            likelihood *= witten_bell_smoothing(ngram_dict, tokens[i:i + NGRAM_SIZE])
        return likelihood
    elif smoothing == 'k' or smoothing == 'kn':
        likelihood = 1
        for i in range(len(tokens) - NGRAM_SIZE + 1):
            likelihood *= kneser_ney_smoothing(ngram_dict, kneserd, tokens[i:i + NGRAM_SIZE])
        return likelihood


def perplexity(ngram_dict: dict, sentence: list, smoothing: str, kneserd=0.75) -> float:
    """
    Calculates the perplexity of the input sentence
    :param ngram_dict: n-gram dictionary
    :param sentence: input sentence
    :param smoothing: smoothing method
    :param kneserd: discounting factor for Kneser-Ney smoothing
    :return: perplexity of the sentence
    """
    prob = sentence_likelihood(ngram_dict, sentence, smoothing, kneserd)
    # print(sentence, prob)
    prob = max(prob, 1e-15)

    return pow(prob, -1 / len(sentence))


def get_all_perps(path: str):
    # path = "corpus/Pride and Prejudice - Jane Austen.txt"
    in_text = open(path, "r", encoding="utf-8")
    sentences = in_text.read()
    sentences = sentence_tokenizer(sentences, 1)
    # remove all sentences with less than NGRAM_SIZE tokens
    sentences = [sentence for sentence in sentences if len(sentence) >= NGRAM_SIZE]
    random.seed(time.time())
    random_sentences = random.sample(sentences, 1000)
    in_text.close()

    in_text = open(path, "r", encoding="utf-8")
    trainLines = sentences.copy()
    # remove test lines from training lines

    for sentence in random_sentences:
        trainLines.remove(sentence)

    print(trainLines)
    # combined_text = " ".join(" ".join(trainLines))
    combined_text = ""
    for line in trainLines:
        combined_text += " ".join(line) + " "

    # print(combined_text)
    # exit(33)

    random_sentences = [sentence for sentence in random_sentences if len(sentence) >= NGRAM_SIZE]
    tokens = rem_low_freq(get_token_list(combined_text), 1)

    for n in range(NGRAM_SIZE):
        ngramDicts[n + 1] = construct_ngram(n + 1, tokens)

    # get 1000 random sentences from the corpus using random library

    wb_perplexities = []
    toWrite = []
    with alive_bar(len(random_sentences)) as bar:
        for sentence in random_sentences:
            wb_perplexities.append(perplexity(ngramDicts, sentence, 'wb'))
            # sentence<space>perplexity
            toWrite.append(" ".join(sentence) + "\t" + str(wb_perplexities[-1]))
            bar()

    wb_avg = sum(wb_perplexities) / len(wb_perplexities)
    print(f'Witten-Bell average perplexity: {wb_avg}')
    outputfile = open(f"2020115006_LM{MODEL + 1}_test-perplexity.txt", "w", encoding="utf-8")
    outputfile.write(f"{wb_avg}\n")
    outputfile.write("\n".join(toWrite))
    outputfile.close()
    # calculate perplexity for each sentence using Kneser-Ney smoothing

    kn_perplexities = []
    toWrite = []
    with alive_bar(len(random_sentences)) as bar:
        for sentence in random_sentences:
            kn_perplexities.append(perplexity(ngramDicts, sentence, 'kn'))
            # sentence<space>perplexity
            # outputfile.write(" ".join(sentence) + "\t" + str(kn_perplexities[-1]) + "\n")
            toWrite.append(" ".join(sentence) + "\t" + str(kn_perplexities[-1]))
            bar()

    kn_avg = sum(kn_perplexities) / len(kn_perplexities)

    print(f'Kneser-Ney average perplexity: {kn_avg}')
    outputfile = open(f"2020115006_LM{MODEL}_test-perplexity.txt", "w", encoding="utf-8")
    # write average perplexity at the top
    outputfile.write(f'{kn_avg}\n')
    outputfile.write("\n".join(toWrite))
    outputfile.close()

    wb_perplexities = []
    toWrite = []
    with alive_bar(len(trainLines)) as bar:
        for sentence in trainLines:
            wb_perplexities.append(perplexity(ngramDicts, sentence, 'wb'))
            # sentence<space>perplexity
            # outputfile.write(" ".join(sentence) + "\t" + str(wb_perplexities[-1]) + "\n"
            toWrite.append(" ".join(sentence) + "\t" + str(wb_perplexities[-1]))
            bar()

    wb_avg = sum(wb_perplexities) / len(wb_perplexities)
    print(f'Witten-Bell average perplexity: {wb_avg}')
    # calculate perplexity for each sentence using Kneser-Ney smoothing

    # write average perplexity at the top
    outputfile = open(f"2020115006_LM{MODEL + 1}_train-perplexity.txt", "w", encoding="utf-8")

    outputfile.write(f'{wb_avg}\n')
    outputfile.write("\n".join(toWrite))
    outputfile.close()

    kn_perplexities = []
    toWrite = []
    with alive_bar(len(trainLines)) as bar:
        for sentence in trainLines:
            kn_perplexities.append(perplexity(ngramDicts, sentence, 'kn'))
            # sentence<space>perplexity
            # outputfile.write(" ".join(sentence) + "\t" + str(kn_perplexities[-1]) + "\n")
            toWrite.append(" ".join(sentence) + "\t" + str(kn_perplexities[-1]))
            bar()

    kn_avg = sum(kn_perplexities) / len(kn_perplexities)
    # seek to top
    outputfile = open(f"2020115006_LM{MODEL}_train-perplexity.txt", "w", encoding="utf-8")
    # write average perplexity at the top
    outputfile.write(f'{kn_avg}\n')
    outputfile.write("\n".join(toWrite))

    outputfile.close()
    print(f'Kneser-Ney average perplexity: {kn_avg}')


if __name__ == '__main__':
    # if len(args) == 3:
    #     smoothingAl = args[1]
    #     path = args[2]
    #     fullText = open(path, "r", encoding="utf-8").read()
    #     tokens = rem_low_freq(get_token_list(fullText), 1)
    #     ngramDicts = {}
    #     for n in range(NGRAM_SIZE):
    #         ngramDicts[n + 1] = construct_ngram(n + 1, tokens)
    #     sentence = input("Enter sentence: ")
    #     sentence = get_token_list(sentence)
    #     print(perplexity(ngramDicts, sentence, smoothingAl))
    #     exit(0)
    
    with open('datasetCW.pkl', 'rb') as f:
        data = pickle.load(f)
        
    for i in range(5):
        print(data[i])

    data = data[:40000]

    with open('output.txt', 'w') as output_file:
        for i in data:
            output_file.write(str(i[1]) + '\n')

    with open('incorrect.txt', 'w') as incorrect_file:
        for i in data:
            incorrect_file.write(str(i[0]) + '\n')
    
    input_text = open('output.txt', 'r', encoding="utf-8").read()
    print(type(input_text))
    tokens = rem_low_freq(get_token_list(input_text), 1)
    ngramDicts = {}
    for n in range(NGRAM_SIZE):
        ngramDicts[n + 1] = construct_ngram(n + 1, tokens)
    
    with open('output2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['sentence', 'perplexity', 'label', 'type'])
        test_set = random.sample(data, 20000)
        for i in test_set:
            sentence = get_token_list(i[0])
            if len(sentence) == 0:
                continue
            perp = perplexity(ngramDicts, sentence, 'wb')
            writer.writerow([i[0], perp, 'incorrect', 'wb'])

            sentence = get_token_list(i[1])
            if len(sentence) == 0:
                continue
            perp = perplexity(ngramDicts, sentence, 'wb')
            writer.writerow([i[1], perp, 'correct', 'wb'])


    print("Enter sentence: ")
    sentence = input()
    sentence = get_token_list(sentence)
    print(perplexity(ngramDicts, sentence, 'wb'))
    exit(0)
    ngramDicts = {}
    MODEL = 1
    get_all_perps("corpus/Pride and Prejudice - Jane Austen.txt")
    MODEL = 3
    get_all_perps("corpus/Ulysses - James Joyce.txt")
