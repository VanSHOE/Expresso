import pandas as pd
from itertools import islice
from tqdm import tqdm
from icecream import ic
import pickle as pkl

dataset = []

chunksize = 10 ** 6
with pd.read_csv('C4_200M.tsv-00000-of-00010', sep='\t', chunksize=chunksize) as reader:
    for chunk in reader:
        for index, row in tqdm(chunk.iterrows(), total=chunk.shape[0]):
            dataset.append((row[0], row[1]))


ic(len(dataset))
pkl.dump(dataset, open('datasetCW.pkl', 'wb'))
