import torch
from torch.autograd import Variable
from vocab import *
from config import *
import numpy as np
import random

np.random.seed(0)

# Taken from https://github.com/sanyam5/skip-thoughts
# Then modified to keep multiple dataset content.

class DataLoader:
    EOS = 0  # to mean end of sentence
    UNK = 1  # to mean unknown token

    maxlen = MAXLEN

    def __init__(self, text_file=None, sentences=None, word_dict=None):

        if text_file:
            corpus, datasets = text_file
            sentences = {}
            print("Loading corpus at {}".format(corpus))
            print("Datasets used:")
            for name, path in datasets.items():
                print(name)
                with open(path, "rt") as f:
                    sentences[name] = f.readlines()
                   
            print("Making dictionary for these words")
            word_dict = build_and_save_dictionary(sentences, source=corpus)

        assert sentences and word_dict, "Please provide the file to extract from or give sentences and word_dict"

        self.sentences = sentences
        self.word_dict = word_dict
        print("Making reverse dictionary")
        self.revmap = list(self.word_dict.items())
        
        self.categories = list(sentences.keys())
        self.n_sentences = [len(stcs) for _,stcs in sentences.items()]
        tot_stcs = sum(self.n_sentences)
        self.p = [x/tot_stcs for x in self.n_sentences]
        print(self.n_sentences)
        print(self.p)

    def convert_sentence_to_indices(self, sentence, tensor=True):
        indices = [
                      # assign an integer to each word, if the word is too rare assign unknown token
                      self.word_dict.get(w) if self.word_dict.get(w, VOCAB_SIZE + 1) < VOCAB_SIZE else self.UNK

                      for w in sentence.split()  # split into words on spaces
                  ][: self.maxlen - 1]  # take only maxlen-1 words per sentence at the most.

        # last words are EOS
        indices += [self.EOS] * (self.maxlen - len(indices))

        indices = np.array(indices)
        if tensor:
            indices = Variable(torch.from_numpy(indices))
            if USE_CUDA:
                indices = indices.cuda(CUDA_DEVICE)

        return indices

    def convert_indices_to_sentences(self, indices):

        def convert_index_to_word(idx):

            idx = idx.item()
            if idx == 0:
                return "EOS"
            elif idx == 1:
                return "UNK"
            
            search_idx = idx - 2
            if search_idx >= len(self.revmap):
                return "NA"
            
            word, idx_ = self.revmap[search_idx]

            assert idx_ == idx
            return word

        words = [convert_index_to_word(idx) for idx in indices]

        return " ".join(words)

    def fetch_batch(self, batch_size):
        chosen_cat = np.random.choice(len(self.categories), p=self.p)
        cat = self.categories[chosen_cat]
        first_index = random.randint(0, self.n_sentences[chosen_cat] - batch_size)
        
        batch = []
        lengths = []

        for i in range(first_index, first_index + batch_size):
            sent = self.sentences[cat][i]
            ind = self.convert_sentence_to_indices(sent)
            batch.append(ind)
            lengths.append(min(len(sent.split()), MAXLEN))

        batch = torch.stack(batch)
        lengths = np.array(lengths)

        return batch, chosen_cat, lengths
