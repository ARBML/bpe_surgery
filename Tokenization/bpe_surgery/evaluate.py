import time 
from .const import RESERVERD_WORDS, UNK 
from tqdm import tqdm

class Evaluate:
    def __init__(self, corpus, gold_tokens, tokenizer):
        self.corpus = corpus
        self.gold_tokens = gold_tokens
        self.tokenizer = tokenizer 

    #https://aclanthology.org/2020.findings-emnlp.414.pdf
    def evaluate_on_token_length(self):
        vocab = self.tokenizer.vocab
        pbar = tqdm(total=len(vocab))
        freq = []
        for word in vocab:
            if word not in RESERVERD_WORDS:
                freq.append(len(word)) 
            pbar.update(1)
        return freq 

    #https://aclanthology.org/2020.findings-emnlp.414.pdf
    def evaluate_on_segmentation(self):
        freq = 0
        word_tokens = self.tokenizer.tokenize(self.corpus)
        assert len(word_tokens) == len(self.gold_tokens) 
        
        pbar = tqdm(total=len(self.gold_tokens)) 
        for i, word_gold_token in enumerate(self.gold_tokens):
            freq += len(set(word_tokens[i]) & set(word_gold_token))/len(set(word_tokens[i]) | set(word_gold_token)) 
            pbar.update(1)

        return freq/len(self.gold_tokens)

    def evaluate_on_compression_factor(self, normalized=True):
            factor = 0
            word_tokens = self.tokenizer.tokenize(self.corpus)
            words = self.corpus.split(" ")

            assert len(words) == len(word_tokens)
            pbar = tqdm(total=len(words)) 
            for i, word in enumerate(words):
                factor += (
                    len(word) + 1
                    if UNK in word_tokens[i]
                    else len(word_tokens[i])
                )
                pbar.update(1)
            if normalized:
                normalized_factor = factor / (
                    sum(len(word) + 1 for word in words)
                )
                return normalized_factor
            return factor

    def evaluate_on_inf_speed(self):
        start_time = time.time()
        self.tokenizer.tokenize(self.corpus)
        end_time = time.time()
        return end_time - start_time 

    def evaluate_on_num_tokens(self):
        tokens = self.tokenizer.tokenize(self.corpus)
        return sum(len(word_tokens) for word_tokens in tokens)


