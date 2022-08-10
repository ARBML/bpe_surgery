import time 
from const import RESERVERD_WORDS, UNK 
from tqdm.notebook import tqdm

#https://aclanthology.org/2020.findings-emnlp.414.pdf
def evaluate_on_token_length(tokenizer):
    vocab = tokenizer.vocab
    pbar = tqdm(total=len(vocab))
    freq = []
    for word in vocab:
        if word not in RESERVERD_WORDS:
            freq.append(len(word)) 
        pbar.update(1)
    return freq 

#https://aclanthology.org/2020.findings-emnlp.414.pdf
def evaluate_on_segmentation(corpus, gold_tokens, tokenizer):
    freq = 0
    words = corpus.split(" ")
    pbar = tqdm(total=len(words)) 
    for word in words:
        word_tokens = tokenizer._tokenize_word(word)
        word_gold_tokens = gold_tokens[word]
        freq += len(set(word_tokens) & set(word_gold_tokens))/len(word_tokens)
        pbar.update(1)
    return freq/len(words)

def evaluate_on_compression_factor(corpus, tokenizer, normalized=True):
        factor = 0
        words = corpus.split()
        pbar = tqdm(total=len(words)) 
        for word in words:
            tokens = tokenizer._tokenize_word(word)
            factor += (
                len(word) + 1
                if UNK in tokens
                else len(tokens)
            )
            pbar.update(1)
        if normalized:
            normalized_factor = factor / (
                sum(len(word) + 1 for word in words)
            )
            return normalized_factor
        return factor

def evaluate_on_inf_speed(corpus, tokenizer):
    start_time = time.time()
    tokenizer.tokenize(corpus)
    end_time = time.time()
    return end_time - start_time 

def evaluate_on_num_tokens(corpus, tokenizer):
    num_tokens = 0
    words = corpus.split(" ") 
    pbar = tqdm(total=len(words)) 
    for word in words:
        word_tokens = tokenizer._tokenize_word(word)
        num_tokens += len(word_tokens)
        pbar.update(1)
    return num_tokens


