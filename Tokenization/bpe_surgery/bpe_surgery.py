import nltk
from collections import Counter
import re
import morfessor
import random 
from farasa.segmenter import FarasaSegmenter
import pickle 
from tqdm.notebook import tqdm
from .const import SOW, UNK, PAD, MORPH_SEP, SEG_SEP, SOS, EOS
import os 

class bpe:
  """
  Tokenizer main class
  """
  def __init__(self, vocab_size = 100, verbose = False, morph = False, morph_with_sep = False, seg = False, prob = 0,
               lang = 'ar', lower_case = True, prefixes = [], suffixes = []):
    self.special_tokens = [PAD, UNK, SOW, SOS, EOS]
    self.vocab = [PAD, UNK, SOW, SOS, EOS]
    self.sow = SOW
    self.sos = SOS
    self.eos = EOS  
    self.morph = morph
    self.prob = prob
    self.seg = seg
    self.morph_with_sep = morph_with_sep     
    self.merges = []
    self.vocab_size = vocab_size
    self.verbose = verbose
    self.lang = lang
    self.prefixes = prefixes 
    self.suffixes = suffixes
    self.affixes = []
    self.name = 'bpe'

    if self.morph:
      self.name += '-morph'
      if lang == 'en':
        io = morfessor.MorfessorIO()
        self.segmenter = io.read_binary_model_file('morfessor.bin')
      elif lang == 'ar':
        self.segmenter = FarasaSegmenter()

    if self.seg:
      self.name += '-seg'
      self.segmenter = FarasaSegmenter()

    self.name += f'-{lang}'
    self.name += f'-{vocab_size}'
    self.lower_case = lower_case
    nltk.download('punkt', quiet=True) 

  def split_affixes(self, affixes):
    """
    splits affixes based on approximations 
    returns: a [list] of tuples ed▁ => [((e, d), (ed, ▁)]
    """
    merges = []
    for affix in affixes:
      if affix.startswith(MORPH_SEP):
        dir = 'rtl'
      else:
        dir = 'ltr'
      chars = list(affix)
      curr_merge = []

      # combine ltr: [i, n , g] => [(i, n), (in, g)] or rtl: [(n, g), (i, ng)]
      while len(chars) > 1:
        if dir == 'ltr':
          curr_merge.append((chars[0], chars[1]))
          chars[1] = chars[0]+chars[1]
          chars = chars[1:]
        else:
          curr_merge.append((chars[-2], chars[-1]))
          chars[-2] = chars[-2]+chars[-1]
          chars = chars[:-1]


      # add the possible merges for the affix
      for merge in curr_merge:
        if merge not in merges:
          merges.append(merge)
    return merges

  def extract_affixes(self, t):
    """
    Extract all affixes given a spcific language using the segmenter
    returns: a [list] of affixes wanted => ['ed</w>']
    """
    affixes = set()
    if self.lang == 'en':
      for word in t.split(' '):
        if len(word) == 0:
          continue
        morphemes = self.segmenter.viterbi_segment(word)[0]
        if len(morphemes) == 1:
          continue
        
        max_len = max([len(morpheme) for morpheme in morphemes])
        for morpheme in morphemes:
          affix = morpheme
          if len(affix) < max_len:
            if word.startswith(affix):
              affix = SOW+affix
              if self.morph_with_sep:
                affix += MORPH_SEP
            elif self.morph_with_sep:
              affix = MORPH_SEP + affix
            affixes.add(affix)

    if self.lang == 'ar':
      for word in self.segmenter.segment(t).split(' '):
        if len(word) == 0:
          continue
        
        morphemes = word.split('+')
        
        if len(morphemes) == 1:
          continue
        
        
        max_len = max([len(morpheme) for morpheme in morphemes])
        for morpheme in morphemes:
          affix = morpheme
          if len(affix) < max_len: #exclude the main stem from the list of affixes TODO: make sure this works
            if word.startswith(affix):
              affix = SOW+affix
              if self.morph_with_sep:
                affix += MORPH_SEP
            elif self.morph_with_sep:
              affix = MORPH_SEP + affix
            affixes.add(affix)

    return affixes
  
  def has_morph_sep(self, bigram):
    """
    check if it has morph separator 
    returns: bool value
    """
    if (bigram[0] + MORPH_SEP in self.vocab) or (MORPH_SEP + bigram[1] in self.vocab):
      # print("can't merge ", bigram)
      return True 
    else:
      return False

  def get_pairs(self, t, check_morph_sep = False):
    """
    get pairs of bigrams with frequency 
    returns: a [counter] of the bigrams in the corpus
    """
    grams = Counter()
    tokens = t.split(' ') 
    
    for i in range(len(tokens[:-1])):
      bigram = (tokens[i], tokens[i+1])
      if bigram[0] == UNK:
        continue
      if len(bigram[0]) * len(bigram[1]) == 0:
        continue
      if check_morph_sep and self.has_morph_sep(bigram):
        continue
      if not bigram[-1].startswith(SOW): # don't combine across words
        grams[bigram] += 1
    # print("================")
    return grams

  def get_pairs_dict(self, t, check_morph_sep = False):
    """
    get pairs of bigrams with frequency 
    returns: a [counter] of the bigrams in the corpus
    """
    grams = Counter()

    for word in t:
      tokens = word.split(' ')
      for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        if bigram[0] == UNK:
          continue
        if check_morph_sep and self.has_morph_sep(bigram):
          continue
        grams[bigram] += t[word]
    return grams
  
  def merge_dict(self, t, bigram):
    """
    join a bigram in a given text corpus
    returns: [dict] with merged bigram
    """
    new_tokens = Counter()
    
    for word in t:
      tokens = word.split(' ')
      out_word = []
      i = 0 
      while i < len(tokens):
        if ('').join(tokens[i:i+2]) == ('').join(bigram):
          out_word.append(''.join(bigram))
          i += 1
        else:
          out_word.append(tokens[i])
        i += 1
      new_tokens[' '.join(out_word)] =  t[word]
    return new_tokens
  
  def merge(self, t, bigram):
    """
    join a bigram in a given text corpus
    returns: dict with merged bigram
    """
    tokens = t.split(' ')
    new_tokens = []
    i = 0 
    while i < len(tokens):
      if ('').join(tokens[i:i+2]) == ('').join(bigram):
        new_tokens.append(('').join(bigram))
        i += 1
      else:
        new_tokens.append(tokens[i])
      i += 1
    return (' ').join(new_tokens)

  def preprocess(self, t):
    """
    split format hello => _ h e l l o 
    returns: processed text
    """

    t = t.replace("\n", "")

    # sp doesn't split on characters like lock-up =/> lock up 
    t = re.sub('([.,?;!-])', ' ', t)

    # not clear how to deal with such special characters like made-up, it seems in sentencepiece it removes the - but 
    # it doesn't to be the same for for \'
    # note that sentecepiece doesnt' seem to split on continued characters like he,then which is annoying.
    t = re.sub('\'', '', t)
    # t = re.sub(' +', ' ', t)

    if self.lower_case:
      t = t.lower()

    return t
  
  def apply_merges(self, corpus, merges):
    """Given a set of merges it applies the merges on the corpus

    Args:
        corpus (dict): dict contains tokens and frequency
        merges (list): list of tuples 

    Returns:
        dict: corpus iwth applied merges
    """
    while True:
        pair_to_merge = None 
        pairs = self.get_pairs_dict(self.corpus, check_morph_sep= False)
        for pair in merges:
          if pair in pairs:
            pair_to_merge = pair
            break
        
        if pair_to_merge:
          self.corpus = self.merge_dict(self.corpus, pair)
        else:
          break
    return corpus 

  def train(self, text = None, file = None):
    """
    train on either a plain text or a file
    returns: [None] 
    """

    if text:
      t = text
    elif file:
      t = open(file, 'r').read()
    else:
      raise("Must use corpus using plain text or a file")

    CONTINUE_PRETRAINED = len(self.merges) != 0

    t = self.preprocess(t)

    if self.seg:
      print("apply pre segmentation ...")
      t = self.segmenter.segment(t)

    self.corpus = Counter()
    for word in t.split(' '):
      if len(word) > 0:
        self.corpus[f'{SOW} '+(' ').join(list(word.strip()))] += 1

  
    if CONTINUE_PRETRAINED:
      print("Continue pretraining from vocab_size : ", len(self.vocab))
      self.corpus = self.apply_merges(self.corpus, self.merges)
    else:
      self.vocab += [char for char in set(t.replace(' ', ''))]


    if len(self.vocab) > self.vocab_size:
        raise Exception('Minimum vocab size is ', len(self.vocab))

    if self.morph and not CONTINUE_PRETRAINED:
      
      if len(self.prefixes) == 0 and len(self.suffixes) == 0:
        print('extracting affixes automatically ...')
        self.affixes = self.extract_affixes(t)
      else:
        print('Use provided affixes')
        if self.morph_with_sep:
          self.affixes = [SOW+affix+MORPH_SEP for affix in self.prefixes]+[MORPH_SEP+affix for affix in self.suffixes]
        else:
          self.affixes = [SOW+affix for affix in self.prefixes]+[affix for affix in self.suffixes]

      if self.verbose:
        print(self.affixes)
      init_merges = self.split_affixes(self.affixes)
      
      for merge in init_merges:
        if len(self.vocab) >= self.vocab_size:
          break
        self.vocab.append(('').join(merge))
        self.merges.append(merge)
      
      self.corpus = self.apply_merges(self.corpus, self.merges)
      if self.verbose:
        print(self.corpus)     

    step = 0
    
    pbar = tqdm(total=self.vocab_size - len(self.vocab))
    while True:
      grams = self.get_pairs_dict(self.corpus, check_morph_sep= self.morph_with_sep)

      # this is from sentence piece, it seems to break up ties in this way
      r = sorted(grams.items(), key=lambda item: (-item[1], item[0]))
      r = [item for item in r if item[-1] == r[0][-1]]
      r = sorted(r, key=lambda item: ('').join(item[0]))
      grams_count = sorted(r, key=lambda item: len(('').join(item[0])))

      if self.verbose:
        print(grams_count)
      # stop conditions
      if len(grams_count) == 0:
        print('no more bigrams to merge')
        break
      if len(self.vocab) >= self.vocab_size:
        print('vocab size reached')
        break

      # randomly choose some grams  
      if self.prob > random.random():
        idx = random.randint(0, len(grams_count) - 1)
        best_pair = grams_count[idx][0]
      else:
        best_pair = grams_count[0][0]

      # build vocab and merges
      self.vocab.append(('').join(best_pair))
      self.corpus = self.merge_dict(self.corpus, best_pair)
      self.merges.append(best_pair)
      if self.verbose:
        print(f'step: {step}, merges: {self.merges}, vocab: {self.vocab}')
      pbar.update(1)
      step += 1
    pbar.close()

  def _encode_word(self, word):
    """
    encode a single word
    returns: ids
    """
    tokens = self._tokenize_word(word, remove_sow=False)
    return [self.vocab.index(token) for token in tokens]

  def _encode_sentence(self, sentence, add_boundry = False, out_length = None):
    """
    encode a senteces
    returns: [list of int]
    """
    output = []

    for word in sentence.split(' '):
      if len(word) > 0:
        output.append(self._encode_word(word))

    output = [item for sublist in output for item in sublist]
    
    if add_boundry:
      output = [3]+output+[4]

    if out_length is None:
      return output
    else:
      if out_length > len(output):
        return output + [self.vocab.index(PAD)] * max(out_length - len(output), 0)
      else:
        return output[:out_length-1]+[4]

  def encode(self, data = None, from_path = None, out_length = None):
    """
    encode a text corpus from raw data our from a file
    returns: [list of int]
    """
    if from_path:
      with open(from_path, 'r') as f:
        sentences = f.read().splitlines()
    elif data:
      sentences = data.copy()
    else:
      raise('Error, not correct input')

    output = []
    if self.seg:
      sentences = self.segmenter.segment(sentences)

    pbar = tqdm(total=len(sentences)) 
    for stmt in sentences:
      output.append(self._encode_sentence(stmt, out_length = out_length))
      pbar.update(1)
    pbar.close()
    
    return output
  
  def encode_sentences(self, data = None, add_boundry = False, out_length = None):
    """
    encode a text corpus from raw data our from a file
    returns: [list of int]
    """
    sentences = data.copy()

    output = []
    if self.seg:
      sentences = self.segmenter.segment(sentences)

    pbar = tqdm(total=len(sentences)) 
    for stmt in sentences:
      output.append(self._encode_sentence(stmt, add_boundry = add_boundry, out_length = out_length))
      pbar.update(1)
    pbar.close()
    
    return output

  def decode(self, ids):
    """
    Decode a list of ids
    returns: [tokens]
    """
    if type(ids[0]) is list:
      output = []
      for inst in ids:
        output.append([self.vocab[id] for id in inst])
      return output
    else:
      return [self.vocab[id] for id in ids]
  
  def decode_sentences(self, ids):
    """
    list of lists of ids
    returns: [tokens] _he ll _hi there _k
    """
    output = []
    for inst in ids:
        stmt = ''
        stmt = ''
        for _id in inst:
            token = self.vocab[_id]
            if token.startswith(self.sow):
                stmt = stmt + " "
            if token in self.special_tokens:
                continue
            stmt += token.replace(self.sow, '')
        

    output.append(stmt.strip())
      
    return output 

  def tokenize(self, sentence, remove_sow = True):
    """
    tokenize a sentence
    returns: [tokens]
    """
    tokens = []

    if self.seg:
      sentence = self.segmenter.segment(sentence)
      
    words = sentence.split(' ')
    pbar = tqdm(total=len(words))
    
    for word in words:
      if len(word) > 0:
        tokens.append(self._tokenize_word(word, remove_sow = remove_sow))
      pbar.update(1)
    return tokens

  def _tokenize_word(self, t, remove_sow = True):
    """
    tokenize a single word
    returns: [tokens] 
    """
    t = SOW + ' ' + (' ').join([char if char in self.vocab else UNK for char in list(t)])

    while True:
      pairs = self.get_pairs(t, check_morph_sep = False)
      best_pair = None 

      for pair in self.merges:
        if pair in pairs:
          best_pair = pair
          break

      # stopping criteria no more merges 
      if best_pair is None:
        break 
      t = self.merge(t, best_pair)
    t = t.strip()
    if remove_sow:
      return t.replace(f'{SOW}', '').strip().split(' ')
    else:
      return t.split(' ')

  def calculate_num_tokens(self, data = None, from_path = None,):
    """
    calculate number of tokens generated by the tokenizer
    returns: [int] 
    """
    return sum([len(tokens) for tokens in self.encode(data= data, from_path = from_path, out_length = None)])
     

  def save(self, path):
    """
    save merges using file name
    returns: [None] 
    """
    os.makedirs(path, exist_ok=True)

    with open(f'{path}/tok.model', 'wb') as handle:
      pickle.dump([self.vocab, self.merges], handle, protocol=pickle.HIGHEST_PROTOCOL)

  def load(self, path):
    with open(f'{path}/tok.model', 'rb') as handle:
      self.vocab, self.merges = pickle.load(handle)