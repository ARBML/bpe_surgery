from io import open
from conllu import parse_incr
import re 
import glob 

def preprocess(text):
    special_tokens = ',،:.!)(?؟)"'
    text = text.replace("ـــ", "")
    for token in special_tokens:
        text = text.replace(token, f" {token} ").strip()
    return text

def parse_file(file, out):
    data_file = open(file, "r", encoding="utf-8")
    for tokenlist in parse_incr(data_file):
        try:
            org_text = preprocess(tokenlist.metadata['text']).split(' ')
            tok_text = tokenlist.metadata['treeTokens'].split(' ')
            j = 0
            int_tok = []
            for token in tok_text:
                int_tok.append(token.replace('+',''))
                if not('+' in token):
                    if org_text[j] == "".join(int_tok):
                        out[org_text[j]]= int_tok
                    int_tok = []
                    j += 1
        except:
            pass
    return out

def get_gold_segmentations(dataset = "cameltb"):
    if dataset == "cameltb":
        conll_files = glob.glob('camelTB/data/annotated/**/**.conllx')
        out = {}
        for file in conll_files:
            out = parse_file(file, out)
        return out 
    else:
        raise('error')