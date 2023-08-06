## Morphology-aware tokenization for Arabic

```
pip install -e .
```

## Usage

```
from bpe_surgery import bpe
tokenizer = bpe(vocab_size=23,)
tokenizer.train(file_path='test_ar_sm.txt')
tokenizer.tokenize("السلام عليكم شيء جميل")
```

Output

```
[['ال', 'س', 'ل', 'ا', 'م'],
 ['ع', 'ل', 'ي', 'ك', 'م'],
 ['<unk>', 'ي', '<unk>'],
 ['جم', 'ي', 'ل']]
```