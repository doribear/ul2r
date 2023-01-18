from tokenizers import Tokenizer, pre_tokenizers, processors, trainers, models, decoders
from konlpy.tag import Mecab
from tqdm import tqdm

import os


mecab = Mecab()
train_corpus = list(map(lambda x : os.path.join('corpus', x), os.listdir('corpus')))
train_corpus = list(map(lambda x : list(map(lambda y : mecab.morphs(y.strip()), tqdm(open(x).readlines()))), train_corpus))
train_corpus = sum(train_corpus, [])
train_corpus = list(map(lambda x : ' '.join(x), train_corpus))



tokenizer = Tokenizer(models.WordPiece(unk_token = '<unk>'))
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

trainer = trainers.WordPieceTrainer(min_frequency = 30, special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '<unk>'])

tokenizer.train_from_iterator(train_corpus, trainer)
tokenizer.enable_padding(pad_id = tokenizer.token_to_id('[PAD]'), length = 125)
tokenizer.enable_truncation(max_length = 125)
tokenizer.decoder = decoders.WordPiece()
tokenizer.post_processor = processors.BertProcessing(('[SEP]', tokenizer.token_to_id('[SEP]')), ('[CLS]', tokenizer.token_to_id('[CLS]')))

tokenizer.save('tokenizer.json')