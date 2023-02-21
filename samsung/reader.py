
from collections import defaultdict
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

PAD_TOKEN = 'PAD'
EOS_TOKEN = 'EOS'
GO_TOKEN = 'GO'
UNK_TOKEN = 'UNK'


class Reader(object):
    def __init__(self, train_path=None, token_2_id=None,
                 special_tokens=(), min_count=1, sep='\t'):
        if token_2_id:
            self.token_2_id = token_2_id
        else:
            token_counts = defaultdict(int)
            for tokens in self.read_tokens(train_path):
                for i in tokens:
                    token_counts[i] += 1
            new_token_counts = {}
            for i, j in token_counts.items():
                if j >= min_count:
                    new_token_counts[i] = j
            self.token_counts = new_token_counts
            count_pairs = sorted(self.token_counts.items(), key=lambda k: (-k[1], k[0]))
            vocab, _ = list(zip(*count_pairs))
            vocab = list(vocab)
            vocab[0:0] = special_tokens
            full_token_id = list(zip(vocab, range(len(vocab))))
            self.token_2_id = dict(full_token_id)
        self.id_2_token = {int(v): k for k, v in self.token_2_id.items()}
        self.sep = sep

    def read_tokens(self, path):
       
        raise NotImplementedError("Must implement read_tokens")

    def unknown_token(self):
        raise NotImplementedError("Must implement unknow_tokens")

    def read_samples_by_string(self, path):
        raise NotImplementedError("Must implement read_samples")

    def convert_token_2_id(self, token):
        token_id = token if token in self.token_2_id else self.unknown_token()
        return self.token_2_id[token_id]

    def convert_id_2_token(self, id):
        return self.id_2_token[id]

    def is_unknown_token(self, token):
        return token not in self.token_2_id or token == self.unknown_token()

    def sentence_2_token_ids(self, sentence):
        return [self.convert_token_2_id(w) for w in sentence.split()]

    def token_ids_2_tokens(self, word_ids):
        return [self.convert_id_2_token(w) for w in word_ids]

    def read_samples(self, path):
        for source_words, target_words in self.read_samples_by_string(path):
            source = [self.convert_token_2_id(w) for w in source_words]
            target = [self.convert_token_2_id(w) for w in target_words]
            # head: "GO"; last: "EOS"
            target.insert(0, GO_ID)
            target.append(EOS_ID)
            yield source, target

    def read_samples_tokens(self, path):
        for source_words, target_words in self.read_samples_by_string(path):
            target = target_words
            # head: "GO"; last: "EOS"
            target.insert(0, GO_TOKEN)
            target.append(EOS_TOKEN)
            yield source_words, target

    def build_dataset(self, path):
        print('Read data, path:{0}'.format(path))
        sources, targets = [], []
        for source, target in self.read_samples_tokens(path):
            sources.append(source)
            targets.append(target)
        return sources, targets
