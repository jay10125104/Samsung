import numpy as np
from keras.callbacks import Callback
from generator.corpus_reader import str2id, id2str
from generator.reader import GO_TOKEN, EOS_TOKEN
def gen_target(input_text, model, char2id, id2char, maxlen=400, topk=3, max_target_len=50):
    xid = np.array([str2id(input_text, char2id, maxlen)] * topk)
    yid = np.array([[char2id[GO_TOKEN]]] * topk)
    scores = [0] * topk
    for i in range(max_target_len):
        proba = model.predict([xid, yid])[:, i, :]
        log_proba = np.log(proba + 1e-6) 
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]
        _scores = []
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[j][k]])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:] 
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == char2id[EOS_TOKEN]:
                return id2str(_yid[k][1:-1], id2char)
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    return id2str(yid[np.argmax(scores)][1:-1], id2char)
class Evaluate(Callback):
    def __init__(self, model, attn_model_path, char2id, id2char, maxlen):
        super(Evaluate, self).__init__()
        self.lowest = 1e10
        self.model = model
        self.attn_model_path = attn_model_path
        self.char2id = char2id
        self.id2char = id2char
        self.maxlen = maxlen

    def on_epoch_end(self, epoch, logs=None):
        sents = [
            "Field &amp; Main Bank purchased a new position in PowerShares Fin . Preferred Port . ( NYSEARCA : PGF )  "
            "in the fourth quarter , according to its most recent disclosure with the SEC . The institutional investor "
            "purchased 22,550 shares of the exchange traded fund 's stock , valued at approximately $ 425,000 . "
            "Other large investors also recently modified their holdings of the company . Cedar Hill Associates LLC "
            "acquired a new stake in shares of PowerShares Fin . Preferred Port .",
            ]
        for sent in sents:
            target = gen_target(sent, self.model, self.char2id, self.id2char, self.maxlen)
            print('input:' + sent)
            print('output:' + target)
        if logs['val_loss'] <= self.lowest:
            self.lowest = logs['val_loss']
            self.model.save_weights(self.attn_model_path)
