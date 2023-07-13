import os
data_dir = './data'
train_path = os.path.join(data_dir, 'train_sample.txt')
test_path = os.path.join(data_dir, 'test_sample.txt')
output_dir = './output'
save_vocab_path = os.path.join(output_dir, 'vocab.txt')
attn_model_path = os.path.join(output_dir, 'attn_model.weight')
batch_size = 32
epochs = 50
rnn_hidden_dim = 128
maxlen = 400
min_count = 5
dropout = 0.0
use_gpu = False
sep = '\t'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
