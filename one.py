def load_word_dict(save_path):
    dict_data = dict()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split()
            try:
                dict_data[items[0]] = items[1]
            except IndexError:
                logger.error('error', line)
    return dict_data
path = r'C:\Users\JAY\OneDrive\Desktop\samsung\data\test_sample.txt'
a = load_word_dict(path)
print(a)