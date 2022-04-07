class Preprocess(object):
    def __init__(self, data, window=5, unk='<UNK>', max_vocab=20000):
        self.window = window
        self.unk = unk
        # self.data = open(file, encoding='UTF-8')
        self.data = data
        self.max_vocab = max_vocab

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[max(i - self.window, 0):i]
        right = sentence[i + 1:i + 1 + self.window]
        return iword, [
            self.unk for _ in range(self.window - len(left))
        ] + left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self):
        max_vocab = self.max_vocab
        print("building vocab...")
        step = 0
        # word count
        self.word_frequency = {self.unk: 1}
        for line in self.data:
            step += 1
            if step % 1000 == 0:
                print("working on {}kth line".format(step // 1000), end='\r')
            for word in line:
                self.word_frequency[word] = self.word_frequency.get(word, 0) + 1
        print("")

        word_list = [self.unk] + sorted(self.word_frequency,
                                        key=self.word_frequency.get,
                                        reverse=True)[:max_vocab - 1]
        self.idx2word = {idx: word for idx, word in enumerate(word_list)}
        self.word2idx = {word: idx for idx, word in enumerate(word_list)}
        self.vocab = set([word for word in self.word2idx])
        print("build done")
        return self.idx2word, self.word2idx, self.vocab, self.word_frequency


import numpy as np
np.random.seed(2022)
data = np.array(np.random.randint(0, 10312, (10312, 80)), dtype=str)
preprocess = Preprocess(data)
idx2word, word2idx, _, _ = preprocess.build()
print(word2idx)
preprocess.convert()
