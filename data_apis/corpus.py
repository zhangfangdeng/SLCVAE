#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
import pickle as pkl
from collections import Counter
import numpy as np
import nltk
import sys

reload(sys)
sys.setdefaultencoding('utf8')

class SWDADialogCorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self, corpus_path, max_vocab_cnt=10000, word2vec=None, word2vec_dim=None):
        """
        :param corpus_path: the folder that contains the SWDA dialog corpus
        """
        self._path = corpus_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.dialog_id = 0
        self.meta_id = 1
        self.utt_id = 2
        self.sil_utt = ["<s>", "<sil>", "</s>"]
        self.train_corpus = self.process("./data/train.source.data","./data/train.target.data")
        self.valid_corpus = self.process("./data/valid.source.data","./data/valid.target.data")
        self.test_corpus = self.process("./data/test.source.data","./data/test.target.data")
        self.build_vocab(max_vocab_cnt)
        self.load_word2vec()
        print("Done loading corpus")

    def process(self, file1, file2):
        """new_dialog: [(a, 1/0), (a,1/0)], new_meta: (a, b, topic), new_utt: [[a,b,c)"""
        """ 1 is own utt and 0 is other's utt"""
        f1 = open(file1, "r")
        f2 = open(file2, "r")

        line1 = f1.readline().strip("\n").decode('utf-8')
        line2 = f2.readline().strip("\n").decode('utf-8')

        source_data = []
        target_data = []
        while line1:
            # line1 = ["<s>"] + nltk.WordPunctTokenizer().tokenize(line1) + ["</s>"]
            line1 = ["<s>"] + list(line1.split(" ")) + ["</s>"]
            source_data.append(line1)
            line1 = f1.readline().strip("\n").decode('utf-8')


        while line2:
            # line2 = ["<s>"] + nltk.WordPunctTokenizer().tokenize(line2) + ["</s>"]
            line2 = ["<s>"] + list(line2.split(" ")) + ["</s>"]
            target_data.append(line2)
            line2 = f2.readline().strip("\n").decode('utf-8')
        new_dialog = {"source":source_data,"target":target_data}

        return new_dialog

    def build_vocab(self, max_vocab_cnt):
        all_words = []
        for lines in self.train_corpus["source"]:
            for line in lines:
                all_words.append(line)
        for lines in self.train_corpus["target"]:
            for line in lines:
                all_words.append(line)
        for lines in self.test_corpus["source"]:
            for line in lines:
                all_words.append(line)
        for lines in self.test_corpus["target"]:
            for line in lines:
                all_words.append(line)
        for lines in self.valid_corpus["source"]:
            for line in lines:
                all_words.append(line)
        for lines in self.valid_corpus["target"]:
            for line in lines:
                all_words.append(line)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus), len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))

        self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab["<unk>"]

    def load_word2vec(self):
        if self.word_vec_path is None:
            return
        with open(self.word_vec_path, "rb") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        # clean up lines for memory efficiency
        self.word2vec = []
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            self.word2vec.append(vec)
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))

    def get_utt_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(self.train_corpus)
        id_valid = _to_id_corpus(self.valid_corpus)
        id_test = _to_id_corpus(self.test_corpus)
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_dialog_corpus(self):
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                # convert utterance and feature into numeric numbers
                results.append([self.rev_vocab.get(t, self.unk_id) for t in dialog])
            return results

        id_train_source = _to_id_corpus(self.train_corpus["source"])
        id_train_target = _to_id_corpus(self.train_corpus["target"])
        id_valid_source = _to_id_corpus(self.valid_corpus["source"])
        id_valid_target = _to_id_corpus(self.valid_corpus["target"])
        id_test_source = _to_id_corpus(self.test_corpus["source"])
        id_test_target = _to_id_corpus(self.test_corpus["target"])
        return {'train_source': id_train_source, 'train_target': id_train_target,
                'valid_source': id_valid_source, 'valid_target': id_valid_target,
                'test_source': id_test_source, 'test_target': id_test_target}

    def get_meta_corpus(self):
        def _to_id_corpus(data):
            results = []
            for m_meta, o_meta, topic in data:
                results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
            return results

        id_train = _to_id_corpus(self.train_corpus[self.meta_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.meta_id])
        id_test = _to_id_corpus(self.test_corpus[self.meta_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

