import numpy as np
import random as rnd
from numpy import random
from word_representation import word_to_idx, idx_to_vector


class DataIterator(object):
    adjectives_ = set()
    adjectives = None
    nouns_ = set()
    nouns = None

    def __init__(self, file_name, batch_size, k_noise_sample=5, random_seed=42):
        self.file_name = file_name
        self.batch_size = batch_size
        self.word_dim = idx_to_vector.shape[1]
        self.r = random.RandomState(seed=random_seed)
        self.r_ = rnd.Random(random_seed)
        self.adjective_noun_phrases = []
        self.k_noise_sample = k_noise_sample
        with open(self.file_name) as f:
            for line in f:
                self.adjective_noun_phrases.append(line.decode('utf-8').split())
        DataIterator.adjectives_.update(adj for adj, _ in self.adjective_noun_phrases)
        DataIterator.nouns_.update(nn for _, nn in self.adjective_noun_phrases)
        DataIterator.adjectives = list(DataIterator.adjectives_)
        DataIterator.nouns = list(DataIterator.nouns_)

    def get_infinite_iterator(self):
        epoch = 0
        while True:
            for matrix, labels in self.get_iterator():
                yield matrix, labels
            print 'epoch {}!'.format(epoch)
            epoch += 1

    def get_iterator(self):
        self.r.shuffle(self.adjective_noun_phrases)
        for i, (adj, noun) in enumerate(self.adjective_noun_phrases):
            if i % self.batch_size == 0:
                if i != 0:
                    yield matrix, labels
                k = 0
                matrix, labels = self._get_empty_matricies()
            matrix[k, 0:self.word_dim] = idx_to_vector[word_to_idx[adj]]

            for j in xrange(self.k_noise_sample):
                random_adj = self.r_.choice(DataIterator.adjectives)
                matrix[k+1+j, 0:self.word_dim] = idx_to_vector[word_to_idx[random_adj]]
            matrix[k:k+1+self.k_noise_sample, self.word_dim:2*self.word_dim] = idx_to_vector[word_to_idx[noun]]
            k += self.k_noise_sample + 1

    def _get_empty_matricies(self):
        matrix = np.empty(shape=((1+self.k_noise_sample) * self.batch_size, 2 * self.word_dim))
        labels = np.array(sum([[1.0], self.k_noise_sample * [0.0]] * self.batch_size, []), ndmin=2).T
        return matrix, labels

    @staticmethod
    def get_features(adj, noun):
        if adj not in word_to_idx or noun not in word_to_idx:
            raise ValueError('{} or {} is absent in our simple modest small vocab!'.format(adj, noun))
        word_dim = idx_to_vector.shape[1]
        matrix = np.empty(shape=(1, 2 * word_dim))
        matrix[0, 0:word_dim] = idx_to_vector[word_to_idx[adj]]
        matrix[0, word_dim:2*word_dim] = idx_to_vector[word_to_idx[noun]]
        return matrix