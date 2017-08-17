# encoder_decoder_with attention
import numpy as np
import chainer
from chainer import Variable, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L

class Translator(chainer.Chain):
    def __init__(self, debug = False, source = 'en.txt', target = 'ja.txt', embed_size = 100):
        self.embed_size = embed_size

        self.source_lines, self.source_word2id, _                   = self.load_language(source)
        self.target_lines, self.target_word2id, self.target_id2word = self.load_language(target)

        source_size = len(self.source_word2id)
        target_size = len(self.target_word2id)
        super(Translator, self).__init__(
            embed_x = L.EmbedID(source_size, embed_size),
            embed_y = L.EmbedID(target_size, embed_size),
            H       = L.LSTM(embed_size, embed_size),
            Wc1     = L.Linear(embed_size, embed_size),
            Wc2     = L.Linear(embed_size, embed_size),
            W       = L.Linear(embed_size, target_size),
        )
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self)

        if debug:
            print("embed_size: {0}".format(embed_size), end="")
            print(", source_size: {0}".format(source_size), end="")
            print(", target_size: {0}".format(target_size))

    def learn(self, debug = False):
        line_num = len(self.source_lines) - 1
        for i in range(line_num):
            source_words = self.source_lines[i].split()
            target_words = self.target_lines[i].split()

            self.H.reset_state()
            self.zerograds()
            loss = self.loss(source_words, target_words)
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()

            if debug:
                print("{0} / {1} line finished.".format(i + 1, line_num))

    def test(self, source_words):
        bar_h_i_list = self.h_i_list(source_words, True)
        x_i = self.embed_x(Variable(np.array([self.source_word2id['<eos>']], dtype=np.int32)))
        h_t = self.H(x_i)
        c_t = self.c_t(bar_h_i_list, h_t.data[0], True)

        result = []
        bar_h_t = F.tanh(self.Wc1(c_t) + self.Wc2(h_t))
        wid = np.argmax(F.softmax(self.W(bar_h_t)).data[0])
        result.append(self.target_id2word[wid])

        loop = 0
        while (wid != self.target_word2id['<eos>']) and (loop <= 30):
            y_i = self.embed_y(Variable(np.array([wid], dtype=np.int32)))
            h_t = self.H(y_i)
            c_t = self.c_t(bar_h_i_list, h_t.data, True)

            bar_h_t = F.tanh(self.Wc1(c_t) + self.Wc2(h_t))
            wid = np.argmax(F.softmax(self.W(bar_h_t)).data[0])
            result.append(self.target_id2word[wid])
            loop += 1
        return result

    # loss
    def loss(self, source_words, target_words):
        bar_h_i_list = self.h_i_list(source_words)
        x_i = self.embed_x(Variable(np.array([self.source_word2id['<eos>']], dtype=np.int32)))
        h_t = self.H(x_i)
        c_t = self.c_t(bar_h_i_list, h_t.data[0])

        bar_h_t    = F.tanh(self.Wc1(c_t) + self.Wc2(h_t))
        tx         = Variable(np.array([self.target_word2id[target_words[0]]], dtype=np.int32))
        accum_loss = F.softmax_cross_entropy(self.W(bar_h_t), tx)
        for i in range(len(target_words)):
            wid = self.target_word2id[target_words[i]]
            y_i = self.embed_y(Variable(np.array([wid], dtype=np.int32)))
            h_t = self.H(y_i)
            c_t = self.c_t(bar_h_i_list, h_t.data)

            bar_h_t    = F.tanh(self.Wc1(c_t) + self.Wc2(h_t))
            next_wid   = self.target_word2id['<eos>'] if (i == len(target_words) - 1) else self.target_word2id[target_words[i+1]]
            tx         = Variable(np.array([next_wid], dtype=np.int32))
            loss       = F.softmax_cross_entropy(self.W(bar_h_t), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

    # h_i list
    def h_i_list(self, words, test = False):
        h_i_list = []
        #volatile = 'on' if test else 'off'
        for word in words:
            wid = self.source_word2id[word]
            x_i = self.embed_x(Variable(np.array([wid], dtype=np.int32)))
            h_i = self.H(x_i)
            h_i_list.append(np.copy(h_i.data[0]))
        return h_i_list

    # context vector c_t
    def c_t(self, bar_h_i_list, h_t, test = False):
        s = 0.0
        for bar_h_i in bar_h_i_list:
            s += np.exp(h_t.dot(bar_h_i))

        c_t = np.zeros(self.embed_size)
        for bar_h_i in bar_h_i_list:
            alpha_t_i = np.exp(h_t.dot(bar_h_i)) / s
            c_t += alpha_t_i * bar_h_i
        volatile = 'on' if test else 'off'
        c_t = Variable(np.array([c_t]).astype(np.float32))
        return c_t

    # read into words
    def load_language(self, filename):
        word2id = {}
        lines = open(filename).read().split('\n')
        for i in range(len(lines)):
            sentence = lines[i].split()
            for word in sentence:
                if word not in word2id:
                    word2id[word] = len(word2id)
        word2id['<eos>'] = len(word2id)
        id2word = {v:k for k, v in word2id.items()}
        return [lines, word2id, id2word]

    def load_model(self, filename):
        serializers.load_npz(filename, self)

    def save_model(self, filename):
        serializers.save_npz(filename, self)