import argparse

import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.ops import lookup_ops

import time
import os

tf.enable_eager_execution()
tf.set_random_seed(42)

logging = tf.logging
logging.set_verbosity(logging.INFO)


def log_msg(msg):
    logging.info(f'{time.ctime()}: {msg}')


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-eos', default='<eos>')
    parser.add_argument('-unk_index', default=0, type=int)

    parser.add_argument('-cell', default='lstm')

    parser.add_argument('-opt', default='adam')
    parser.add_argument('-lr', default=0.002, type=float)

    parser.add_argument('-t', default=8, type=int)

    parser.add_argument('-bs', default=32, type=int)
    parser.add_argument('-nd', default=256, type=int)
    parser.add_argument('-nh', default=256, type=int)
    parser.add_argument('-dropout', default=0.2, type=float)
    parser.add_argument('-clip_ratio', default=10.0, type=float)

    parser.add_argument('-vocab')
    parser.add_argument('-train')
    parser.add_argument('-valid')

    parser.add_argument('-save_dir')
    parser.add_argument('-ckpt_prefix', default='ptb-lm')

    parser.add_argument('-num_epochs', default=100, type=int)
    parser.add_argument('-stats_step', default=500, type=int)
    parser.add_argument('-eval_step', default=5000, type=int)
    return parser.parse_args()


class Embedding(tf.keras.Model):
    def __init__(self, V, d):
        super(Embedding, self).__init__()
        self.W = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d]))

    def call(self, word_indexes):
        return tf.nn.embedding_lookup(self.W, word_indexes)


class StaticRNN(tf.keras.Model):
    def __init__(self, h, cell):
        super(StaticRNN, self).__init__()
        if cell == 'lstm':
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=h)
        elif cell == 'gru':
            self.cell = tf.nn.rnn_cell.GRUCell(num_units=h)
        else:
            self.cell = tf.nn.rnn_cell.BasicRNNCell(num_units=h)

    def call(self, word_vectors, num_words):
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        outputs, final_state = tf.nn.static_rnn(cell=self.cell, inputs=word_vectors_time, sequence_length=num_words,
                                                dtype=tf.float32)
        return outputs


class LanguageModel(tf.keras.Model):
    def __init__(self, V, d, h, cell):
        super(LanguageModel, self).__init__()
        self.word_embedding = Embedding(V, d)
        self.rnn = StaticRNN(h, cell)
        self.output_layer = tf.keras.layers.Dense(units=V)

    def call(self, datum, train=False, dropout=0.):
        word_vectors = self.word_embedding(datum[0])
        if train:
            word_vectors = tf.nn.dropout(word_vectors, keep_prob=1 - dropout)
        rnn_outputs_time = self.rnn(word_vectors, datum[2])

        # We want to convert it back to shape batch_size x TimeSteps x h
        rnn_outputs = tf.stack(rnn_outputs_time, axis=1)
        if train:
            rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=1 - dropout)
        logits = self.output_layer(rnn_outputs)
        return logits


def create_dataset(sentences_file, vocab_table, batch_size, eos, t):
    # Create a Text Line dataset, which returns a string tensor
    dataset = tf.data.TextLineDataset(sentences_file)

    # Convert to a list of words..
    dataset = dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=t)

    # Create target words right shifted by one, append EOS, also return size of each sentence...
    dataset = dataset.map(lambda words: (words, tf.concat([words[1:], [eos]], axis=0), tf.size(words)), num_parallel_calls=t)

    # Lookup words, word->integer, EOS->1
    dataset = dataset.map(lambda src_words, tgt_words, num_words: (vocab_table.lookup(src_words),
                                                                   vocab_table.lookup(tgt_words), num_words), num_parallel_calls=t)

    # [None] -> src words, [None] -> tgt_words, [] length of sentence
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([None], [None], []))

    #dataset = dataset.prefetch(1)
    return dataset


def loss_fun(model, datum, train=False):
    logits = model(datum, train)
    mask = tf.sequence_mask(datum[2], dtype=tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1]) * mask
    return tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(datum[2]), dtype=tf.float32)


def train_loss(model, datum):
    return loss_fun(model, datum, train=True)


def compute_ppl(model, dataset):
    total_loss = 0.
    total_words = 0
    for batch_num, datum in enumerate(dataset):
        num_words = int(tf.reduce_sum(datum[2]))
        avg_loss = loss_fun(model, datum)
        total_loss = avg_loss * num_words
        total_words += num_words
    loss = total_loss / float(num_words)
    return np.exp(loss)


def clip_gradients(grads_and_vars, clip_ratio):
    gradients, variables = zip(*grads_and_vars)
    clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
    return zip(clipped, variables)


def check_if_ppl_better(best_valid_ppl, lm, valid_dataset, root, ckpt_prefix, epoch_num, step_num):
    ppl = compute_ppl(lm, valid_dataset)
    if ppl < best_valid_ppl:
        save_path = root.save(ckpt_prefix)
        log_msg(
            f'Epoch: {epoch_num} Step: {step_num} ppl improved: {ppl: 0.4f} old: {best_valid_ppl: 0.4f} path: {save_path}')
        return True, ppl
    else:
        log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl worse: {ppl: 0.4f} old: {best_valid_ppl: 0.4f}')
        return False, ppl


def main():
    args = setup_args()
    log_msg(args)

    vocab_table = lookup_ops.index_table_from_file(args.vocab, default_value=args.unk_index)
    train_dataset = create_dataset(args.train, vocab_table, args.bs, args.eos, args.t)
    valid_dataset = create_dataset(args.valid, vocab_table, args.bs, args.eos, args.t)

    loss_and_grads_fun = tfe.implicit_value_and_gradients(train_loss)
    lm = LanguageModel(int(vocab_table.size()), d=args.nd, h=args.nh, cell=args.cell)

    log_msg('Model built!')
    best_valid_ppl = compute_ppl(lm, valid_dataset)
    log_msg(f'Start ppl: {best_valid_ppl: 0.4f}')

    if args.opt == 'adam':
        opt = tf.train.AdamOptimizer(args.lr)
    else:
        opt = tf.train.GradientDescentOptimizer(args.lr)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    ckpt_prefix = os.path.join(args.save_dir, args.ckpt_prefix)
    root = tfe.Checkpoint(optimizer=opt, model=lm, optimizer_step=tf.train.get_or_create_global_step())
    for epoch_num in range(args.num_epochs):
        log_msg(f'Epoch: {epoch_num} START')
        batch_loss = []
        for step_num, train_datum in enumerate(train_dataset, start=1):
            loss_value, gradients = loss_and_grads_fun(lm, train_datum)
            batch_loss.append(loss_value)

            if step_num % args.stats_step == 0:
                log_msg(f'Epoch: {epoch_num} Step: {step_num} Avg Loss: {np.average(np.asarray(loss_value)): 0.4f}')
                batch_loss = []

            if step_num % args.eval_step == 0:
                better, ppl = check_if_ppl_better(best_valid_ppl, lm, valid_dataset, root, ckpt_prefix, epoch_num, step_num)
                if better:
                    best_valid_ppl = ppl

            opt.apply_gradients(clip_gradients(gradients, args.clip_ratio))
        log_msg(f'Epoch: {epoch_num} END')
        better, ppl = check_if_ppl_better(best_valid_ppl, lm, valid_dataset, root, ckpt_prefix, epoch_num, step_num=-1)
        if better:
            best_valid_ppl = ppl


if __name__ == '__main__':
    main()