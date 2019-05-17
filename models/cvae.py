#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import re
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
from tensorflow.python.ops import variable_scope

import decoder_fn_lib
import utils
from models.seq2seq import dynamic_rnn_decoder
from utils import gaussian_kld
from utils import get_bi_rnn_encode
from utils import get_bow
from utils import get_rnn_encode
from utils import norm_log_liklihood
from utils import sample_gaussian


class BaseTFModel(object):
    global_t = tf.placeholder(dtype=tf.int32, name="global_t")
    learning_rate = None
    scope = None

    @staticmethod
    def print_model_stats(tvars):
        total_parameters = 0
        for variable in tvars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            print("Trainable %s with %d parameters" % (variable.name, variable_parametes))
            total_parameters += variable_parametes
        print("Total number of trainable parameters is %d" % total_parameters)

    @staticmethod
    def get_rnncell(cell_type, cell_size, keep_prob, num_layer):
        # thanks for this solution from @dimeldo
        cells = []
        for _ in range(num_layer):
            if cell_type == "gru":
                cell = rnn_cell.GRUCell(cell_size)
            else:
                cell = rnn_cell.LSTMCell(cell_size, use_peepholes=False, forget_bias=1.0)

            if keep_prob < 1.0:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            cells.append(cell)

        if num_layer > 1:
            cell = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        else:
            cell = cells[0]

        return cell

    @staticmethod
    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

    def train(self, global_t, sess, train_feed):
        raise NotImplementedError("Train function needs to be implemented")

    def valid(self, *args, **kwargs):
        raise NotImplementedError("Valid function needs to be implemented")

    def batch_2_feed(self, *args, **kwargs):
        raise NotImplementedError("Implement how to unpack the back")

    def optimize(self, sess, config, loss, log_dir, scope=None):
        if log_dir is None:
            return
        # optimization

        # tvars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print(tvars2)
        # exit()
        # tvars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/contextRNN")
        # tvars1 += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/decoder")

        if scope is "CVAE":
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/wordEmbedding")
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/contextRNN")
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/recognitionNetwork")
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/priorNetwork")
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/generationNetwork")
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/decoder")
        else:
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/label_encoder")
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/lecontextRNN")
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/ggammaNet")
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/generationNetwork1")

        grads = tf.gradients(loss, tvars)
        # if self.scope is None:
        #     tvars = tf.trainable_variables()
        # else:
        #     tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        # grads = tf.gradients(loss, tvars)

        if config.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, tf.constant(config.grad_clip))
        # add gradient noise
        if config.grad_noise > 0:
            grad_std = tf.sqrt(config.grad_noise / tf.pow(1.0 + tf.to_float(self.global_t), 0.55))
            grads = [g + tf.truncated_normal(tf.shape(g), mean=0.0, stddev=grad_std) for g in grads]

        if config.op == "adam":
            print("Use Adam")
            optimizer = tf.train.AdamOptimizer(config.init_lr)
        elif config.op == "rmsprop":
            print("Use RMSProp")
            optimizer = tf.train.RMSPropOptimizer(config.init_lr)
        else:
            print("Use SGD")
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        # self.print_model_stats(tvars)
        train_log_dir = os.path.join(log_dir, "checkpoints")
        print("Save summary to %s" % log_dir)
        self.train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        return optimizer.apply_gradients(zip(grads, tvars))


class KgRnnCVAE(BaseTFModel):

    def __init__(self, sess, config, api, log_dir, forward, scope=None):
        # self.self_label = tf.placeholder(dtype=tf.bool,shape=(None), name="self_label")
        self.self_label = False
        self.vocab = api.vocab
        self.rev_vocab = api.rev_vocab
        self.vocab_size = len(self.vocab)
        self.sess = sess
        self.scope = scope
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.context_cell_size = config.cxt_cell_size
        self.sent_cell_size = config.sent_cell_size
        self.dec_cell_size = config.dec_cell_size

        with tf.name_scope("io"):
            self.input_contexts = tf.placeholder(dtype=tf.int32, shape=(None, None), name="dialog_context")
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="context_lens")

            self.output_tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_token")
            self.output_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_lens")

            # optimization related variables
            self.learning_rate = tf.Variable(float(config.init_lr), trainable=False, name="learning_rate")
            self.learning_rate_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, config.lr_decay))
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

        max_input_len = array_ops.shape(self.input_contexts)[1]
        max_out_len = array_ops.shape(self.output_tokens)[1]
        batch_size = array_ops.shape(self.input_contexts)[0]


        with variable_scope.variable_scope("wordEmbedding"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, config.embed_size], dtype=tf.float32)
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], dtype=tf.float32,
                                         shape=[self.vocab_size, 1])
            embedding = self.embedding * embedding_mask

            input_embedding = embedding_ops.embedding_lookup(embedding, self.input_contexts)

            output_embedding = embedding_ops.embedding_lookup(embedding, self.output_tokens)


            if config.sent_type == "rnn":
                sent_cell = self.get_rnncell("gru", self.sent_cell_size, config.keep_prob, 1)
                input_embedding, sent_size = get_rnn_encode(input_embedding, sent_cell, scope="sent_rnn")
                output_embedding, _ = get_rnn_encode(output_embedding, sent_cell, self.output_lens,
                                                     scope="sent_rnn", reuse=True)
            elif config.sent_type == "bi_rnn":
                fwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                bwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                input_embedding, sent_size = get_bi_rnn_encode(input_embedding, fwd_sent_cell, bwd_sent_cell,self.context_lens, scope="sent_bi_rnn")
                output_embedding, _ = get_bi_rnn_encode(output_embedding, fwd_sent_cell, bwd_sent_cell, self.output_lens, scope="sent_bi_rnn", reuse=True)
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

            # reshape input into dialogs
            if config.keep_prob < 1.0:
                input_embedding = tf.nn.dropout(input_embedding, config.keep_prob)

        with variable_scope.variable_scope("contextRNN"):
            enc_cell = self.get_rnncell(config.cell_type, self.context_cell_size, keep_prob=1.0, num_layer=config.num_layer)
            # and enc_last_state will be same as the true last state
            input_embedding = tf.expand_dims(input_embedding, axis=2)
            _, enc_last_state = tf.nn.dynamic_rnn(
                enc_cell,
                input_embedding,
                dtype=tf.float32,
                sequence_length=self.context_lens)

            if config.num_layer > 1:
                if config.cell_type == 'lstm':
                    enc_last_state = [temp.h for temp in enc_last_state]

                enc_last_state = tf.concat(enc_last_state, 1)
            else:
                if config.cell_type == 'lstm':
                    enc_last_state = enc_last_state.h

        # input [enc_last_state, output_embedding] -- [c, x] --->z
        with variable_scope.variable_scope("recognitionNetwork"):
            recog_input = tf.concat([enc_last_state, output_embedding], 1)
            self.recog_mulogvar = recog_mulogvar = layers.fully_connected(recog_input, config.latent_size * 2, activation_fn=None, scope="muvar")
            recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=1)

        with variable_scope.variable_scope("priorNetwork"):
            # P(XYZ)=P(Z|X)P(X)P(Y|X,Z)
            prior_fc1 = layers.fully_connected(enc_last_state, np.maximum(config.latent_size * 2, 100),
                                               activation_fn=tf.tanh, scope="fc1")
            prior_mulogvar = layers.fully_connected(prior_fc1, config.latent_size * 2, activation_fn=None,
                                                    scope="muvar")
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)

            # use sampled Z or posterior Z
            latent_sample = tf.cond(self.use_prior,
                                    lambda: sample_gaussian(prior_mu, prior_logvar),
                                    lambda: sample_gaussian(recog_mu, recog_logvar))


        with variable_scope.variable_scope("label_encoder"):
            le_embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], dtype=tf.float32,
                                         shape=[self.vocab_size, 1])
            le_embedding = self.embedding * le_embedding_mask



            le_input_embedding = embedding_ops.embedding_lookup(le_embedding, self.input_contexts)

            le_output_embedding = embedding_ops.embedding_lookup(le_embedding, self.output_tokens)

            if config.sent_type == "rnn":
                le_sent_cell = self.get_rnncell("gru", self.sent_cell_size, config.keep_prob, 1)
                le_input_embedding, le_sent_size = get_rnn_encode(le_input_embedding, le_sent_cell, scope="sent_rnn")
                le_output_embedding, _ = get_rnn_encode(le_output_embedding, le_sent_cell, self.output_lens,
                                                     scope="sent_rnn", reuse=True)
            elif config.sent_type == "bi_rnn":
                le_fwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                le_bwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                le_input_embedding, le_sent_size = get_bi_rnn_encode(le_input_embedding, le_fwd_sent_cell, le_bwd_sent_cell,self.context_lens, scope="sent_bi_rnn")
                le_output_embedding, _ = get_bi_rnn_encode(le_output_embedding, le_fwd_sent_cell, le_bwd_sent_cell, self.output_lens, scope="sent_bi_rnn", reuse=True)
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

            # reshape input into dialogs
            if config.keep_prob < 1.0:
                le_input_embedding = tf.nn.dropout(le_input_embedding, config.keep_prob)

        # [le_enc_last_state, le_output_embedding]
        with variable_scope.variable_scope("lecontextRNN"):
            enc_cell = self.get_rnncell(config.cell_type, self.context_cell_size, keep_prob=1.0, num_layer=config.num_layer)
            # and enc_last_state will be same as the true last state
            le_input_embedding = tf.expand_dims(le_input_embedding, axis=2)
            _, le_enc_last_state = tf.nn.dynamic_rnn(
                enc_cell,
                le_input_embedding,
                dtype=tf.float32,
                sequence_length=self.context_lens)

            if config.num_layer > 1:
                if config.cell_type == 'lstm':
                    le_enc_last_state = [temp.h for temp in le_enc_last_state]

                le_enc_last_state = tf.concat(le_enc_last_state, 1)
            else:
                if config.cell_type == 'lstm':
                    le_enc_last_state = le_enc_last_state.h
            best_en = tf.concat([le_enc_last_state,le_output_embedding],1)


        with variable_scope.variable_scope("ggammaNet"):
            enc_cell = self.get_rnncell(config.cell_type, 200, keep_prob=1.0, num_layer=config.num_layer)
            # and enc_last_state will be same as the true last state
            input_embedding = tf.expand_dims(best_en, axis=2)
            _, zlabel = tf.nn.dynamic_rnn(
                enc_cell,
                input_embedding,
                dtype=tf.float32,
                sequence_length=self.context_lens)

            if config.num_layer > 1:
                if config.cell_type == 'lstm':
                    zlabel = [temp.h for temp in enc_last_state]

                zlabel = tf.concat(zlabel, 1)
            else:
                if config.cell_type == 'lstm':
                    zlabel = zlabel.h

        with variable_scope.variable_scope("generationNetwork"):
            gen_inputs = tf.concat([enc_last_state, latent_sample], 1)

            dec_inputs = gen_inputs
            selected_attribute_embedding = None

            # Decoder_init_state
            if config.num_layer > 1:
                dec_init_state = []
                for i in range(config.num_layer):
                    temp_init = layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state-%d" % i)
                    if config.cell_type == 'lstm':
                        temp_init = rnn_cell.LSTMStateTuple(temp_init, temp_init)

                    dec_init_state.append(temp_init)

                dec_init_state = tuple(dec_init_state)
            else:
                dec_init_state = layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state")
                if config.cell_type == 'lstm':
                    dec_init_state = rnn_cell.LSTMStateTuple(dec_init_state, dec_init_state)

        with variable_scope.variable_scope("generationNetwork1"):
            gen_inputs_sl = tf.concat([le_enc_last_state, zlabel], 1)

            dec_inputs_sl = gen_inputs_sl
            selected_attribute_embedding = None

            # Decoder_init_state
            if config.num_layer > 1:
                dec_init_state_sl = []
                for i in range(config.num_layer):
                    temp_init = layers.fully_connected(dec_inputs_sl, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state-%d" % i)
                    if config.cell_type == 'lstm':
                        temp_init = rnn_cell.LSTMStateTuple(temp_init, temp_init)

                    dec_init_state_sl.append(temp_init)

                dec_init_state_sl = tuple(dec_init_state_sl)
            else:
                dec_init_state_sl = layers.fully_connected(dec_inputs_sl, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state")
                if config.cell_type == 'lstm':
                    dec_init_state_sl = rnn_cell.LSTMStateTuple(dec_init_state_sl, dec_init_state_sl)

        with variable_scope.variable_scope("decoder"):
            dec_cell = self.get_rnncell(config.cell_type, self.dec_cell_size, config.keep_prob, config.num_layer)
            dec_cell = OutputProjectionWrapper(dec_cell, self.vocab_size)

            if forward:
                loop_func = decoder_fn_lib.context_decoder_fn_inference(None, dec_init_state, embedding,
                                                                        start_of_sequence_id=self.go_id,
                                                                        end_of_sequence_id=self.eos_id,
                                                                        maximum_length=self.max_utt_len,
                                                                        num_decoder_symbols=self.vocab_size,
                                                                        context_vector=selected_attribute_embedding)
                loop_func_sl = decoder_fn_lib.context_decoder_fn_inference(None, dec_init_state_sl, le_embedding,
                                                                             start_of_sequence_id=self.go_id,
                                                                             end_of_sequence_id=self.eos_id,
                                                                             maximum_length=self.max_utt_len,
                                                                             num_decoder_symbols=self.vocab_size,
                                                                             context_vector=selected_attribute_embedding)

                dec_input_embedding = None
                dec_input_embedding_sl = None
                dec_seq_lens = None
            else:
                loop_func = decoder_fn_lib.context_decoder_fn_train(dec_init_state, selected_attribute_embedding)
                loop_func_sl = decoder_fn_lib.context_decoder_fn_train(dec_init_state_sl, selected_attribute_embedding)

                dec_input_embedding = embedding_ops.embedding_lookup(embedding, self.output_tokens)
                dec_input_embedding_sl = embedding_ops.embedding_lookup(le_embedding, self.output_tokens)

                dec_input_embedding = dec_input_embedding[:, 0:-1, :]
                dec_input_embedding_sl = dec_input_embedding_sl[:, 0:-1, :]

                dec_seq_lens = self.output_lens - 1

                if config.keep_prob < 1.0:
                    dec_input_embedding = tf.nn.dropout(dec_input_embedding, config.keep_prob)
                    dec_input_embedding_sl = tf.nn.dropout(dec_input_embedding_sl,config.keep_prob)

                # apply word dropping. Set dropped word to 0
                if config.dec_keep_prob < 1.0:
                    keep_mask = tf.less_equal(tf.random_uniform((batch_size, max_out_len-1), minval=0.0, maxval=1.0),
                                              config.dec_keep_prob)
                    keep_mask = tf.expand_dims(tf.to_float(keep_mask), 2)
                    dec_input_embedding = dec_input_embedding * keep_mask
                    dec_input_embedding_sl = dec_input_embedding_sl * keep_mask
                    dec_input_embedding = tf.reshape(dec_input_embedding, [-1, max_out_len-1, config.embed_size])
                    dec_input_embedding_sl = tf.reshape(dec_input_embedding_sl, [-1, max_out_len - 1, config.embed_size])

            dec_outs, _, final_context_state = dynamic_rnn_decoder(dec_cell, loop_func,
                                                                   inputs=dec_input_embedding,
                                                                   sequence_length=dec_seq_lens)

            dec_outs_sl, _, final_context_state_sl = dynamic_rnn_decoder(dec_cell, loop_func_sl,
                                                                   inputs=dec_input_embedding_sl,
                                                                   sequence_length=dec_seq_lens)

            if final_context_state is not None:
                final_context_state = final_context_state[:, 0:array_ops.shape(dec_outs)[1]]
                mask = tf.to_int32(tf.sign(tf.reduce_max(dec_outs, axis=2)))
                self.dec_out_words = tf.multiply(tf.reverse(final_context_state, axis=[1]), mask)
            else:
                self.dec_out_words = tf.argmax(dec_outs, 2)

            if final_context_state_sl is not None:
                final_context_state_sl = final_context_state_sl[:, 0:array_ops.shape(dec_outs_sl)[1]]
                mask_sl = tf.to_int32(tf.sign(tf.reduce_max(dec_outs_sl, axis=2)))
                self.dec_out_words_sl = tf.multiply(tf.reverse(final_context_state_sl, axis=[1]), mask_sl)
            else:
                self.dec_out_words_sl = tf.argmax(dec_outs_sl, 2)

        if not forward:
            with variable_scope.variable_scope("loss"):

                labels = self.output_tokens[:, 1:]
                label_mask = tf.to_float(tf.sign(labels))

                rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels)

                rc_loss = tf.reduce_sum(rc_loss * label_mask, reduction_indices=1)
                self.avg_rc_loss = tf.reduce_mean(rc_loss)

                sl_rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs_sl, labels=labels)
                sl_rc_loss = tf.reduce_sum(sl_rc_loss * label_mask, reduction_indices=1)
                self.sl_rc_loss = tf.reduce_mean(sl_rc_loss)
                # used only for perpliexty calculation. Not used for optimzation
                self.rc_ppl = tf.exp(tf.reduce_sum(rc_loss) / tf.reduce_sum(label_mask))

                """ as n-trial multimodal distribution. """

                kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
                self.avg_kld = tf.reduce_mean(kld)
                if log_dir is not None:
                    kl_weights = tf.minimum(tf.to_float(self.global_t)/config.full_kl_step, 1.0)
                else:
                    kl_weights = tf.constant(1.0)

                self.label_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=latent_sample,logits=zlabel))


                self.kl_w = kl_weights
                self.elbo = self.avg_rc_loss + kl_weights * self.avg_kld

                self.cvae_loss = self.elbo + + 0.1 * self.label_loss
                self.sl_loss = self.sl_rc_loss

                tf.summary.scalar("rc_loss", self.avg_rc_loss)
                tf.summary.scalar("elbo", self.elbo)
                tf.summary.scalar("kld", self.avg_kld)


                self.summary_op = tf.summary.merge_all()

                self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar)
                self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar)
                self.est_marginal = tf.reduce_mean(rc_loss - self.log_p_z + self.log_q_z_xy)


            self.train_sl_ops = self.optimize(sess, config, self.sl_loss, log_dir, scope="SL")
            self.train_ops = self.optimize(sess, config, self.cvae_loss, log_dir, scope="CVAE")

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)


    def batch_2_feed(self, batch, global_t, use_prior, repeat=1):
        source,target,lens_source,lens_target = batch


        feed_dict = {self.input_contexts: source, self.context_lens:lens_source,
                     self.output_tokens: target, self.output_lens: lens_target,
                     self.use_prior: use_prior
                     }
        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key is self.use_prior:
                    tiled_feed_dict[key] = val
                    continue
                multipliers = [1]*len(np.array(val).shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if global_t is not None:
            feed_dict[self.global_t] = global_t

        return feed_dict

    def train(self, global_t, sess, train_feed, update_limit=5000):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        kl_losses = []
        bow_losses = []
        local_t = 0
        start_time = time.time()
        loss_names =  ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"]
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break
            feed_dict = self.batch_2_feed(batch, global_t, use_prior=False)

            feed_dict_CVAE = feed_dict
            self.self_label = False
            _, _, sum_op, elbo_loss, rc_loss, rc_ppl, kl_loss = sess.run([self.train_ops, self.train_sl_ops, self.summary_op,
                                                                       self.cvae_loss,self.avg_rc_loss,
                                                                       self.rc_ppl, self.avg_kld],
                                                                       feed_dict)

            # feed_dict_SL = feed_dict
            # self.self_label = True
            # _, label_loss = sess.run([self.label_ops, self.avg_label_loss], feed_dict)



            self.train_summary_writer.add_summary(sum_op, global_t)
            elbo_losses.append(elbo_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)

            global_t += 1
            local_t += 1
            if local_t % (train_feed.num_batch / 10) == 0:
                kl_w = sess.run(self.kl_w, {self.global_t: global_t})
                self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
                                loss_names, [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "kl_w %f" % kl_w)

        # finish epoch!
        epoch_time = time.time() - start_time
        avg_losses = self.print_loss("Epoch Done", loss_names,
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses],
                                     "step time %.4f" % (epoch_time / train_feed.num_batch))

        return global_t, avg_losses[0]

    def valid(self, name, sess, valid_feed):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        bow_losses = []
        kl_losses = []

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)

            # self.self_label = True
            elbo_loss, rc_loss, rc_ppl, kl_loss = sess.run(
                [self.elbo, self.avg_rc_loss,
                 self.rc_ppl, self.avg_kld], feed_dict)
            elbo_losses.append(elbo_loss)
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)
            kl_losses.append(kl_loss)

        avg_losses = self.print_loss(name, ["elbo_loss", "rc_loss", "rc_peplexity", "kl_loss"],
                                     [elbo_losses, rc_losses, rc_ppls, kl_losses], "")
        return avg_losses[0]

    def test(self, sess, test_feed, num_batch=None, repeat=5, dest=sys.stdout):
        local_t = 0
        recall_bleus = []
        prec_bleus = []

        while True:
            batch = test_feed.next_batch()

            if batch is None or (num_batch is not None and local_t > num_batch):
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=repeat)

            # self.self_label = True

            word_outs = sess.run(self.dec_out_words, feed_dict)

            sample_words = np.split(word_outs, repeat, axis=0)
            true_srcs = feed_dict[self.input_contexts]
            true_src_lens = feed_dict[self.context_lens]
            true_outs = feed_dict[self.output_tokens]
            local_t += 1


            if dest != sys.stdout:
                if local_t % (test_feed.num_batch / 10) == 0:
                    print("%.2f >> " % (test_feed.ptr / float(test_feed.num_batch))),

            for b_id in range(test_feed.batch_size):
                # print the dialog context
                src_str = " ".join([self.vocab[e] for e in true_srcs[b_id].tolist() if e != 0])
                dest.write("Src : %s\n" % (src_str))

                # print the true outputs
                true_tokens = [self.vocab[e] for e in true_outs[b_id].tolist() if e not in [0, self.eos_id, self.go_id]]
                true_str = " ".join(true_tokens).replace(" ' ", "'")
                # print the predicted outputs
                dest.write("Target >> %s\n" % (true_str))
                print "target >> %s" %(true_str)
                local_tokens = []
                for r_id in range(repeat):
                    pred_outs = sample_words[r_id]
                    pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist() if e != self.eos_id and e != 0]
                    pred_str = " ".join(pred_tokens).replace(" ' ", "'")
                    dest.write("Sample %d >> %s\n" % (r_id,  pred_str))
                    print "pred >> %s" % (pred_str)
                    local_tokens.append(pred_tokens)

                max_bleu, avg_bleu = utils.get_bleu_stats(true_tokens, local_tokens)
                recall_bleus.append(max_bleu)
                prec_bleus.append(avg_bleu)
                # make a new line for better readability
                dest.write("\n")

        avg_recall_bleu = float(np.mean(recall_bleus))
        avg_prec_bleu = float(np.mean(prec_bleus))
        avg_f1 = 2*(avg_prec_bleu*avg_recall_bleu) / (avg_prec_bleu+avg_recall_bleu+10e-12)
        report = "Avg recall BLEU %f, avg precision BLEU %f and F1 %f (only 1 reference response. Not final result)" \
                 % (avg_recall_bleu, avg_prec_bleu, avg_f1)
        print report
        dest.write(report + "\n")
        print("Done testing")


