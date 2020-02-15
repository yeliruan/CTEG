#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf
from config import Config
from dataloader import *
import time
from generator import Generator


if __name__ == '__main__':
    if __name__ == '__main__':
        log_file = "log.txt"
        # set random seed for reproduce
        tf.set_random_seed(88)
        np.random.seed(88)

        config_g = Config().generator_config_zhihu
        config_d = Config().discriminator_config_zhihu
        training_config = Config().training_config_zhihu

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # load vocab
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        vocab_dict = np.load(training_config["word_dict"]).item()
        idx2word = {v: k for k, v in vocab_dict.items()}
        print(len(vocab_dict))
        config_g["vocab_dict"] = vocab_dict
        config_g["pretrain_wv"] = np.load(training_config["pretrain_wv"])
        # assert pre-train word embedding
        assert config_g["embedding_size"] == config_g["pretrain_wv"].shape[1]

        G = Generator(config_g)
        G.build_placeholder()
        G.build_graph()
        # si 话题seq 270000个样例，每个样例话题数不固定
        # sl 每个样例话题的个数
        # slbl 对应每个样例所属的话题27000*100
        # ti 对应每个样例目标essay
        # tl 每个essagy的长度
        # prepare dataset loader

        si_tst, sl_tst, slbl_tst, ti_tst, tl_tst, tst_mem = load_npy(Config().test_data_path_zhihu)
        g_test_dataloader = GenDataLoader(config_g["batch_size"], si_tst, sl_tst, ti_tst, tl_tst,
                                          max_len=config_g["max_len"], source_label=slbl_tst, memory=tst_mem)

        g_test_dataloader.create_batch()

        sess = tf.Session(config=tf_config)


        sess.run(tf.global_variables_initializer())

        for _ in range(g_test_dataloader.num_batch):
            topic_idx, topic_len, target_idx, target_len, source_label, mem = g_test_dataloader.next_batch()
            samples = G.generate_essay(sess, topic_idx, topic_len, memory=mem, padding=True)
            print(samples)
            for topic_ids,essay_ids in zip(topic_idx,samples):
                print("topic:"+" ".join([idx2word[id] for id in topic_ids]))
                print("生成essay："+"".join([idx2word[id] for id in essay_ids]))
