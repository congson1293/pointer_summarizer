# -*- encoding: utf-8 -*-

from pyfasttext import FastText
import numpy as np
from config import emb_dim


fasttext = FastText('../fasttext/test_model.bin')

padding_embedd = np.random.uniform(-0.01, 0.01, (1, emb_dim))
start_decoding_embedd = np.random.uniform(-0.01, 0.01, (1, emb_dim))
stop_decoding_embedd = np.random.uniform(-0.01, 0.01, (1, emb_dim))
unknown_decoding_embedd = np.random.uniform(-0.01, 0.01, (1, emb_dim))