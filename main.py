from __future__ import print_function

import os
import sys
import torch
import io
import time
import numpy as np
from PIL import Image
import torch.onnx
from datetime import datetime
from torch.autograd import Variable
from miscc.config import cfg
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import time
import random

from matplotlib.pyplot import imshow

from flask import Flask

#%matplotlib inline


def vectorize_caption(wordtoix, caption, copies=2):
    # create caption vector
    tokens = caption.split(" ")
    cap_v = []
    for t in tokens:
        t = t.strip().encode("ascii", "ignore").decode("ascii")
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    # expected state for single generation
    captions = np.zeros((copies, len(cap_v)))
    for i in range(copies):
        captions[i, :] = np.array(cap_v)
    cap_lens = np.zeros(copies) + len(cap_v)

    # print(captions.astype(int), cap_lens.astype(int))
    # captions, cap_lens = np.array([cap_v, cap_v]), np.array([len(cap_v), len(cap_v)])
    # print(captions, cap_lens)
    # return captions, cap_lens

    return captions.astype(int), cap_lens.astype(int)


def generate(caption, copies=2):
    # load word vector
    captions, cap_lens = vectorize_caption(wordtoix, caption, copies)
    n_words = len(wordtoix)

    # only one to generate
    batch_size = captions.shape[0]

    nz = cfg.GAN.Z_DIM
    captions = Variable(torch.from_numpy(captions), volatile=True)
    cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)

    #######################################################
    # (1) Extract text embeddings
    #######################################################
    hidden = text_encoder.init_hidden(batch_size)
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    mask = captions == 0

    #######################################################
    # (2) Generate fake images
    #######################################################
    noise.data.normal_(0, 1)
    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

    # G attention
    cap_lens_np = cap_lens.cpu().data.numpy()

    # storing to blob storage
    container_name = "images"
    full_path = "https://attgan.blob.core.windows.net/images/%s"
    prefix = datetime.now().strftime("%Y/%B/%d/%H_%M_%S_%f")
    imgs = []
    # only look at first one
    # j = 0
    for j in range(batch_size):
        for k in range(len(fake_imgs)):
            im = fake_imgs[k][j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            #             im = Image.fromarray(im)
            imgs.append(im)
    imshow(imgs[-1])

app = Flask(__name__)

@app.route('/<text>')
def hello_world(text):
  return generate(text)

if __name__ == "__main__":
    x = pickle.load(open("data/birds/captions.pickle", "rb"))
    ixtoword = x[2]
    wordtoix = x[3]
    del x

    word_len = len(wordtoix)

    text_encoder = RNN_ENCODER(word_len, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.eval()

    netG = G_NET()
    state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    netG.eval()

    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Flask
    app.run()
