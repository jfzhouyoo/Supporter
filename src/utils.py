import torch
import random
import numpy as np
import torch.nn as nn

import nltk
import os
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

_lemmatizer = WordNetLemmatizer()

import sys
import logging
import logging.handlers
from pytz import timezone, utc
from datetime import datetime

import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Attention(nn.Module):
    
    def __init__(self,
                 query_size,
                 memory_size=None,
                 hidden_size=None,
                 mode="mlp",
                 return_attn_only=False,
                 project=False,
                 device=None):
        super(Attention, self).__init__()
        assert (mode in ["dot", "general", "mlp"]), (
            "Unsupported attention mode: {mode}"
        )

        self.query_size = query_size
        self.memory_size = memory_size or query_size
        self.hidden_size = hidden_size or query_size
        self.mode = mode
        self.return_attn_only = return_attn_only
        self.project = project
        self.device = device

        if mode == "general":
            self.linear_query = nn.Linear(
                self.query_size, self.memory_size, bias=False)
        elif mode == "mlp":
            self.linear_query = nn.Linear(
                self.query_size, self.hidden_size, bias=True)
            self.linear_memory = nn.Linear(
                self.memory_size, self.hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        if self.project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size + self.memory_size,
                          out_features=self.hidden_size),
                nn.Tanh())

    def __repr__(self):
        main_string = "Attention({}, {}".format(self.query_size, self.memory_size)
        if self.mode == "mlp":
            main_string += ", {}".format(self.hidden_size)
        main_string += ", mode='{}'".format(self.mode)
        if self.project:
            main_string += ", project=True"
        main_string += ")"
        return main_string

    def forward(self, query, memory, mask=None):
        if self.mode == "dot":
            assert query.size(-1) == memory.size(-1)
            attn = torch.bmm(query, memory.transpose(1, 2))
        elif self.mode == "general":
            assert self.memory_size == memory.size(-1)
            key = self.linear_query(query)
            attn = torch.bmm(key, memory.transpose(1, 2))
        else:
            hidden = self.linear_query(query).unsqueeze(
                2) + self.linear_memory(memory).unsqueeze(1)
            key = self.tanh(hidden)
            attn = self.v(key).squeeze(-1)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            attn.masked_fill_(mask, -float("inf"))

        weights_tmp = self.softmax(attn)   # need avoid the problem of "all -inf"
        weights = weights_tmp.clone()  # fix a bug, the attn_score may be all -inf
        weights[torch.isnan(weights) == True] = torch.tensor(0, dtype=torch.float, device=self.device)

        weighted_memory = torch.bmm(weights, memory)

        if self.return_attn_only:
            return weights

        if self.project:
            project_output = self.linear_project(
                torch.cat([weighted_memory, query], dim=-1))
            return project_output, weights
        else:
            return weighted_memory, weights

def tokenize(example, ppln):
    for fn in ppln:
        example = fn(example)
    return example


def kw_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower, pos_tag, to_basic_form])


def simp_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower])


def nltk_tokenize(string):
    return nltk.word_tokenize(string)


def lower(tokens):
    if not isinstance(tokens, str):
        return [lower(token) for token in tokens]
    return tokens.lower()


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def to_basic_form(tokens):
    if not isinstance(tokens, tuple):
        return [to_basic_form(token) for token in tokens]
    word, tag = tokens
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    else:
        return word
    return _lemmatizer.lemmatize(word, pos)


def truecasing(tokens):
    ret = []
    is_start = True
    for word, tag in tokens:
        if word == 'i':
            ret.append('I')
        elif tag[0].isalpha():
            if is_start:
                ret.append(word[0].upper() + word[1:])
            else:
                ret.append(word)
            is_start = False
        else:
            if tag != ',':
                is_start = True
            ret.append(word)
    return ret

def shanghai_format(sec, what):
    utc_dt = utc.localize(datetime.utcnow())
    my_tz = timezone("Asia/Shanghai")
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()

def get_logger(logname):
    logging.Formatter.converter = shanghai_format
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s  [%(levelname)s]  %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def str2bool(v):
    """
    str2bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



