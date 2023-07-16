import os
import nltk
import json
import torch
import pickle
import numpy as np
from collections import defaultdict
from copy import deepcopy
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

class Dataset(data.Dataset):
    def __init__(self, args, data, tokenizer, emotion_statistic):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.CLS_id = tokenizer.cls_token_id
        self.EOS_id = tokenizer.eos_token_id
        self.SEP_id = tokenizer.sep_token_id
        self.emotion_statistic = emotion_statistic

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # context+strategy、context_role、context_kws、context_infer_kws、next_uttr_infer_kws
        # context_positive_emo_labels、context_negative_emo_labels、next_uttr_pos_emo_labels、next_uttr_neg_emo_labels
        # target_strategy、target_pos_kws_labels、target_neg_kws_labels、next_uttr_pos_kws_labels、next_uttr_neg_kws_labels
        item = {}

        # context
        item["context"], item["context_role"], item["context_token_type"], \
        item["ori_context"], item["ori_context_role"], item["ori_context_token_type"] = self.process_context(
            self.data["context"][index],
            self.data["context_strategy_seqs"][index],
            self.data["context_role"][index],
            self.data["context_positive_kws"][index], 
            self.data["context_negative_kws"][index],
        )
        item["context_txt"] = self.data["context"][index]

        # infered kws
        item["context_infer_pos_kws"], item["context_infer_pos_role"], item["context_infer_pos_token_type"], \
        item["context_infer_neg_kws"], item["context_infer_neg_role"], item["context_infer_neg_token_type"], \
        item["next_uttr_infer_pos_kws"], item["next_uttr_infer_pos_role"], item["next_uttr_infer_pos_token_type"], \
        item["next_uttr_infer_neg_kws"], item["next_uttr_infer_neg_role"], item["next_uttr_inder_neg_token_typr"] = self.process_infered_kws(
            len(self.data["context"][index])+1,
            self.data["context_infer_pos_kws"][index], 
            self.data["context_infer_neg_kws"][index],
            self.data["next_uttr_infer_pos_kws"][index], 
            self.data["next_uttr_infer_neg_kws"][index]
        )

        # emotion labels
        item["context_pos_emo_labels"], item["context_neg_emo_labels"], item["next_uttr_pos_emo_labels"], item["next_uttr_neg_emo_labels"] = self.process_emotion_labels(
            self.data["context_pos_emo_labels"][index],
            self.data["context_neg_emo_labels"][index],
            self.data["next_uttr_pos_emo_labels"][index],
            self.data["next_uttr_neg_emo_labels"][index]
        )

        # kws labels
        item["target_pos_kws_labels"], item["target_neg_kws_labels"], item["next_uttr_pos_kws_labels"], item["next_uttr_neg_kws_labels"] = self.process_kws_labels(
            self.data["target_positive_kws"][index],
            self.data["target_negative_kws"][index],
            self.data["next_uttr_positive_kws"][index],
            self.data["next_uttr_negative_kws"][index]
        )
        
        # target
        item["target"], item["target_role"], item["target_lm_labels"] = self.process_target(
            self.data["target"][index], self.data["strategy"][index])
        # item["target_txt"] = ["[" + self.data["strategy"][index] +"]"] + self.data["target"][index]
        item["target_txt"] = self.data["target"][index]

        # rewards ready
        item["dialog_turn"] = self.data["dialog_turn"][index]
        item["context_seeker_sum_emo_score"] = self.data["context_seeker_sum_emo_score"][index]
        item["next_uttr_emotion_score"] = self.data["next_uttr_emotion_score"][index]
        item["context_last_infer_kws"] = self.data["context_last_infer_kws"][index]
        item["next_uttr_infer_kws"] = self.data["next_uttr_infer_kws"][index]

        item["context_txt"] = self.data["context"][index]
        item["context_strategy_seqs_txt"] = self.data["context_strategy_seqs"][index]
        item["context_role_txt"] = self.data["context_role"][index]
        item["context_positive_kws_txt"] = self.data["context_positive_kws"][index] 
        item["context_negative_kws_txt"] = self.data["context_negative_kws"][index]
        item["next_uttr_txt"] = self.data["next_uttr"][index]
        item["next_uttr_positive_kws_txt"] = self.data["next_uttr_positive_kws"][index]
        item["next_uttr_negative_kws_txt"] = self.data["next_uttr_negative_kws"][index]

        item["context_emotion_scores"] = self.data["context_emotion_scores"][index]
        item["context_role"] = self.data["context_role"][index]

        return item

    def process_context(self, context, context_strategy, context_role, context_positive_kws, context_negative_kws):
        context_ids = [self.CLS_id]
        context_role_ids = [self.CLS_id]
        context_token_type_ids = [self.CLS_id]
        # context uttr
        for (i, uttr), strategy, role in zip(enumerate(context), context_strategy, context_role):
            if role == "seeker":
                uttr_encode_ids = self.tokenizer.encode(" ".join(uttr)) + [self.EOS_id]
            elif role == "supporter":
                uttr_encode_ids = self.tokenizer.encode("["+strategy+"]") + self.tokenizer.encode(" ".join(uttr)) + [self.EOS_id]
            else:
                raise ValueError("The Label of Role is Error!")

            context_ids += uttr_encode_ids
            spk = self.args.seeker_idx if role == "seeker" else self.args.supporter_idx
            context_role_ids = context_role_ids + [spk] * len(uttr_encode_ids)
            context_token_type_ids = context_token_type_ids + [i+1] * len(uttr_encode_ids)

            if i == len(context)-1:
                context_ids += [self.SEP_id]
                context_role_ids += [spk]
                context_token_type_ids += [i+1]
        
        while len(context_ids) > self.args.max_context_length:
            cut_idx = context_ids.index(self.EOS_id, -self.args.max_context_length+1)
            context_ids = [self.CLS_id] + context_ids[cut_idx:]
            context_role_ids = [self.CLS_id] + context_role_ids[cut_idx:]
            context_token_type_ids = [self.CLS_id] + context_token_type_ids[cut_idx:]
        assert len(context_ids) == len(context_role_ids) == len(context_token_type_ids)
        
        ori_context_ids = deepcopy(context_ids)
        ori_context_role_ids = deepcopy(context_role_ids)
        ori_context_token_type_ids = deepcopy(context_token_type_ids)

        # context kws
        assert len(context_positive_kws) == len(context_negative_kws)
        context_kws_ids = []
        context_kws_role_ids = []
        context_kws_token_type_ids = []
        for (i, role), context_pos_kws, context_neg_kws in zip(enumerate(context_role), context_positive_kws, context_negative_kws):
            context_pos_neg_kws = list(set(context_pos_kws + context_neg_kws))
            pos_neg_kws_ids = self.tokenizer.encode(" ".join(context_pos_neg_kws)) + [self.EOS_id]
            context_kws_ids += pos_neg_kws_ids
            spk = self.args.context_kws_seeker_idx if role == "seeker" else self.args.context_kws_supporter_idx
            context_kws_role_ids = context_kws_role_ids + [spk] * len(pos_neg_kws_ids)
            context_kws_token_type_ids = context_kws_token_type_ids + [i+1] * len(pos_neg_kws_ids)
        while len(context_kws_ids) > self.args.max_context_kws_length:
            cut_idx = context_kws_ids.index(self.EOS_id, -self.args.max_context_kws_length+1)
            context_kws_ids = context_kws_ids[cut_idx:]
            context_kws_role_ids = context_kws_role_ids[cut_idx:]
            context_kws_token_type_ids = context_kws_token_type_ids[cut_idx:]
        assert len(context_kws_ids) == len(context_kws_role_ids) == len(context_kws_token_type_ids)

        context_ids += context_kws_ids
        context_role_ids += context_kws_role_ids
        context_token_type_ids += context_kws_token_type_ids
        
        return context_ids, context_role_ids, context_token_type_ids, ori_context_ids, ori_context_role_ids, ori_context_token_type_ids

    def process_infered_kws(self, turn_id, context_infer_pos_kws, context_infer_neg_kws, next_uttr_infer_pos_kws, next_uttr_infer_neg_kws):
        def process_kws(infer_kws, role_id, turn_id):
            infer_kws = list(set([kws for _,_,kws,_ in infer_kws[:self.args.max_infer_kws_length]]))
            infer_kws_ids = self.tokenizer.encode(" ".join(infer_kws))
            infer_role_ids = len(infer_kws_ids) * [role_id]
            infer_token_type_ids = len(infer_kws_ids) * [turn_id]
            return infer_kws_ids, infer_role_ids, infer_token_type_ids
        context_infer_pos_kws_ids, context_infer_pos_role_ids, context_infer_pos_token_type_ids = process_kws(
            context_infer_pos_kws, self.args.context_infer_kws_idx, turn_id
        )
        context_infer_neg_kws_ids, context_infer_neg_role_ids, context_infer_neg_token_type_ids = process_kws(
            context_infer_neg_kws, self.args.context_infer_kws_idx, turn_id
        )
        next_uttr_infer_pos_kws_ids, next_uttr_infer_pos_role_ids, next_uttr_infer_pos_token_type_ids = process_kws(
            next_uttr_infer_pos_kws, self.args.next_uttr_infer_kws_idx, turn_id
        )
        next_uttr_infer_neg_kws_ids, next_uttr_infer_neg_role_ids, next_uttr_infer_neg_token_type_ids = process_kws(
            next_uttr_infer_neg_kws, self.args.next_uttr_infer_kws_idx, turn_id
        )
        return context_infer_pos_kws_ids, context_infer_pos_role_ids, context_infer_pos_token_type_ids, \
               context_infer_neg_kws_ids, context_infer_neg_role_ids, context_infer_neg_token_type_ids, \
               next_uttr_infer_pos_kws_ids, next_uttr_infer_pos_role_ids, next_uttr_infer_pos_token_type_ids, \
               next_uttr_infer_neg_kws_ids, next_uttr_infer_neg_role_ids, next_uttr_infer_neg_token_type_ids
    
    def process_emotion_labels(self, context_pos_emo, context_neg_emo, next_uttr_pos_emo, next_uttr_neg_emo):
        context_pos_emo_labels = [self.emotion_statistic[emo]["idx"] for emo in context_pos_emo]
        context_neg_emo_labels = [self.emotion_statistic[emo]["idx"] for emo in context_neg_emo]
        next_uttr_pos_emo_labels = [self.emotion_statistic[emo]["idx"] for emo in next_uttr_pos_emo]
        next_uttr_neg_emo_labels = [self.emotion_statistic[emo]["idx"] for emo in next_uttr_neg_emo]
        return context_pos_emo_labels, context_neg_emo_labels, next_uttr_pos_emo_labels, next_uttr_neg_emo_labels

    def process_kws_labels(self, target_positive_kws, target_negative_kws, next_uttr_positive_kws, next_uttr_negative_kws):
        target_pos_kws_labels = self.tokenizer.encode(" ".join(target_positive_kws))
        target_neg_kws_labels = self.tokenizer.encode(" ".join(target_negative_kws))
        next_uttr_pos_kws_labels = self.tokenizer.encode(" ".join(next_uttr_positive_kws))
        next_uttr_neg_kws_labels = self.tokenizer.encode(" ".join(next_uttr_negative_kws))
        return target_pos_kws_labels, target_neg_kws_labels, next_uttr_pos_kws_labels, next_uttr_neg_kws_labels
    
    def process_target(self, target, strategy):
        target_ids = self.tokenizer.encode("["+strategy+"]") + self.tokenizer.encode(" ".join(target)) + [self.EOS_id]
        target_role_ids = [self.args.supporter_idx] * len(target_ids)
        return target_ids, target_role_ids, target_ids

    def collate_fn(self, data):        
        data_tensor = {}
        ignore_keys = {"context_txt", "target_txt", "dialog_turn", "context_seeker_sum_emo_score", "next_uttr_emotion_score", "context_last_infer_kws", "next_uttr_infer_kws", "context_role", "context_emotion_scores"}
        for key in data[0].keys():
            if key in ignore_keys or "txt" in key:
                data_tensor[key] = [item[key] for item in data]
            elif key == "target_lm_labels":
                data_tensor[key] = pad_sequence(
                    [torch.tensor(item[key], dtype=torch.long)
                    for item in data],
                    batch_first=True, padding_value=-100).to(self.args.device)
            else:
                data_tensor[key] = pad_sequence(
                    [torch.tensor(item[key], dtype=torch.long)
                    for item in data],
                    batch_first=True, padding_value=0).to(self.args.device)
        return data_tensor
