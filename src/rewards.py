import os
import json
import nltk
import torch
import pickle
import numpy as np
from .utils import kw_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
stopwords = stopwords.words("english")

class RewardAgent:
    def __init__(self, args, tokenizer, rewards_model):
        self.args = args
        # self.tokenizer = tokenizer
        self.vad_dict = json.load(open(args.vad_dict, "r", encoding="utf-8"))
        self.total_kws = pickle.load(open(args.kws_vocab, "rb"))
        self.forward_rewards_model = rewards_model[0]
        self.backward_rewards_model = rewards_model[1]
        self.tokenizer = rewards_model[2]
        self.emo_score_model = pipeline("sentiment-analysis", model=self.args.pretrained_emo_score_model, return_all_scores=True, device=self.args.gpu)

    def process_context(self, context, context_strategy, context_role, context_pos_kws, context_neg_kws):
        context_ids = [self.tokenizer.cls_token_id]
        for uttr, strategy, role, pos_kws, neg_kws in zip(context, context_strategy, context_role, context_pos_kws, context_neg_kws):
            if role == "seeker":
                utterance = " ".join(uttr)
            elif role == "supporter":
                utterance = "["+strategy+"]" + " " + " ".join(uttr)
            utterance += " [KWS] "
            kws = pos_kws + neg_kws
            utterance  = utterance + " ".join(kws)
            context_ids = context_ids + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance)) + [self.tokenizer.sep_token_id]
        
        while len(context_ids) > self.args.max_context_length:
            cut_idx = context_ids.index(self.tokenizer.sep_token_id, -self.args.max_context_length+1)
            context_ids = [self.tokenizer.cls_token_id] + context_ids[cut_idx:]
        
        context_token_type_ids = [0] * len(context_ids)
        context_attenttion_mask = [1] * len(context_ids)
        
        return context_ids, context_token_type_ids, context_attenttion_mask
    
    def process_response(self, uttr, kws, is_next_uttr=False):
        if is_next_uttr:
            utterance = " ".join(uttr) + " [KWS] " + " ".join(kws)
            utterance_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance)) + [self.tokenizer.sep_token_id]
            uttr_token_type_ids = [0] * len(utterance_ids)
        else:
            utterance = uttr + " [KWS] " + " ".join(kws)  
            utterance_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance)) + [self.tokenizer.sep_token_id]
            uttr_token_type_ids = [1] * len(utterance_ids)
        uttr_attention_mask = [1] * len(utterance_ids)
        return utterance_ids, uttr_token_type_ids, uttr_attention_mask
    
    @torch.no_grad()
    def calc_coherence(self, data, responses, response_kws):
        def rewards_model_dataset(idx, data, response, response_kws, is_forward=False, is_backward=False):
            response_ids, response_token_type_ids, response_attention_mask = self.process_response(
                response, 
                response_kws
            )
            if is_forward:
                context_ids, context_token_type_ids, context_attention_mask = self.process_context(
                    data["context_txt"][idx],
                    data["context_strategy_seqs_txt"][idx],
                    data["context_role_txt"][idx],
                    data["context_positive_kws_txt"][idx],
                    data["context_negative_kws_txt"][idx]
                )
                input_ids = context_ids + response_ids
                token_type_ids = context_token_type_ids + response_token_type_ids
                attention_mask = context_attention_mask + response_attention_mask
            if is_backward:
                next_uttr_ids, next_uttr_token_type_ids, next_uttr_attention_mask = self.process_response(
                    data["next_uttr_txt"][idx],
                    data["next_uttr_positive_kws_txt"][idx] + data["next_uttr_negative_kws_txt"][idx],
                    is_next_uttr = True
                )
                input_ids = next_uttr_ids + response_ids
                token_type_ids = next_uttr_token_type_ids + response_token_type_ids
                attention_mask = next_uttr_attention_mask + response_attention_mask
            return torch.tensor(input_ids, dtype=torch.long).to(self.args.device), \
                   torch.tensor(token_type_ids, dtype=torch.long).to(self.args.device), \
                   torch.tensor(attention_mask, dtype=torch.long).to(self.args.device)
       
        context_last_infer_kws = data["context_last_infer_kws"]
        context_coherence_rewards = list()
        next_uttr_infer_kws = data["next_uttr_infer_kws"]
        future_coherence_rewards = list()
        for con_inf_kws, next_inf_kws, res_kws, (idx, response) in zip(context_last_infer_kws, next_uttr_infer_kws, response_kws, enumerate(responses)):
            # context
            con_occur_kws = [kws for kws in res_kws if kws in con_inf_kws]
            con_coher_reward = np.exp(len(con_occur_kws) / (len(res_kws) + 1e-20)) / np.exp(1)
            con_input_ids, con_token_type_ids, con_attention_mask = rewards_model_dataset(idx, data, response, res_kws, is_forward=True)
            con_outputs = self.forward_rewards_model(
                input_ids = con_input_ids.view(1, -1),
                token_type_ids = con_token_type_ids.view(1, -1),
                attention_mask = con_attention_mask.view(1, -1),
            )
            con_logits = torch.softmax(con_outputs["logits"], dim=1).cpu().numpy()
            con_coher_reward = con_coher_reward * con_logits[0][1]
            context_coherence_rewards.append(con_coher_reward)
            # future
            next_occur_kws = [kws for kws in res_kws if kws in next_inf_kws]
            next_coher_reward = np.exp(len(next_occur_kws) / (len(res_kws) + 1e-20)) / np.exp(1)
            next_input_ids, next_token_type_ids, next_attention_mask = rewards_model_dataset(idx, data, response, res_kws, is_backward=True)
            next_outputs = self.backward_rewards_model(
                input_ids = next_input_ids.view(1, -1),
                token_type_ids = next_token_type_ids.view(1, -1),
                attention_mask = next_attention_mask.view(1, -1)
            )
            next_logits = torch.softmax(next_outputs["logits"], dim=1).cpu().numpy()
            next_coher_reward = next_coher_reward * next_logits[0][1]
            future_coherence_rewards.append(next_coher_reward)
        return np.mean(context_coherence_rewards), np.array(context_coherence_rewards), \
               np.mean(future_coherence_rewards), np.array(future_coherence_rewards)

    # def context_coherence(self, data, response_kws):
    #     context_last_infer_kws = data["context_last_infer_kws"]
    #     context_coherence_rewards = list()
    #     for res_kws, infer_kws in zip(response_kws, context_last_infer_kws):
    #         occur_kws = [kws for kws in res_kws if kws in infer_kws]
    #         coher_reward = 0.0
    #         if len(occur_kws) > 0:
    #             coher_reward = len(occur_kws) / len(res_kws)
    #         context_coherence_rewards.append(coher_reward)
    #     return np.mean(context_coherence_rewards), np.array(context_coherence_rewards)

    # def future_coherence(self, data, response_kws):
    #     next_uttr_infer_kws = data["next_uttr_infer_kws"]
    #     future_coherence_rewards = list()
    #     for res_kws, infer_kws in zip(response_kws, next_uttr_infer_kws):
    #         occur_kws = [kws for kws in res_kws if kws in infer_kws]
    #         coher_reward = 0.0
    #         if len(occur_kws) > 0:
    #             coher_reward = len(occur_kws) / len(res_kws)
    #         future_coherence_rewards.append(coher_reward)
    #     return np.mean(future_coherence_rewards), np.array(future_coherence_rewards)

    def turn_level_elicitation(self, data, response_emotion_score):
        next_uttr_emotion_score = data["next_uttr_emotion_score"]
        dialog_turn = data["dialog_turn"]
        assert len(next_uttr_emotion_score) == len(response_emotion_score)
        # turn_level_rewards = [
        #     # np.exp(-np.abs(res_emo_score - next_uttr_emo_score))
        #     np.cos(np.pi * np.abs(res_emo_score - next_uttr_emo_score) /2)
        #     for res_emo_score, next_uttr_emo_score in zip(response_emotion_score, next_uttr_emotion_score)
        # ]
        turn_level_rewards = list()
        for turn, res_emo_score, next_uttr_emo_score in zip(dialog_turn, response_emotion_score, next_uttr_emotion_score):
            if turn >= self.args.max_dialog_turn:
                current_turn = self.args.max_dialog_turn - 1 + (turn - self.args.max_dialog_turn) / turn
            else:
                current_turn = turn
            reward = np.cos((np.pi*current_turn)/(2*self.args.max_dialog_turn)) * np.cos(np.pi * np.abs(res_emo_score - next_uttr_emo_score)/2)
            turn_level_rewards.append(reward)
        return np.mean(turn_level_rewards), np.array(turn_level_rewards)

    # def conversation_level_elicitation(self, data, response_emotion_score):
    #     dialog_turn = data["dialog_turn"]
    #     context_seeker_sum_emo_score = data["context_seeker_sum_emo_score"]
    #     assert len(dialog_turn) == len(context_seeker_sum_emo_score) == len(response_emotion_score)
    #     conversation_level_rewards = list()
    #     for turn, context_emo_socre, repsonse_emo_score in zip(dialog_turn, context_seeker_sum_emo_score, response_emotion_score):
    #         emotion_diatance = repsonse_emo_score - context_emo_socre / turn
    #         # if turn < self.args.empathy_turn: # empathy or elicitation
    #         #     if emotion_diatance < 0.0:
    #         #         conversation_level_rewards.append(0.0)
    #         #     else:
    #         #         conversation_level_rewards.append(0.5) # !!!
    #         # else: # elicitation
    #         #     # reward = np.sin((np.pi*turn)/(2*self.args.empathy_turn)) * emotion_diatance
    #         #     reward = np.sin((np.pi*self.args.empathy_turn)/(2*turn)) * emotion_diatance
    #         #     conversation_level_rewards.append(reward)
    #         if turn >= self.args.max_dialog_turn:
    #             current_turn = self.args.max_dialog_turn - 1 + (turn - self.args.max_dialog_turn) / turn
    #         else:
    #             current_turn = turn
    #         reward = np.cos((np.pi*current_turn)/(2*self.args.max_dialog_turn)) * emotion_diatance
    #         conversation_level_rewards.append(reward)
    #     return np.mean(conversation_level_rewards), np.array(conversation_level_rewards)

    def conversation_level_elicitation(self, data, response_emotion_score):
        dialog_turn = data["dialog_turn"]
        context_seeker_sum_emo_score = data["context_seeker_sum_emo_score"]
        context_emotion_scores = data["context_emotion_scores"]
        context_roles = data["context_role"]
        assert len(dialog_turn) == len(context_seeker_sum_emo_score) == len(response_emotion_score)
        conversation_level_rewards = list()
        for current_max_turn, context_sum_emo_socre, repsonse_emo_score, con_emo_socres, con_roles in zip(dialog_turn, context_seeker_sum_emo_score, response_emotion_score, context_emotion_scores, context_roles):
            reward = 0.0 # for calculating total reward
            seeker_num = np.sum([1 for role in con_roles if role=="seeker"])
            turn = current_max_turn - seeker_num # for checking dialog turn
            assert len(con_emo_socres) == len(con_roles)
            assert turn >= 0
            for emo_score, role in zip(con_emo_socres, con_roles):
                if role == "seeker":
                    turn += 1
                    assert turn <= current_max_turn
                    if turn >= self.args.max_dialog_turn:
                        current_turn = self.args.max_dialog_turn - 1 + (turn - self.args.max_dialog_turn) / turn
                    else:
                        current_turn = turn
                    reward += np.cos((np.pi*current_turn)/(2*self.args.max_dialog_turn)) * (repsonse_emo_score - emo_score)
            conversation_level_rewards.append(reward)
        return np.mean(conversation_level_rewards), np.array(conversation_level_rewards)

    # def get_turn_emotion_score(self, utterance):
    #     emotion_score = list()
    #     uttr_words = kw_tokenize(utterance)
    #     for word in uttr_words:
    #         if word not in stopwords and word.isalpha(): # and word in self.vad_dict:
    #             word_vad = self.vad_dict[word][0] if word in self.vad_dict else 0.5
    #             emotion_score.append(word_vad)
    #     avg_score = 0.5
    #     if len(emotion_score) > 0:
    #         avg_score = sum(emotion_score)/len(emotion_score)
    #     return avg_score
    
    # def get_conv_emotion_score(self, utterance):
    #     emotion_score = list()
    #     uttr_words = kw_tokenize(utterance)
    #     for word in uttr_words:
    #         if word not in stopwords and word.isalpha(): # and word in self.vad_dict:
    #             word_vad = self.vad_dict[word][0] if word in self.vad_dict else 0.5
    #             emotion_score.append(word_vad)
    #     avg_score = 0.5
    #     if len(emotion_score) > 0:
    #         avg_score = sum(emotion_score)/len(emotion_score)
    #     return avg_score
    
    def get_emotion_score(self, utterance):
        emotion_scores = self.emo_score_model(utterance)
        score = 0.0
        positive_emotion = ["joy", "surprise"]
        for item in emotion_scores[0]:
            if item["label"] in positive_emotion:
                score += item["score"]
        return score

    def get_rewards(self, data, responses, is_evaluate=False):
        # emotion elicitation rewards
        response_turn_emotion_score = [
            self.get_emotion_score(utterance)
            for utterance in responses
        ]
        response_conv_emotion_score = [
            self.get_emotion_score(utterance)
            for utterance in responses
        ]
        turn_level_elicit_reward, batch_turn_level_rewards = self.turn_level_elicitation(data, response_turn_emotion_score)
        conversation_level_elicit_reward, batch_conversation_level_rewards = self.conversation_level_elicitation(data, response_conv_emotion_score)

        # word coherence rewards
        response_kws = [
            list(set([kws for kws in kw_tokenize(utterance) if kws in self.total_kws]))
            for utterance in responses
        ]
        # context_coherence_reward, batch_context_rewards = self.context_coherence(data, response_kws)
        # future_coherence_reward, batch_future_rewards = self.future_coherence(data, response_kws)

        context_coherence_reward, batch_context_rewards, \
        future_coherence_reward, batch_future_rewards = self.calc_coherence(data, responses, response_kws)

        eval_rewards = turn_level_elicit_reward + conversation_level_elicit_reward + \
                context_coherence_reward + future_coherence_reward

        rewards = self.args.turn_reward_weight * turn_level_elicit_reward + \
                  self.args.conversation_reward_weight * conversation_level_elicit_reward + \
                  self.args.context_reward_weight * context_coherence_reward + \
                  self.args.future_reward_weight * future_coherence_reward
        batch_rewards = self.args.turn_reward_weight * batch_turn_level_rewards + \
                        self.args.conversation_reward_weight * batch_conversation_level_rewards + \
                        self.args.context_reward_weight * batch_context_rewards + \
                        self.args.future_reward_weight * batch_future_rewards
        # assert rewards == np.mean(batch_rewards)
        return rewards, batch_rewards, eval_rewards
    
    def coherence_rewards(self):
        '''
        生成的response中的关键词在conversational graph中与context连贯，并且与next utterance连贯
        '''
        pass

    def elicitation_rewards(self):
        '''
        turn-level elicitation: 生成的response中词的emotion score与next utterance中词的emotion score接近, 此时即考虑了negative elicitation
        conversation-level elicitation: 情感距离 ED = (n-1) * ES_n - (ES_1 + ES_2 + ... + ES_n-1), ES = (e_1 + e_2 + ... + e_m) / m
        0 if n <= max_turn and ED < 0
        1 if n <= max_turn and ED >= 0
        sin((ED * π * n)/(2*max_turn)) if n > max_turn
        '''
        pass