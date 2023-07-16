import os
import json
import random
import pickle
import torch
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from sklearn.model_selection import ParameterGrid
from pytorch_lightning import seed_everything
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from main_rewards import RewardsModelAgent
from src.dataloader import Dataset
from metric.myMetrics import Metric
from src.utils import set_seed, get_logger, str2bool, kw_tokenize
from src.model import Supporter
from src.transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    #AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from src.transformers import (BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig)

def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, default="data/dataset_preproc.p")
    parser.add_argument("--emotion_statistic", type=str, default="data/emotion_statistic.json")
    parser.add_argument("--vad_dict", type=str, default="data/VAD.json")
    parser.add_argument("--conv_graph", type=str, default="data/ConstructConvGraph/conv_graph.json")
    parser.add_argument("--kws_vocab", type=str, default="data/ConstructConvGraph/total_kws.pkl")
    parser.add_argument("--comet", type=str, default="data/ConstructDataset/Comet")
    parser.add_argument("--max_context_length", type=int, default=256)
    parser.add_argument("--max_context_kws_length", type=int, default=128)
    parser.add_argument("--max_infer_kws_length", type=int, default=128)
    parser.add_argument("--max_emotion_labels", type=int, default=10)
    parser.add_argument("--max_num_kws", type=int, default=128)
    parser.add_argument("--seeker_idx", type=int, default=0)
    parser.add_argument("--supporter_idx", type=int, default=1)
    parser.add_argument("--context_kws_seeker_idx", type=int, default=0)
    parser.add_argument("--context_kws_supporter_idx", type=int, default=1)
    parser.add_argument("--context_infer_kws_idx", type=int, default=1)
    parser.add_argument("--next_uttr_infer_kws_idx", type=int, default=1)

    # model parameters
    parser.add_argument("--pretrained_blender_model", type=str, default="blender", help="facebook/blenderbot_small-90M")
    parser.add_argument("--pretrained_blender_config", type=str, default="blender", help="facebook/blenderbot_small-90M")
    parser.add_argument("--pretrained_blender_tokenizer", type=str, default="blender", help="facebook/blenderbot_small-90M")
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--pretrained_emo_score_model", type=str, default="emotion")

    parser.add_argument("--num_emo_experts", type=int, default=4)
    parser.add_argument("--num_kws_experts", type=int, default=4)
    parser.add_argument("--max_num_actions", type=int, default=2)
    parser.add_argument("--policy_dropout", type=float, default=0.5)

    parser.add_argument("--empathy_turn", type=int, default=6)
    parser.add_argument("--max_dialog_turn", type=int, default=10)
    parser.add_argument("--turn_reward_weight", type=float, default=1.0)
    parser.add_argument("--conversation_reward_weight", type=float, default=1.0)
    parser.add_argument("--context_reward_weight", type=float, default=1.0)
    parser.add_argument("--future_reward_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    
    # train parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--pretrain_epochs", type=int, default=5)
    parser.add_argument("--early_epochs", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=120)
    parser.add_argument("--max_grad_norm", type=int, default=1.0)
    parser.add_argument("--expert_mse_weight", type=float, default=1e-5)

    # other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--use_baseline_rl", type=str2bool, default=False)
    parser.add_argument("--sliding_window", type=int, default=20)
    parser.add_argument("--is_train", type=str2bool, default=False)
    parser.add_argument("--is_test", type=str2bool, default=False)
    parser.add_argument("--is_evaluate", type=str2bool, default=False)
    parser.add_argument("--is_evaluate_coher_elicit", type=str2bool, default=False)
    parser.add_argument("--is_grid_search", type=str2bool, default=False)
    parser.add_argument("--is_load_grid", type=str2bool, default=False)
    parser.add_argument("--load_grid_idx", type=int, default=20)
    parser.add_argument("--is_pretrain", type=str2bool, default=False)
    parser.add_argument("--is_with_pretrain", type=str2bool, default=False)
    parser.add_argument("--is_interact", type=str2bool, default=False)
    parser.add_argument("--simulator", type=str, default="dialoggpt", help="[blenderbot, blenderbot-vanilla, dialoggpt, dialoggpt-vanilla]")
    parser.add_argument("--num_interaction_turn", type=int, default=10)
    
    # save
    parser.add_argument("--save_method", type=str, default="ppl")
    parser.add_argument("--save_step", type=int, default=100)
    parser.add_argument("--save_model_path", type=str, default="save/model/")
    parser.add_argument("--save_simulator_path", type=str, default="simulator/save_model/")
    parser.add_argument("--save_log", type=str, default="save/log/")
    parser.add_argument("--save_tensorboard", type=str, default="save/tensorboard/")
    parser.add_argument("--save_test_log", type=str, default="save/log/test.log")
    parser.add_argument("--save_evaluate_log", type=str, default="save/log/evaluate.log")
    parser.add_argument("--save_grid_search_log", type=str, default="save/log/grid_search.log")
    parser.add_argument("--save_grid_search_eval_log", type=str, default="save/log/grid_search_eval.log")
    parser.add_argument("--save_interact_log", type=str, default="save/log/interact.log")
    parser.add_argument("--results_file", type=str, default="save/results.json")
    parser.add_argument("--interaction_result_file", type=str, default="save/interaction_results.json")

    args = parser.parse_args()
    cuda_id = "cuda:" + str(args.gpu)
    args.device = torch.device(cuda_id) if torch.cuda.is_available() else 'cpu'
    return args

class Agent:
    def __init__(
        self, 
        args, 
        additional_special_tokens,
        emotion_statistic,
        kws_vocab
    ):
        self.args = args
        self.pretrained_blender_config = BlenderbotSmallConfig.from_pretrained(args.pretrained_blender_config)
        self.pretrained_blender_tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.pretrained_blender_tokenizer)
        self.pretrained_blender_tokenizer.add_tokens(additional_special_tokens)
        self.pretrained_blender_tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        self.pretrained_blender_tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        # preprocess data
        self.preprocess_data()
        # prepare rewards model
        rewards_model = self.get_rewards_model(additional_special_tokens)
        # prepare model
        self.model = Supporter(
            args, 
            self.pretrained_blender_tokenizer,
            self.pretrained_blender_config,
            emotion_statistic,
            kws_vocab,
            rewards_model
        ).to(args.device)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        # log
        self.log_writer = SummaryWriter(log_dir=args.save_tensorboard)
    
    def prepare_optimizer(self, epochs):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr = self.args.learning_rate, 
            eps = self.args.adam_epsilon
        )
        total_steps = len(self.training_dataloader) // self.args.gradient_accumulation_steps * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps = self.args.warmup_steps, 
            num_training_steps = total_steps
        )
    
    def preprocess_data(self):
        [data_train, data_dev, data_test] = pickle.load(open(self.args.dataset, "rb"))
        emotion_statistic = json.load(open(self.args.emotion_statistic, "r", encoding="utf-8"))
        
        # train
        training_dataset = Dataset(self.args, data_train, self.pretrained_blender_tokenizer, emotion_statistic)
        self.training_dataloader = DataLoader(
            dataset=training_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=training_dataset.collate_fn,
        )
        # dev
        dev_dataset = Dataset(self.args, data_dev, self.pretrained_blender_tokenizer, emotion_statistic)
        self.dev_dataloader = DataLoader(
            dataset=dev_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=dev_dataset.collate_fn,
        )
        # test
        testing_dataset = Dataset(self.args, data_test, self.pretrained_blender_tokenizer, emotion_statistic)
        self.test_dataloader = DataLoader(
            dataset=testing_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=testing_dataset.collate_fn,
        )
    
    def save(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location=self.args.device))
    
    def write_log(self, losses, iter_counter, description):
        not_tensor = {"lr"}
        for key, value in losses.items():
            if key in not_tensor:
                self.log_writer.add_scalars(key, {description: value}, iter_counter)
            else:
                self.log_writer.add_scalars(key, {description: value.item()}, iter_counter)

    def pretrain(self, save_path, logger=None):
        self.prepare_optimizer(self.args.pretrain_epochs)
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        iter_counter = 1
        best_ppl = 1000.0
        for epoch in range(1, self.args.pretrain_epochs+1):
            train_data_iteration = tqdm(
                self.training_dataloader,
                desc=f"Pretraining epoch: {epoch}",
                total=len(self.training_dataloader),
                bar_format="{l_bar}{r_bar}"
            )
            for train_data in train_data_iteration:
                self.model.train()
                # modeling
                loss_dict = self.model(train_data, is_pretrain=True)
                # optimize
                loss_dict["loss"].backward()
                if iter_counter % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.write_log({"lr": self.scheduler.get_lr()[0]}, iter_counter, "pretraning")
                self.write_log(loss_dict, iter_counter, "pretraining")
                # dev
                if iter_counter % self.args.save_step == 0:
                    dev_loss_dict = self.dev(epoch, iter_counter, logger=logger, is_pretrain=True)
                    self.write_log(dev_loss_dict, iter_counter, "pretraining-dev")
                    dev_ppl = dev_loss_dict["ppl"]
                    if dev_ppl <= best_ppl:
                        best_ppl = dev_ppl
                        self.save(save_path)
                iter_counter += 1
        print("Pretrain Done! & Saved!")
    
    def train(self, save_path, logger=None):
        # prepare_optimizer
        self.prepare_optimizer(self.args.train_epochs)
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        iter_counter = 1
        best_ppl = 1000.0
        best_rewards = 0.0
        # training
        for epoch in range(1, self.args.train_epochs+1):
            # baseline-based-rl
            if self.args.use_baseline_rl:
                rewards_sum = [0.0]
                sliding_idx = 0
            train_data_iteration = tqdm(
                self.training_dataloader,
                desc=f"Training epoch: {epoch}",
                total=len(self.training_dataloader),
                bar_format="{l_bar}{r_bar}"
            )
            for train_data in train_data_iteration:
                self.model.train()
                # baseline-based rl
                if self.args.use_baseline_rl:
                    if len(rewards_sum) >= self.args.sliding_window:
                        baseline_val = (rewards_sum[sliding_idx] - rewards_sum[sliding_idx-self.args.sliding_window]) / self.args.sliding_window
                    else:
                        baseline_val = rewards_sum[sliding_idx]
                else:
                    baseline_val = 0.0
                # modeling
                loss_dict = self.model(train_data, baseline_val=baseline_val, is_joint_train=True)
                # optimize
                loss_dict["loss"].backward()
                if iter_counter % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.write_log({"lr": self.scheduler.get_lr()[0]}, iter_counter, "traning")
                # baseline-based rl
                if self.args.use_baseline_rl:
                    rewards_sum.append(rewards_sum[sliding_idx-1] + loss_dict["rewards"])
                    sliding_idx += 1
                # write log
                self.write_log(loss_dict, iter_counter, "training")
                # dev
                if iter_counter % self.args.save_step == 0:
                    dev_loss_dict = self.dev(epoch, iter_counter, logger=logger, is_joint_train=True)
                    self.write_log(dev_loss_dict, iter_counter, "developing")
                    if epoch <= self.args.early_epochs:
                        iter_counter += 1
                        continue
                    if self.args.save_method == "ppl":
                        dev_ppl = dev_loss_dict["ppl"]
                        if dev_ppl <= best_ppl:
                            best_ppl = dev_ppl
                            self.save(save_path)
                    elif self.args.save_method == "rewards":
                        dev_rewards = dev_loss_dict["rewards"]
                        if dev_rewards > best_rewards:
                            best_rewards = dev_rewards
                            self.save(save_path)
                    else:
                        raise ValueError("Save Method Error!")
                # update iter
                iter_counter += 1

    def dev(self, epoch, iter_counter, is_pretrain=False, is_joint_train=False, logger=None):
        self.model.eval()
        loss_list = defaultdict(list)
        loss_dict = defaultdict()
        eval_loss = 0.0
        eval_num = list()
        dev_data_iteration = tqdm(
            self.dev_dataloader,
            desc=f"dev epoch: {epoch}, iter: {iter_counter}",
            total=len(self.dev_dataloader),
            bar_format="{l_bar}{r_bar}"
        )
        with torch.no_grad():
            for dev_data in dev_data_iteration:
                return_loss_dict = self.model(dev_data, is_pretrain=is_pretrain, is_joint_train=is_joint_train)
                for key, value in return_loss_dict.items():
                    loss_list[key].append(value.item())
                eval_loss += return_loss_dict["gen_loss"].item() * (dev_data["target_lm_labels"].cpu().numpy() != -100).astype(np.int).sum()
                eval_num.append((dev_data["target_lm_labels"].cpu().numpy() != -100).astype(np.int).sum())
            for key, value in loss_list.items():
                loss_dict[key] = np.mean(value)
            loss_dict["ppl"] = torch.exp(torch.tensor(eval_loss / sum(eval_num)))
        if logger is not None:
            logger.info(str(loss_dict))
        return loss_dict

    def test(self, load_path, logger=None):
        self.model.eval()
        loss_list = defaultdict(list)
        loss_dict = defaultdict()
        eval_loss = 0.0
        eval_num = list()
        save_results = list()
        test_logger = logger
        # Load Model
        self.load(load_path)
        test_data_iteration = tqdm(
            self.test_dataloader,
            desc="testing...",
            total=len(self.test_dataloader),
            bar_format="{l_bar}{r_bar}"
        )
        with torch.no_grad():
            for test_data in test_data_iteration:
                return_loss_dict, response_results = self.model(test_data, is_test=True)
                for key, value in return_loss_dict.items():
                    loss_list[key].append(value.item())
                eval_loss += return_loss_dict["gen_loss"].item() * (test_data["target_lm_labels"].cpu().numpy() != -100).astype(np.int).sum()
                eval_num.append((test_data["target_lm_labels"].cpu().numpy() != -100).astype(np.int).sum())
                results = defaultdict()
                results["context"] = [" ".join(txt) for txt in test_data["context_txt"][0]]
                results["target"] = " ".join(test_data["target_txt"][0])
                results["generation"] = response_results[0][response_results[0].index("]")+1:].strip()
                results["strategy_generation"] = response_results[0].strip()
                # for rewards
                results["dialog_turn"] = test_data["dialog_turn"][0]
                results["context_seeker_sum_emo_score"] = test_data["context_seeker_sum_emo_score"][0]
                results["next_uttr_emotion_score"] = test_data["next_uttr_emotion_score"][0]
                results["context_last_infer_kws"] = test_data["context_last_infer_kws"][0]
                results["next_uttr_infer_kws"] = test_data["next_uttr_infer_kws"][0]
                results["context_txt"] = test_data["context_txt"][0]
                results["context_strategy_seqs_txt"] = test_data["context_strategy_seqs_txt"][0]
                results["context_role_txt"] = test_data["context_role_txt"][0]
                results["context_positive_kws_txt"] = test_data["context_positive_kws_txt"][0] 
                results["context_negative_kws_txt"] = test_data["context_negative_kws_txt"][0]
                results["next_uttr_txt"] = test_data["next_uttr_txt"][0]
                results["next_uttr_positive_kws_txt"] = test_data["next_uttr_positive_kws_txt"][0]
                results["next_uttr_negative_kws_txt"] = test_data["next_uttr_negative_kws_txt"][0]
                results["context_role"] = test_data["context_role"][0]
                results["context_emotion_scores"] = test_data["context_emotion_scores"][0]
                save_results.append(results)
            for key, value in loss_list.items():
                loss_dict[key] = np.mean(value)
            loss_dict["ppl"] = torch.exp(torch.tensor(eval_loss / sum(eval_num)))
        if test_logger is not None:
            test_logger.info(str(loss_dict))
        with open(self.args.results_file, "w", encoding="utf-8") as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        return loss_dict
    
    def evaluate(self, logger=None):
        evaluate_log = logger
        results_file = json.load(open(self.args.results_file, "r", encoding="utf-8"))
        target = [results["target"] for results in results_file]
        generation = [results["generation"] for results in results_file]
        automatic_metrics = Metric(self.pretrained_blender_tokenizer)
        for tar, gen in zip(target, generation):
            automatic_metrics.forword([tar], gen)
        automatic_results, _ = automatic_metrics.close()
        if logger is not None:
            evaluate_log.info(str(automatic_results))
        return automatic_results
    
    def evaluate_coher_elict_metrics(self, logger=None):
        evaluate_log  = logger
        results_file = json.load(open(self.args.results_file, "r", encoding="utf-8"))
        results = defaultdict(list)
        for line in results_file:
            for key, value in line.items():
                results[key].append(value)
        # Elicitation Scores
        response_turn_emotion_score = [
            self.model.reward_agent.get_emotion_score(utterance)
            for utterance in results["generation"]
        ]
        response_conv_emotion_score = [
            self.model.reward_agent.get_emotion_score(utterance)
            for utterance in results["generation"]
        ]
        turn_level_elicit_reward, _ = self.model.reward_agent.turn_level_elicitation(results, response_turn_emotion_score)
        total_conversation_level_elicit_reward, conversation_level_elicit_rewards_array = self.model.reward_agent.conversation_level_elicitation(results, response_conv_emotion_score)
        # conversation_level_elicit_reward = np.mean([reward for reward, turn in zip(conversation_level_elicit_rewards_array, results["dialog_turn"]) if turn >= self.args.empathy_turn])
        f1_elicit_reward = 2*turn_level_elicit_reward*total_conversation_level_elicit_reward / (turn_level_elicit_reward+total_conversation_level_elicit_reward)
        # Coherence Scores
        response_kws = [
            list(set([kws for kws in kw_tokenize(utterance) if kws in self.model.reward_agent.total_kws]))
            for utterance in results["generation"]
        ]
        # context_coherence_reward, _ = self.model.reward_agent.context_coherence(results, response_kws)
        # future_coherence_reward, _ = self.model.reward_agent.future_coherence(results, response_kws)
        context_coherence_reward, _, future_coherence_reward, _ = self.model.reward_agent.calc_coherence(results, results["strategy_generation"], response_kws)
        f1_coherence_reward = 2*context_coherence_reward*future_coherence_reward/(context_coherence_reward+future_coherence_reward)
        coher_elicit_scores = {
            "turn_level_elicit_reward": turn_level_elicit_reward,
            # "conversation_level_elicit_reward": conversation_level_elicit_reward,
            "conversation_level_elicit_reward": total_conversation_level_elicit_reward,
            "F1_elicit_reward": f1_elicit_reward,
            "context_coherence_reward": context_coherence_reward,
            "future_coherence_reward": future_coherence_reward,
            "F1_coherence_reward": f1_coherence_reward
        }
        if logger is not None:
            evaluate_log.info("Evaluate Coherence and Elicitation Scores......")
            evaluate_log.info(str(coher_elicit_scores))
        return coher_elicit_scores
    
    def get_rewards_model(self, additional_special_tokens):
        additional_special_tokens.append("[KWS]")
        from main_rewards import get_args as get_rewards_args
        rewards_args = get_rewards_args()
        rewards_args.device = self.args.device
        save_forward_path = rewards_args.save_rewards_path + "forward-rewardsmodel.ckpt"
        save_backward_path = rewards_args.save_rewards_path + "backward-rewardsmodel.ckpt"
        rewards_model_agent = RewardsModelAgent(rewards_args, additional_special_tokens)
        rewards_model_agent.model.eval()
        with torch.no_grad():
            rewards_model_agent.model.load_state_dict(torch.load(save_forward_path, map_location=rewards_args.device))
            forward_rewards_model = deepcopy(rewards_model_agent.model)
            rewards_model_agent.model.load_state_dict(torch.load(save_backward_path, map_location=rewards_args.device))
            backward_rewards_model = deepcopy(rewards_model_agent.model)
        return forward_rewards_model, backward_rewards_model, rewards_model_agent.pretrained_tokenizer
    
def grid_search_train(args, param_grid, additional_special_tokens, emotion_statistic, kw_vocab):
    # log
    grid_search_log = get_logger(args.save_grid_search_log)
    grid_search_eval_log = get_logger(args.save_grid_search_eval_log)
    # obtain grid param
    grid_counter = ParameterGrid(param_grid)
    grid_list = [dict(grid.items()) for grid in grid_counter]
    random.shuffle(grid_list)
    # load pretrain model
    save_pretrain_path = args.save_model_path + "pretrain_supporter.ckpt"
    # for saving model
    save_grid_search_path = args.save_model_path + "supporter-grid.ckpt"
    save_path = args.save_model_path + "supporter.ckpt"
    # save search results
    if args.is_load_grid:
        search_results = {'ppl': 14.9057, 'bleu-1': 17.565652152747056, 'bleu-2': 6.730932192425283, 'dist-1': 4.470718003515907, 'dist-2': 24.95505741644348}
    else:
        search_results = {"ppl":100.0, "bleu-1":0.0, "bleu-2":0.0, "dist-1":0.0, "dist-2": 0.0}
    # grid search
    for idx, grid in enumerate(grid_list):
        grid_search_log.info(str(idx) + ": Grid Parameters = " + str(grid))
        ############## For unexpected interruption #######################
        if args.is_load_grid:
            if idx < args.load_grid_idx: continue
        ############## For unexpected interruption########################
        # change arguments
        grid_args = vars(args)
        for key in grid:
            grid_args[key] = grid[key]
        grid_search_log.info(str(idx) + ": Agent Initialization......")
        agent = Agent(args, additional_special_tokens, emotion_statistic, kw_vocab)
        if args.is_pretrain:
            grid_search_log.info(str(idx) + ": Pretraning Start......")
            agent.pretrain(save_pretrain_path)
        if os.path.exists(save_pretrain_path) and args.is_with_pretrain:
            grid_search_log.info(str(idx) + ": Agent Load Pretrain Model from " + save_pretrain_path)
            agent.load(save_pretrain_path)
        grid_search_log.info(str(idx) + ": Training Start......")
        agent.train(save_grid_search_path, logger=grid_search_eval_log)
        grid_search_log.info(str(idx) + ": Testing Start......")
        test_loss_dict = agent.test(save_grid_search_path)
        grid_search_log.info(str(idx) + ": Testing Loss = " + str(test_loss_dict))
        grid_search_log.info(str(idx) + ": Evaluating Start......")
        automatic_results = agent.evaluate()
        grid_search_log.info(str(idx) + ": Automatic Evaluation Results = " + str(automatic_results))
        automatic_results["ppl"] = test_loss_dict["ppl"]
        # check search targets
        achieve_target_count = 0
        for target in search_results:
            if target == "ppl":
                if automatic_results["ppl"] <= search_results["ppl"]:
                    achieve_target_count += 1
            else:
                if automatic_results[target] >= search_results[target]:
                    achieve_target_count += 1
        # check whether achieving target
        if achieve_target_count >= 3:
            agent.save(save_path)
            for target in search_results:
                search_results[target] = automatic_results[target]
            grid_search_log.info(str(idx) + ": Search Results = " + str(search_results))
            grid_search_log.info("=============Save Once!=================")
        del agent

def main():
    # arguments
    args = get_args()
    
    # for reproducibility
    set_seed(args.seed)
    seed_everything(args.seed)
    # for model
    additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    emotion_statistic = json.load(open(args.emotion_statistic,"r",encoding="utf-8"))
    kw_vocab = pickle.load(open(args.kws_vocab,"rb"))
    agent = Agent(args, additional_special_tokens, emotion_statistic, kw_vocab)
    #for save model
    save_path = args.save_model_path + "supporter.ckpt"
    # for save pretrain model
    save_pretrain_path = args.save_model_path + "pretrain_supporter.ckpt"
    # logger
    eval_logger = get_logger(args.save_evaluate_log)
    test_logger = get_logger(args.save_test_log)
    eval_logger.info(str(vars(args)))
    test_logger.info(str(vars(args)))
    # for pretraining
    if args.is_pretrain:
        eval_logger.info("Pretraining Start......")
        agent.pretrain(save_pretrain_path, logger=eval_logger)
    # for training
    if args.is_train:
        if args.is_grid_search:
            print("Grid Search Training Start......")
            param_grid = {
                "learning_rate": [2e-5],
                "pretrain_epochs": [5],
                "early_epochs": [0],
                "train_epochs": [3],
                "max_num_actions": [2],
                "turn_reward_weight": [0.1, 1, 10],
                "conversation_reward_weight": [1, 10, 0.1],
                "context_reward_weight": [1],
                "future_reward_weight": [0.1]
            }
            grid_search_train(args, param_grid, additional_special_tokens, emotion_statistic, kw_vocab)
        else:
            if os.path.exists(save_pretrain_path) and args.is_with_pretrain:
                eval_logger.info("Load Pretrained Model from " + save_pretrain_path)
                agent.load(save_pretrain_path)
            eval_logger.info("Training Start......") 
            agent.train(save_path, logger=eval_logger)
    # for testing
    if args.is_test and os.path.exists(save_path):
        test_logger.info("Testing Start.....")
        test_loss_dict = agent.test(save_path, logger=test_logger)
    # for evaluting
    if args.is_evaluate and os.path.exists(args.results_file):
        test_logger.info("Evaluating Start......")
        automatic_results = agent.evaluate(logger=test_logger)
    if args.is_evaluate_coher_elicit and os.path.exists(args.results_file):
        coher_elicit_scores = agent.evaluate_coher_elict_metrics(logger=test_logger)


if __name__ == "__main__":
    main()

