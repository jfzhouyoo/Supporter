import os
import sys
import warnings
warnings.filterwarnings("ignore")
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig,BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from src.utils import set_seed, get_logger, str2bool
from rewards.rewards_dataloader import RewardsDataset as Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, default="rewards/rewardsdataset.pkl")
    parser.add_argument("--max_context_length", type=int, default=256)
    parser.add_argument("--seeker_idx", type=int, default=0)
    parser.add_argument("--supporter_idx", type=int, default=1)
    parser.add_argument("--is_situation", type=str2bool, default=False)

    # model parameter
    parser.add_argument("--pretrained_model", type=str, default="./rewards/bert/")
    parser.add_argument("--pretrained_config", type=str, default="./rewards/bert/")
    parser.add_argument("--pretrained_tokenizer", type=str, default="./rewards/bert/")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--early_epochs", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=120)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--n_class", type=int, default=2)

    parser.add_argument("--is_train", type=str, default=False)
    parser.add_argument("--is_test", type=str, default=False)
    parser.add_argument("--is_evaluate", type=str, default=False)
    parser.add_argument("--is_pretrain", type=str2bool, default=False)
    parser.add_argument("--is_with_pretrain", type=str2bool, default=False)
    parser.add_argument("--is_grid_search", type=str2bool, default=False)
    parser.add_argument("--is_evaluate_coher_elicit", type=str2bool, default=False)
    parser.add_argument("--is_load_grid", type=str2bool, default=False)
    parser.add_argument("--load_grid_idx", type=int, default=20)

    parser.add_argument("--max_length", type=int, default=512)

    # other
    parser.add_argument("--save_method", type=str, default="ppl")
    parser.add_argument("--direction", type=str, default="forward")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_step", type=int, default=100)
    parser.add_argument("--save_rewards_path", type=str, default="rewards/save_model/")
    parser.add_argument("--save_rewards", type=str, default="rewards/save_model/tensorboard/")
    parser.add_argument("--save_rewards_test_log", type=str, default="rewards/save_model/log/rewards_test.log")
    parser.add_argument("--rewards_results_file", type=str, default="rewards/save_model/rewards_results.json")
    parser.add_argument("--save_rewards_evaluate_log", type=str, default="rewards/save_model/log/rewards_evaluate.log")
    parser.add_argument("--results_file", type=str, default="save/results.json")

    parser.add_argument("--save_test_log", type=str, default="save/log/test.log")
    parser.add_argument("--save_evaluate_log", type=str, default="save/log/evaluate.log")
    parser.add_argument("--turn_reward_weight", type=float, default=1.0)
    parser.add_argument("--conversation_reward_weight", type=float, default=1.0)
    parser.add_argument("--context_reward_weight", type=float, default=1.0)
    parser.add_argument("--future_reward_weight", type=float, default=1.0)
    
    args = parser.parse_args()
    cuda_id = "cuda:" + str(args.gpu)
    args.device = torch.device(cuda_id) if torch.cuda.is_available() else 'cpu'
    return args

class RewardsModelAgent:
    def __init__(self, args, additional_special_tokens):
        self.args = args
        # prepare model
        self.pretrained_config = BertConfig.from_pretrained(args.pretrained_config)
        self.pretrained_config.num_labels = args.n_class
        self.pretrained_tokenizer = BertTokenizer.from_pretrained(args.pretrained_tokenizer)
        self.pretrained_tokenizer.add_tokens(additional_special_tokens)
        self.model = BertForSequenceClassification.from_pretrained(args.pretrained_model, config=self.pretrained_config).to(args.device)
        self.model.resize_token_embeddings(len(self.pretrained_tokenizer))
        # preprocess data
        self.preprocess_data()
        # prepare optimizer
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
        total_steps = len(self.training_dataloader) // self.args.gradient_accumulation_steps * self.args.train_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps = self.args.warmup_steps, 
            num_training_steps = total_steps
        )
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.log_writer = SummaryWriter(log_dir=args.save_rewards)

    def preprocess_data(self):
        [data_train, data_dev, data_test] = pickle.load(open(self.args.dataset, "rb"))

        # train
        training_dataset = Dataset(self.args, data_train, self.pretrained_tokenizer)
        self.training_dataloader = DataLoader(
            dataset=training_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=training_dataset.collate_fn,
        )
        # dev
        dev_dataset = Dataset(self.args, data_dev, self.pretrained_tokenizer)
        self.dev_dataloader = DataLoader(
            dataset=dev_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=dev_dataset.collate_fn,
        )
        # test
        testing_dataset = Dataset(self.args, data_test, self.pretrained_tokenizer)
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
    
    def train(self, save_path):
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        iter_counter = 1
        best_acc = 0.0
        for epoch in range(1, self.args.train_epochs+1):
            train_data_iteration = tqdm(
                self.training_dataloader,
                desc=f"Training epoch: {epoch}",
                total=len(self.training_dataloader),
                bar_format="{l_bar}{r_bar}"
            )
            for train_data in train_data_iteration:
                self.model.train()
                outputs = self.model(
                    input_ids = train_data["input_ids"],
                    token_type_ids = train_data["token_type_ids"],
                    attention_mask = train_data["attention_mask"],
                    labels = train_data["label"]
                )
                loss, logits = outputs["loss"], outputs["logits"]
                # optimize
                loss.backward()
                if iter_counter % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.log_writer.add_scalars("lr", {"training": self.scheduler.get_lr()[0]}, iter_counter)
                # write log
                self.log_writer.add_scalars("loss", {"training": loss.item()}, iter_counter)
                # dev
                if iter_counter % self.args.save_step == 0:
                    dev_loss, dev_acc, dev_f1 = self.dev(epoch, iter_counter)
                    self.log_writer.add_scalars("acc", {"developing": dev_acc}, iter_counter)
                    self.log_writer.add_scalars("loss", {"developing": dev_loss}, iter_counter)
                    self.log_writer.add_scalars("f1", {"developing": dev_f1}, iter_counter)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        self.save(save_path)
                # update iter
                iter_counter += 1

    def dev(self, epoch, iter_counter):
        self.model.eval()
        dev_data_iteration = tqdm(
            self.dev_dataloader,
            desc=f"dev epoch: {epoch}, iter: {iter_counter}",
            total=len(self.dev_dataloader),
            bar_format="{l_bar}{r_bar}"
        )
        dev_loss_list = list()
        pred_labels = []
        truth_labels = []
        with torch.no_grad():
            for dev_data in dev_data_iteration:
                outputs = self.model(
                    input_ids = dev_data["input_ids"],
                    token_type_ids = dev_data["token_type_ids"],
                    attention_mask = dev_data["attention_mask"],
                    labels = dev_data["label"]
                )
                loss, logits = outputs["loss"], outputs["logits"]
                dev_loss_list.append(loss.item())
                logits = logits.detach().cpu().numpy()
                preds = np.argmax(logits, axis=1).flatten()
                pred_labels.extend(preds.tolist())
                truth_labels.extend(dev_data["label"].cpu())
                assert len(pred_labels) == len(truth_labels)
            dev_loss = np.mean(dev_loss_list)
            accuracy = accuracy_score(truth_labels, pred_labels)
            f1 = f1_score(truth_labels, pred_labels, average='macro')
        return dev_loss, accuracy, f1

    def test(self, load_path):
        self.model.eval()
        self.load(load_path)
        test_logger = get_logger(self.args.save_rewards_test_log)
        test_data_iteration = tqdm(
            self.test_dataloader,
            desc="testing...",
            total=len(self.test_dataloader),
            bar_format="{l_bar}{r_bar}"
        )
        test_loss_list = list()
        pred_labels = []
        truth_labels = []
        with torch.no_grad():
            for test_data in test_data_iteration:
                outputs = self.model(
                    input_ids = test_data["input_ids"],
                    token_type_ids = test_data["token_type_ids"],
                    attention_mask = test_data["attention_mask"],
                    labels = test_data["label"]
                )
                loss, logits = outputs["loss"], outputs["logits"]
                test_loss_list.append(loss.item())
                logits = logits.detach().cpu().numpy()
                preds = np.argmax(logits, axis=1).flatten()
                pred_labels.extend(preds.tolist())
                truth_labels.extend(test_data["label"].cpu())
                assert len(pred_labels) == len(truth_labels)
            test_loss = np.mean(test_loss_list)
            accuracy = accuracy_score(truth_labels, pred_labels)
            f1 = f1_score(truth_labels, pred_labels, average='macro')
        test_logger.info(f"test_loss: {test_loss} , test_acc: {accuracy}, test_f1: {f1}")

def main():
    args = get_args()
    set_seed(args.seed)
    ####### debug ##########
    # args.is_train = True
    # args.batch_size = 2
    # args.simulator = "dialoggpt"
    ####### debug ##########
    additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]", "[KWS]"]
    rewardsmodel = RewardsModelAgent(args, additional_special_tokens)
    save_path = args.save_rewards_path + f"{args.direction}-rewardsmodel.ckpt"
    if args.is_train:
        print("RewardsModel Training Start......")
        rewardsmodel.train(save_path)
    
    if args.is_test and os.path.exists(save_path):
        print("RewardsModel Testing Start......")
        rewardsmodel.test(save_path)
    
if __name__=="__main__":
    main()
