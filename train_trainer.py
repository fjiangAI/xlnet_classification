#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : xlnet_baseline                 #
#                                                                   #
#                   @File Name    : train_trainer.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2020/11/12 20:21                 #
#                                                                   #
#                   @Last Update  : 2020/11/12 20:21                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from callback.progressbar import ProgressBar
import glob

from transformers import Trainer, XLNetTokenizer, \
    XLNetForSequenceClassification

from config_args import deal_parser, set_args_again
from custom_dataset import get_dataset
from utils import acc_and_f1, seed_everything, create_dir

WEIGHTS_NAME = "pytorch_model.bin"


class ExperimentTrainer:
    def __init__(self, training_args):
        self.training_args = training_args
        self.dataset = {}
        self.trainer = None

    def set_dataset(self, train_dataset, eval_dataset=None, test_dataset=None):
        self.dataset["train_dataset"] = train_dataset
        self.dataset["val_dataset"] = eval_dataset
        self.dataset["test_dataset"] = test_dataset

    def train(self, model):
        """
        Instantiation trainer.
        You can add more args in Trainer.
        :param model:
        :return:
        """
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.dataset["train_dataset"],
            eval_dataset=self.dataset["val_dataset"],
        )
        self.trainer.train()

    def _get_checkpoints(self, args):
        """
        get the checkpoints you saved.
        :param args:
        :return:
        """
        checkpoints = [(0, args.output_dir)]
        if args.predict_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in
                sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = [(int(checkpoint.split('-')[-1]), checkpoint) for checkpoint in checkpoints if
                           checkpoint.find('checkpoint') != -1]
            checkpoints = sorted(checkpoints, key=lambda x: x[0])
        print("Test the following checkpoints: %s", checkpoints)
        return checkpoints

    def test(self, args):
        results = []
        checkpoints = self._get_checkpoints(args)
        for _, checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = XLNetForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = self._test(args, model, prefix=prefix)
            results.extend([(k + '_{}'.format(global_step), v) for k, v in result.items()])
        output_test_file = os.path.join(args.output_dir, "checkpoint_test_results.txt")
        with open(output_test_file, "w") as writer:
            for key, value in results:
                writer.write("%s = %s\n" % (key, str(value)))

    def _save_result(self, model_type, preds, out_label_ids):
        """
        save the test result.
        :param model_type:
        :param preds:
        :param out_label_ids:
        :return:
        """
        create_dir("./outputs/")
        create_dir("./outputs/" + model_type)
        np.save("./outputs/" + model_type + "/" + "predict.npy", preds)
        np.save("./outputs/" + model_type + "/" + "real.npy", out_label_ids)

    def _test(self, args, model, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        test_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        test_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (
            args.output_dir,)

        results = {}
        for test_task, test_output_dir in zip(test_task_names, test_outputs_dirs):
            test_dataset = self.dataset["test_dataset"]
            if not os.path.exists(test_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(test_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

            # Test!
            print("***** Running test {} *****".format(prefix))
            print("  Num examples = %d", len(test_dataset))
            print("  Batch size = %d", args.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            pbar = ProgressBar(n_total=len(test_dataloader), desc="Testing")
            for step, batch in enumerate(test_dataloader):
                model.eval()
                with torch.no_grad():
                    inputs = {'input_ids': batch["input_ids"].to(args.device),
                              'attention_mask': batch['attention_mask'].to(args.device),
                              'token_type_ids': batch['token_type_ids'].to(args.device),
                              "labels": batch["labels"].to(args.device)}
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs.loss, outputs.logits
                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                pbar(step)
            print(' ')
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
            self._save_result(args.model_type, preds, out_label_ids)
            preds = np.argmax(preds, axis=1)
            result = acc_and_f1(preds, out_label_ids, average="micro")
            results.update(result)
        return results


def load_model_and_tokenizer(root):
    """
    load model
    :param root:
    :return:
    """
    tokenizer = XLNetTokenizer.from_pretrained(root)
    model = XLNetForSequenceClassification.from_pretrained(root, return_dict=True)
    return model, tokenizer


def show_info(args):
    """
    show the run info.
    :param args:
    :return:
    """
    print("args.data_dir:" + args.data_dir)
    print("args.model_name" + args.model_name)
    print("args.output_dir" + args.output_dir)


def main():
    # 1. Set args
    args = deal_parser()
    args, training_args = set_args_again(args)
    # 2. Show info about run
    show_info(args)
    # 3. Set seed
    seed_everything(args.seed)
    # 4. load trainer,model,tokenizer and dataset.
    experiment_trainer = ExperimentTrainer(training_args)
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    train_dataset, val_dataset, test_dataset = get_dataset(tokenizer, dataset_root=args.data_dir)
    experiment_trainer.set_dataset(train_dataset=train_dataset, eval_dataset=val_dataset, test_dataset=test_dataset)
    # 5. train and test
    experiment_trainer.train(model=model)
    experiment_trainer.test(args)


if __name__ == '__main__':
    main()
