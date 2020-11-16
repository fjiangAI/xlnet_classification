#!/usr/bin/env python
# encoding: utf-8
"""
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : xlnet_classification             #
#                                                                   #
#                   @File Name    : custom_dataset.py               #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #
#                   @Start Date   : 2020/11/16 9:47                 #
#                                                                   #
#                   @Last Update  : 2020/11/16 9:47                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
"""
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_custom_split(split_dir):
    """
    get text and labels of the data file.
    :param split_dir: the file name
    :return: texts(list),labels(list)
    """
    texts = []
    labels = []
    labels_maping = {'Joint': 0,
                     'Sequence': 1,
                     'Progression': 2,
                     "Contrast": 3,
                     "Supplement": 4,
                     "Result-Cause": 5,
                     "Cause-Result": 6,
                     "Background": 7,
                     "Behavior-Purpose": 8,
                     "Purpose-Behavior": 9,
                     "Elaboration": 10,
                     "Summary": 11,
                     "Evaluation": 12,
                     "Statement-Illustration": 13,
                     "Illustration-Statement": 14
                     }
    with open(split_dir, encoding='utf-8', mode="r") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            # line: id \t text_a \t text_b \t label \n
            items = line.split('\t')
            # if there is only one input.
            # texts.append(items[1])
            # here is the pair of input.
            texts.append([items[1], items[2]])
            labels.append(labels_maping[items[3]])
    return texts, labels


def get_dataset(tokenizer, dataset_root=""):
    """
    get the dataset
    :param tokenizer: the tokenizer you want to use encoding the input
    :param dataset_root: the data file root.
    :return: train_dataset, val_dataset, test_dataset
    """
    train_texts, train_labels = read_custom_split(dataset_root + '/train.tsv')
    test_texts, test_labels = read_custom_split(dataset_root + '/test.tsv')
    val_texts, val_labels = read_custom_split(dataset_root + '/test.tsv')
    train_encodings = tokenizer(train_texts, return_tensors='pt', padding=True)
    val_encodings = tokenizer(val_texts, return_tensors='pt', padding=True)
    test_encodings = tokenizer(test_texts, return_tensors='pt', padding=True)
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    return train_dataset, val_dataset, test_dataset
