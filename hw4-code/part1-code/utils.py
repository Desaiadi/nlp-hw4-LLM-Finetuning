import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # WordNet synonym replacement with 35% probability
    words = word_tokenize(example["text"])
    transformed_words = []
    
    for word in words:
        # 35% chance to replace each word
        if random.random() < 0.35:
            # Get synonyms from WordNet
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.append(synonym)
            
            # Replace with random synonym if available
            if synonyms:
                transformed_words.append(random.choice(synonyms))
            else:
                transformed_words.append(word)
        else:
            transformed_words.append(word)
    
    # Detokenize back to string
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_words)

    ##### YOUR CODE ENDS HERE ######

    return example