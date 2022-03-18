#from .graph2text.finetune import SummarizationModule, Graph2TextModule
from .graph2text.finetune import Graph2TextModule
#import argparse
#import pytorch_lightning as pl
#import os
#import sys
#from pathlib import Path
#import pdb
from typing import Dict, List, Tuple, Union, Optional
import torch
import re

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2

DATA_DIR = 'verbalisation/graph2text/data/webnlg'
OUTPUT_DIR = 'verbalisation/graph2text/outputs/port_test'
CHECKPOINT = 'verbalisation/graph2text/outputs/t5-base_13881/val_avg_bleu=68.1000-step_count=5.ckpt'
MAX_LENGTH = 384
SEED = 42


class VerbModule():
    
    def __init__(self, override_args: Dict[str, str] = None): 
        # Model
        if not override_args:
            override_args = {
                'data_dir': DATA_DIR,
                'output_dir': OUTPUT_DIR,
            }
        self.g2t_module = Graph2TextModule.load_from_checkpoint(CHECKPOINT, strict=False, **override_args)
        # Unk replacer
        self.vocab = self.g2t_module.tokenizer.get_vocab()
        self.convert_some_japanese_characters = True
        self.unk_char_replace_sliding_window_size = 2
        self.unknowns = []

    def __generate_verbalisations_from_inputs(self, inputs: Union[str, List[str]]):
        try:
            inputs_encoding = self.g2t_module.tokenizer.prepare_seq2seq_batch(
                inputs, truncation=True, max_length=MAX_LENGTH, return_tensors='pt'
            )
            
            self.g2t_module.model.eval()
            with torch.no_grad():
                gen_output = self.g2t_module.model.generate(
                    inputs_encoding['input_ids'],
                    attention_mask=inputs_encoding['attention_mask'],
                    use_cache=True,
                    decoder_start_token_id = self.g2t_module.decoder_start_token_id,
                    num_beams= self.g2t_module.eval_beams,
                    max_length= self.g2t_module.eval_max_length,
                    length_penalty=1.0    
                )
        except Exception:
            print(inputs)
            raise

        return gen_output

    def __decode_sentences(self, encoded_sentences: Union[str, List[str]]):
        if type(encoded_sentences) == str:
            encoded_sentences = [encoded_sentences]
        decoded_sentences = [self.g2t_module.tokenizer.decode(i, skip_special_tokens=True) for i in encoded_sentences]
        return decoded_sentences
        
    def verbalise_sentence(self, inputs: Union[str, List[str]]):
        if type(inputs) == str:
            inputs = [inputs]
        
        gen_output = self.__generate_verbalisations_from_inputs(inputs)
        
        decoded_sentences = self.__decode_sentences(gen_output)

        if len(decoded_sentences) == 1:
            return decoded_sentences[0]
        else:
            return decoded_sentences

    def verbalise_triples(self, input_triples: Union[Dict[str, str], List[Dict[str, str]], List[List[Dict[str, str]]]]):
        if type(input_triples) == dict:
            input_triples = [input_triples]

        verbalisation_inputs = []
        for triple in input_triples:
            if type(triple) == dict:
                assert 'subject' in triple
                assert 'predicate' in triple
                assert 'object' in triple
                verbalisation_inputs.append(
                    f'translate Graph to English: <H> {triple["subject"]} <R> {triple["predicate"]} <T> {triple["object"]}'
                )
            elif type(triple) == list:
                input_sentence = ['translate Graph to English:']
                for subtriple in triple:
                    assert 'subject' in subtriple
                    assert 'predicate' in subtriple
                    assert 'object' in subtriple
                    input_sentence.append(f'<H> {subtriple["subject"]}')
                    input_sentence.append(f'<R> {subtriple["predicate"]}')
                    input_sentence.append(f'<T> {subtriple["object"]}')
                verbalisation_inputs.append(
                    ' '.join(input_sentence)
                )

        return self.verbalise_sentence(verbalisation_inputs)
        
    def verbalise(self, input: Union[str, List, Dict]):
        if (type(input) == str) or (type(input) == list and type(input[0]) == str):
            return self.verbalise_sentence(input)
        elif (type(input) == dict) or (type(input) == list and type(input[0]) == dict):
            return self.verbalise_triples(input)
        else:
            return self.verbalise_triples(input)
                
    def add_label_to_unk_replacer(self, label: str):
        N = self.unk_char_replace_sliding_window_size
        self.unknowns.append({})
        
        # Some pre-processing of labels to normalise some characters
        if self.convert_some_japanese_characters:
            label = label.replace('（','(')
            label = label.replace('）',')')
            label = label.replace('〈','<')
            label = label.replace('／','/')
            label = label.replace('〉','>')        
        
        label_encoded = self.g2t_module.tokenizer.encode(label)
        label_tokens = self.g2t_module.tokenizer.convert_ids_to_tokens(label_encoded)
        label_token_to_string = self.g2t_module.tokenizer.convert_tokens_to_string(label_tokens)
        unk_token_to_string = self.g2t_module.tokenizer.convert_tokens_to_string([self.g2t_module.tokenizer.unk_token])
                
        #print(label_encoded,label_tokens,label_token_to_string)
        
        match_unks_in_label = re.findall('(?:(?: )*<unk>(?: )*)+', label_token_to_string)
        if len(match_unks_in_label) > 0:
            # If the whole label is made of UNK
            if (match_unks_in_label[0] + self.g2t_module.tokenizer.eos_token) == label_token_to_string:
                #print('Label is all unks')
                self.unknowns[-1][label_token_to_string.strip()] = label
            # Else, there should be non-UNK characters in the label
            else:
                #print('Label is NOT all unks')
                # Analyse the label with a sliding window of size N (N before, N ahead)
                for idx, token in enumerate(label_tokens):
                    idx_before = max(0,idx-N)
                    idx_ahead = min(len(label_tokens), idx+N+1)
                    
                                       
                    # Found a UNK
                    if token == self.g2t_module.tokenizer.unk_token:
                        
                        # In case multiple UNK, exclude UNKs seen after this one, expand window to other side if possible
                        if len(match_unks_in_label) > 1:
                            #print(idx)
                            #print(label_tokens)
                            #print(label_tokens[idx_before:idx_ahead])
                            #print('HERE!')
                            # Reduce on the right, expanding on the left
                            while self.g2t_module.tokenizer.unk_token in label_tokens[idx+1:idx_ahead]:
                                idx_before = max(0,idx_before-1)
                                idx_ahead = min(idx+2, idx_ahead-1)
                                #print(label_tokens[idx_before:idx_ahead])
                            # Now just reduce on the left
                            while self.g2t_module.tokenizer.unk_token in label_tokens[idx_before:idx]:
                                idx_before = min(idx-1,idx_before+2)
                                #print(label_tokens[idx_before:idx_ahead])

                        span = self.g2t_module.tokenizer.convert_tokens_to_string(label_tokens[idx_before:idx_ahead]).replace('</s>','')        
                        # First token of the label is UNK                        
                        if idx == 1 and label_tokens[0] == '▁':
                            #print('Label begins with unks')
                            to_replace = '^' + re.escape(span).replace(
                                    re.escape(unk_token_to_string),
                                    '.+?'
                                )
                            
                            replaced_span = re.search(
                                to_replace,
                                label
                            )[0]
                            self.unknowns[-1][span.strip()] = replaced_span
                        # Last token of the label is UNK
                        elif idx == len(label_tokens)-2 and label_tokens[-1] == self.g2t_module.tokenizer.eos_token:
                            #print('Label ends with unks')
                            pre_idx = self.g2t_module.tokenizer.convert_tokens_to_string(label_tokens[idx_before:idx])
                            pre_idx_unk_counts = pre_idx.count(unk_token_to_string)
                            to_replace = re.escape(span).replace(
                                    re.escape(unk_token_to_string),
                                    f'[^{re.escape(pre_idx)}]+?'
                                ) + '$'
                            
                            if pre_idx.strip() == '':
                                to_replace = to_replace.replace('[^]', '(?<=\s)[^a-zA-Z0-9]')
                            
                            replaced_span = re.search(
                                to_replace,
                                label
                            )[0]
                            self.unknowns[-1][span.strip()] = replaced_span
                            
                        # A token in-between the label is UNK                            
                        else:
                            #print('Label has unks in the middle')
                            pre_idx = self.g2t_module.tokenizer.convert_tokens_to_string(label_tokens[idx_before:idx])

                            to_replace = re.escape(span).replace(
                                re.escape(unk_token_to_string),
                                f'[^{re.escape(pre_idx)}]+?'
                            )
                            #If there is nothing behind the ??, because it is in the middle but the previous token is also
                            #a ??, then we would end up with to_replace beginning with [^], which we can't have
                            if pre_idx.strip() == '':
                                to_replace = to_replace.replace('[^]', '(?<=\s)[^a-zA-Z0-9]')
        
                            replaced_span = re.search(
                                to_replace,
                                label
                            )
                            
                            if replaced_span:
                                span = re.sub(r'\s([?.!",](?:\s|$))', r'\1', span.strip())
                                self.unknowns[-1][span] = replaced_span[0]  

    def replace_unks_on_sentence(self, sentence: str, loop_n : int = 3, empty_after : bool = False):
        # Loop through in case the labels are repeated, maximum of three times
        while '<unk>' in sentence and loop_n > 0:
            loop_n -= 1
            for unknowns in self.unknowns:
                for k,v in unknowns.items():
                    # In case it is because the first letter of the sentence has been uppercased
                    if not k in sentence and k[0] == k[0].lower() and k[0].upper() == sentence[0]:
                        k = k[0].upper() + k[1:]
                        v = v[0].upper() + v[1:]
                    # In case it is because a double space is found where it should not be
                    elif not k in sentence and len(re.findall(r'\s{2,}',k))>0:
                        k = re.sub(r'\s+', ' ', k)
                    #print(k,'/',v,'/',sentence)
                    sentence = sentence.replace(k.strip(),v.strip(),1)
                    #sentence = re.sub(k, v, sentence)
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            sentence = re.sub(r'\s([?.!",](?:\s|$))', r'\1', sentence)
        if empty_after:
            self.unknowns = []
        return sentence

if __name__ == '__main__':

    verb_module = VerbModule()
    verbs = verb_module.verbalise('translate Graph to English: <H> World Trade Center <R> height <T> 200 meter <H> World Trade Center <R> is a <T> tower')
    print(verbs)