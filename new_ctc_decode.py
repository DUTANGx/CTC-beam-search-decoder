# coding: utf-8
import numpy as np
import pandas as pd
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DICT = os.path.join(BASE, 'dict')


class CTC_DecoderX(object):
    def __init__(self):
        with open(os.path.join(DICT, 'pinyin_dic.txt')) as dic:
            self.vocab = json.load(dic)
        self.reverse_vocab = dict([(v, k) for k, v in self.vocab.items()])
        # in reverse dict the last token should be empty
        self.reverse_vocab[self.vocab['_']] = ''
        self.vocab_size = len(self.vocab.items())

    def beam_search_step(self, context):
        """
        search [beam_size] tokens for next step
        :param context: vector of size vocab_size
        """
        previous_probs = self.top_probs.reshape((-1, 1))
        # use log add scheme
        context = np.log(context.reshape((1, -1)) + 1e-8)
        # current probs is of shape [top_num, beam_size], Cij means the probability ith path connect to jth candidate
        current_probs = previous_probs + context
        # argsort to find top num
        flatten_sorted_args = np.argsort(current_probs.ravel())[:-1 - self.top_num:-1]
        path_ids, token_ids = np.unravel_index(flatten_sorted_args, (self.top_num, self.vocab_size))
        selected_paths = self.top_paths[path_ids]
        selected_tokens = token_ids.reshape((-1, 1))
        # update new top paths and top probs
        self.top_paths = np.concatenate((selected_paths, selected_tokens), axis=1)
        self.top_probs = current_probs.ravel()[flatten_sorted_args[:self.top_num]]

    def merge_path(self, paths):
        """
        merge all redundant tokens
        """
        redundant_mask = (paths[:, 1:] - paths[:, :-1]) != 0
        # discard all blanks
        discard = lambda x: x[x != self.vocab_size - 1]
        merged_paths = [
            discard(np.append(paths[i, 0], paths[i, 1:][(redundant_mask[i])])) for i in
            range(paths.shape[0])]
        # merged_paths = np.concatenate((paths[:,0].reshape(-1,1),merged_paths), axis=1)
        return merged_paths

    def convert_to_tuple(self, paths):
        converted = [tuple(path) for path in paths]
        return converted

    def translate(self, paths):
        """
        translate the merged paths into charaters
        :param paths: list
        :return: list
        """
        if type(paths) is not list:
            paths = [paths]
        single_translate = lambda x: [self.reverse_vocab.get(k) for k in x]
        translated = [single_translate(path) for path in paths]
        return translated

    def map_reduce_prob(self, converted_paths, probs):
        """
        add the probs of same paths
        :param converted_paths:
        :param probs:
        :param is_logadd:
        :return:
        """
        # deal with logged probs
        probs = np.exp(probs)
        top1 = None
        top1_prob = 0
        path_prob_dict = {}
        for path, prob in zip(converted_paths, probs):
            existed_prob = path_prob_dict.get(path)
            if existed_prob:
                prob += existed_prob
            path_prob_dict[path] = prob
            if prob > top1_prob:
                top1_prob = prob
                top1 = path

        return top1, top1_prob, path_prob_dict

    def beam_search(self, AM_outputs, in_len, top_num=None, beam_size=100):
        if top_num:
            self.top_num = top_num
        else:
            self.top_num = beam_size
        self.beam_size = beam_size
        # # initialize top paths filled with blanks
        # self.top_paths = np.ones((self.top_num, 1)) * (self.vocab['_'])
        # self.top_probs = np.zeros(self.top_num)
        # handle first frame
        first_context = AM_outputs[0, 0, :]
        first_context = np.log(first_context + 1e-8)
        first_top_orders = np.argsort(first_context)[:-1 - self.top_num:-1]
        self.top_paths = first_top_orders.reshape((-1, 1))
        self.top_probs = first_context[(first_top_orders)]
        # loop through afterwards frames
        for context_i in range(1, int(in_len)):
            context = AM_outputs[0, context_i, :]
            self.beam_search_step(context)

        merged_paths = self.merge_path(self.top_paths)
        converted_paths = self.convert_to_tuple(merged_paths)
        top1, top1_prob, path_prob_dict = self.map_reduce_prob(converted_paths, self.top_probs)

        return top1, top1_prob, path_prob_dict

    def decode(self, AM_outputs, in_len, top_num=None, beam_size=100):
        top1, top1_prob, path_prob_dict = self.beam_search(AM_outputs, in_len, top_num, beam_size)
        sorted_keys = sorted(path_prob_dict.keys(),
                             key=lambda x: path_prob_dict.get(x),
                             reverse=True)
        top_1 = sorted_keys[0]
        return top_1, sorted_keys, path_prob_dict


class CTC_DecoderX_withPrior(CTC_DecoderX):
    def __init__(self, alpha):
        super().__init__()
        """
        prior is a vocab_size * vocab_size matrix
        prior_ij means the transaction prob from ith char to jth char
        """
        self.prior = pd.read_csv(os.path.join(DICT, 'wiki2gramprobk100.csv'), index_col=0).values
        # self.prior = pd.read_csv(os.path.join(DICT, 'bigramTrueFalse.csv'), index_col=0).values
        self.prior[-1, :] = 1
        self.prior[:, -1] = 1
        # self.prior[self.prior != 1] = 0.5
        self.alpha = alpha

    @staticmethod
    def find_frontier(top_paths, blank_index):
        frontier = np.array(top_paths[:, -1])
        blank_mask = frontier == blank_index
        i = top_paths.shape[1] - 2
        while blank_mask.any() and (i >= 0):
            frontier[blank_mask] = top_paths[blank_mask, i]
            blank_mask = frontier == blank_index
            i -= 1

        return frontier

    def language_model_bigram(self, path):
        pLM = 0
        for i in range(0, len(path) - 1):
            pLM += np.log(self.prior[path[i], path[i + 1]])
        return np.exp(pLM / len(path))

    def beam_search_step(self, context, in_len):
        """
        search [beam_size] tokens for next step masked by prior
        :param context: vector of size vocab_size
        """
        ''' changing frontier '''
        frontier = self.find_frontier(self.top_paths, self.vocab_size - 1)

        previous_probs = self.top_probs.reshape((-1, 1))
        # adding mask
        mask = np.log(self.prior[frontier, :]) + self.top_priors.reshape((-1, 1))
        context = context.reshape((1, -1))
        # use log add scheme
        current_probs = previous_probs + np.log(context + 1e-8)
        current_probs_with_prior = self.alpha * current_probs + (1 - self.alpha) * mask / in_len

        ''' same as decoderX but sort on current probs with prior'''
        # argsort to find top num
        flatten_sorted_args = np.argsort(current_probs_with_prior.ravel())[:-1 - self.top_num:-1]
        path_ids, token_ids = np.unravel_index(flatten_sorted_args, (self.top_num, self.vocab_size))
        selected_paths = self.top_paths[path_ids]
        selected_tokens = token_ids.reshape((-1, 1))
        # update new top paths and top probs
        self.top_paths = np.concatenate((selected_paths, selected_tokens), axis=1)
        self.top_probs = current_probs.ravel()[flatten_sorted_args[:self.top_num]]
        self.top_priors = mask.ravel()[flatten_sorted_args[:self.top_num]]

    def beam_search(self, AM_outputs, in_len, top_num=None, beam_size=100):
        if top_num:
            self.top_num = top_num
        else:
            self.top_num = beam_size
        self.beam_size = beam_size
        # # initialize top paths filled with blanks
        # self.top_paths = np.ones((self.top_num, 1)) * (self.vocab['_'])
        # self.top_probs = np.zeros(self.top_num)
        # handle first frame
        first_context = AM_outputs[0, 0, :]
        first_context = np.log(first_context + 1e-8)
        first_top_orders = np.argsort(first_context)[:-1 - self.top_num:-1]
        self.top_paths = first_top_orders.reshape((-1, 1))
        self.top_probs = first_context[(first_top_orders)]
        self.top_priors = np.zeros_like(self.top_probs)
        # loop through afterwards frames
        for context_i in range(1, int(in_len)):
            context = AM_outputs[0, context_i, :]
            self.beam_search_step(context, context_i + 1)

        # self.top_priors /= in_len
        merged_paths = self.merge_path(self.top_paths)
        converted_paths = self.convert_to_tuple(merged_paths)
        top1, top1_prob, path_prob_dict = self.map_reduce_prob(converted_paths, self.top_probs)

        return top1, top1_prob, path_prob_dict

    def decode(self, AM_outputs, in_len, top_num=None, beam_size=100):
        top1, top1_prob, path_prob_dict = self.beam_search(AM_outputs, in_len, top_num, beam_size)
        sorted_keys = [top1]
        if len(list(path_prob_dict.values())) != 1:
            p_model_max = np.array(list(path_prob_dict.values())).max()
            p_model_min = np.array(list(path_prob_dict.values())).min()

            p_lm_dict = dict(zip(list(path_prob_dict.keys()),
                                       [self.language_model_bigram(x) for x in path_prob_dict.keys()]))
            p_lm_max = np.array(list(p_lm_dict.values())).max()
            p_lm_min = np.array(list(p_lm_dict.values())).min()

            normalize_p_model = lambda x: (path_prob_dict.get(x) - p_model_min) / (p_model_max - p_model_min)
            normalize_p_lm = lambda x: (p_lm_dict.get(x) - p_lm_max) / (p_lm_max - p_lm_min)

            sorted_keys = sorted(list(path_prob_dict.keys()),
                                 key=lambda x: self.alpha * normalize_p_model(x) + (
                                         1 - self.alpha) * normalize_p_lm(x),
                                 reverse=True)
            top1 = sorted_keys[0]

        return top1, sorted_keys, path_prob_dict


class CTC_DecoderX_to_NoTone(CTC_DecoderX):
    def __init__(self):
        with open(os.path.join(DICT, 'pinyin_dic.txt')) as dic:
            original_vocab = json.load(dic)
        # get the index of first pinyin
        red_dic = {}
        for k, v in original_vocab.items():
            new_k = k[:-1]
            value = red_dic.get(new_k)
            if value is not None:
                red_dic[new_k] = min(value, v)
            else:
                red_dic[new_k] = v
        self.indexes = sorted(list(red_dic.values()))
        self.indexes.append(len(original_vocab.items()))

        with open(os.path.join(DICT, 'pinyin_dic_order.txt')) as dic:
            self.vocab = json.load(dic)
        self.reverse_vocab = dict([(v, k) for k, v in self.vocab.items()])
        # in reverse dict the last token should be empty
        self.reverse_vocab[self.vocab['_']] = ''
        self.vocab_size = len(self.vocab.items())

    def input_transformation_to_NoTone(self, AM_output):
        xxxxx = [AM_output[:, :, self.indexes[i]:self.indexes[i + 1]].sum(axis=-1)[:, :, np.newaxis] for i in
                 range(len(self.indexes) - 1)]
        new_AM_output = np.concatenate(xxxxx, axis=-1)
        return new_AM_output

    def beam_search(self, AM_outputs, in_len, top_num=None, beam_size=100):
        AM_outputs = self.input_transformation_to_NoTone(AM_outputs)
        return super().beam_search(AM_outputs, in_len, top_num, beam_size)


class CTC_DecoderX_LateLM(CTC_DecoderX):
    def __init__(self, alpha):
        super().__init__()
        self.prior = pd.read_csv(os.path.join(DICT, 'wiki2gramprobk100.csv'), index_col=0).values
        self.alpha = alpha

    def language_model_bigram(self, path):
        pLM = 0
        for i in range(0, len(path) - 1):
            pLM += np.log(self.prior[path[i], path[i + 1]])
        return np.exp(pLM / len(path))

    def decode(self, AM_outputs, in_len, top_num=None, beam_size=100):
        top1, top1_prob, path_prob_dict = self.beam_search(AM_outputs, in_len, top_num, beam_size)
        sorted_keys = [top1]
        if len(list(path_prob_dict.values())) != 1:
            p_model_max = np.array(list(path_prob_dict.values())).max()
            p_model_min = np.array(list(path_prob_dict.values())).min()

            p_lm_dict = dict(zip(list(path_prob_dict.keys()),
                                       [self.language_model_bigram(x) for x in path_prob_dict.keys()]))
            p_lm_max = np.array(list(p_lm_dict.values())).max()
            p_lm_min = np.array(list(p_lm_dict.values())).min()

            normalize_p_model = lambda x: (path_prob_dict.get(x) - p_model_min) / (p_model_max - p_model_min)
            normalize_p_lm = lambda x: (p_lm_dict.get(x) - p_lm_max) / (p_lm_max - p_lm_min)

            sorted_keys = sorted(list(path_prob_dict.keys()),
                                 key=lambda x: self.alpha * normalize_p_model(x) + (
                                         1 - self.alpha) * normalize_p_lm(x),
                                 reverse=True)
            top1 = sorted_keys[0]

        return top1, sorted_keys, path_prob_dict


if __name__ == '__main__':
    # decoder = CTC_DecoderX()
    # decoder = CTC_DecoderX_withPrior(0.5)
    decoder = CTC_DecoderX_LateLM(5)
    AM_outputs = np.load('test_sample.npy')
    top_1, sorted_keys, path_prob_dict = decoder.decode(AM_outputs, 116, beam_size=100)
    for path in sorted_keys:
        print(' '.join(decoder.translate(path)[0]))
        # print('P_Model = {}.'.format(path_prob_dict[path]))
        print('P_Model = {}; P_LM = {}.'.format(path_prob_dict[path], decoder.language_model_bigram(path)))
    # frontier = decoder.find_frontier(np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 4, 4], [0, 4, 4, 4], [4, 4, 4, 4]]), 4)
    # print(frontier)
