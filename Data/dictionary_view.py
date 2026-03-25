import os
import numpy as np
import json
from Data.language import Language
from Data.Utils.string_representation import string_repr
from GeneralUtils.utils import *


class DictionaryView:
    def __init__(self,
                 dictionary_view_name,
                 dictionaries_views_folder,
                 dictionaries_folder=None,
                 load_view_file=False,
                 language_name=None,
                 language_folder=None,
                 dictionary_names='all',
                 dictionary_names_to_probabilities=None,
                 max_word_length=None
                 ):
        self.dictionary_view_name = dictionary_view_name
        self.dictionaries_views_folder = dictionaries_views_folder
        self.dictionary_view_file = os.path.join(dictionaries_views_folder, dictionary_view_name + '.json')
        if not load_view_file:
            self.dictionaries_folder = dictionaries_folder
            self.language_name = language_name
            self.language_folder = language_folder
            self.max_word_length = max_word_length
            if language_name is not None and language_folder is not None:
                self.language = Language(language_name, language_folder, True)
            else:
                self.language = None
            self.dictionary_names = dictionary_names
            self.dictionary_names = [dictionary_name for dictionary_name in os.listdir(dictionaries_folder) if (os.path.isfile(os.path.join(dictionaries_folder, dictionary_name)) and ((dictionary_name[:-4] in dictionary_names) or dictionary_names == 'all'))]
            if dictionary_names_to_probabilities is not None:
                self.dictionary_names_to_probabilities = dict()
                for k, v in dictionary_names_to_probabilities.items():
                    self.dictionary_names_to_probabilities[k + '.txt'] = dictionary_names_to_probabilities[k]
                for k in self.dictionary_names:
                    if k not in self.dictionary_names_to_probabilities.keys():
                        self.dictionary_names_to_probabilities[k] = 0.0
            else:
                self.dictionary_names_to_probabilities = {
                    dictionary_name: 1.0 / float(len(self.dictionary_names)) for dictionary_name in self.dictionary_names
                }
            self.__load_dictionaries_vocab()
            self.dictionary_probs = [self.dictionary_names_to_probabilities[dictionary_name] for dictionary_name in self.dictionary_names]
            self.dictionary_len = sum([len(dictionary) for dictionary in self.dictionary.values()])
            self.__save()
        else:
            self.__load()

    def __load_dictionaries_vocab(self):
        self.dictionary = {}
        dictionaries_to_delete = []
        for dictionary_name in self.dictionary_names:
            self.dictionary[dictionary_name] = []
            if dictionary_name in self.dictionary_names_to_probabilities.keys() and self.dictionary_names_to_probabilities[dictionary_name] > 0.0:
                with open(os.path.join(self.dictionaries_folder, dictionary_name), mode='r', encoding='utf-8') as dict_file:
                    for l in dict_file:
                        word = l
                        if word[-1] == '\n':
                            word = word[:-1]
                        if len(word) > 0 and (self.max_word_length is None or len(word) <= self.max_word_length):
                            if self.language is not None:
                                if not self.language.is_accept(word):
                                    break
                                else:
                                    self.dictionary[dictionary_name].append(word)
                np.random.shuffle(self.dictionary[dictionary_name])
            if len(self.dictionary[dictionary_name]) == 0:
                dictionaries_to_delete.append(dictionary_name)
        for dictionary_name in dictionaries_to_delete:
            for other_dictionary_name in self.dictionary_names:
                if other_dictionary_name != dictionary_name:
                    self.dictionary_names_to_probabilities[other_dictionary_name] += (self.dictionary_names_to_probabilities[dictionary_name] / float(len(self.dictionary_names_to_probabilities.keys()) - 1))
            del self.dictionary[dictionary_name]
            del self.dictionary_names_to_probabilities[dictionary_name]
            self.dictionary_names.remove(dictionary_name)
        self.current_dictionary_name_idx = 0
        self.current_dictionary_word_idx = 0
        self.dictionary_lengths = {dictionary_name: len(self.dictionary[dictionary_name]) for dictionary_name in self.dictionary_names}

    def get_word(self, by_order=False):
        if not by_order:
            random_dictionary = np.random.choice(self.dictionary_names, p=self.dictionary_probs)
            random_idx = np.random.randint(0, self.dictionary_lengths[random_dictionary])
            random_word = self.dictionary[random_dictionary][random_idx]
            return random_word
        else:
            word = self.dictionary[self.dictionary_names[self.current_dictionary_name_idx]][self.current_dictionary_word_idx]
            if self.current_dictionary_word_idx < (len(self.dictionary[self.dictionary_names[self.current_dictionary_name_idx]]) - 1):
                self.current_dictionary_word_idx += 1
            else:
                self.current_dictionary_name_idx = (self.current_dictionary_name_idx + 1) % len(self.dictionary_names)
                self.current_dictionary_word_idx = 0
            return word

    def __save(self):
        data = {
            'dictionary_view_name': self.dictionary_view_name,
            'dictionaries_views_folder': self.dictionaries_views_folder,
            'dictionaries_folder': self.dictionaries_folder,
            'dictionary_view_file': self.dictionary_view_file,
            'language_name': self.language_name,
            'language_folder': self.language_folder,
            'dictionary_names': self.dictionary_names,
            'dictionary_probs': self.dictionary_probs,
            'dictionary_names_to_probabilities': self.dictionary_names_to_probabilities,
            'dictionary_len': self.dictionary_len,
            'max_word_length': self.max_word_length
        }
        save_to_json(data, self.dictionary_view_file)

    def __load(self):
        data = json.load(open(self.dictionary_view_file, mode='r', encoding='utf-8'))
        self.dictionary_view_name = data['dictionary_view_name']
        self.dictionaries_views_folder = data['dictionaries_views_folder']
        self.dictionaries_folder = data['dictionaries_folder']
        self.language_name = data['language_name']
        self.language_folder = data['language_folder']
        self.dictionary_names = data['dictionary_names']
        self.dictionary_probs = data['dictionary_probs']
        self.dictionary_names_to_probabilities = data['dictionary_names_to_probabilities']
        self.dictionary_len = data['dictionary_len']
        self.max_word_length = data['max_word_length']
        if self.language_name is not None and self.language_folder is not None:
            self.language = Language(self.language_name, self.language_folder, True)
        else:
            self.language = None
        self.__load_dictionaries_vocab()

    def __len__(self):
        return self.dictionary_len

    def __str__(self):
        return self.dictionary_view_name

