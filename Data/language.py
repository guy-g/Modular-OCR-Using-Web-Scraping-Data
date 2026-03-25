import json
import os
import langdetect
from GeneralUtils.utils import *


class Language:
    def __init__(self, language_name, languages_folder, load_file=False, charset=None, direction='ltr', charset_to_charset_normalization=None, language_detection=None, language_detection_threshold=0.5):
        self.language_name = language_name
        self.languages_folder = languages_folder
        self.language_file = os.path.join(languages_folder, language_name + '.json')
        if not load_file:
            self.charset = charset
            self.direction = direction
            self.language_detection = language_detection
            self.language_detection_threshold = language_detection_threshold
            if charset_to_charset_normalization is None:
                self.charset_to_charset_normalization = {i: i for i in charset}
                self.accepting_charset = charset
            else:
                self.charset_to_charset_normalization = charset_to_charset_normalization
                self.accepting_charset = ''.join(list(set([j for i in charset_to_charset_normalization.values() for j in i] + [j for j in charset_to_charset_normalization.keys()] + [j for j in charset])))
                for k in charset:
                    if k not in self.charset_to_charset_normalization.keys():
                        self.charset_to_charset_normalization[k] = [k]
            self.__save()
        else:
            self.__load()
    
    def __save(self):
        data = {
            'language_name': self.language_name,
            'languages_folder': self.languages_folder,
            'charset': self.charset,
            'direction': self.direction,
            'accepting_charset': self.accepting_charset,
            'charset_to_charset_normalization': self.charset_to_charset_normalization,
            'language_detection': self.language_detection,
            'language_detection_threshold': self.language_detection_threshold
        }
        save_to_json(data, self.language_file)
    
    def __load(self):
        data = json.load(open(self.language_file, mode='r', encoding='utf-8'))
        self.language_name = data['language_name']
        self.languages_folder = data['languages_folder']
        self.charset = data['charset']
        self.direction = data['direction']
        self.accepting_charset = data['accepting_charset']
        self.charset_to_charset_normalization = data['charset_to_charset_normalization']
        self.language_detection = data['language_detection']
        self.language_detection_threshold = data['language_detection_threshold']

    def normalize_word(self, word):
        normalized_word = ''
        for ch in word:
            for k, v in self.charset_to_charset_normalization.items():
                if k == ch or ch in v:
                    normalized_word += k
                    break
            else:
                normalized_word += ch
                # raise Exception('Word cant be normalized')
        return normalized_word

    def is_accept(self, word):
        for ch in word:
            if ch not in self.accepting_charset:
                return False
        if self.language_detection is not None:
            lang = langdetect.detect_langs(word)
            lang = [i for i in lang if i.lang == self.language_detection]
            if len(lang) > 0 and lang[0].prob >= self.language_detection_threshold:
                return True
            return False
        return True

