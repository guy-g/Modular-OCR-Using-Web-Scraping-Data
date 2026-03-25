import os
import json
import numpy as np
from fontTools.ttLib import TTFont
from Data.language import Language
from GeneralUtils.utils import *


class FontsView:
    def __init__(self,
                 fonts_view_name,
                 fonts_views_folder,
                 fonts_folder=None,
                 demonstration_font_name=None,
                 load_view_file=False,
                 language_name=None,
                 language_folder=None,
                 font_names='all',
                 font_names_to_probabilities=None
                 ):
        self.fonts_view_name = fonts_view_name
        self.fonts_views_folder = fonts_views_folder
        self.fonts_view_file = os.path.join(fonts_views_folder, fonts_view_name + '.json')
        if not load_view_file:
            self.fonts_folder = fonts_folder
            self.demonstration_font_name = demonstration_font_name
            self.language_name = language_name
            self.language_folder = language_folder
            if language_name is not None and language_folder is not None:
                self.language = Language(language_name, language_folder, True)
            else:
                self.language = None
            self.font_names = font_names
            self.font_names = [font_name for font_name in os.listdir(fonts_folder) if (os.path.isfile(os.path.join(fonts_folder, font_name)) and ((font_name[:-4] in font_names) or font_names == 'all'))]
            if font_names_to_probabilities is not None:
                self.font_names_to_probabilities = font_names_to_probabilities
            else:
                self.font_names_to_probabilities = {
                    font_name: 1.0 / float(len(self.font_names)) for font_name in self.font_names
                }
            font_names_to_delete = []
            if self.language is not None:
                for font_name in self.font_names:
                    if not self.font_contains_charset(os.path.join(fonts_folder, font_name), self.language.accepting_charset):
                        font_names_to_delete.append(font_name)
            for font_name in font_names_to_delete:
                for other_font_name in self.font_names:
                    if other_font_name != font_name:
                        self.font_names_to_probabilities[other_font_name] += (self.font_names_to_probabilities[font_name] / float(len(self.font_names_to_probabilities.keys()) - 1))
                del self.font_names_to_probabilities[font_name]
                self.font_names.remove(font_name)
            self.font_probs = [self.font_names_to_probabilities[font_name] for font_name in self.font_names]
            self.font_paths = [os.path.join(self.fonts_folder, fn) for fn in self.font_names]
            if self.demonstration_font_name is None:
                self.demonstration_font_name = self.font_names[0][:-4]
            self.__save()
        else:
            self.__load()
        print('Num fonts: {}'.format(len(self.font_paths)))

    @staticmethod
    def font_contains_charset(font_file, charset):
        font = TTFont(font_file)
        charset_ord = [ord(ch) for ch in charset]
        for char_ord in charset_ord:
            for table in font['cmap'].tables:
                if char_ord not in table.cmap.keys():
                    return False
        return True

    def get_font(self, full_path=False, without_extansion=False):
        random_font = np.random.choice(self.font_names, p=self.font_probs)
        if without_extansion:
            random_font = random_font[:-4]
        if full_path:
            random_font = os.path.join(self.fonts_folder, random_font)
        return random_font

    def get_demonstration_font(self, full_path=False):
        if self.demonstration_font_name is None:
            return self.get_font(full_path=full_path)
        demonstration_font = self.demonstration_font_name
        if full_path:
            demonstration_font = os.path.join(self.fonts_folder, demonstration_font + '.ttf')
        return demonstration_font

    def __save(self):
        data = {
            'fonts_view_name': self.fonts_view_name,
            'fonts_views_folder': self.fonts_views_folder,
            'fonts_folder': self.fonts_folder,
            'fonts_view_file': self.fonts_view_file,
            'demonstration_font_name': self.demonstration_font_name,
            'language_name': self.language_name,
            'language_folder': self.language_folder,
            'font_names': self.font_names,
            'font_names_to_probabilities': self.font_names_to_probabilities,
            'font_probs': self.font_probs,
            'font_paths': self.font_paths
        }
        save_to_json(data, self.fonts_view_file)

    def __load(self):
        data = json.load(open(self.fonts_view_file, mode='r', encoding='utf-8'))
        self.fonts_view_name = data['fonts_view_name']
        self.fonts_views_folder = data['fonts_views_folder']
        self.fonts_folder = data['fonts_folder']
        self.demonstration_font_name = data['demonstration_font_name']
        self.language_name = data['language_name']
        self.language_folder = data['language_folder']
        self.font_names = data['font_names']
        self.font_names_to_probabilities = data['font_names_to_probabilities']
        self.font_probs = data['font_probs']
        self.font_paths = data['font_paths']
        if self.language_name is not None and self.language_folder is not None:
            self.language = Language(self.language_name, self.language_folder, True)
        else:
            self.language = None

    def __len__(self):
        return len(self.font_names)

