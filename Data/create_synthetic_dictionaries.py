import os
import json
import numpy as np
from tqdm import tqdm
import string
import datetime
from datetime import timedelta
from Config.setting import *
from faker import Faker
from Data.language import Language
from GeneralUtils.utils import save_to_json


DATE_SEP = ['/', '-', '.']
CLOCK_SEP = [':', '.']


def create_random_dictionary(dictionary_name='random',
                             dictionary_type='random',
                             dictionary_size=500000,
                             charset=string.ascii_letters + string.digits + string.punctuation,
                             charset_to_charset_normalization=None, max_word_length=38):
    lang = Language('create_random_dictionary', LANGUAGES_PATH, charset=charset,
                    charset_to_charset_normalization=charset_to_charset_normalization)
    chars = list(lang.accepting_charset)
    dictionary = []
    pbar = tqdm(total=dictionary_size)
    i = 0
    while i < dictionary_size:
        if dictionary_type == 'random':
            d = _create_random_string(chars, max_word_length)
        elif dictionary_type == 'dates':
            d = _create_random_date()
        elif dictionary_type == 'faker':
            d = create_faker_string()
        if max_word_length >= len(d) > 0 and lang.is_accept(d):
            dictionary.append(d)
            i += 1
            pbar.update(1)
    with open(os.path.join(DICTIONARIES_PATH, '{}.txt'.format(dictionary_name)), mode='w', encoding='utf-8') as fw:
        for w in dictionary:
            fw.write(w + '\n')

def _create_random_date():
    if np.random.rand() < 0.5:
        year = str(np.random.randint(1900, 2030))
        month = str(np.random.randint(1, 13))
        day = str(np.random.randint(1, 32))
        year, month, day = _random_pad_strs(year, month, day)
        date_sep = np.random.choice(DATE_SEP)
        if np.random.rand() < 0.5:
            date_str = date_sep.join([day, month, year])
        else:
            date_str = date_sep.join([month, day, year])
    else:
        hour = str(np.random.randint(0, 24))
        minute = str(np.random.randint(0, 60))
        second = str(np.random.randint(0, 60))
        hour, minute, second = _random_pad_strs(hour, minute, second)
        clock_sep = np.random.choice(CLOCK_SEP)
        if np.random.rand() < 0.5:
            date_str = clock_sep.join([hour, minute])
        else:
            date_str = clock_sep.join([hour, minute, second])
    return date_str


def _random_pad_strs(s1, s2, s3):
    if np.random.rand() < 0.5:
        if len(s1) == 1:
            s1 = '0' + s1
        if len(s2) == 1:
            s2 = '0' + s2
        if len(s3) == 1:
            s3 = '0' + s3
    if len(s1) == 2 and len(s2) == 2 and len(s3) == 1:
        s3 = '0' + s3
    return s1, s2, s3


def _create_random_string(characters, max_word_length):
    s = ''.join(np.random.choice(characters, size=np.random.randint(1, max_word_length)))
    return s


FAKER = Faker()


def file_path():
    return FAKER.file_path(depth=np.random.randint(1, 10))


def paragraph():
    return FAKER.paragraph(np.random.randint(1, 11))


FAKER_FUNC = [FAKER.address, FAKER.building_number,
              FAKER.country_code, FAKER.country, FAKER.street_address, FAKER.street_name,
              FAKER.license_plate, FAKER.bban, FAKER.iban, FAKER.swift8,
              FAKER.color, FAKER.color_name, FAKER.rgb_color,
              FAKER.company, FAKER.company_suffix, FAKER.catch_phrase,
              FAKER.bs, FAKER.credit_card_expire, FAKER.credit_card_full,
              FAKER.credit_card_provider, FAKER.credit_card_security_code,
              FAKER.currency_code, FAKER.pricetag, FAKER.am_pm, FAKER.century,
              FAKER.date, FAKER.day_of_week, FAKER.iso8601, FAKER.month_name,
              FAKER.time, FAKER.timezone, FAKER.year, file_path, FAKER.unix_device,
              FAKER.latitude, FAKER.ascii_company_email, FAKER.ascii_email, FAKER.ascii_free_email,
              FAKER.domain_name, FAKER.ipv4, FAKER.ipv4_private, FAKER.ipv4_public, FAKER.ipv6,
              FAKER.mac_address, FAKER.uri, FAKER.ripe_id, FAKER.url, FAKER.job, FAKER.first_name,
              FAKER.language_name, FAKER.last_name, FAKER.prefix, FAKER.suffix, FAKER.phone_number,
              FAKER.android_platform_token, FAKER.chrome, FAKER.firefox, FAKER.ios_platform_token,
              FAKER.linux_platform_token, FAKER.mac_platform_token, FAKER.text, paragraph
              ]


def create_faker_string(word=True):
    s = str(np.random.choice(FAKER_FUNC)())
    if word:
        s = np.random.choice(s.replace('\n', ' ').replace('\t', ' ').split(' '))
    return s
