import os
from Data.create_synthetic_dictionaries import *


def create_faker_corpus(
        corpus_name='faker',
        corpus_size=6000,
        charset=string.ascii_letters + string.digits + string.punctuation,
        charset_to_charset_normalization=None
):
    if charset_to_charset_normalization is not None:
        charset_to_charset_normalization[' '] = ' '
    lang = Language('create_random_dictionary', LANGUAGES_PATH, charset=charset + ' ',
                    charset_to_charset_normalization=charset_to_charset_normalization)
    i = 0
    corpus = []
    pbar = tqdm(total=corpus_size)
    while i < corpus_size:
        faker_str = create_faker_string(word=False)
        if lang.is_accept(faker_str):
            corpus.append({"text": faker_str})
            i += 1
            pbar.update(1)
    corpus_dir = os.path.join(CORPUSES_PATH, corpus_name)
    os.makedirs(corpus_dir, exist_ok=True)
    save_to_json(corpus, os.path.join(corpus_dir, '{}.json'.format(corpus_name)))


def create_wiki_corpus(downloaded_wiki_corpus_path, name='wiki_sentences', partitions=27):
    '''
    :param downloaded_wiki_corpus_path: The corpus has to be download from https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences
    :param name: corpus name
    '''
    corpus = []
    with open(downloaded_wiki_corpus_path, mode='r', encoding='utf-8') as f:
        for l in tqdm(f):
            line = l[:-1] if l[-1] == '\n' else l
            corpus.append({"text": line})
    corpus_dir_path = os.path.join(CORPUSES_PATH, name)
    os.makedirs(corpus_dir_path, exist_ok=True)
    num_paragraphs_in_partition = int(np.ceil(len(corpus) / float(partitions)))
    for p in range(partitions):
        pars = corpus[p * num_paragraphs_in_partition: (p + 1) * num_paragraphs_in_partition]
        save_to_json(pars, os.path.join(corpus_dir_path, 'part_{}.json'.format(p + 1)))

