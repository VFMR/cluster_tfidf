import numpy as np

def clean_term(term):
    replacement_dct = {
        #'\\': '/',
        'ã¤': 'ä',
        'ã¼': 'ü',
        'ã¶': 'ö',
        #'Ã\\x9c': 'Ü',´
        'ã\x9f': 'ß',
        'ã\\x9f': 'ß',
        'Ã\x9c': 'Ü',
        'Ã\\x9c': 'Ü',
        'Ã¼': 'Ü',
        'Ã¶': 'Ö' ,
        'Ã\x96': 'Ö',
        'Ã\\x96': 'Ö',
        'Ã¤': 'Ä',
        r'Ã\x84': 'Ä',
        'Ã\x84': 'Ä',
        'Ã\\x84': 'Ä'
        }

    for key, value in replacement_dct.items():
        term = term.replace(key, value)
    
    return term


def myCosine(a, b):
    return a@b / (np.linalg.norm(a)*np.linalg.norm(b))


def cosine_sim_words(x, y, embeddings):
    a = embeddings[x]
    b = embeddings[y]
    return myCosine(a, b)


def count_file_rows(path, encoding='latin1'):
    with open(path, 'r', encoding=encoding) as f:
        n_docs = sum([1 for _ in f.readlines()])-1
    return n_docs


def get_n_docs_from_training_path(path, encoding='latin1'):
    return count_file_rows(path, encoding)