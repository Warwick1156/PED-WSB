import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


def tokenize(text) -> list:
    text = str(text)
    stopwords_list = stopwords.words('english')
    ret = [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stopwords_list]

    return list(set(ret))


def remove_stopwords(list_: list) -> list:
    return [[word for word in tokens if word not in stopwords.words('english')] for tokens in list_]


def remove_numbers(list_: list) -> list:
    return [[re.sub(r'[0-9]', '', word) for word in tokens] for tokens in list_]


def lemmatize(list_: list, pos: str) -> list:
    wnl = WordNetLemmatizer()
    return [[wnl.lemmatize(word, pos=pos) for word in tokens] for tokens in list_]


if __name__ == '__main__':
    print(remove_numbers([['524dds2']]))
    wnl = WordNetLemmatizer()

    print(wnl.lemmatize('investing', pos='v'))