import torch
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import pandas as pd
from functools import partial
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import swifter
from utilities import parallelize_task_by_process
from typing import Union
from sentence_transformers import SentenceTransformer


sts_model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')


# tokenizer = BertTokenizer.from_pretrained('monsoon-nlp/hindi-bert')
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# model = TFBertModel.from_pretrained('monsoon-nlp/hindi-bert')
# model = TFBertModel.from_pretrained('bert-base-multilingual-uncased')


def word_embedding_tf_2d(tokenizer: BertTokenizer, model: TFBertModel, text: str) -> np.ndarray:
    """
    get the vector representation for the sentence
    :param tokenizer:
    :param model:
    :param text:
    :return:
    """
    # tokenize and get the input ids for the tokens
    input_ids = tf.constant(tokenizer.encode(text))[None, :]
    # get the embedding vectors
    outputs = model(input_ids)

    hidden_states = outputs[0]

    # calculate the mean on axis 0 to reduce the vector to 1D
    output_vector = np.mean(hidden_states[0], axis=0)
    # print(output_vector.shape)

    return output_vector


def find_max_len(tokenizer: BertTokenizer, data_df: pd.DataFrame):
    data_df['premise'] = data_df['premise'].swifter.apply(tokenizer.tokenize).swifter.apply(len)

    max_len = data_df['premise'].max()
    print(max_len)
    mean_len = data_df['premise'].mean()
    median_len = data_df['premise'].median()

    print(mean_len)
    print(median_len)


def get_word_embedding(text):
    output_vector = word_embedding_tf_2d(tokenizer, model, text)

    return output_vector


def read_and_vectorize(train_file_path: str, test_file_path: str, bert_model: Union[TFBertModel, None],
                       train_samples: int, test_samples: int, tokenizer: Union[BertTokenizer, None] = None,):
    df_train = pd.read_csv(train_file_path)

    df_test = pd.read_csv(test_file_path)

    label_enc = LabelEncoder()

    # func = partial(word_embedding_tf_2d, tokenizer, bert_model)
    # func = partial(word_embedding_tf_3d, tokenizer, bert_model)
    if train_samples:
        df_train = df_train.sample(n=train_samples)

    # df_train['premise'] = df_train['premise'].swifter.apply(func)
    #
    # df_train['hypothesis'] = df_train['hypothesis'].swifter.apply(func)

    df_train['premise'] = sts_model.encode(df_train['premise'].values).tolist()

    df_train['hypothesis'] = sts_model.encode(df_train['hypothesis'].values).tolist()

    df_train['label'] = label_enc.fit_transform(df_train['label'])

    df_train.info()

    if test_samples:
        df_test = df_test.sample(test_samples)

    # df_test['premise'] = df_test['premise'].swifter.apply(func)
    # df_test['hypothesis'] = df_test['hypothesis'].swifter.apply(func)

    df_test['premise'] = sts_model.encode(df_test['premise'].values).tolist()
    df_test['hypothesis'] = sts_model.encode(df_test['hypothesis'].values).tolist()
    df_test.info()

    return df_train, df_test, label_enc


def get_train_test_data(train_samples, test_samples):
    # df_train, df_test, label_enc = read_and_vectorize('dataset/train.csv', 'dataset/test.csv', tokenizer=tokenizer, bert_model=model,
    #                                                   train_samples=train_samples, test_samples=test_samples)

    df_train, df_test, label_enc = read_and_vectorize('dataset/train.csv', 'dataset/test.csv', tokenizer=None, bert_model=None,
                                                      train_samples=train_samples, test_samples=test_samples)

    return df_train, df_test, label_enc


if __name__ == '__main__':
    sample_text = 'बहुत समय से मिले नहीं'

    # states = word_embedding_tf_2d(tokenizer, model, sample_text)
    # print(states)
    encodings = sts_model.encode(sentences=['बहुत समय से मिले नहीं', 'वह तुरंत ऑगस्टा के बाहर चले गए।'])
    print(encodings.shape)
    # read_and_vectorize('dataset/train.csv', 'dataset/test.csv', tokenizer, model, train_samples=None, test_samples=50)