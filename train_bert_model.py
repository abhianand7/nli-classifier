import keras.backend as K
from keras.models import Model
from keras.layers import Input, Reshape
from keras.layers import Dense, Lambda
from keras.layers import LSTM, Bidirectional
from keras.layers import Dropout, concatenate
from keras.metrics import Precision, Recall, AUC
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import swifter
import pickle
from keras.models import load_model

from typing import Union

from prepare_data import get_train_test_data


def get_siamese_lstm_input(input_len: int, word_embed_size: int) -> Model:
    """
    create a single instance of LSTM model which will be used on the two inputs to create a siamese model
    :param input_len:
    :param word_embed_size:
    :return:
    """
    input_1 = Input(shape=(input_len, word_embed_size))

    conv_layer = Conv1D(filters=32, kernel_size=7, strides=3, padding='same', activation='relu')(input_1)
    max_pool_layer = MaxPooling1D(pool_size=2)(conv_layer)

    lstm_1 = Bidirectional(
        LSTM(
            units=256,
            activation='relu',
            dropout=0.35,
            recurrent_dropout=0.35,
            return_sequences=True
        )
    )(max_pool_layer)

    lstm_2 = Bidirectional(
        LSTM(
            units=256,
            activation='relu',
            dropout=0.35,
            recurrent_dropout=0.35,
            return_sequences=True
        )
    )(lstm_1)

    flatten_out = GlobalAveragePooling1D()(lstm_2)

    model = Model(inputs=[input_1], outputs=[flatten_out])

    return model


def create_lstm_siamese_model(input_len: int, num_classes: int, word_embed_size: int) -> Model:
    """
    create siamese model by combining the single instance of LSTM on the two inputs
    :param input_len:
    :param num_classes:
    :param word_embed_size:
    :return:
    """
    input_1 = Input(shape=(input_len, word_embed_size))

    input_2 = Input(shape=(input_len, word_embed_size))

    model = get_siamese_lstm_input(input_len, word_embed_size)

    input_1_ = model(input_1)
    input_2_ = model(input_2)

    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))

    L1_distance = L1_layer([input_1_, input_2_])

    dense_out = Dense(
        units=num_classes,
        activation='softmax'
    )(L1_distance)

    siamese_net = Model(
        inputs=[input_1, input_2],
        outputs=[dense_out]
    )
    siamese_net.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    siamese_net.summary()
    return siamese_net


def create_lstm_model(input_len: int, num_classes: int, word_embed_size: int) -> Model:
    """
    create a hybrid cnn and lstm network
    :param input_len:
    :param num_classes:
    :param word_embed_size:
    :return:
    """
    input_1 = Input(shape=(input_len, word_embed_size))

    input_2 = Input(shape=(input_len, word_embed_size))

    comb_input = concatenate([input_1, input_2])

    conv_layer = Conv1D(filters=32, kernel_size=7, strides=3, padding='same', activation='relu')(comb_input)
    max_pool_layer = MaxPooling1D(pool_size=2)(conv_layer)

    lstm_1 = Bidirectional(
        LSTM(
            units=512,
            activation='relu',
            dropout=0.35,
            recurrent_dropout=0.35,
            return_sequences=True
        )
    )(max_pool_layer)

    lstm_2 = Bidirectional(
        LSTM(
            units=512,
            activation='relu',
            dropout=0.35,
            recurrent_dropout=0.35,
            return_sequences=True
        )
    )(lstm_1)

    flatten_out = GlobalAveragePooling1D()(lstm_2)

    dense_out = Dense(
        num_classes,
        activation='softmax'
    )(flatten_out)

    model = Model(inputs=[input_1, input_2], outputs=[dense_out])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    model.summary()
    return model


def get_siamese_input(input_length: int) -> Model:
    """
    single instance of siamese dense model
    :param input_length:
    :return:
    """
    input_1 = Input(shape=(input_length, ))

    dense_1 = Dense(
        units=2048,
        activation='relu'
    )(input_1)
    dropout_1 = Dropout(0.35)(dense_1)

    dense_2 = Dense(
        units=4096,
        activation='relu'
    )(dropout_1)

    dropout_2 = Dropout(0.35)(dense_2)

    dense_3 = Dense(
        units=256,
        activation='relu'
    )(dropout_2)

    dropout_3 = Dropout(0.35)(dense_3)

    model = Model(inputs=[input_1], outputs=[dropout_3])

    return model


def create_siamese_model(input_len: int, num_classes: int) -> Model:
    """
    create siamese model with dense layers
    :param input_len:
    :param num_classes:
    :return:
    """
    input_1 = Input(shape=(input_len,))

    input_2 = Input(shape=(input_len,))

    model = get_siamese_input(input_len)

    input_1_ = model(input_1)
    input_2_ = model(input_2)

    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))

    L1_distance = L1_layer([input_1_, input_2_])

    dense_out = Dense(
        units=num_classes,
        activation='softmax'
    )(L1_distance)

    siamese_net = Model(inputs=[input_1, input_2], outputs=[dense_out])

    siamese_net.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    siamese_net.summary()
    return siamese_net


def create_dense_model(input_len: int, num_classes: int) -> Model:
    """
    normal model with dense layers
    :param input_len:
    :param num_classes:
    :return:
    """
    input_1 = Input(shape=(input_len, ))

    input_2 = Input(shape=(input_len, ))

    comb_input = concatenate([input_1, input_2])

    dense_1 = Dense(
        units=2048,
        activation='relu'
    )(comb_input)
    dropout_1 = Dropout(0.35)(dense_1)

    dense_2 = Dense(
        units=4096,
        activation='relu'
    )(dropout_1)

    dropout_2 = Dropout(0.35)(dense_2)

    dense_3 = Dense(
        units=256,
        activation='relu'
    )(dropout_2)

    dropout_3 = Dropout(0.35)(dense_3)

    dense_out = Dense(
        num_classes,
        activation='softmax'
    )(dropout_3)

    model = Model(inputs=[input_1, input_2], outputs=[dense_out])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    model.summary()
    return model


def transform_input_data(data_df: pd.DataFrame, columns: list) -> list:
    """
    :param data_df:
    :param columns:
    :return:
    """
    transformed_data = list()
    for i in columns:
        x = data_df[i].values
        x = np.asarray([np.asarray(i) for i in x])
        transformed_data.append(x)

    return transformed_data


def train(train_data: pd.DataFrame, model: Model, epochs: int, validation_split: float, model_path: str) -> Model:
    """

    :param train_data:
    :param model:
    :param epochs:
    :param validation_split:
    :param model_path:
    :return:
    """
    X = train_data.drop(labels=['label', 'id'], axis=1)

    # make sure input vectors are numpy arrays
    x1, x2 = transform_input_data(X, ['premise', 'hypothesis'])

    Y = train_data['label'].to_numpy()

    # convert categorical variable to vector form
    Y = to_categorical(Y, 3)

    callbacks = list()

    # early stopping if the model stops improving
    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, min_delta=0.001, verbose=1)
    callbacks.append(early_stop)

    model.fit(
        [x1, x2],
        Y,
        batch_size=16,
        epochs=epochs,
        verbose=1,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks
    )
    if model_path:
        model.save(model_path,
                   overwrite=True
                   )

    return model


def test_model(model: Union[Model, str], test_data: pd.DataFrame, label_enc: Union[LabelEncoder, str]) -> pd.DataFrame:
    """
    function for running the trained model and predicting on new inputs
    :param model:
    :param test_data:
    :param label_enc:
    :return:
    """
    X = test_data.drop(labels=['id'], axis=1)

    x1, x2 = transform_input_data(X, columns=['premise', 'hypothesis'])

    if isinstance(model, str):
        model = load_model(model)

    predictions = model.predict([x1, x2])

    predictions = predictions.tolist()
    results = [np.argmax(i) for i in predictions]

    if isinstance(label_enc, str):
        with open(label_enc, 'rb') as fobj:
            label_enc = pickle.load(fobj)

    results = label_enc.inverse_transform(results)

    result_df = pd.DataFrame()
    result_df['id'] = test_data['id']
    result_df['label'] = results

    result_df.info()
    return result_df


def run():
    train_data, test_data, label_enc = get_train_test_data(train_samples=None, test_samples=None)

    # save the label encoder
    with open('label_encoder.pkl', 'wb') as fobj:
        pickle.dump(label_enc, fobj)

    # model = create_dense_model(1024, 3)
    model = create_siamese_model(1024, 3)
    # model = create_lstm_model(1024, 3, 1)
    # model = create_lstm_siamese_model(1024, 3, 1)

    # start training
    model = train(train_data, model, epochs=200, validation_split=0, model_path='model.h5')

    # test model
    result_df = test_model(model, test_data, label_enc)
    result_df.to_csv('submission.csv', index=False)

    # sample_result_df = test_model(model, train_data.sample(100), label_enc)
    # sample_result_df.to_csv('sample_test_result.csv', index=False)


if __name__ == '__main__':
    create_siamese_model(1024, 3)
    # run()
