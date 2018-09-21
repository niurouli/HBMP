from __future__ import print_function
import tempfile
import numpy as np
import keras
from keras.callbacks import *
from keras.models import Model
from keras.regularizers import l2
from keras.activations import softmax
from keras.layers.advanced_activations import LeakyReLU
from final_model.layers import *
from final_model.AttentionLayer import *
from final_model.utils import *
from model.selfLayer import *
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
params = {'l2':4e-5,
          'epochs':50,
        'max_length':42,
        'lr':  0.0005,
         'batch_size':300,
         'lstm_dim':300,
          'emd_dim':300,
         'dropout_rate': 0.4,
          'dense_dim':600,
         'patience':4,
         'activation':'relu',
          'max_word_len':12,
          'layers':2,
          'translate_dim':300,
          'embedding_filepath':'D://代码//DIIN-in-Keras-master//data/glove.840B.300d.txt'
          }


def aggregate(input_1, input_2, num_dense=params['dense_dim']):

    sub_rs = subtract([input_1, input_2])
    mul_rs = multiply([input_1, input_2])
    import tensorflow as tf
    abs_rs = Lambda(lambda x: tf.abs(x))(sub_rs)
    x = concatenate([input_1, input_2,abs_rs,mul_rs ])

    x = Dropout(params['dropout_rate'])(x)
    x = Dense(num_dense)(x)
    x = LeakyReLU(alpha=0.01)(x)


    x = Dropout(params['dropout_rate'])(x)
    x = Dense(num_dense)(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Dropout(params['dropout_rate'])(x)
    x = Dense(num_dense)(x)
    x = LeakyReLU(alpha=0.01)(x)
    return x

def build_model(word_dict,char_dict, num_class=3, MAX_LEN=params['max_length']):
    premise_word_input_layer = keras.layers.Input(
        shape=(MAX_LEN,),
        name='Input_Word1',
    )

    hypothesis_word_input_layer = keras.layers.Input(
        shape=(MAX_LEN,),
        name='Input_Word2',
    )
    word_embd_weights = get_embdding_from_file(word_dict,params['embedding_filepath'] )
    word_embd_layer = keras.layers.Embedding(
        input_dim=len(word_dict),
        output_dim=params['emd_dim'],
        mask_zero=False,
        weights=word_embd_weights,
        trainable=True,
        name='Embedding_Word1',
    )

    hypothesis_word_embedding = word_embd_layer(hypothesis_word_input_layer)
    premise_word_embedding = word_embd_layer(premise_word_input_layer)

    hypothesis_word_embedding = BatchNormalization()(hypothesis_word_embedding)
    premise_word_embedding = BatchNormalization()(premise_word_embedding)

    prems = []
    hypos = []
    scale = Scale()
    Bilstm_forward = LSTM(params['lstm_dim'], return_sequences=True,return_state=True,recurrent_dropout=params['dropout_rate'],kernel_regularizer=l2(params['l2']))
    Bilstm_backward = LSTM(params['lstm_dim'], return_sequences=True,return_state=True,recurrent_dropout=params['dropout_rate'],kernel_regularizer=l2(params['l2']),go_backwards=True)

    prem_lstm_out_forward = Bilstm_forward(premise_word_embedding)
    prem_lstm_out_backward = Bilstm_backward(premise_word_embedding)

    prem_lstm_out = concatenate([prem_lstm_out_forward[0],prem_lstm_out_backward[0]])

    prems.append(GlobalMaxPool1D()(prem_lstm_out))


    hypo_lstm_out_forward = Bilstm_forward(hypothesis_word_embedding)
    hypo_lstm_out_backward = Bilstm_backward(hypothesis_word_embedding)

    hypo_lstm_out = concatenate([hypo_lstm_out_forward[0], hypo_lstm_out_backward[0]])
    hypos.append(GlobalMaxPool1D()(hypo_lstm_out))
    # Encoding
    for i in range(params['layers']):
        Bilstm_forward = LSTM(params['lstm_dim'], return_sequences=True, return_state=True, recurrent_dropout=params['dropout_rate'],
                 kernel_regularizer=l2(params['l2']))
        Bilstm_backward =LSTM(params['lstm_dim'], return_sequences=True, return_state=True, recurrent_dropout=params['dropout_rate'],
                 kernel_regularizer=l2(params['l2']),go_backwards=True)

        prem_lstm_out_forward = Bilstm_forward(premise_word_embedding,initial_state=prem_lstm_out_forward[1:])
        prem_lstm_out_backward = Bilstm_backward(premise_word_embedding,initial_state=prem_lstm_out_backward[1:])

        prem_lstm_out = concatenate([prem_lstm_out_forward[0], prem_lstm_out_backward[0]])
        prems.append(GlobalMaxPool1D()(prem_lstm_out))

        hypo_lstm_out_forward = Bilstm_forward(hypothesis_word_embedding,initial_state=hypo_lstm_out_forward[1:])
        hypo_lstm_out_backward = Bilstm_backward(hypothesis_word_embedding,initial_state=hypo_lstm_out_backward[1:])
        hypo_lstm_out = concatenate([hypo_lstm_out_forward[0], hypo_lstm_out_backward[0]])

        hypos.append(GlobalMaxPool1D()(hypo_lstm_out))


    # Aggregate
    prem = concatenate(prems,axis=1)
    hypo = concatenate(hypos,axis=1)

    x = aggregate(prem, hypo)
    x = Dense(num_class, activation='softmax')(x)

    return Model(inputs=[premise_word_input_layer, hypothesis_word_input_layer], outputs=x)

training = get_data('../data/snli_1.0_train.jsonl')
validation = get_data('../data/snli_1.0_dev.jsonl')
test = get_data('../data/snli_1.0_test.jsonl')
sentences = training[0] + training[1] + validation[0] + validation[1] + test[0] + test[1]

from keras_wc_embd import get_dicts_generator

dict_generator = get_dicts_generator(
    word_min_freq=1,
    char_min_freq=1,
    word_ignore_case=False,
    char_ignore_case=False,
)
for sentence in sentences:
    dict_generator(sentence)

word_dict, char_dict, _ = dict_generator(return_dict=True)

def get_input(sentences, word_unknown=1, char_unknown=1, word_ignore_case=False, char_ignore_case=False):
    sentence_num = len(sentences)
    max_sentence_len = params['max_length']
    max_word_len = params['max_word_len']

    word_embd_input = [[0] * max_sentence_len for _ in range(sentence_num)]
    char_embd_input = [[[0] * max_word_len for _ in range(max_sentence_len)] for _ in range(sentence_num)]

    for sentence_index, sentence in enumerate(sentences):
        for word_index, word in enumerate(sentence):
            if word_index >= max_sentence_len:
                break
            if word_ignore_case:
                word_key = word.lower()
            else:
                word_key = word
            word_embd_input[sentence_index][word_index] = word_dict.get(word_key, word_unknown)
    return np.asarray(word_embd_input), np.asarray(char_embd_input)

train_left_embd_input = get_input(sentences=training[0])
train_right_embd_input = get_input(sentences=training[1])
val_left_embd_input = get_input(sentences=validation[0])
val_right_embd_input = get_input(sentences=validation[1])
test_left_embd_input = get_input(sentences=test[0])
test_right_embd_input = get_input(sentences=test[1])


model = build_model(word_dict=word_dict,char_dict=char_dict)

optimzer =keras.optimizers.adam(lr=params['lr'],beta_1=0.9, beta_2=0.99)
model.compile(optimizer=optimzer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='../png/HBMP.png', show_shapes=True,
           show_layer_names=True)
print('Training')
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
def scheduler(epoch):
    if epoch % 3 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

callbacks = [EarlyStopping(patience=params['patience']), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
history = model.fit([train_left_embd_input[0], train_right_embd_input[0],], training[2], batch_size=params['batch_size'], epochs=params['epochs'],
          validation_data=([val_left_embd_input[0], val_right_embd_input[0]], validation[2]), callbacks=callbacks)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test_left_embd_input[0], test_right_embd_input[0]], test[2], batch_size=64)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
# model.save('esim_mltiLayers.h5',overwrite=True)
