#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
import sys
sys.path.append('../../')
from miaoli.utils.load_data import load_data
from miaoli.utils.get_embeddings import generate_batch_train_data_mem_cla, get_test_embedding_mem_cla
from miaoli.DeNVTC.encoder import encoder_net_cnn


dataset = "20ng"
batch_size = 32
epochs = 100


# load data
train_set, test_set, word_embedding_dict, data_characteristics, word_index_dict, embedding_matrix = load_data(dataset, split_count=1000)

# construct model
x = Input(shape=(data_characteristics["words_dim"] * data_characteristics["embedding_dim"], )) # #words * 300
h = encoder_net_cnn(x, data_characteristics)
predictions = Dense(data_characteristics["num_classes"], activation="softmax", kernel_initializer='random_normal', name="classification", kernel_regularizer="l2")(h)
classifier = Model(x, predictions)
classifier.summary()

classifier_weights_dir = "weights/classification_20ng_weights.h5y"
# if os.path.exists(classifier_weights_dir):
#     classifier.load_weights(classifier_weights_dir)

classifier.compile(loss='categorical_crossentropy',
                   optimizer=Adam(lr=0.0001),
                   metrics=['accuracy'])

class EpochCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        classifier.save_weights(classifier_weights_dir)

        # validation
        x_test, y_test = get_test_embedding_mem_cla(test_set, data_characteristics["words_dim"],
                                                    data_characteristics["embedding_dim"], word_embedding_dict,
                                                    data_characteristics["num_classes"])
        score = classifier.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

classifier.fit_generator(
    generate_batch_train_data_mem_cla(train_set, batch_size, data_characteristics["words_dim"],
                                        data_characteristics["embedding_dim"], word_embedding_dict,
                                        data_characteristics["num_classes"]),
    steps_per_epoch= data_characteristics["all_data_count"] // batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[EpochCallback()],
    validation_data=None,
    validation_steps=None,
    class_weight=None,
    max_queue_size=2,
    workers=1,
    use_multiprocessing=True,
    shuffle=True,
    initial_epoch=0)

