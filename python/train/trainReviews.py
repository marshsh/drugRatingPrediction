import pickle

import doc2emb as d2e


#####################################################
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras import layers
from keras.regularizers import l1, l2

#####################################################


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
from keras import backend

from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score



def soft_acc(y_true, y_pred):
	"""
	For use in Keras model.
	"""
    return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))


def getModel(shapeInp):


	model = Sequential()
	model.add(Dense(256, input_dim=shapeInp, kernel_initializer='normal', activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, kernel_initializer='normal', activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(16, kernel_initializer='normal', activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, kernel_initializer='normal'))

	model.compile(loss='mean_squared_error',
				  optimizer='sgd',
				  metrics=['mae',soft_acc])

	return model



def getEmbedding(embeddingType, Train=False, Validate=False, Test=False):

	# HardCoded FileNames

		w2tFileName = "./drugCom_SMH/smh_r2_l68_w0.1_s3_o0.9_m5train_drugReviews40000.IFSwords2topicsOrd"
		vocFN = "./data/train_drugReviews40000.vocab"
		topicsFN = "./drugCom_SMH/smh_r2_l68_w0.1_s3_o0.9_m5train_drugReviews40000.ordered_models"


		# TRAIN
	if Train or Validate:
		corpusFN = "./data/train_drugReviews40000.corpus4train"
		labelsFN = "./data/train_drugReviews.labels"
	elif Test :
		# TEST
		corpusFN = "./data/test_drugReviews40000.corpus4train"
		labelsFN = "./data/test_drugReviews.labels"
	else :
		print "No Train, Validate or Test indicated."


		vocSize = 0
		topicsNum = 0
		corpusSize = 0

	# Getting vocSize
	with open(vocFN, "r") as f:
		for a in f.readlines():
			vocSize += 1

	# Getting topicsNum
	with open(topicsFN, "r") as f:
		for a in f.readlines():
			topicsNum += 1

	# Getting corpusSize
	with open(corpusFN, "r") as f:
		for a in f.readlines():
			corpusSize += 1


	# Getting TRAIN Document-Embeddings Generator

	if embeddingType = "SMH":
		docGenera = doc2emb.SMHcorpus2emb(corpusFN, w2tFileName, vocSize, topicsNum, Train=Train, Validate=Validate, labelsFN=labelsFN)
		shapeInp = topicsNum
	elif embeddingType = "BOW":
		docGenera = doc2emb.BOWcorpus2emb(corpusFN, vocSize, Train=Train, Validate=Validate, labelsFN=labelsFN)
		shapeInp = vocSize
	else :
		docGenera = doc2emb.BOW_SMH_corpus2emb(corpusFN, vocSize, Train=Train, Validate=Validate, labelsFN=labelsFN)
		shapeInp = vocSize + topicsNum

	return docGeneraTRAIN, shapeInp








def train(embeddingType):

	docGeneraTRAIN, shapeInp = getEmbedding(embeddingType, Train=True)
	docGeneraVAL, _m = getEmbedding(embeddingType, Validate=True)

	model = getModel(shapeInp)

	history = model.fit_generator(docGeneraTRAIN, steps_per_epoch=shapeInp, epochs=300, validation_data=docGeneraVAL
		callbacks=[
        keras.callbacks.callbacks.ModelCheckpoint(
            'model'
            '-epoch_{epoch:02d}'
            '-regr_mae_{val_loss:.2f}',
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1),
        keras.callbacks.callbacks.EarlyStopping(
        	monitor='val_loss', 
        	min_delta=0, 
        	patience=10, 
        	verbose=0, 
        	mode='auto', 
        	baseline=None, 
        	restore_best_weights=False),
        keras.callbacks.tensorboard_v1.TensorBoard(
        	log_dir='./logs', 
        	histogram_freq=0, 
        	batch_size=32, 
        	write_graph=True, 
        	write_grads=False, 
        	write_images=False, 
        	embeddings_freq=0, 
        	embeddings_layer_names=None, 
        	embeddings_metadata=None, 
        	embeddings_data=None, 
        	update_freq='epoch')
    	]
    )


	# score = model.evaluate(x_test, y_test, batch_size=128)






def show_metrics(title, y_true, y_regr, y_clas):
    y_regr_closest = np.round(y_regr)
    
    fmt = '{:<16} | {:>8} | {:>8} | {:>8}'.format
    nums2str = lambda *nums: (f'{n:.3f}' for n in nums)
    
    print(fmt(title, 'MAE', 'KAPPA', 'ACCURACY'))
    print(fmt(' regression', *nums2str( 
            mean_absolute_error(y_true, y_regr),
            cohen_kappa_score(y_true, y_regr_closest),
            accuracy_score(y_true, y_regr_closest)
    )))
    
    print(fmt(' classification', *nums2str( 
            mean_absolute_error(y_true, y_clas),
            cohen_kappa_score(y_true, y_clas),
            accuracy_score(y_true, y_clas)
    )))
    print()



def main():








if __name__ == "__main__":

	main()

