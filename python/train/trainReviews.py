import pickle

import doc2emb

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
from keras import backend

import argparse



def soft_acc(y_true, y_pred):
	"""
	For use in Keras model.
	"""
	return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))


def getModel(shapeInp):
	print "Getting model with shapeInp: " + str(shapeInp)


	model = Sequential()
	model.add(Dense(128, input_dim=shapeInp, kernel_initializer='normal', activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, kernel_initializer='normal', activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(16, kernel_initializer='normal', activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, kernel_initializer='normal'))

	model.compile(loss='mean_absolute_error',
				  optimizer='sgd',
				  metrics=['mae',soft_acc])

	return model



def getEmbedding(embeddingType, Train=False, Validate=False, Test=False):
	print "Getting embeddings: " + embeddingType

	# HardCoded FileNames
	w2tFileName = "./drugCom_SMH/smh_r2_l68_w0.1_s3_o0.9_m5train_drugReviews40000.IFSwords2topicsOrd"
	vocFN = "./data/train_drugReviews40000.vocab"
	topicsFN = "./drugCom_SMH/smh_r2_l68_w0.1_s3_o0.9_m5train_drugReviews40000.ordered_models"


	if Train or Validate:
		# TRAIN
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

	if embeddingType == "SMH":
		docGenera = doc2emb.SMHcorpus2emb(corpusFN, w2tFileName, vocSize, topicsNum, Train=Train, Validate=Validate, labelsFN=labelsFN)
		shapeInp = topicsNum
	elif embeddingType == "BOW":
		docGenera = doc2emb.BOWcorpus2emb(corpusFN, vocSize, Train=Train, Validate=Validate, labelsFN=labelsFN)
		shapeInp = vocSize
	else :
		docGenera = doc2emb.BOW_SMH_corpus2emb(corpusFN, vocSize, Train=Train, Validate=Validate, labelsFN=labelsFN)
		shapeInp = vocSize + topicsNum

	return docGenera, shapeInp








def train(embeddingType,callbacksBool):

	docGeneraTRAIN, shapeInp = getEmbedding(embeddingType, Train=True)
	docGeneraVAL, valSteps = getEmbedding(embeddingType, Validate=True)

	model = getModel(shapeInp)

	if callbacksBool:
		history = model.fit_generator(docGeneraTRAIN, steps_per_epoch=shapeInp, epochs=300, validation_data=docGeneraVAL, validation_steps=valSteps,
			callbacks=[
			keras.callbacks.ModelCheckpoint(
				'modelos/model'
				'-epoch_{epoch:02d}'
				'-regr_mae_{val_loss:.2f}',
				monitor='val_loss',
				verbose=0,
				save_best_only=True,
				save_weights_only=False,
				mode='auto',
				period=1),
			keras.callbacks.EarlyStopping(
				monitor='val_loss', 
				min_delta=0, 
				patience=5, 
				verbose=0, 
				mode='auto', 
				baseline=None, 
				restore_best_weights=False),
			keras.callbacks.TensorBoard(
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
	else :
		history = model.fit_generator(docGeneraTRAIN, steps_per_epoch=shapeInp, workers=0, epochs=1,  validation_data=docGeneraVAL, validation_steps=valSteps)



	print "Done fitting model, now saving history"

	histName = "./history/" + embeddingType + "_history.pkl"

	with open(histName, "w") as f:
		pickle.dump(history, f)

	print "History Saved. \n \n \n \n"




def main():

	parser = argparse.ArgumentParser("Trains NN model on drugCom data reoresented as SMH or BOW vectors")
	parser.add_argument("embeddingType", choices=["SMH", "BOW"],
						help = "Embedding Type")
	parser.add_argument("-cB", "--callbacksBool", action='store_true',
						help = "Use callbacks")

	args = parser.parse_args()

	train(args.embeddingType,args.callbacksBool)


if __name__ == "__main__":

	main()

