import os
import argparse

import keras
from keras import backend
from trainReviews import getDocsEmbedded


def soft_acc(y_true, y_pred):
	"""
	For use in Keras model.
	"""
	return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))


def halveSoft_acc(y_true, y_pred):
	"""
	For use in Keras model.
	"""
	aa = backend.constant(0.5)
	return backend.mean(backend.equal(  backend.round(  y_true * aa  ), backend.round(  y_pred * aa  ) ))


def evaluate(embeddingType, metric):

	# Get TEST documents
	docGeneraTEST, shapeInp = getDocsEmbedded(embeddingType, Test=True)


	# Choose checkpoint Model
	ldir = os.listdir("./modelos")
	models = []
	for fileN in ldir:
		if metric in fileN:
			z = fileN.split("_")
			index = z.index(metric)
			metric_val = float(z[index+1].strip("-")[0])

			models.append( (fileN,metric_val) )

	if not models:
		raise ValueError("No se han entrenado los modelos")

	if metric == "acc":
		mMname = max(models, key=lambda x: x[1])[0]
	else :
		mMname = min(models, key=lambda x: x[1])[0]		

	mMname = "./modelos/" + mMname
	print mMname
	# theModel = keras.models.load_model(mMname,  custom_objects={ 'soft_acc': soft_acc })
	theModel = keras.models.load_model(mMname,  custom_objects={ 'soft_acc': halveSoft_acc })

	loss = theModel.evaluate_generator(docGeneraTEST, steps=shapeInp)
	loss_names =  theModel.metrics_names

	print loss
	print loss_names


	fmt = '{:>16} | {:>16} | {:>16}'.format

	print "*"
	print "*"
	print "*"
	print "*"
	print "*"
	print "*"
	print "*"
	print "*"
	print " "
	print fmt("Loss", "Test MAE", "Test Accuracy (1 point tolerance)")
	print fmt("","","")
	print fmt(*loss)
	print fmt("","","")
	print "*"
	print "*"
	print "*"
	print "*"



def main():

	parser = argparse.ArgumentParser("Trains NN model on drugCom data reoresented as SMH or BOW vectors")
	parser.add_argument("embeddingType", choices=["SMH", "BOW"],
						help = "Embedding Type")
	parser.add_argument("metric", choices=["mae", "acc"],
						help = "Embedding Type")
	parser.add_argument("-cB", "--callbacksBool", action='store_true',
						help = "Use callbacks")

	args = parser.parse_args()

	evaluate(args.embeddingType, args.metric)


if __name__ == "__main__":

	main()



