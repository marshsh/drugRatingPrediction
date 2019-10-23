import os
import pickle




def dumpPickle(fileName, dic):
	pickle_out = open(fileName,"w")
	pickle.dump(dic, pickle_out)
	pickle_out.close()

def loadPickle(fileName):
	print 'Loading Pickle :  ' + fileName
	pickle_in = open(fileName,"r")
	dic = pickle.load(pickle_in)
	pickle_in.close()
	print 'Loading completed ... \n'
	return dic
