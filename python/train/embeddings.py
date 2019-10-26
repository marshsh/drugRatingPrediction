import os
import smh
import numpy as np
import train.tools
import gensim

from collections import Iterable
from math import log1p

from discoverTopics.topics import load_vocabulary, save_topics, save_time, get_models_docfreq, sort_topics, listdb_to_topics

# import arguments as a


def smh_reduced_topicN( filePrefix, topicN=1000, topTopicWords=10, reCalculate=False, logNormal=False):
	print '*** smh_reduced_topicN with ... {} topics ***'.format(topicN)

	smhVectors = smh_get_embeddings( filePrefix, reCalculate=reCalculate, logNormal=logNormal)

	topicOrderVec = smh_TopicsOrder(filePrefix, topTopicWords=topTopicWords, reCalculate=reCalculate)

	newSMHvectors = {}

	filterIndexes = topicOrderVec[:topicN]

	for key, vecVal in smhVectors.items():
		vec = np.asarray(vecVal)
		newSMHvectors[key] = vec[filterIndexes]

	return newSMHvectors




def smh_TopicsOrder(filePrefix, topTopicWords = 10, reCalculate=False):

	extension = getSMHextension()

	# if os.path.exists(filePrefix + '.topicsOrder' + extension) and (not reCalculate) :
	# 	return tools.loadPickle(filePrefix + '.topicsOrder' + extension)

	if not os.path.exists(filePrefix + '.topicsRaw' + extension) and (not reCalculate) :
		smh_get_model(filePrefix)
	
	topicsRawPath = getFileExtension( filePrefix, '.topicsRaw' + extension)
	model = smh.listdb_load(topicsRawPath)

	smh_sort_Model_topicsIds( filePrefix, model, topTopicWords = topTopicWords)

	return tools.loadPickle(filePrefix + '.topicsOrder' + extension)


	


def smh_get_embeddings( filePrefix, reCalculate=False, logNormal=False):
	print '*** smh_get_embeddings ***'

	extension = getSMHextension()

	# the SMH vectors have already been calculated and saved
	if os.path.exists(filePrefix + '.smh_vectors' + extension) and (not reCalculate) :
		return tools.loadPickle(filePrefix + '.smh_vectors' + extension)

	# the vectors have not been calculated, but the topic distribution have been saved

	if os.path.exists(filePrefix + '.topicsRaw' + extension) and (not reCalculate) :
		return smh_embeddings_from_model( filePrefix, logNormal=logNormal )


	# We calculate all from the documents' bags of words
	smh_get_model(filePrefix)
	smhVectors = smh_embeddings_from_model( filePrefix , logNormal=logNormal )

	tools.dumpPickle( filePrefix + '.smh_vectors' + extension, smhVectors )

	return smhVectors


def contextSMH_get_embeddings( filePrefix, windowSize = 5, reCalculate=False, logNormal=False):

	extension = getSMHextension()

	if os.path.exists(filePrefix + '.context' + '.' + str(windowSize) + '.' + extension) and (not reCalculate) :
		contextVec = tools.loadPickle(filePrefix + '.context' + '.' + str(windowSize) + '.' + extension)
		return contextVec




	# Load saved context vectors
	if os.path.exists(filePrefix + '.ctxtBefore' + '.' + str(windowSize) + '.' + extension) and \
	  os.path.exists(filePrefix + '.ctxtBefore' + '.' + str(windowSize) + '.' + extension) and (not reCalculate) :
		print 'Loading contextVecBefore and ... \n'
		contextVecBefore = tools.loadPickle(filePrefix + '.ctxtBefore' + '.' + str(windowSize) + '.' + extension)
		contextVecAfter = tools.loadPickle(filePrefix + '.ctxtAfter' + '.' + str(windowSize) + '.' + extension)
		# print contextVecBefore.keys()
	else:
		# the SMH vectors have already been calculated and saved, but CTXT vectors haven't
		if os.path.exists(filePrefix + '.smh_vectors' + extension) and (not reCalculate) :
			smhVectors = tools.loadPickle(filePrefix + '.smh_vectors' + extension)
		else :
			print 'Loading smhVectors \n'
			smhVectors = smh_get_embeddings( filePrefix, reCalculate=reCalculate, logNormal=logNormal )

		print 'Calculating contextVecBefore \n'
		contextVecBefore, contextVecAfter = contextSMH(filePrefix, smhVectors, windowSize, logNormal=logNormal)

		tools.dumpPickle(filePrefix + '.ctxtBefore' + '.' + str(windowSize) + '.' + extension, contextVecBefore )
		tools.dumpPickle(filePrefix + '.ctxtAfter' + '.' + str(windowSize) + '.' + extension, contextVecAfter )


	# print ' \n Concatenation of embeddings.'
	# for key in contextVecBefore.keys():
	# 	embeddings_dic[key] =  contextVecBefore[key] + contextVecAfter[key]
	# print 'Embeddings concatenated. \n'

	print 'Adding ContextAfter and ContextBefore into new dictionary'

	embeddings_dic = {}


	print 'Length of contextVecBefore.keys() : ', len(contextVecBefore.keys()) 
	sizeVectors = len(contextVecBefore.itervalues().next())

	for key in contextVecBefore.keys():
		embeddings_dic[key] =  [ contextVecBefore[key][x] + contextVecAfter[key][x] for x in range(sizeVectors)]
	print 'Embeddings concatenated. \n'


	tools.dumpPickle(filePrefix + '.context' + '.' + str(windowSize) + '.' + extension, embeddings_dic)


	return embeddings_dic


#################################################################################################
# SMH Vectors


def smh_get_model( filePrefix):
	print '\n*** smh_get_model *** \n \n tuple_size = {}, coocurrence_threshold = {}, overlap = {} \n \n'.format(a.TUPLE_SIZE, a.COOCURRENCE_THRESHOLDS, a.OVERLAP)

	corpusFile = getFileExtension( filePrefix, '.corpus')
	ifsFile = getFileExtension( filePrefix, '.ifs')


	corpus = smh.listdb_load(corpusFile)
	ifs = smh.listdb_load(ifsFile)
	print 'Loaded .ref and .ifs'
	discoverer = smh.SMHDiscoverer( tuple_size=a.TUPLE_SIZE, cluster_table_size = a.CLUSTER_TABLE_SIZE, cooccurrence_threshold=a.COOCURRENCE_THRESHOLDS, overlap=a.OVERLAP, min_cluster_size=a.MIN_CLUSTER_SIZE)
                 

	# threshold 0.02, 0.04, 0.06
	# tuple_size = 2, 3

	# 

	print 'Fitting SMH Discoverer'
	models = discoverer.fit(ifs, expand = corpus)
	extension = getSMHextension()
	models.save(filePrefix + '.topicsRaw' + extension)
	print "SMH Model saved (a ldb with lists of topics' tokens)"



def smh_sort_Model_topicsIds_from_docFreq( filePrefix, model, topTopicWords = 10):
    """
    Sorts topics based on amount of documents that are associted with topic
    
    Returns numpy array
    """
    print "Sorting SMH model topics with first _{}_ most frequent words on topic.".format(topTopicWords)

    topic_scores = np.zeros(model.ldb.size)


    for topicId, topicLDB in enumerate(model.ldb):
    	# for each topic in the model we create a 'topicFreqs' vector with all the word frequencies

    	topicFreqs = [0 for x in range(topicLDB.size)]
    	for itemId, itemInLDB in enumerate(topicLDB):
    		freq = itemInLDB.freq
    		topicFreqs[itemId] = freq

    	topicFreqs.sort(reverse=True)

    	# We get the first '_topTopicWords_' words most comon in topic and average their frequency
        if topTopicWords:
            if len(topicFreqs) >= topTopicWords:
                topic_scores[topicId] = np.mean(topicFreqs[:topTopicWords])
        else:
            topic_scores[i] = np.mean(topicFreqs)


    # We order TopicIds acording to their score
    topic_indices = np.argsort(topic_scores)[::-1]
    # Save TopicIds order
    extension = getSMHextension()
    fileName = filePrefix + '.topicsOrder' + extension
    tools.dumpPickle(fileName,topic_indices)

    print "Finished Sorting and saved topics order."




def smh_sort_Model_topicsIds( filePrefix, model, topTopicWords = 10):
    """
    Sorts topics based on scores
    Scores are calculated taking the average of their top '_topTopicWords_' most frequent words
    
    Returns numpy array
    """
    print "Sorting SMH model topics with first _{}_ most frequent words on topic.".format(topTopicWords)

    topic_scores = np.zeros(model.ldb.size)


    for topicId, topicLDB in enumerate(model.ldb):
    	# for each topic in the model we create a 'topicFreqs' vector with all the word frequencies

    	topicFreqs = [0 for x in range(topicLDB.size)]
    	for itemId, itemInLDB in enumerate(topicLDB):
    		freq = itemInLDB.freq
    		topicFreqs[itemId] = freq

    	topicFreqs.sort(reverse=True)

    	# We get the first '_topTopicWords_' words most comon in topic and average their frequency
        if topTopicWords:
            if len(topicFreqs) >= topTopicWords:
                topic_scores[topicId] = np.mean(topicFreqs[:topTopicWords])
        else:
            topic_scores[i] = np.mean(topicFreqs)


    # We order TopicIds acording to their score
    topic_indices = np.argsort(topic_scores)[::-1]
    # Save TopicIds order
    extension = getSMHextension()
    fileName = filePrefix + '.topicsOrder' + extension
    tools.dumpPickle(fileName,topic_indices)

    print "Finished Sorting and saved topics order."



# All preparations needed to use SMH, are done in the script prepare_db.sh
def smh_embeddings_from_model( filePrefix, logNormal=False ):

	extension = getSMHextension()

	if logNormal:
		return smh_logNormal_embeddings( filePrefix, reCalculate=True )

	topicsRawPath = getFileExtension( filePrefix, '.topicsRaw' + extension)
	model = smh.listdb_load(topicsRawPath)

	vocpath = getFileExtension( filePrefix, '.vocab')
	print "Loading vocabulary from", vocpath
	vocabulary, docfreq = load_vocabulary(vocpath)



	smhVectors = {}

	for topicId in range(model.ldb.size):
		for itemInList in range( model.ldb[topicId].size ):

			token = model.ldb[topicId][itemInList].item
			freq = model.ldb[topicId][itemInList].freq

			word = vocabulary[token]

			if word not in smhVectors:
				smhVectors[ word ] = [ 0 for n in range(model.ldb.size)]			


			smhVectors[word][topicId] = freq

	# Already saving in calling method

	return smhVectors


def smh_logNormal_embeddings( filePrefix ):

	smhVectorsDic = smh_get_embeddings( filePrefix, reCalculate=reCalculate )
	smhLogN = {}

	for word, vector in smhVectorsDic.items():
		smhLogN[word] = logNormalize(vector)

	return smhLogN


def logNormalize(vector):
	"""
	Returns a log-Normalization of given vector.
	"""
	logVector = [ log1p(x) for x in vector ]
	suma = sum(logVector)    
	r = [ float(x)/suma for x in logVector]

	return r

 


################################################################################
# SMH with context vectors


def contextSMH(filePrefix, smhVectors, windowSize, logNormal=False ):

	documentsFile = filePrefix + '.ref'

	contextVecBefore = {}
	contextVecAfter = {}

	sizeVectors = len(smhVectors[smhVectors.keys()[0]])

	with open(documentsFile, 'r') as f:
		for line in f.readlines():
			line = line.split(' ')
			length = len(line)

			for i, word in enumerate(line):

				if smhVectors.get(word) != None:
					if contextVecAfter.get(word) == None :
						contextVecAfter[word] = smhVectors.get(word)
						contextVecBefore[word] = smhVectors.get(word)
				
				for h in range(1,windowSize+1):
					if i+h < length :
						if smhVectors.get(line[ i+h ]) != None:
							if smhVectors.get(word) != None:
								if contextVecAfter.get(word) != None:
									contextVecAfter[word] = [ contextVecAfter[word][x] + smhVectors.get(line[ i+h ])[x] for x in range(sizeVectors)  ]
					if i-h > -1 :
						if smhVectors.get(line[ i-h ]) != None:
							if smhVectors.get(word) != None:
								if contextVecBefore.get(word) != None:
									contextVecBefore[word] = [ contextVecBefore[word][x] + smhVectors[line[ i-h ]][x] for x in range(sizeVectors)  ]
				
	# tools.dumpPickle(contextVecBefore, filePrefix + '.ctxtBefore' + '.' + str(windowSize) )
	# tools.dumpPickle(contextVecAfter, filePrefix + '.ctxtAfter' + '.' + str(windowSize) )

	if logNormal:
		for word, vector in contextVecBefore.items():
			contextVecBefore[word] = logNormalize(vector)
		for i, word in contextVecAfter.items():
			contextVecAfter[word] = logNormalize(vector)



	return contextVecBefore, contextVecAfter						




#####################################################################################
# Usefull functions


def getSMHextension(embType='', tupSize=None, coo=None, overlap=None, minClustS=None, topicN=None):

	tupSize = tupSize or a.TUPLE_SIZE
	coo = coo or a.COOCURRENCE_THRESHOLDS
	overlap = overlap or a.OVERLAP
	minClustS = minClustS or a.MIN_CLUSTER_SIZE
	topicN_val = topicN or a.TOPIC_N

	extension = '[mTupS_{}][coo_{}][ovlp_{}][mClustS_{}]'.format(tupSize,coo,overlap,minClustS)
	
	if topicN or 'smh_reduced' in embType:
		extension += '[topicN_{}]'.format(topicN_val)
	return extension


def getFileExtension( filePrefix, extension):

	filePrefix = filePrefix[0:filePrefix.rfind(os.sep)] + os.sep


	filePath = ''
	for fileN in os.listdir(filePrefix):
		if extension in fileN:
			filePath = filePrefix + fileN
			print "\n {} \n".format(filePath)
			return filePath



