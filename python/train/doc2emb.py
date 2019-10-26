"""
Functions that return the corresponding embedding of a document.
"""
from smh import listdb_load
from python.discoverTopics.topics import load_vocabulary, save_topics, save_time, get_models_docfreq, sort_topics, listdb_to_topics
import codecs
import random





def BOWcorpus2emb(corpusFN, vocSize, Train=False, Validate=False, labelsFN=None):
	"""
	Returns a generator of bow embeddings, taking documents from the pointed .corpus file
	If using labelsFN, check that corpusFN and labelsFN are the same length (and correspond to each other)
	"""

	if Train and Validate:
		Validate = False

	# Setting the Train/Validate division prameters
	if Train or Validate:
		train_size = 0
		with open(corpusFN, "r") as f:
			for line in f:
				train_size += 1

		# Use to divide train / validate
		valNum = 0.2
		valSize = int(train_size*valNum)

		random.seed(12345678)
		valIndexes = set(random.sample(range(train_size),valSize))


	while True:
		if labelsFN:
			f = open(labelsFN, "r")

		corpus = listdb_load(corpusFN)

		for i, doc in enumerate(corpus.ldb):
			# Block to handle Train/Validate division
			if Train:
				if index in valIndexes:
					continue
			elif Validate:
				if index not in valIndexes:
					continue
			# Block ends.

			emb = [0 for i in range(vocSize)]

			for wordBundle in doc:

				if wordBundle.item < vocSize:
					emb[wordBundle.item] = wordBundle.freq

			if labelsFN:
				label = f.readline()
				yield emb, label
			else :
				yield emb

		if labelsFN:
			f.close()



def load_words2topics(w2tFileName):
	"""
	Loads list data base maps words to topics into dictionary.
	Returns array of tuples: [(docID,docFreq)_i]
	"""

	words2topics = {}

	w2t_ldb = listdb_load(w2tFileName)

	for wID, wordTopics in enumerate(w2t_ldb.ldb):
		listTopics = []
		for topic in wordTopics:
			listTopics.append((topic.item, topic.freq))
		words2topics[wID] = listTopics

	return words2topics


def SMHcorpus2emb(corpusFN, w2tFileName, vocSize, topicsNum, Train=False, Validate=False, labelsFN=None):
	"""
	Returns a generator of embeddings of documents, taking documents from the pointed (.corpus) 
	file, and returning for each document, a vetor whose entries represent the amount of influence 
	of a topic_i in the document.
	"""

	smh_genera = _aux_SMH(corpusFN, w2tFileName, vocSize, topicsNum, Train=Train, Validate=Validate, labelsFN=labelsFN)

	for item in smh_genera:
		yield item




def _aux_SMH(corpusFN, w2tFileName, vocSize, topicsNum, Train=False, Validate=False, labelsFN=None, allVectors=False):

	words2topics = load_words2topics(w2tFileName)

	bow_genera = BOWcorpus2emb(corpusFN, vocSize, labelsFN=labelsFN)


	for bundle in bow_genera:
		# Bifurcation with labels / without labels
		if labelsFN:
			bow_doc = bundle[0]
			label = bundle[1]
		else :
			bow_doc = bundle


		topics_emb = [0 for i in range(topicsNum)]

		for word in bow_doc:
			wordTcs = words2topics[word] # gets Topics related to word
			# sums topic frequencies to embedding
			for doc, freq in wordTcs:
				if doc < topicsNum:
					topics_emb[doc] += freq


		# Concatenates BOW vec with Topics vec if it's indicated (BOW before Topics)
		if allVectors:
			topic_emb = bow_doc + topic_emb

		# Yields embedding
		if labelsFN:
			yield topic_emb, label
		else :
			yield topic_emb





def BOW_SMH_corpus2emb(corpusFN, w2tFileName, vocSize, topicsNum, Train=False, Validate=False, labelsFN = None):
	"""
	Concatenation of both SMHcorpus2emb() and BOWcorpus2emb() embeddings (BOW before Topics)
	"""

	vec_genera = _aux_SMH(corpusFN, w2tFileName, vocSize, topicsNum, Train=Train, Validate=Validate, labelsFN=labelsFN, allVectors=True)

	for item in vec_genera:
		yield item


