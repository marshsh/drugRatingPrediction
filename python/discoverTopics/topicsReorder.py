"""
Tool to reorder SMH topics according to amount of documents associated with the topic.
"""

from smh import listdb_load
from topics import load_vocabulary, save_topics, save_time, get_models_docfreq, sort_topics, listdb_to_topics
import codecs

import argparse


def reorderTopics(modelName):
	
	newTopicsName = modelName.rstrip("models") + "ordered_models"
	newExplicitTopicsName = modelName.rstrip("models") + "ordered_models_words"
	vocpath = "./data/train_drugReviews40000.vocab"
	ifspath = "./data/train_drugReviews40000.ifs"

	models = listdb_load(modelName)
	vocabulary, docfreq  = load_vocabulary(vocpath)

	# Generate dictionary of lists of documents associated to wordTokens
	ifs_dic = load_ifs_dic(ifspath)


	# Generate list of documents associated to topic
	associaNum = {}
	lenModels = 0
	for topicID, topic in enumerate(models.ldb):
		topicDocs = {}
		lenModels += 1

		for wordToken in topic:
			word_Docs = ifs_dic[wordToken.item]
			for doc in word_Docs:
				if doc not in topicDocs:
					topicDocs[doc] = True

		associaNum[topicID] = len(topicDocs)


	# Sorting Topics
	sortedIDs = sorted(associaNum.items(), key=lambda x: x[1], reverse=True)
	sortedDic = {} # sends topicID (line Num in listdb) to place in sorted list
	for i, item in enumerate(sortedIDs):
		sortedDic[item[0]] = i


	# Creating string array version of listdb with ordered topics, and an array with the words
	new_ldb = [ "" for i in range(lenModels) ]
	explicit_new = [ "" for i in range(lenModels) ]

	for topicID, topic in enumerate(models.ldb):		
		topicPlace = sortedDic[topicID]
		topicStr = str(topic.size)
		explStr = str(associaNum[topicID]) + " ... " + str(topic.size)


		for itemBund in topic:
			topicStr += " " + str(itemBund.item) + ":" + str(itemBund.freq)
			explStr += " " + vocabulary[itemBund.item] + ":" + str(itemBund.freq)

		new_ldb[topicPlace] = topicStr
		explicit_new[topicPlace] = explStr


	# Save ordered listdb
	with codecs.open(newTopicsName, 'w', 'utf-8') as f:
		for line in new_ldb:
			f.write(line + "\n")

	with codecs.open(newExplicitTopicsName, 'w', 'utf-8') as f:
		for line in explicit_new:
			f.write(line + "\n")







def load_ifs_dic(ifspath):
	ifs_ldb = listdb_load(ifspath)

	ifs_dic = {}

	for wordToken, m in enumerate(ifs_ldb.ldb):
		docs = []
		for j in m:
			docs.append(j.item)

		ifs_dic[wordToken] = docs

	return ifs_dic





def main():
	try :
		parser = argparse.ArgumentParser()
		parser.add_argument("modelName",
						help="directory of the topics, the (.model)")
		args = parser.parse_args()

		reorderTopics(args.modelName)
	except SystemExit:
		print "for help use --help"
		sys.exit(2)


if __name__ == "__main__":
	main()
