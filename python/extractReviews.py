import os
import pickle


train_reviews = []
train_ratings = []


print '    Saving TRAIN reviews and ratings'
with open( os.getcwd() + "/data/raw/drugsComTrain_raw.tsv", "r") as f:

	for line in f.readlines():
		s_line = line.split("\t")
		
		if len(s_line) >= 5:
			train_reviews.append(s_line[3])
			train_ratings.append(s_line[4])


with open("./data/train_reviews.RAW_pickle", "w") as file:
	pickle.dump(train_reviews, file)

with open("./data/train_rate.pickle", "w") as file:
	pickle.dump(train_ratings, file)

print'     ... Done'

# "********************************************************"

test_reviews = []
test_ratings = []

print '    Saving TEST reviews and ratings'
with open( os.getcwd() + "/data/raw/drugsComTest_raw.tsv", "r") as f:

	for line in f.readlines():
		s_line = line.split("\t")

		if len(s_line) >= 5:
			test_reviews.append(s_line[3])
			test_ratings.append(s_line[4])


with open("./data/test_reviews.RAW_pickle", "w") as file:
	pickle.dump(test_reviews, file)

with open("./data/test_rate.pickle", "w") as file:
	pickle.dump(test_ratings, file)

print'    ... Done'




