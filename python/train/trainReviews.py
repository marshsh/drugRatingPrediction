import pickle

import doc2emb as d2e

import random




def train():



	# Get training size.
	train_size = 0
	with open("./data/train_drugReviews40000.corpus4train", "r") as f:
		for line in f:
			train_size += 1

	# Use to divide train / validate
	valNum = 0.2
	valSize = int(train_size*valNum)

	random.seed(12345678)
	valIndexes = set(random.sample(range(train_size),valSize))















def main():








if __name__ == "__main__":

	main()

