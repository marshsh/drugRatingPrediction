#!/bin/bash


echo "Making /data directory"
mkdir `pwd`/data

echo "Downloading english stopwords"
wget -cO ./data/stopwords_english.txt \
	 https://raw.githubusercontent.com/pan-webis-de/authorid/master/data/stopwords_english.txt


echo "Downloading Drug Review Dataset"
wget -cO drugs_raw.zip \
	 https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip
unzip -u drugs_raw.zip -d `pwd`/data/raw

echo
echo "Extracting Reviews and Ratings to pickle list:"
python ./python/extractReviews.py

echo
echo "Generating TRAIN Drug Reviews SMH reference text (lemmatized)"
python python/reviews2ref.py "/data/raw/drugsComTrain_raw.tsv" "train_drugReviews"

echo
echo "Generating TEST Drug Reviews SMH reference text (lemmatized)"
python python/reviews2ref.py "/data/raw/drugsComTest_raw.tsv" "test_drugReviews"



echo "Generating BOWs from 20 newsgroups reference text"
python python/ref2corpus.py "./data/train_drugReviews.ref" "./data/stopwords_english.txt" "./data" -c 40000

echo "Genereting inverted file from corpus"
smhcmd ifindex "./data/train_drugReviews40000.corpus" "./data/train_drugReviews40000.ifs"

echo "Done processing drugCom corpus"
echo
echo
