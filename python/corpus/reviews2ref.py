#!/usr/bin/env python
# -*- coding: utf-8
#
# Mariana Gleason Freidberg ... adapted from:
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2017 --> 2019
#
# -------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------
"""
Function to store the .tsv drugCom reviews in a corpus format (lematized and tokenized by nltk using "tokenizer.py").
"""
import os
import codecs
import argparse

from tokenizer import line2tokens

	


def review2ref(raw_tsv, out_name):
	"""
	Extracts the drug reviews and its ratings, vectorizes the reviews, stores them in
	a database of lists and saves it to file.
	"""
	out_refpath = os.getcwd() + "/data/" + out_name + ".ref"
	out_labelspath = os.getcwd() + "/data/" + out_name + ".labels"

	fref = codecs.open(out_refpath, 'w', 'utf-8')
	flabels = codecs.open(out_labelspath, 'w', 'utf-8')

	with open( os.getcwd() + raw_tsv, "r") as in_raw:

		for line in in_raw.readlines():
			s_line = line.split("\t")
			
			if len(s_line) >= 5:
				review = s_line[3]
				rating = s_line[4]

				line = review.decode('utf-8').strip()
				line2 = line.replace('\n',' ').replace('\r',' ').replace('`',' ')
				tokened_line = line2tokens(line2)

				fref.write(' '.join(tokened_line) + '\n')
				flabels.write(str(rating) + '\n')

	fref.close()
	flabels.close()


def main():
	"""
	Main function
	"""
	try:
		parser = argparse.ArgumentParser()
		parser = argparse.ArgumentParser(
			description="Transforms drugCom.tsv and creates reference and labels files")
		parser.add_argument("raw_tsv",
							help="directory of drugCom.tsv")
		parser.add_argument("out_name",
							help="name of .ref and .labels files")
		args = parser.parse_args()

		review2ref(args.raw_tsv, args.out_name)
			
	except SystemExit:
		print "for help use --help"
		sys.exit(2)

if __name__ == "__main__":
	main()
