
from python.reviews2ref import review2ref as r2r



raw = "/data/raw/drugsComTrain_raw.tsv"
out = "train_drugReviews"

r2r("/data/raw/drugsComTrain_raw.tsv","drugReviews")
