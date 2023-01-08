from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from absl import flags
from absl.flags import FLAGS
import pickle
import math
import os
import argparse
#knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#--------------
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
help="path to output label encoder")
args = vars(ap.parse_args())
#flags.DEFINE_string('embed', './embeddings.pickle', 'path to serialized db of facial embeddings')
#flags.DEFINE_string('recognizer', './output', 'path to output model trained to recognize faces')
#flags.DEFINE_string('le', './output', 'path to output label encoder')

print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())
print(data["embeddings"])
# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
lables = le.fit_transform(data["names"])
#trainX, testX, trainY, testY  = train_test_split(data["embeddings"][0], lables, test_size= 0.10, random_state=42)
#print(trainX)
#le = LabelBinarizer()
#labels = le.fit_transform(data["names"]).flatten()
print("[INFO] training model...")
#recognizer = SVC(C=10.0, kernel="linear", probability=True)
recognizer = KNeighborsClassifier(n_neighbors=int(round(math.sqrt(40))), algorithm='ball_tree', weights='distance', n_jobs=-1)
recognizer.fit(data["embeddings"][0], lables)
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
# write the label encoder to disk<font></font>
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()