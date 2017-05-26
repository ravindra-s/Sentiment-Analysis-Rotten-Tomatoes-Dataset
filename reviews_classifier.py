import logging
import logging.config
import pandas as pd
import csv
from preprosessor import preprocess_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import OrderedDict

logger = logging.getLogger(__name__)


def cross_validation_model_analysis(all_features, all_labels):
	""" Run permutations of vectorizers and estimators on the dataset and report results."""

	logger.info("Total size of the dataset (num of phrases): " + str(len(all_features)))

	# Trim down the dataset as per your requirement and hardware capabilities.
	# This is required because running models on the entire dataset may cause memory issues.

	# all_features = all_features[:22000]
	# all_labels = all_labels[:22000]
	
	logger.info("Beginning pipeline execution")
	

	estimators_dict = OrderedDict([('MultiNomialNB', MultinomialNB()), \
						('LinearSVC', LinearSVC()), \
						('LogisticRegression', LogisticRegression(n_jobs = -1)), \
						('DecisionTreeClassifier', DecisionTreeClassifier(min_samples_split=2000, \
																			min_samples_leaf=200))])
	transformers_dict = OrderedDict([('tf-idf', TfidfVectorizer()), \
						  				('bag_of_words', CountVectorizer(ngram_range=(1, 2)))])
	
	steps = []

	f = open('results_summary_f1score.txt','w')
	f.write("A quick look at f1-scores of all the models\n")
	f.write("-"*50)


	for transformer_name, transformer in transformers_dict.items():
		f.write("\nFeature type: " + transformer_name + "\n\n")
		logger.info("*"*100)
		logger.info("For feature transformation using : " + transformer_name)
		steps.append((transformer_name, transformer))

		for estimator_name, estimator in estimators_dict.items():
			
			logger.info("Running the model with : " + estimator_name)
			steps.append((estimator_name, estimator))
			model = Pipeline(steps)
			predicted_labels = cross_val_predict(model, all_features,all_labels, \
												 cv = 5, n_jobs = -1, verbose = 100)
		
			recall = round(recall_score(all_labels, predicted_labels, average = 'weighted'),2)
			precision = round(precision_score(all_labels, predicted_labels, average = 'weighted'),2)
			f1 = round(f1_score(all_labels, predicted_labels, average = 'weighted'),2)
			report = classification_report(all_labels, predicted_labels)
			conf_matrix = confusion_matrix(all_labels, predicted_labels)
		
			logger.info("recall score: " + str(recall))
			logger.info("precision score: " + str(precision))
			logger.info("f1 score: " + str(f1))
			logger.info("confusion matrix: \n" + str(conf_matrix))
			logger.info("classification report: \n" + report)

			logger.info("*"*70)

			f.write(estimator_name + " : " + str(f1) + "\n")

			# remove current estimator and make room for the next one
			del steps[1] 	

		f.write("-"*50)

		# remove current transformer and make room for the next one 
		del steps[0]		

	f.close()

def main():

	# 0. Logging
	logging.basicConfig(level=logging.DEBUG,
						format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
						datefmt='%m-%d %H:%M',
						filename='reviews_classification.log',
						filemode='w')

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	# 1. Preprocessing of dataset 
	train_file = "train.tsv"	
	train_features, train_labels = preprocess_dataset(train_file, is_train = True)

	# Run cross validation model analysis on the entire dataset 
	cross_validation_model_analysis(train_features, train_labels)

	# Train the model on entire dataset and test it on the test data provided by Kaggle
	classifiy_reviews(train_features, train_labels, "test.tsv")


def classifiy_reviews(train_features, train_labels, test_dataset_file):
	""" Run the model on the test data provided. Test data isn't labeled.
		It will generate a csv of output file which can then be uploaded for submission on Kaggle.

	"""

	df = pd.read_csv(test_dataset_file, delimiter="\t")
	phrase_ids = df['PhraseId'].tolist()
	test_features = preprocess_dataset(test_dataset_file, is_train = False)

	logger.info("Train data size: " + str(len(train_features)))
	logger.info("Train label size: " + str(len(train_labels)))

	logger.info("Begin vectorisation of features")
	vectorizer = TfidfVectorizer()	
	train_features = vectorizer.fit_transform(train_features)
	test_features = vectorizer.transform(test_features)
	logger.info("Finished vectorisation of features")

	logger.info("Begin classification of the test data")
	svc = MLPClassifier(learning_rate = 'adaptive', max_iter = 100, shuffle = True, \
						verbose = 100, warm_start = True, early_stopping = True)
	svc.fit(train_features, train_labels)
	predictions = svc.predict(test_features)
	logger.info("Finished classification of the test data")

	logger.info("Test data size: " + str(test_features.shape[0]))
	logger.info("Test label (predictions) size: " + str(predictions.shape[0]))

	with open('results.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(["PhraseId","Sentiment"])
		writer.writerows(zip(phrase_ids, predictions))

if __name__ == '__main__':

	main()
