from sys import argv
from sklearn import svm
from sklearn.metrics import precision_recall_curve, confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import sem

import numpy as np

#usage of this script: python3 SVM_preprocess.py training_set.tsv benchmark_set.tsv

def extract_fold_classes(dataset, n_folds):
	'''given a dataset and the n_folds it returns 2 lists: a list of lists of sequences + a lits of lists of classes'''
	all_folds, all_classes = [], []
	
	for n in range(n_folds):
		fold_seq = []
		fold_class = []
		
		with open(dataset,'r') as filename:
			for line in filename:
				split_line = line.rstrip().split('\t')
				if split_line[0] == 'UniProtKB accession':	#skip header
					pass
				else:
					real_class = split_line[3]
					fold = split_line[4]
					seq = split_line[5]
					
					if int(n) == int(fold):
						fold_seq.append(seq)
						if real_class == 'SP':
							fold_class.append(1)
						else:
							fold_class.append(0)
		all_folds.append(fold_seq)
		all_classes.append(fold_class)
	
	return all_folds, all_classes


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def encoding_seq(seq,k):
	'''encode a single sequence and return the encoded vector'''
	vector = []
	order='ARNDCQEGHILKMFPSTWYV'
	
	dic = {char:0 for char in order}	#20-dimensional empty dictionary
	for res in seq[:k]:			#count occurence of each residue
		dic[res] = dic.get(res)+1
	
	for i in dic:
		vector.append(dic[i]/k)

	return vector


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def encode_folds(folds,k):
	'''takes the 5 folds and encodes them, returns a list of 5 inner lists where each list is an encoded fold'''
	all_encoded_folds =[]
	
	for fold in folds:
		encoded_fold = []
		for seq in fold:
			vector = encoding_seq(seq,k)
			encoded_fold.append(vector)
			
		all_encoded_folds.append(encoded_fold)
	
	return all_encoded_folds


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def parameters_opt(all_folds, all_classes):
	c_values = [1,2,4]
	gamma_values = [0.5, 1, 'scale']
	k_values = [20, 22, 24]
	
	combo = []
	avg_metrics = []		#in the end I want the avg performance for each combination (27 combo)
	
	for k in k_values:
		encoded_folds = encode_folds(all_folds, k)		#list of lists of the encoded folds with a particular k value
		test_fold = []
		for c in c_values:
			for g in gamma_values:
				
				metrics = []
				
				for i in range(len(encoded_folds)):			#loop over the number of folds
					test_fold_seq = encoded_folds[i]
					test_fold_class = all_classes[i]
					
					train_fold_seq = []			#all the seq from the other folds 
					train_fold_classes = []			#all the classes from the other folds
					
					
					for j in range(len(encoded_folds)):
						if j != i:					#put the other folds in a training set
							for seq in encoded_folds[j]:
								train_fold_seq.append(seq)
							for item in all_classes[j]:
								train_fold_classes.append(item)
					
					
					mySVC = svm.SVC(C=c, kernel='rbf', gamma=g)
					mySVC.fit(train_fold_seq, train_fold_classes)		#train the model based on the seq and their classes
					y_pred = mySVC.predict(test_fold_seq)			#predict the seq in the test set
					
					metrics.append(get_performance_metrics(test_fold_class, y_pred))	#in the end metrics is a list of 27 lists (one for each 3x3x3 combo)
														#each list has 5 lists with the metrics for each CV run on that combo
				
				
				#compute the metrics average for each 3x3x3 combo
				arr = np.array(metrics)
				mean_scores = list(np.mean(arr, axis=0))		#mean of each column
				
				#then I add parameters c,k,g to the list and append everything to combo
				mean_scores.append(c)
				mean_scores.append(k)
				mean_scores.append(g)

				combo.append(mean_scores)
	return combo		
	
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def CV_optimized_averages(all_folds, all_classes, c, k, g):
	'''use the optimal c, k, g to re-run CV on each fold and compute the average MCC, F1, accuracy, recall, precision'''
	
	metrics = []
	
	#mean_metrics = []
	std_metrics = []
	
	encoded_folds = encode_folds(all_folds, k)
	
	for i in range(len(encoded_folds)):		#loop over the number of folds
		test_fold_seq = encoded_folds[i]
		test_fold_class = all_classes[i]
			
		train_fold_seq = []			#all the seq from the other folds 
		train_fold_classes = []			#all the classes from the other folds
					
					
		for j in range(len(encoded_folds)):
			if j != i:					#put the other folds in a training set
				for seq in encoded_folds[j]:
					train_fold_seq.append(seq)
				for item in all_classes[j]:
					train_fold_classes.append(item)
					
					
		mySVC = svm.SVC(C=c, kernel='rbf', gamma=g)
		mySVC.fit(train_fold_seq, train_fold_classes)		#train the model based on the seq and their classes
		y_pred = mySVC.predict(test_fold_seq)			#predict the seq in the test set
					
		metrics.append(get_performance_metrics(test_fold_class, y_pred))	# I do 5 CV run and for each run I get some performance metrics



	#compute the metrics average for each column (columns are: MCC, F1, accuracy, recall, precision)
	arr = np.array(metrics)
	mean_opt_metrics = list(np.mean(arr, axis=0))		#list with the mean of the metrics
	std_opt_metrics = list(sem(arr, axis=0))		#list with the standard deviation of each mean
	
	

	return mean_opt_metrics, std_opt_metrics
	

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def extract_bench_seq_class(dataset, k):
	'''extracts benchmark sequences and classes - the seq are already encoded. Returns 2 lists'''
	bench_seq = []
	bench_class = []
	
	with open(dataset,'r') as filename:
		for line in filename:
			split_line = line.rstrip().split('\t')
			if split_line[0] == 'UniProtKB accession':	#skip header
				pass
			else:
				real_class = split_line[3]
				seq = split_line[4]
				
				encoded_seq = encoding_seq(seq,k)
				bench_seq.append(encoded_seq)
				
				if real_class == 'SP':
					bench_class.append(1)
				else:
					bench_class.append(0)
	
	return bench_seq, bench_class


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def predict_bench(all_folds, all_classes, bench_seq, bench_class, c, k, g):
	'''fit the SVM to the full training and then predicts the class of the benchmark seq - returns the performance metrics'''
	
	all_train_seq = []
	all_train_class = []
	
	for fold in all_folds:				#all the training seq encoded in a single list
		for seq in fold:
			encoded_seq = encoding_seq(seq,k)
			all_train_seq.append(encoded_seq)

	for fold in all_classes:			#all the training classes in a single list
		for item in fold:
			all_train_class.append(item)
	
	
	mySVC_bench = svm.SVC(C=c, kernel='rbf', gamma=g)
	mySVC_bench.fit(all_train_seq, all_train_class)			#train the model based on ALL the training seq 
	y_pred_bench = mySVC_bench.predict(bench_seq)			#predict the bench
	
	mcc_bench, F1, accuracy, recall, precision = get_performance_metrics(bench_class, y_pred_bench)
	
	
	return mcc_bench, F1, accuracy, recall, precision, y_pred_bench


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_performance_metrics(real_class, y_pred):
	'''takes the real_classes and the y_pred and computes MCC, CM, F1, accuracy, recall, precision'''
	
	MCC = matthews_corrcoef(real_class, y_pred)
	F1 = f1_score(real_class, y_pred)
	accuracy = accuracy_score(real_class, y_pred)
	recall = recall_score(real_class, y_pred)
	precision = precision_score(real_class, y_pred)
	
	return MCC, F1, accuracy, recall, precision
	
	
	
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	
	


if __name__ == '__main__':

	tr = argv[1]						#training tsv dataset
	bm = argv[2]						#benchmark tsv dataset
	
	all_folds, all_classes = extract_fold_classes(tr,5)	#all_folds is a list 5 lists of seq & all_classes is a list of 5 lists of classes (SP:1, NO_SP:0)
	
	combo = parameters_opt(all_folds, all_classes)		#all the 3x3x3 combinations - MCC, F1, accuracy, recall, precision, c, k, g
	
	best = max(combo)					#combo with max MCC
	
	c = best[5]
	k = best[6]
	g = best[7]
		
	print('Combination with highest MCC:')
	print('MCC %s, F1 %s, accuracy %s, recall %s, precision %s, c %s, k %s, g %s' % (best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7]))
	print('\n')
	
	#Once I found the best paramaters I run again the CV with the best combo of parameters and calculate the mean of MCC, F1, accuracy, recall, precision of each fold
	mean_opt_metrics, std_opt_metrics = CV_optimized_averages(all_folds, all_classes, c, k, g)
	
	print('CV scores with optimized parameters:')
	print('MCC %s, F1 %s, accuracy %s, recall %s, precision %s' % (mean_opt_metrics[0], mean_opt_metrics[1], mean_opt_metrics[2], mean_opt_metrics[3], mean_opt_metrics[4]))
	print('MCC sem %s, F1 sem %s, accuracy sem %s, recall sem %s, precision sem %s' % (std_opt_metrics[0], std_opt_metrics[1], std_opt_metrics[2], std_opt_metrics[3], std_opt_metrics[4]))
	
	print('\n')
	
	
	#Now I work on the benchmarking: first I extract the classes and the sequences (and encode them)
	
	bench_seq, bench_class = extract_bench_seq_class(bm,k)
	mcc_bench, F1_bench, accuracy_bench, recall_bench, precision_bench, y_pred_bench  = predict_bench(all_folds, all_classes, bench_seq, bench_class, c, k, g)
	print('Scores for the benchmark predictions:')
	print('MCC bench ', mcc_bench)
	print('F1 ', F1_bench)
	print('Accuracy ', accuracy_bench)
	print('Recall ', recall_bench)
	print('Precision ', precision_bench)
	

# With the grid search we optimized the k,c,gamma combinations. We have 3x3x3 combinations of k,c,gamma. 
# For each CV run (of each combo) we calculate the precision/recall/accuracy/F1/CM/MCC and then average them
# We report in the table the average metrics for the best combination (combo with max MCC)
# in the supplementary material we report the average metrics for each 3x3x3 combo
	
	

	
	

