import os
#os.environ['SF_BACKEND'] = 'tensorflow'
import multiprocessing
import slideflow as sf
import logging
import numpy as np
import pandas as pd
import pprint
import re
import itertools
import json
import csv
import tempfile
import pyarrow
from tabulate import tabulate  
from sklearn import metrics
from sklearn.metrics import confusion_matrix

########## Sara's Simplified Slideflow Stats 

def testing_os_pathing():
	print(os.path.abspath(__file__))

# ============================================================================ #
######## --------- UTILITY FUNCTIONS ----------- #########
def to_onehot(val, max):
	"""Converts value to one-hot encoding.

	Args:
		val (int): Value to encode
		max (int): Maximum value (length of onehot encoding)
	"""
	onehot = np.zeros(max, dtype=np.int64)
	onehot[val] = 1
	return onehot

def one_hot(array):
    '''Convert arrays with multiple unique values to one-hot binary encoding, 
	with columns equal to number of unique values.'''
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

def merge_dict(dict1, dict2):
	for key, val in dict1.items():
		if type(val) == dict:
			if key in dict2 and type(dict2[key] == dict):
				merge_dict(dict1[key], dict2[key])
		else:
			if key in dict2:
				dict1[key] = dict2[key]

	for key, val in dict2.items():
		if not key in dict1:
			dict1[key] = val

	return dict1

def merge_two_dicts(dict1, dict2):
	for key, val in dict1.items():
		if type(val) == dict:
			if key in dict2 and type(dict2[key] == dict):
				merge_dict(dict1[key], dict2[key])
		else:
			if key in dict2:
				dict1[key] = dict2[key]

	for key, val in dict2.items():
		if not key in dict1:
			dict1[key] = val

	return dict1

def merge_dict_list(dict_list):
		merged_dict = {}
		for i in range(len(dict_list)):
				dict_to_merge = dict_list[i]
				if i == 0:
						merged_dict = dict_to_merge
				else:
						merged_dict = merge_two_dicts(merged_dict, dict_to_merge)

		return merged_dict

def flatten(t):
	'''Flatten list.'''
	return [item for sublist in t for item in sublist]

def find_max_epoch(model_dir):
	'''Given a model directory, find the maximum epoch trained to for that model. 
	Useful when the model & its results saved at multiple epochs.

	Args:
		model_dir (str): Path to model directory.
	
	Returns:
		(int): Maximum epoch trained to.
	'''
	epoch_list = [re.findall(r'(?<=epoch)\d(?=.\w+)', f) for f in os.listdir(model_dir) if re.findall(r'(?<=epoch)\d(?=.\w+)', f)]
	# return the max epoch for a given model directory, theoretically the best epoch
	return max(list(set(flatten(epoch_list))))

def get_params(model_dir, keys_list):
	# TODO: put in optional keys list from params.json
	'''Utility function to pull out desired values from params.json file, given a list of
	
	Args:
		model_dir (str): Path to model directory with params.json. Must end with a "/".
		keys_list (list): List of param names to use as keys to access values in params.json. 
	
	Returns:
		params_dict (dict): Dictionary with returned params, keys are from keys_list and 
		values are extracted from params.json.
	'''
	f = open(os.path.join(model_dir,"params.json"))
	params = json.load(f)
	# create dict to return from keys
	params_dict = dict.fromkeys(keys_list)
	# pull out items from imported json dict
	for key in keys_list: 
		params_dict[key] = params[key]
	f.close()

	return params_dict

# ============================================================================ #
########### ---- FILE ACCESS FUNCTIONS ------ ###########
def combine_pred_csvs(model_paths, save_path=None):
	'''For bootstrapping/crossval purposes, will load in multiple prediction csvs and combine them
	if save_path is provided, will save newly created aggregated CSV to desired location.
	
	Args:
		models_paths (list): Paths to prediction CSVs to combine. Should all be of same level.
		save_path (str, Optional): Path where to save combined prediction CSV.

	Returns:
		preds_all (DataFrane): Combined predictions.
	'''
	preds_all = []
	for name in model_paths:
		try:
			preds = pd.read_csv(name)
		except UnicodeDecodeError:
			preds = pd.read_parquet(name)
		preds_all.append(preds)
	preds_all = pd.concat(preds_all)

	if save_path is not None:
		preds_all.to_csv(save_path)

	return preds_all

def get_model_names(models_dir, model_label, printplz=False):
	# TODO: crosscheck with get_model_paths below
	'''Get the list of model paths given a model_label.
	
	Args:
		models_dir (str): Path to project models directory.
		model_label (str): Label matching model group (bootstrapped/crossvalidated experiment)
		printplz (bool): Default False. If True, prints the model names. 

	Returns:
		models_path_list (list): list of model paths.
	'''
	models_path_list = [models_dir + f for f in os.listdir(models_dir) if re.search(rf'{model_label}', f)]
	# print the pred_paths to the console
	if printplz is True:
		print(models_path_list)

	return models_path_list

# def get_model_paths(models_dir, model_patterns):
# 	# this function is meant to pull out paths of desired models
# 	list_of_models = []
# 	for i in model_patterns:
# 		pattern = i
# 		list_of_models += [f for f in os.listdir(models_dir) if re.match(rf'{i}', f)]

def get_pred_paths(model_dir, level, epoch=None):
	'''Use to get list of prediction CSV filepaths for a given model directory, level, and epoch.
	
	Args:
		model_dir (str): Path to model directory that contains prediction files.
		level (str): Patient, slide, tile level you want to return prediction file for.
		epoch (int): Epoch for whcih to return prediction file for. Default None, will return single epochs.

	Returns:
		pred_path_list (list): List of prediction file paths for specific level and epoch.
	'''
	# TODO: change to if epoch 
	# get list of desired model paths from kfold run and return list of prediction files
	pred_path_list = []

	# find all prediction CSVs
	path_list = [f for f in os.listdir(model_dir) if re.match(r'^.*predictions.*$', f)]
	# filter for tile, slide, or patient
	path_list = [f for f in path_list if re.match(rf'^{level}.*$', f)]
	# filter by specific epoch if epoch is given (if there are multiple), else should just be one, return that for each model
	if epoch is not None:
		pred_path_list += [model_dir + "/" + f for f in path_list if re.match(rf'^.*epoch{epoch}.*$', f)]
	else:
		pred_path_list += [model_dir + "/" + f for f in path_list if re.match(rf'^.*epoch.*$', f)]

	# return list of what should be three
	return pred_path_list

def find_unique_models(models_dir):
	'''Given a models_dir, find all unique experiments run; aka return all unique model group labels.
	
	Args:
		models_dir (str): Path to project's models directory.

	Returns:
		unique_runs (list): Labels correspondong to unique model group labels. 
	'''
	model_names = os.listdir(models_dir)
	model_groups = [re.split("-kfold\d", re.split("\d{5}-", name)[1])[0] for name in model_names]
	unique_runs = list(set(model_groups))

	return unique_runs


# ============================================================================ #
###### --------- GET METRICS FUNCTIONS --------  ####
def return_pred_objects(pred_csv_path, level='tile'):
	# TODO: extract level from pred_csv_path
	'''For a given prediction CSV path, reads in the prediction file and finds objects required for calculating
	metrics.

	Args:
		pred_csv_path_list (list): List of paths to prediction CSV files. All CSVs should from the same k-fold 
		cross-validated experiment (a model group).
		level (str): results level corresponding to CSVs (tile, slide, patient)
		
	Returns:
		num_cat (int): Number of outcome categories. 
		pred_cols (list): List of prediction column names from prediction file..
			Example: [Subgroup-y_pred0, Subgroup-y_pred1, Subgroup-y_pred2]
		y_pred (ndarray): Softmax prediction values. Each array length equal to number of unique outcome values. 
			Number of rows is equivalent to number of prediction objects at that level.
			Example: array([[1.4923677e-02, 7.2541076e-01, 2.5966552e-01], ...,
				[7.5657427e-01, 2.8979644e-02, 2.1444611e-01]], dtype=float32)
		true_cols (list): List of true columns names from prediction file.
			Example: ['Subgroup-y_true']
		y_true (ndarray): Onehot prediction values. Each array length equal to number of unique outcome values. 
			Number of rows is equivalent to number of prediction objects at that level.
			Example: array([[0., 0., 1.], ..., [0., 0., 1.]])
		cases (list): Patient IDs corresponding to each line in predictions file. 
		onehot_predictions (ndarray): predictions but as one-hot encodings instead of softmax.
			Example: array([[0, 1, 0], [0, 1, 0], [1, 0, 0],...])
	'''
	# load in pred CSV as pandas dataframe
	try:
		preds = pd.read_csv(pred_csv_path)
	except UnicodeDecodeError:
		preds = pd.read_parquet(pred_csv_path)

	# patient/slide vs. tile level
	label = 'y_pred' if 'percent_tiles_positive' not in [col for col in preds.columns] else 'percent_tiles_positive'
	
	# pred columns
	pred_cols = [col for col in preds.columns if label in col]
	y_pred = np.array(preds.loc[:,pred_cols])

	# number of categories
	num_cat = len(pred_cols)

	# true columns
	if re.match(rf'.*.csv', pred_csv_path):
		true_cols = [col for col in preds.columns if 'y_true' in col]
		y_true = np.array(preds.loc[:,true_cols])
		#y_true = np.array([to_onehot(i, num_cat) for i in y_true])
	else:
		true_cols = [col for col in preds.columns if 'y_true' in col]
		y_true = np.array(preds.loc[:,true_cols])
		y_true = one_hot(y_true.transpose()[0])
		# make sure that y_true has the same number of columns as y_pred
		if y_true.shape[1] != num_cat:
			y_true = np.concatenate((y_true, np.zeros((y_true.shape[0], num_cat - y_true.shape[1]))), axis=1)

	# get patient list
	if level == 'patient':
		cases = list(preds['patient'])
	else:
		cases = list(preds['slide'])

	# one-hot predictions
	onehot_predictions = np.array([to_onehot(x, num_cat) for x in np.argmax(y_pred, axis=1)])
	#print(onehot_predictions)

	return num_cat, pred_cols, y_pred, true_cols, y_true, cases, onehot_predictions


def basic_metrics(y_true, onehot_predictions):
	'''Generates basic performance metrics, including sensitivity, specificity, and accuracy. Must be onehot_predictions.
	Only for single outcome, so y_pred and y_true from return_pred_objects should be sliced for single outcome.
	
	Args:
		y_true (ndarray): Onehot prediction values for single outcome value. Array length equal to number of 
			prediction objects at that level.
			Example: array([0., 0., 1., 0., 1.])
		onehot_predictions (ndarray): One-hot encoded predictions for single outcome value. Array length equal to number of 
			prediction objects at that level.
			Example: array([0, 0, 1, 0, 1])

	Returns:
		metrics_dict (dict): dictionary of calculated metric values.
			Example: {'accuracy': 0.33,
					'sensitivity': 0.0,
					'specificity': 0.5,
					'precision': 0.0,
					'recall': 0.0,
					'f1_score': 0.0,
					'kappa': -0.5}
	'''
	#print(y_true)
	assert(len(y_true) == len(onehot_predictions))
	assert([y in [0,1] for y in y_true])
	assert([y in [0,1] for y in onehot_predictions])

	TP = 0 # True positive
	TN = 0 # True negative
	FP = 0 # False positive
	FN = 0 # False negative

	for i, yt in enumerate(y_true):
		yp = onehot_predictions[i]
		if yt == 1 and yp == 1:
			TP += 1
		elif yt == 1 and yp == 0:
			FN += 1
		elif yt == 0 and yp == 1:
			FP += 1
		elif yt == 0 and yp == 0:
			TN += 1
	
	addtl_metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'recall', 'f1_score', 'kappa']

	metrics_dict = {key: {} for key in addtl_metrics}

	try:
		metrics_dict['accuracy'] = round((TP + TN) / (TP + TN + FP + FN), 2)
	except ZeroDivisionError:
		metrics_dict['accuracy'] = np.nan
	try:
		metrics_dict['sensitivity'] = round(TP / (TP + FN), 2)
	except ZeroDivisionError:
		metrics_dict['sensitivity'] = np.nan
	try:
		metrics_dict['specificity'] = round(TN / (TN + FP), 2)
	except ZeroDivisionError:
		metrics_dict['specificity'] = np.nan
	metrics_dict['precision'] = round(metrics.precision_score(y_true, onehot_predictions, zero_division=0), 2) 
	metrics_dict['recall'] = round(metrics.recall_score(y_true, onehot_predictions), 2)
	metrics_dict['f1_score'] = round(metrics.f1_score(y_true, onehot_predictions), 2)
	metrics_dict['kappa'] = round(metrics.cohen_kappa_score(y_true, onehot_predictions), 2)

	return metrics_dict

def get_aucs(y_true, y_pred):
	'''Generate AUROC and AUPRC from outcome probabilities (continuous, not onehot). Only for single outcome,
	so y_pred and y_true from return_pred_objects should be sliced for single outcome.
	
	Args:
		y_true (ndarray): Onehot prediction values for single outcome value. Array length equal to number of 
			prediction objects at that level.
			Example: array([0., 0., 1., 0., 1.])
		y_pred (ndarray): Softmax prediction values for single outcome value. Array length equal to number of 
			prediction objects at that level.
			Example: array([0, 0, 1, 0, 1])

	Returns:
		metrics_dict (dict): Dict with keys 'AUROC', 'AUPRC'.
			Example: {'AUROC': 0.5, 'AUPRC': 0.5}

	'''
	assert(len(y_true) == len(y_pred))

	auc_metrics = ['AUROC', 'AUPRC']

	metrics_dict = {key: {} for key in auc_metrics}
	# generate AUROC
	fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
	roc_auc = metrics.auc(fpr, tpr)

	# generate AUPRC
	auprc = metrics.average_precision_score(y_true, y_pred)
	metrics_dict['AUROC'] = round(roc_auc, 2)
	metrics_dict['AUPRC'] = round(auprc, 2)

	return metrics_dict

def get_metrics_outcome(y_true, y_pred, onehot_predictions, cat_num):
	'''Get all metrics for a single outcome value and return dictionary of metrics. 
	
	Args:
		y_true (ndarray): Onehot prediction values for single outcome value. Array length equal to number of 
			prediction objects at that level.
			Example: array([0., 0., 1., 0., 1.])
		y_pred (ndarray): Softmax prediction values for single outcome value. Array length equal to number of 
			prediction objects at that level.
			Example: array([1.4923677e-02, 2.1980377e-05, 9.9800617e-01, 5.6900114e-02, 7.5657427e-01], dtype=float32)
		onehot_predictions (ndarray): One-hot encoded predictions for single outcome value. Array length equal to number of 
			prediction objects at that level.
			Example: array([0, 0, 1, 0, 1])
		cat_num (int): Integer outcome value metrics are being generated for.

	Returns:
		cat_dict (dict): Dict of metrics returned for particular outcome value (key), values are metrics.
			Example: {0: {'accuracy': 0.77,
				'sensitivity': 0.72,
				'specificity': 0.79,
				'precision': 0.63,
				'recall': 0.72,
				'f1_score': 0.67,
				'kappa': 0.5,
				'AUROC': 0.8,
				'AUPRC': 0.7,
				'ConfusionMatrix': array([[44786, 11612],
						[ 7806, 20084]])}}
	'''
	cat_dict = {}
	basic_metrics_dict = basic_metrics(y_true, onehot_predictions)
	aucs_dict = get_aucs(y_true, y_pred)
	cm_dict = get_confusion_matrix(y_true, onehot_predictions)
	cat_dict[cat_num] = merge_dict_list([basic_metrics_dict, aucs_dict, cm_dict])

	return cat_dict

def gen_metrics_all_outcomes(num_cat, y_pred, y_true, onehot_predictions):
	'''Generate metrics for all categories and returned combined dictionary. Metrics will be generated for
	all outcome vategories. 
	
	Args:
		num_cat (int): Number of outcome categories. 
		y_pred (ndarray): Softmax prediction values. Each array length equal to number of unique outcome values. 
			Number of rows is equivalent to number of prediction objects at that level.
			Example: array([[1.4923677e-02, 7.2541076e-01, 2.5966552e-01], ...,
				[7.5657427e-01, 2.8979644e-02, 2.1444611e-01]], dtype=float32)
		y_true (ndarray): Onehot prediction values. Each array length equal to number of unique outcome values. 
			Number of rows is equivalent to number of prediction objects at that level.
			Example: array([[0., 0., 1.], ..., [0., 0., 1.]])
		onehot_predictions (ndarray): predictions but as one-hot encodings instead of softmax.
			Example: array([[0, 1, 0], [0, 1, 0], [1, 0, 0],...])

	Returns:
		merged_cat_dict (dict): Dict of metrics for model, keys are integer outcome values. 
			Example: {0: {'accuracy': 0.77,
				'sensitivity': 0.72,
				'specificity': 0.79,
				'precision': 0.63,
				'recall': 0.72,
				'f1_score': 0.67,
				'kappa': 0.5,
				'AUROC': 0.8,
				'AUPRC': 0.7,
				'ConfusionMatrix': array([[44786, 11612],
						[ 7806, 20084]])},
				1: {'accuracy': 0.84, ...},
				2: {'accuracy': 0.84, ...}}
	'''
	cat_dict_list = []
	for i in range(num_cat):
		cat_dict_list += [get_metrics_outcome(y_true[:,i], y_pred[:,i], onehot_predictions[:,i], i)]
	
	merged_cat_dict = merge_dict_list(cat_dict_list)

	return merged_cat_dict

def get_metrics_from_pred(pred_csv_path, level):
	'''Generate merged metrics dictionary for all outcomes for a certain level.
	
	Args:
		pred_csv_path (list): Path to prediction CSV file.
		level (str): results level corresponding to CSVs (tile, slide, patient)

	Returns:
		merged_metrics_dict (dict): Dict of metrics for model, keys are integer outcome values. 
			Example: Same return as above gen_metrics_all_outcomes.
	'''
	num_cat, pred_cols, y_pred, true_cols, y_true, cases, onehot_predictions = return_pred_objects(pred_csv_path, level=level)
	merged_metrics_dict = gen_metrics_all_outcomes(num_cat, y_pred, y_true, onehot_predictions)
	return merged_metrics_dict

# ============================================================================ #
######### --------- CONFUSION MATRICES ----------- ##########

def get_confusion_matrix(y_true, onehot_predictions, labels=False):
	'''Generate 2x2 Confusion Matrix from onehot predictions for a single outcome value (not multicategorical/multilabel). 
	To get multicategorical/multilabel CM, use aggregated_confmat() below. 
	Simplest function, will just get you a single, pretty confusion matrix. 
	Returns as a dict to make it easy to combine with metrics dictionary.'''
	cm_obj = metrics.confusion_matrix(y_true, onehot_predictions)
	if labels is True:
		cm_obj = pretty_confusionmatrix(cm_obj)
	return {"ConfusionMatrix": cm_obj}

def pretty_confusionmatrix(cm_obj):
	# TODO: adjust for other options
	# input CM should have format like: [[1 2], [0 3]], this assigns proper labels to it
	cmtx = pd.DataFrame(cm_obj, index=['true:yes', 'true:no'], columns=['pred:yes', 'pred:no'])
	#cmtx.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
	return cmtx

def aggregated_confmat(pred_csv_path_list, title=None, format="plot", save_path=None, **kwargs):
	# TODO: in theory, i could remove the level arg and get that from the pred_csv_path_list instead
	"""Produce an aggregated Confusion Matrix that is N x N where N is the number of unique outcome values, as 
	well as a list of N length containing 2x2 CMs corresponding to each outcome. Also can plot & save plot.
	
	Args:
		pred_csv_path_list (list): List of paths to prediction CSV files. All CSVs should from the same k-fold 
		cross-validated experiment (a model group).
		#level (str): results level corresponding to CSVs (tile, slide, patient)
		title (str): Title of plot.
		format (str): Format of output. Options are "plot", "2x2" or "NxN". If "plot", will plot the confusion matrix.
		save_path (str): Path where to save the confusion matrix.
		**kwargs: Additional arguments to pass to the ConfusionMatrixDisplay.from_predictions() function.
	
	Returns:
		cm2 (NumPy array): An N X N NumPy array where N is the number of unique outcome values, the y axis is the 
		True labels and the x axis is the false.
		mlb_cm (list of arrays): Returns a list of 2x2 CMs
		The function will also plot the confusion matrix as well
		# TODO: THOUGH... this could use some actual labels

	Example:
	NxN format, cm2:
	[[4 1 0 0 0 0 0 0]
	[2 0 3 1 0 0 0 0]
	[0 0 5 0 0 0 0 0]
	[0 0 2 6 0 0 0 0]
	[0 0 2 2 2 0 0 0]]

	2x2 format, mlb_cm:
	[[[29  3]  [[30  1]  [[25  7]  [[23  6]  [[31  0]
	 [ 1  4]]  [ 6  0]]  [ 0  5]]  [ 2  6]]  [ 4  2]]]
	"""

	if type(pred_csv_path_list) is list:
		# combine all the prediction files into one large file
		aggregated_pred_csv = combine_pred_csvs(pred_csv_path_list)
		# save temp file to use
		with tempfile.NamedTemporaryFile(mode='w', delete=False) as csvfile:
			aggregated_pred_csv.to_csv(csvfile)
		pred_csv_path = csvfile.name
		# level
		level = pred_csv_path_list[0].split("/")[-1].split("_")[0]
	else:
		pred_csv_path = pred_csv_path_list
		level = pred_csv_path.split("/")[-1].split("_")[0]
	_, _, _, _, y_true, _, onehot_predictions = return_pred_objects(pred_csv_path, level)
	class_pred = np.argmax(onehot_predictions, axis=1) # reversing onehot encoding https://www.codegrepper.com/code-examples/python/reverse+one+hot+encoding+python+numpy
	class_true = np.argmax(y_true, axis=1)
	# if pred_csv_path_list is a temp file, use pred_csv_path_list[0] to get the path to the first file
	# use regex expression to identify if the path is a temp file or not
	if re.search(r'tmp', pred_csv_path):
		oc_lbls = get_params(pred_csv_path_list[0].split(level)[0], ["outcome_labels"])['outcome_labels']
	else:
		oc_lbls = get_params(pred_csv_path.split(level)[0], ["outcome_labels"])['outcome_labels']
	
	from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay

	# cm is a normal confusion matrix 
	if format == "2x2":
		cm2 = confusion_matrix(class_true,class_pred)
		return cm2

	# multilabel confusion matrix returns a list of 2x2 CMs, one CM for each unique outcome
	if format == "NxN":
		mlb_cm = multilabel_confusion_matrix(y_true, onehot_predictions)
		return mlb_cm

	# plotting
	if format == "plot":
		disp = ConfusionMatrixDisplay.from_predictions(class_true, class_pred, display_labels=list(oc_lbls.values()), xticks_rotation='vertical', **kwargs)
		#disp.ax_.set_title(title)
		# import matplotlib.pyplot as plt
		# plt.show()
		# if save_path:
		# 	disp.figure_.savefig(save_path, bbox_inches='tight')

		return disp


#def get_usable_dicts()	

# ============================================================================ #
######## --------- SLIDE MANIFEST NUMBERS ----------- ##########
def get_sm_paths(model_dirs):
	'''Takes a list of paths to each model directories and returns list of paths to slide manifests in each directory.'''
	# return list of slide manifest paths
	return [os.path.join(model_dir,"slide_manifest.csv") for model_dir in model_dirs]

def get_manifest_numbers(sm_path):
	# TODO: Could add option to label unique outcomes using a column for the outcomes. 
	'''Takes path to slide manifest file as argument. Will return slide manifest numbers for training model OR for 
	training model used for evaluation (if SM path is an evaluation directory).
	
	Returns a dataframe with columns "train", "val", "total" with rows
	as each unique outcome value. Row index corresponds to each unique outcome value.
	
	train	val	total
	0	9	3	12
	1	14	1	15
	2	21	8	29	
	'''
	# get slide manifest numbers for an individual model
	sm = pd.read_csv(sm_path)
	# get numbers
	if "-eval-" in sm_path:
		# get numbers for training from training model directory
		train_sm_path = get_params(sm_path.split("slide_manifest.csv")[0], ["model_path"])['model_path'] + "/slide_manifest.csv"
		train_sm = pd.read_csv(train_sm_path)
		sm_nums = pd.concat([train_sm[train_sm['dataset']=="training"]["outcome_label"].value_counts(), sm[sm['dataset']=="validation"]["outcome_label"].value_counts()], axis=1).sort_index()
		sm_nums.columns = ["train", "val"]
		sm_nums['total'] = sm_nums.train + sm_nums.val
	else:
		# get numbers for 
		sm_nums = pd.concat([sm[sm['dataset']=="training"]["outcome_label"].value_counts(), sm[sm['dataset']=="validation"]["outcome_label"].value_counts(), sm["outcome_label"].value_counts()], axis=1).sort_index()
		sm_nums.columns = ["train", "val", "total"]

	# apply int to dataframe sm_nums and also coerce NaNs to 0
	sm_nums = sm_nums.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

	return sm_nums

def get_manifest_numbers_mg(sm_path_list):
	# TODO: maybe combine this with get_manifest_numbers() and add an argument to specify if it's a model group or not
	'''Takes a list of paths to slide manifests, which should each be from a kfold from a cross-validation experiment.
	Will return a dictionary with keys corresponding to each kfold number and the values as the SM dataframe returned from
	get_manifest_numbers(). 

	Returns: Dictionary like below:
	{'1':    train  val  total
	0      9    3     12
	1     14    1     15
	2     21    8     29,
	...
	'3':    train  val  total
	0      8    4     12
	1     11    4     15
	2     19   10     29}
	
	'''
	# get manifest numbers from each kfold in model group
	sm_kfold_dicts = {}
	for file in sm_path_list:
		# get kfold name
		kfold_num = re.findall(r'(?<=kfold)\d', file)[0]
		sm_kfold_dicts[kfold_num] = get_manifest_numbers(file)
	return sm_kfold_dicts

def avg_sm_split(sm_kfold_dicts):
	'''From slide manifest train/val number dictionaries pulled from each kfold, get avg & summarize.
	
	Args:
		sm_kfold_dicts (dict): Dict of DataFrames containing train, val, totals for each outcome, keys are kfold num. 
			Example: sm_kfold_dicts = get_manifest_numbers_mg(sm_paths)

	Returns: Dataframe like below with addtl total col (below is tabulate printed)
		train                val                   total
	--  -------------------  ------------------  -------
	0  8 (8/9/8/8/9)        4 (4/3/4/4/3)            12
	1  12 (13/12/12/11/14)  3 (2/3/3/4/1)            15
	2  21 (20/24/21/19/21)  8 (9/5/8/10/8)           29
	Example: 
		df = avg_sm_split(sm_kfold_dicts)
		from tabulate import tabulate
		print(tabulate(df, showindex=True, headers=df.columns, ))
	
	'''
	# get keys (kfolds) and put them in ascending order
	kfolds = list(sm_kfold_dicts.keys())
	kfolds.sort()
	
	# get training values
	train_vals = pd.concat([sm_kfold_dicts[i]['train'] for i in kfolds], axis=1)
	train_vals.columns = kfolds
	# find average
	train_vals['average'] = round(train_vals.mean(axis=1)).astype(int)
	# convert all to str
	train_vals = train_vals.applymap(str)
	# join entire thing into string with averages
	train_nums = train_vals['average'] + " (" + train_vals[kfolds].stack().groupby(level=0).apply('/'.join) + ")"

	# get validation values
	val_vals = pd.concat([sm_kfold_dicts[i]['val'] for i in kfolds], axis=1)
	val_vals.columns = kfolds
	# find average
	val_vals['average'] = round(val_vals.mean(axis=1)).astype(int)
	# convert all to str
	val_vals = val_vals.applymap(str)
	# join entire thing into string with averages
	val_nums = val_vals['average'] + " (" + val_vals[kfolds].stack().groupby(level=0).apply('/'.join) + ")" 

	# add up totals from avgs into new column "total"
	total_nums = train_vals['average'].astype(int) + val_vals['average'].astype(int)

	# get total train/validation numbers
	train_total = sum(train_vals['average'].astype(int))
	val_total = sum(val_vals['average'].astype(int))

	# concat into one df
	sm_nums = pd.concat([train_nums, val_nums, total_nums], axis=1).sort_index()
	sm_nums.columns = ["train","val","total"]

	return sm_nums, train_total, val_total

def get_sm_stats(model_dirs, label_outcomes=False):
	'''From model group model directories, get slide manifest paths, pull out slide manifests
	from each kfold, then find train/val numbers and averages for each kfold.
	If label_outcomes is True, dataframe index will be labeled with outcome values.

	Args:
		models_dirs (list): List to model directories for model group (typically crossvalidated experiment)
		label_outcomes (bool): Default False. Determines if returned sm_stats should be labeled with outcomes.

	Returns: 
		Tuple consisting of dataframe with columns for train, val, and total, where entries for train & val 
		contain the averaged number across all k-folds and then the individual k-fold numbers. 2nd and 3rd 
		tuple entries are total numbers for training & validation. 

	Example:
	(                       train                 val  total
	outcome                                                
	ERMS           8 (8/9/8/8/9)       slide4 (4/3/4/4/3)     12
	HGESS    12 (13/12/12/11/14)       3 (2/3/3/4/1)     15
	...
	PEComa   12 (13/13/12/11/13)       4 (3/3/4/5/3)     16
	UTROSCT      9 (9/10/10/8/9)       3 (3/2/2/4/3)     12, 122, 43)
	'''
	# get sm paths
	sm_path_list  = get_sm_paths(model_dirs)
	# get sm numbers for each kfold
	sm_kfold_dicts = get_manifest_numbers_mg(sm_path_list)
	# find averages and concatenate numbers to return train/val numbers dataframe
	sm_stats, train_total, val_total = avg_sm_split(sm_kfold_dicts)
	# add totals as last row
	sm_stats = pd.concat([sm_stats, pd.DataFrame([int(train_total), int(val_total), int(train_total+val_total)], 
                                  index=['train', 'val', 'total']).T], axis=0)
	# label with outcome labels if desired
	if label_outcomes is True:
		oc_lbls = get_params(model_dirs[0], ["outcome_labels"])['outcome_labels']
		oc_lbl_df = pd.DataFrame.from_dict(oc_lbls, orient='index', columns=['outcome'])
		oc_lbl_df.loc[len(oc_lbl_df.index)] = ['total']
		# concat oc_lbl_df to sm_stats reset index
		sm_stats = pd.concat([oc_lbl_df.reset_index(drop=True), sm_stats.reset_index(drop=True)], axis=1)

	return sm_stats

# ============================================================================ 
# MODEL GROUP & KFOLD FUNCTIONS FOR AGGREGATING METRICS

def find_results_kfolds(pred_csv_path_list, level):
	# TODO: may want to combine this function with the below as they both pretty much do the same thing.
	# I can also see it being prudent to just make everything numpy arrays
	'''Find metrics for all models (should be kfolds from crossval experiment). 
	Return a single dict with kfold numbers as keys and results as values.
	
	Args:
		pred_csv_path_list (list): List of paths to prediction CSVs.
		level (str): Prediction level (patient, slide, tile).

	Returns:
		kfold_dicts (dict): Dict with kfolds as str keys, values are nested dicts of metric 
		results for each outcome value, where those keys are int. 

	Example: See below find_crossval_plus_avg_results, but this one only returns individuak kfold results.
	'''	
	kfold_dicts = {}

	for file in pred_csv_path_list:
		# get kfold name
		kfold_num = re.findall(r'(?<=kfold)\d', file)[0]
		kfold_dicts[kfold_num] = get_metrics_from_pred(file, level=level)

	return kfold_dicts

def get_avg_crossval_results(pred_csv_path_list, level):
	# TODO: may want to combine this function with the below as they both pretty much do the same thing.
	'''Aggregate results from different kfolds into one CSV file and then find metrics for each outcome over 
	kfolds/models provided in pred_csv_path_list. Basically averaged crossval performance.
	Default metrics are : accuracy, sensitivity, specificity, recall, f1_score, kappa, AUROC, AUPRC, ConfusionMatrix

	Args:
		pred_csv_path_list (list): List of paths to prediction CSVs.
		level (str): Prediction level (patient, slide, tile).

	Returns:
		aggregated_crossval_metrics_dict (dict): Nested dict like below where second level nest is the integer 
		outcome values as keys and values are the averaged metrics.
	
	Example: See below find_crossval_plus_avg_results, but this one only returns averaged results.
	'''
	# aggregate results from different kfolds into one CSV file and then find metrics (basically averaged crossval performance)
	aggregated_pred_csv = combine_pred_csvs(pred_csv_path_list)

	# save temp file to use
	with tempfile.NamedTemporaryFile(mode='w', delete=False) as csvfile:
		aggregated_pred_csv.to_csv(csvfile)

	aggregated_crossval_metrics_dict = {}
	# get path of temp csv file using .name attribute & use to get metrics
	aggregated_crossval_metrics_dict["average"] = get_metrics_from_pred(csvfile.name, level=level)

	return aggregated_crossval_metrics_dict

def find_crossval_plus_avg_results(pred_csv_path_list, level):
	'''Essentially combines crossvalidated kfold results (from find_results_kfolds) plus averaged results 
	over kfolds (get_avg_crossval_results) into one dictionary.
	Alternative is get_avg_results, which returns a DataFrame. 
	
	Args:
		pred_csv_path_list (list): List of paths to prediction CSVs.
		level (str): Prediction level (patient, slide, tile).

	Returns:
		crossval_dict (dict): Nested dict like below where second level nest is the integer outcome values 
		as keys and values are the averaged metrics. Kfolds as str keys, values are nested dicts of metric 
		results for each outcome value, where those keys are int. 
	
	Example:
		{'average': {0: {'accuracy': 0.72,
			'sensitivity': 0.54,
			'specificity': 0.81,
			'precision': 0.61,
			'recall': 0.54,
			'f1_score': 0.57,
			'kappa': 0.36,
			'AUROC': 0.75,
			'AUPRC': 0.59,
			'ConfusionMatrix': array([[132489,  30827],
					[ 40379,  47313]])},
			1: {...},
			2: {...}},
		'3': {0: {'accuracy': 0.88, ...}
			1: {...},
			2: {...}},
		'2': {0: {'accuracy': 0.90, ...}
			1: {...},
			2: {...}},
		'1': {0: {'accuracy': 0.69, ...}
			1: {...},
			2: {...}}}
	'''
	# find both kfold & aggregated dicts and combine into one big dictionary
	kfold_dicts = find_results_kfolds(pred_csv_path_list, level)
	aggregated_crossval_metrics_dict = get_avg_crossval_results(pred_csv_path_list, level)
	crossval_dict = merge_two_dicts(kfold_dicts, aggregated_crossval_metrics_dict)

	return crossval_dict

def concat_results_df(kfold_dict, average_dict, metric_list=["AUROC","AUPRC","accuracy","precision","recall",
	"sensitivity","specificity","ConfusionMatrix"], outcome_labels=None):
	''' Find concatenated results dataframe of summarized averaged + kfold results and averaged results.
	Likely redundant with get_avg_results() below but this one is from the dictionary values.

	Args:
		kfold_dict (dict): Dict of kfold results from find_results_kfolds()
		average_dict (dict): Dict of averaged kfold results from get_avg_crossval_results().
		metric_list (list): Metrics you want to see summarized.
	 	outcome_labels (dict): Expects to be dictionary of numeric values mapping to outcome labels.
	
	Returns:
		result_df (DataFrame): Outcomes (rows) and metrics (cols) with each cell being the averaged value and 
			then the k-fold values in parentheses.
			Example:
					AUROC					AUPRC					ConfusionMatrix
				0	0.75 (0.8/0.74/0.74)	0.59 (0.7/0.5/0.6)		[[132489 30827] [ 40379 47313]] ([[44786 1...
				1	0.9 (0.9/0.94/0.85)		0.78 (0.77/0.84/0.73)	[[170750 18908] [ 19058 42292]] ([[56871 ...
				2	0.72 (0.74/0.7/0.74)	0.64 (0.61/0.67/0.67)	[[102819 46223] [ 36521 65445]] ([[37657 1..
		avg_df (DataFrame): Outcomes (rows) and metrics (cols) as averaged results across kfolds from kfold_dict.
			Example:
					AUROC	AUPRC	ConfusionMatrix
				0	0.75	0.59	[[132489, 30827], [40379, 47313]]
				1	0.9		0.78	[[170750, 18908], [19058, 42292]]
				2	0.72	0.64	[[102819, 46223], [36521, 65445]]
	
	'''
	# TODO make option to just summarize kfolds vs. averaged dictionary as well                                                                           
                                                             
	# convert kfold_dict to DataFrame
	df_list = [pd.DataFrame.from_records(kfold_dict[i]).T for i in list(kfold_dict.keys())]
	# example of what df_list looks like
	# temp = pd.DataFrame.from_records(m_dict)
	# temp.T
	#   accuracy sensitivity specificity precision recall f1_score kappa AUROC AUPRC     ConfusionMatrix
	# 0      1.0         1.0         1.0       1.0    1.0      1.0   1.0   1.0   1.0  [[10, 0], [0, 25]]
	# 1      1.0         1.0         1.0       1.0    1.0      1.0   1.0   1.0   1.0  [[25, 0], [0, 10]]								
	
	# average df only
	avg_df = pd.DataFrame.from_records(average_dict["average"]).T
	avg_df = avg_df[metric_list]

    # assume 3 kfolds                                                                                                                                     
    # https://stackoverflow.com/questions/39291499/how-to-concatenate-multiple-column-values-into-a-single-column-in-pandas-datafra                       
    # example = avg_df["AUPRC"].astype(str) + " (" + df_list[0]["AUPRC"].astype(str) + "/" + df_list[1]["AUPRC"].astype(str) + "/" + df_list[2]["AUPRC"].astype(str) + ")"                                                                                                                                              
	newcol_list = []
	for metric_name in metric_list:
		kfoldslist = avg_df[metric_name].astype(str) + " ("
		for i in range(len(df_list)):
			if i != (len(df_list)-1):
				kfoldslist = kfoldslist + df_list[i][metric_name].astype(str) + "/"
			else:
				kfoldslist = kfoldslist + df_list[i][metric_name].astype(str) + ")"
		newcol_list += [kfoldslist]

	# concat columns back into one another
	result_df = pd.concat(newcol_list, axis=1)

	# get outcome labels & use them to set index of result_df
	if outcome_labels is not None:
		# convert integer string to just int to allow for labeling on index
		outcome_labels = {int(k):v for k,v in outcome_labels.items()}
		oc_lbl_df = pd.DataFrame.from_dict(outcome_labels, orient='index', columns=['outcome'])
		result_df = result_df.set_index(oc_lbl_df['outcome'])
		avg_df = avg_df.set_index(oc_lbl_df['outcome'])

	return result_df, avg_df

def get_avg_results(models_dir, mg, level, metric_list=["AUROC","AUPRC","accuracy","precision","recall","sensitivity","specificity","ConfusionMatrix"]):
	'''For a given model group (a group of models, typically a k-fold cross-validated/bootstrapped experiment)
	at a certain desired result level (tile, slide, or patient), this function finds the average performance
	across the k-folds for a given list of metrics and for one set of hyperparameters.
	Alternative to find_crossval_plus_avg_results, which returns nested dictionaries. 

	Args:
		models_dir (str): Path to project models/ directory.
		mg (str): label for specific model group to get averaged results for 
			Ex: 'dx_short-299_ES_unbalanced_bootstrap-HP0'
		level (str): desired results level, either "tile", "slide", or "patient"
		metric_list (list): A list of desired metrics. Default are ["AUROC","AUPRC","accuracy","precision",
		"recall","sensitivity","specificity","ConfusionMatrix"]

	Returns:
		avg_df (Dataframe): Columns are metrics in given metric_list, rows are each outcome value,
		cells are the average performance across k-folds of the outcome for a given metric. Index is the 
		labeled outcomes.

	Example:
	        AUROC AUPRC accuracy precision recall sensitivity specificity        ConfusionMatrix
	outcome                                                              
	ERMS     0.99  0.91     0.96      0.65   0.79        0.79        0.97    [[185, 6], [3, 11]]
	HGESS    0.92  0.59     0.88      0.42   0.48        0.48        0.92  [[170, 14], [11, 10]]
	IMT      0.94  0.87     0.92      0.85   0.65        0.65        0.98   [[167, 4], [12, 22]]
	'''
	# search through model folders in models dir to get desired models from label (outcome-exp_label-HP#)                                                 
	model_dirs = get_model_names(models_dir, mg)                                                                                                          
	# use each model_dir to get the desired epoch                                                                                                         
	epochs = [find_max_epoch(mdl_dir) for mdl_dir in model_dirs]                                                                                          
	# cycle through each model dir and epoch                                                                                                              
	pred_path_list = flatten([get_pred_paths(mdl, level, ep) for mdl,ep in zip(model_dirs,epochs)])                                                   
	# for this model group, get the results using pred_path_list (which has 3 CSVs for a specific level & epoch)                                          
	kfold_dict = find_results_kfolds(pred_path_list, level)    
	# use kfold dicts to get average
	avg_dict = get_avg_crossval_results(pred_path_list, level)
	avg_df = pd.DataFrame.from_records(avg_dict['average']).T
	# filter for desired metrics
	if metric_list is not None:
		avg_df = avg_df[metric_list]
	oc_lbls = get_params(pred_path_list[0].split(level)[0], ["outcome_labels"])['outcome_labels']
	# convert integer string to just int to allow for labeling on index
	oc_lbls = {int(k):v for k,v in oc_lbls.items()}
	oc_lbl_df = pd.DataFrame.from_dict(oc_lbls, orient='index', columns=['outcome'])
	# avg_df = avg_df.set_index(oc_lbl_df['outcome']) # old way, where outcome labels are index values
	avg_df = pd.concat([oc_lbl_df, avg_df], axis=1)

	return avg_df

def get_eval_results(pred_csv_path, level, metric_list=["AUROC","AUPRC","accuracy","precision","recall","sensitivity","specificity","ConfusionMatrix"]):
	'''For a given prediction file from an evaluation model, return the evaluation performance metrics for a
	desired results level (tile, slide, patient) as a Pandas Dataframe.
	
	Args:
		pred_csv_path (str): Path to prediction file.
		level (str): desired results level, either "tile", "slide", or "patient"
		metric_list (list): A list of desired metrics. Default are ["AUROC","AUPRC","accuracy","precision",
		"recall","sensitivity","specificity","ConfusionMatrix"]

	Returns:
		eval_df (Pandas Dataframe): Columns are metrics in given metric_list, rows are each outcome value,
		cells are the evaluation performance of the outcome for a given metric. Index is the labeled outcomes.

	Example:
	        AUROC AUPRC accuracy sensitivity specificity    ConfusionMatrix
	outcome                                                                
	ERMS     0.97  0.86     0.92         0.8        0.94  [[30, 2], [1, 4]]
	HGESS    0.65  0.32     0.81         0.0        0.97  [[30, 1], [6, 0]]
	IMT      0.98  0.92     0.78         1.0        0.75  [[24, 8], [0, 5]]

	'''
	# find metrics for evaluation for a specific model
	eval_dict = get_metrics_from_pred(pred_csv_path, level)
	eval_df = pd.DataFrame.from_records(eval_dict).T
	# filter for desired metrics
	if metric_list is not None:
		eval_df = eval_df[metric_list]
	# outcome labels
	oc_lbls = get_params(pred_csv_path.split(level)[0], ["outcome_labels"])['outcome_labels']
	# convert integer string to just int to allow for labeling on index
	oc_lbls = {int(k):v for k,v in oc_lbls.items()}
	oc_lbl_df = pd.DataFrame.from_dict(oc_lbls, orient='index', columns=['outcome'])
	# eval_df = eval_df.set_index(oc_lbl_df['outcome']) # old way, has outcomes as index and not as a column
	eval_df = pd.concat([oc_lbl_df, eval_df], axis=1)

	return eval_df

def get_all_desired_results(models_dir, mg, level, metric_list=["AUROC","AUPRC","accuracy","precision","recall","sensitivity","specificity"]):
	'''Aggregate function to return all useful metrics, averaged across all models in model group, for desired metrics in metric list.
	Includes slide manifest numbers, averaged metrics, and confusion matrices.

	Args: 
		models_dir (str): Path to project's models directory.
		mg (str): Label of models corresponding to kfolds from cross-validated experiment.
		level (str): Patient, slide, tile.
		metric_list (list): List of metric names to return averaged results for. 
			Full list: "AUROC", "AUPRC", "accuracy", "precision", "recall", "f1", "kappa", "sensitivity", "specificity", "ConfusionMatrix"
	
	Returns:
		List containing the following 5 objects:
		1. sm_stats (DataFrame): DataFrame of train, val, and total cases numbers (in cols) for each outcome (rows). 
			outcome	train			val			total
			ERMS    8 (8/8/8)   	4 (4/4/4)   12
			HGESS   10 (10/10/10)   5 (5/5/5)   15
			Name: DataFrame, dtype: object]
		2. train_total (int): Total number of cases in training dataset.
		3. val_total (int): Total number of cases in validation dataset.
		4. result_df (DataFrame): Dataframe of outcomes (rows) and averaged metrics (cols) across models in model group.
			outcome	 AUROC					AUPRC
			ERMS     1.0 (0.98/1.0/1.0) 	0.95 (0.8/1.0/1.0)
			UTROSCT  0.88 (0.95/0.98/0.74)  0.62 (0.55/0.86/0.42)
			Name: DataFrame, dtype: object]
		5. cm_series (Series): Series object where index is outcome labels and elements are 2x2 Confusion Matrices for each outcome.
			outcome
			ERMS        [[146, 7], [0, 12]]
			UTROSCT      [[152, 1], [6, 6]]
			Name: ConfusionMatrix, dtype: object]
	'''
	# search through model folders in models dir to get desired models from label (outcome-exp_label-HP#)                                                 
	model_dirs = get_model_names(models_dir, mg)                                                                                                          
	# use each model_dir to get the desired epoch                                                                                                         
	epochs = [find_max_epoch(mdl_dir) for mdl_dir in model_dirs]                                                                                          
	# cycle through each model dir and epoch                                                                                                              
	pred_path_list = flatten([get_pred_paths(mdl, level, ep) for mdl,ep in zip(model_dirs,epochs)])                                                   
	# for this model group, get the results using pred_path_list (which has 3 CSVs for a specific level & epoch)                                          
	kfold_dict = find_results_kfolds(pred_path_list, level)                                                                                           
	# Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.                  
	avg_dict = get_avg_crossval_results(pred_path_list, level)                                                                                   
	oc_lbls = get_params(model_dirs[0], ["outcome_labels"])                                                                                               

	all_desired_results = {}

	# get slide manifest info
	all_desired_results['Slide Manifest'] = get_sm_stats(model_dirs, label_outcomes=True)

	# get concatenated result & averaged dfs
	result_df, avg_df = concat_results_df(kfold_dict, avg_dict, metric_list=metric_list, outcome_labels=oc_lbls['outcome_labels'])
	all_desired_results['Results DataFrame'] = result_df
                                                                                                         
	# pull out averaged CMs and display as its own dataframe where columns are outcomes, use tabulate to print table  
	if 'ConfusionMatrix' in metric_list:
		all_desired_results["Confusion Matrix"] = avg_df['ConfusionMatrix']

	return all_desired_results

def results_slide_tile_eval(avgd_slide, avgd_tile, eval_slide, eval_tile, metric_name):
    # TODO: reconcile with others, will use with compare_exp()
    '''Return crossval averaged results and eval results for a model, for a given metric (metric_name).'''
    df = pd.concat([avgd_slide[metric_name], eval_slide[metric_name], avgd_tile[metric_name], eval_tile[metric_name]], axis=1)
    df2 = df.set_axis(["TrainSlide", 'EvalSlide', 'TrainTile', 'EvalTile'], axis=1, inplace=False)
    return df2

# ============================================================================ #
####### ----- PRINT FUNCTIONS ----- ########

# TODO make GUI https://pbpython.com/dataframe-gui-overview.html
# TODO look at best ways to compare models

def print_cm_table(cm_series):
	''' Prints cm_series as tabulated table. Will print max 4 columns before rolling to second row. 

	Args:
		cm_series (Series): Series object pulled from ConfusionMatrix of results DataFrame, ConfusionMatrices for each outcome. 

	Returns:
		None, but print:
		+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
		| NTRK                        | PEComa                      | UTROSCT                     | LGESS                       |
		+=============================+=============================+=============================+=============================+
		| pred:yes  pred:no           | pred:yes  pred:no           | pred:yes  pred:no           | pred:yes  pred:no           |
		| true:yes       146        7 | true:yes       140       10 | true:yes       134        2 | true:yes       118       13 |
		| true:no          0       12 | true:no          7        8 | true:no          8       21 | true:no          6       28 |
		+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
	'''
	# first get unique outcomes                                                                                                                           
	idx = list(set(cm_series.index.values))                                                                                                               
	# pull out rows of series by labeled index, then sum, and convert back to Series                                                                      
	cm_srs = [pretty_confusionmatrix(cm) for cm in list(cm_series)]
	# print with tabulate    
	# make prettier printing by wrapping CMs longer then 4 
	if len(cm_srs) > 3:
		print(tabulate([cm_srs[:4]], headers=idx[:4], tablefmt='grid'))
		print(tabulate([cm_srs[4:]], headers=idx[4:], tablefmt='grid'))
	else:                                                                                                                      
		print(tabulate([cm_srs], headers=idx, tablefmt='grid'))

def print_results(models_dir, mg, level, metric_list=["AUROC","AUPRC","accuracy","precision","recall","sensitivity","specificity"]):
	'''Example of print out for print_results(models_dir, mg, "tile", metric_list=["AUROC","AUPRC","accuracy"])

	==================== RESULTS FOR dx_short-299_ES_unbalanced_crossval-HP0 ====================
	['/home/pearsonlab/PROJECTS/UCH_BENNETT/models/00095-dx_short-299_ES_unbalanced_crossval-HP0-kfold3', '/home/pearsonlab/PROJECTS/UCH_BENNETT/models/00093-
  	dx_short-299_ES_unbalanced_crossval-HP0-kfold1', '/home/pearsonlab/PROJECTS/UCH_BENNETT/models/00094-dx_short-299_ES_unbalanced_crossval-HP0-kfold2']

	TRAINING & VALIDATION NUMBERS
	outcome    train          val              total
	---------  -------------  -------------  -------
	ERMS       8 (8/8/8)      4 (4/4/4)           12
	HGESS      10 (10/10/10)  5 (5/5/5)           15
	IMT        19 (19/19/20)  10 (10/10/9)        29
	...
	UTROSCT    8 (8/8/8)      4 (4/4/4)           12
	Train split: 111, Val split: 54

	AVERAGED RESULTS
	outcome    AUROC                  AUPRC                  accuracy
	---------  ---------------------  ---------------------  ---------------------
	ERMS       0.97 (0.94/0.98/0.98)  0.75 (0.51/0.81/0.91)  0.93 (0.9/0.96/0.93)
	HGESS      0.81 (0.78/0.88/0.72)  0.36 (0.37/0.52/0.14)  0.87 (0.88/0.88/0.85)
	IMT        0.92 (0.92/0.9/0.94)   0.76 (0.77/0.75/0.79)  0.9 (0.88/0.91/0.9)
	...
	UTROSCT    0.89 (0.87/0.92/0.88)  0.58 (0.51/0.71/0.51)  0.94 (0.93/0.95/0.94)

	CONFUSION MATRICES (truncated)
	+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
	| UTROSCT                     | IMT                         | LMS                         | PEComa                      |
	+=============================+=============================+=============================+=============================+
	| pred:yes  pred:no           | pred:yes  pred:no           | pred:yes  pred:no           | pred:yes  pred:no           |
	| true:yes    215527    13583 | true:yes    209355    18209 | true:yes    196225     9243 | true:yes    186327    18454 |
	| true:no       3964    17870 | true:no      14244     9136 | true:no      16627    28849 | true:no      14270    31893 |
	+-----------------------------+-----------------------------+-----------------------------+-----------------------------+
	'''
	# generate results first; all_desired_results = [sm_stats, train_total, val_total, result_df, cm_series]      
	all_desired_results = get_all_desired_results(models_dir=models_dir, mg=mg, level=level, metric_list=metric_list)

	print("\n")                                                                                                                                    
	print(f'==================== RESULTS FOR {mg} ====================')                                                                                                                                                                                                               
	# print model names in group                                             
	model_dirs = get_model_names(models_dir, mg, printplz=True)                                                                                                          

	# get slide manifest info
	print("\nTRAINING & VALIDATION NUMBERS")
	print(tabulate(all_desired_results['Slide Manifest'], headers='keys', tablefmt='simple_grid'))
	#print(f"Train split: {all_desired_results[1]}, Val split: {all_desired_results[2]}")

	# print results_df    
	print("\nAVERAGED RESULTS")
	print(tabulate(all_desired_results['Results DataFrame'], headers='keys', tablefmt='simple_grid'))

	# get & print confusion matrices
	# TODO: choose to print slide or tile level
	if 'Confusion Matrix' in list(all_desired_results.keys()):
		print("\nCONFUSION MATRICES")                                                                                                                         
		# pull out averaged CMs and display as its own dataframe where columns are outcomes, use tabulate to print table                                                                                                                                             
		print_cm_table(all_desired_results['Confusion Matrix'])  

# TODO: may want to use dataframe display methods with 
def display_side_by_side(dfs:list, captions:list, tablespacing=5):
    from IPython.display import display, HTML
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
        

    Output:
    Displays in Markdown format inline iPython Jupyter notebooks:
            Foo                     Foo Bar                 FooBar
            A	B	C	D           A	B	C	D           E	F	G	H
        0	0	1	2	3       0	0	1	2	3       0	0	1	2	3
        1	4	5	6	7       1	4	5	6	7       1	4	5	6	7
        2	8	9	10	11      2	8	9	10	11      2	8	9	10	11 

    Example: 
        df1 = pd.DataFrame(np.arange(12).reshape((3,4)),columns=['A','B','C','D',])
        df2 = pd.DataFrame(np.arange(16).reshape((4,4)),columns=['A','B','C','D',])
        df3 = pd.DataFrame(np.arange(16).reshape((4,4)),columns=['E','F','G','H',])
        display_side_by_side([df1,df2,df3], captions=['Foo','Foo Bar',"FooBar"]) 

    Source:
        https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
    
    display_side_by_side([avg_results_slide_dx, eval_df_slide_dx, avg_results_tile_dx, eval_df_tile_dx], 
        ['Avg Slide', "Eval Slide", 'Avg Tile', 'Eval Tile'])
    """
    output = ""
    for (caption, df) in zip(captions, dfs):
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption).hide(axis='index').format(precision=2)._repr_html_()
        output += tablespacing * "\xa0"
    display(HTML(output))

    import dataframe_image as dfi
    
def export_df_png(df, save_path='df_styled.png'):
    """Export DataFrame object as a png. 
    
    Input:
        df (DataFrame): DataFrame object to be exported.
        save_path (str): Path name where to save file. 
        """
    # I can export the dataframes as PNGs to display them
    import dataframe_image as dfi
    df_styled = df.style.background_gradient().format(precision=2)
    dfi.export(df_styled, save_path)
    
# def compare_exp_results(model_labels):
# 	# TODO
	# TODO: See Sid's function
# 	'''For comparison of different experiments (i.e. altered training methods) for a particular outcome. 
# 	Takes a list of model group labels.
	
# 	'''
# 	#subgroup_diff_auroc = subgroup_bs_auroc.subtract(subgroup_uq_auroc) # want to make this for multiple AUROCs
# 	# AUROC
# 	# TrainSlide   -0.04
# 	# EvalSlide    -0.22
# 	# TrainTile    -0.01
# 	# EvalTile     -0.12
	

# 	print("Not done")


# ============================================================================ #
####### ----- MISCLASSIFICATION FUNCTIONS ----- ########

def get_misclassifications(pred_path):
	'''Get misclassified cases (tiles, slides or patients) along with predictions dataframe.
	Function figures out what level and outcome labels from the information in the eval/train directory,
	which is found from the prediction file path.

	Args:
		pred_path (str) file path to predictions file (CSV or Parquet) generated during model evaluation/validation. 
	
	Returns: 
		y_all (showing tile level return)
			patient  y_pred  y_true  Subgroup-y_pred0  Subgroup-y_pred1 Subgroup-y_pred2 misclassification  loc_x  loc_y       location  
		0     139       2       2          0.002550      3.173180e-04		0.997133           correct   1760   3095   [1760, 3095] 
		1     106       0       0          0.998573      4.274233e-07   	0.001427           correct   4657  10743  [4657, 10743]  
		2     205       0       0          0.996991      2.723996e-05   	0.002981           correct   2173   9377   [2173, 9377]  
		3      56       0       0          0.684632      3.152895e-01   	0.000078           correct  13952   3563  [13952, 3563]  
		4     101       1       1          0.012249      9.850478e-01   	0.002703           correct   1304   9501   [1304, 9501] 
	Column headers are slightly different depending on what level predictions are for. 
	Patient and tile level predictions have "patient" as identifier, slide level has "slide"
	Tile level predictions contain the additional location, loc_x, loc_y variables.
	'''
	# SLIDE LEVEL is different format, needs to be fixed
	level = pred_path.split("/")[-1].split("_")[0]
	preds = pd.read_parquet(pred_path)
	if level == "slide":
		cases = preds["slide"].astype(str)
	else:
		cases = preds["patient"].astype(str)
	preds_only = preds.filter(regex=("y_pred"))
	outcome = list(preds_only.columns)[0].split("-y_pred")[0]
	y_pred = preds_only.idxmax(1)
	f = lambda x: x.split("pred")[1]
	y_pred = y_pred.apply(f)
	y_pred = pd.Series.to_frame(y_pred)
	y_true = preds.filter(regex=("y_true"))
	y_true_name = outcome + "-y_true"
	y_all = pd.concat([cases,y_pred,y_true], axis=1)
	y_all = y_all.rename(columns={level:level,0:"y_pred",y_true_name:"y_true"})
	y_all["y_pred"] = y_all["y_pred"].astype(int)
	y_all["y_true"] = y_all["y_true"].astype(int)
	y_all['misclassification'] = np.where((y_all["y_pred"] == y_all["y_true"]), "correct", "wrong")
	y_all[list(preds_only.columns)] = preds_only
	if level == "tile":
		y_all[["loc_x","loc_y"]] = preds[["loc_x", "loc_y"]]
		y_all['location'] = y_all[["loc_x","loc_y"]].values.tolist()
		y_all['slide'] = preds['slide'].astype(str)
	return y_all

# function to get missing slides
def get_missing_slides_df(pred_path, extra_anns_cols=None):
	# TODO: need to make generalized, not UCH_BENNETT specific
	# NOTE: I think this honestly is REALLY redundant with the previous function... probably can get rid of
	'''From a predictions file, finds the misclassified [tiles/slides/patients] and returns a Dataframe
	of information, most importantly which lines are misclassified and the corresponding slide paths.
	Really, sort of serves as a wrapper to get_misclassifications().

	Args:
		extra_anns_cols can be a list of additional columns from the annotation file that one wishes to include
		in the final Dataframe.
	
	Returns:
	Dataframe equivalent to y_all from get_misclassifications() but with additional columns from annotation
	file, filter columns, and column for slide paths. 
	incorrect slides: a List object of the slide paths of the misclassified slides. Can be used to generate heatmaps.

	'''
	y_all = get_misclassifications(pred_path)
	model_dir = "/".join(pred_path.split("/")[:-1])
	outcome = get_params(model_dir, ["outcomes"])['outcomes'][0]
	# get annotations
	anns = pd.read_csv(get_params(model_dir, ["annotations"])['annotations'])
	# filter annotations using filters used for eval/training & subset df for impt cols only - https://stackoverflow.com/questions/34157811/filter-a-pandas-dataframe-using-values-from-a-dict
	filters = get_params(model_dir, ["filters"])['filters']
	ind = [True] * len(anns)
	for col, vals in filters.items():
		ind = ind & (anns[col].isin(vals))
	anns = anns[ind]
	# add in desired columns from annotation file
	if extra_anns_cols is not None:
		anns = anns[["patient","slide",outcome] + list(filters.keys()) + extra_anns_cols]
	else:
		anns = anns[["patient","slide",outcome] + list(filters.keys())]
	anns["patient"] = anns["patient"].astype(str)
	# get slide paths using params.json in model dir
	config = get_params(model_dir, ["dataset_config"])['dataset_config']
	source = get_params(model_dir, ["sources"])['sources'][0]
	f = open(config)
	slides_dir = json.load(f)[source]['slides']
	# join y_all to the anns info
	if "patient" in y_all.columns and "location" in y_all.columns:
		# tile level
		df = pd.merge(y_all,anns,how="left",on=["patient","slide"])
		df = df.astype({'patient': str, 'slide': str,'y_pred': int, 'y_true': int, 'loc_x': int, 'loc_y': int})
	elif "patient" in y_all.columns:
		# patient level
		df = pd.merge(y_all,anns,how="left",on=["patient"])
		df = df.astype({'patient': str, 'slide': str,'y_pred': int, 'y_true': int})
	else:
		# slide level
		df = pd.merge(y_all,anns,how="left",on=["slide"])
		df = df.astype({'patient': str, 'slide': str,'y_pred': int, 'y_true': int})
	# get list of slides
	slides_list = ["/" + s + "." for s in list(df['slide'])]
	slide_paths = [os.path.join(slides_dir, f) for f in os.listdir(slides_dir)]
	slide_matches = []
	# search slides_dir and match file name to slide name
	for s in slides_list:
		for f in slide_paths:
			if s in f:
				# join slides_dir and f to get full path
				slide_matches += [f]
	df['slide_path'] = slide_matches
	
	# get list of incorrectly classified (slides/patients/tiles) with slide paths
	incorrect_list = df[df["misclassification"] == "wrong"]["slide_path"].to_list()

	return df, incorrect_list


# ============================================================================ #
####### ----- HEATMAP FUNCTIONS ----- ########
def create_cmap_dict_top3(dfrow):
	'''Takes in a row of the predictions dataframe, finds the top three highest predicted values for the given 
	slide, and maps them to the red, blue, and green colorspaces.'''
	# from the outcome dataframe of predictions, we want to understand which were the top three most missed 
	rowpreds = dfrow.filter(regex=r'-y_pred', axis=1).transpose()
	top3 = [i.split("y_pred")[1] for i in rowpreds.sort_values(rowpreds.columns[0], ascending=False).index.tolist()]
	cmap = {'r': int(top3[0]), 'b': int(top3[1]), 'g': int(top3[2])}
	return cmap

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

# def create_cmap_function(logit_list):
# 	import seaborn as sns
# 	if len(logit_list) >= 12:
# 		sns_pal = sns.color_palette("Paired", len(logit_list))
# 	else:
# 		# using this should make it the same colors as the UMAPs
# 		sns_pal = sns.color_palette('hls', len(logit_list)) # may need to make specific heatmap

# 	max_index = np.argmax(logit_list, axis=0)
# 	#max_index = logit_list.index(max_value)
# 	return sns_pal[max_index]

def create_cmap(logit_list):
	'''Function to create a colormap from a logit list where each outcome is a unique color taken from a 
	seaborn color palette. Color palette is continuous, we use number of logits to create unique bins
	in spectrum and assign each outcome its own color at a specific interval.
	
	Args:
		logit_list (list): list of logit predictions for a given tile.

	Returns:
		cmap (dict): Custom color map
	'''
	# https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar 
	import seaborn as sns
	import matplotlib as mpl
	if len(logit_list) >= 12:
		sns_pal = sns.color_palette("Paired", len(logit_list))
	else:
		# using this should make it the same colors as the UMAPs
		sns_pal = sns.color_palette('hls', len(logit_list)) # may need to make specific heatmap
	cmap_list = [a + (1.0,) for a in list(sns_pal)]
	cmap = mpl.colors.LinearSegmentedColormap.from_list(
		'Custom cmap', cmap_list, len(logit_list))
	bounds = np.linspace(0, len(logit_list), len(logit_list)+1)
	norm = mpl.colors.BoundaryNorm(bounds, len(logit_list))
	return cmap

def prep_hm_display(incorrect_slide_row, sg_model_path_ft):
    # TODO: very ugly, not elegant, could use some work, but it works
    '''Preparing a display string for the heatmap of the incorrect slide, using the 
    incorrect_slide_row dataframe and the sg_model_path_ft model path.
    Example:
        prep_hm_display(sg_missing_df[sg_missing_df['slide_path'].isin([sg_incorrect_list[0]])], sg_model_path_ft)
    '''
    oc = get_params(sg_model_path_ft, ["outcomes"])['outcomes'][0]
    oc_lbls = get_params(sg_model_path_ft, ["outcome_labels"])['outcome_labels']
    oc_lbls = {int(k):v for k,v in oc_lbls.items()}
    temp = incorrect_slide_row.drop(columns=[oc, "Exclude", "External_Train_histo", "slide_path", "slide"])
    temp = temp.to_dict(orient='records')
    temp = temp[0]
    
    # clean up dictionary, names, and placements
    temp['Patient'] = temp.pop("patient")
    #temp['Slide'] = temp.pop("slide")
    temp["y_pred"] = oc_lbls[temp['y_pred']]
    temp["y_true"] = oc_lbls[temp['y_true']]
    # convert key "y_pred" to "Pred"
    temp["Pred"] = temp.pop("y_pred")
    # convert key "y_true" to "True"
    temp["True"] = temp.pop("y_true")
    # rename classification to "Classified"
    temp["Classified"] = temp.pop("misclassification")
    # find keys that contain "y_pred" in the dictionary, split on "y_pred" and take the second element, reassign to key
    for k in list(temp.keys()):
        if "-y_pred" in k:
            temp[oc_lbls[int(k.split("y_pred")[1])]] = round(temp.pop(k), 3)

    # convert dictionary to text string where each key starts a new line
    display_string = "\n".join([f"{k}: {v}" for k,v in temp.items()])

    return display_string

def save_custom_logit_hm(hm, outdir, model_path, text_box_display=None):
	'''Plots & saves a heatmap with a custom logit colormap overlaid. Each logit corresponds to an outcome 
	value and has been assigned a unique color from the seaborn colormap.
	
	Args:
		hm (class obj): Slideflow Heatmap class object.
		outdir (str): path to directory where to save heatmaps.
		model_path (str): path to model used to create Heatmap.

	Returns:
		Plots Heatmap with custom logit colormap overlaid + legend as well as raw Heatmap thumbnail. 
		Saves both in outdir location.
	
	'''
	### Code to create the new heatmap
	import matplotlib.colors as mcol
	import matplotlib.pyplot as plt

	# set up plot 
	# TODO: figure out if it should be horizontal or vertical, based on dimensions of slide
	# if hm.slide.dimensions[0] > hm.slide.dimensions[1]:
	# 	fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30, 20))
	# else:
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 20))

	oc = get_params(model_path, ["outcomes"])['outcomes'][0]
	plt.subplots_adjust(wspace=0, hspace=0.05)
	# set plot title
	plt.suptitle(f"Slide: {hm.slide.name} - {oc} Heatmap", fontsize=24)
	plt.subplots_adjust(top=0.94)

	# choose one of the axes to plot the annotated heatmap on & set important parameters
	ax = hm._prepare_ax(ax=axes[0])
	thumb_kwargs = dict(roi_color='k', linewidth=2)
	implot = hm.plot_thumbnail(ax=ax, show_roi=True, **thumb_kwargs) # ax.imshow
	heatmap_alpha: float = 0.6
	ax.set_facecolor("black")
	divnorm = mcol.TwoSlopeNorm(
		vmin=0,
		vcenter=0.5,
		vmax=1
	)

	# ### Create color grid & masked array
	# https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay
	# heatmap logits are currently in a 3 dimension numpy array (num tiles x axis * num tiles y axis * num logits)
	# want to make it 2D where logit prediction is equal at each 
	def get_color_grid(logit_list):
		# converting the logits into a single tile mapping
		if np.amax(logit_list) == -1:
			return np.nan
		else:
			# normalizing the prediction to make it continuous to fit it within a binned color map 
			return np.argmax(logit_list, axis=0) / len(logit_list) 

	color_grid = np.array([[get_color_grid(logit) for logit in row] for row in hm.predictions])
	
	masked_arr = np.ma.masked_where(
		color_grid == np.nan,
		color_grid)

	# create colormap using create_cmap() function
	cmap_new = create_cmap(hm.predictions[0,0,:].tolist())

	### set alphas -- currently not using
	# def get_alphas(logit_list):
	# 	# weight by softmax prediction value
	# 	return logit_list[np.argmax(logit_list, axis=0)]
	# weights = np.array([[get_alphas(logit) for logit in row] for row in hm.predictions])	
	# Create an alpha channel of linearly increasing values moving to the right.
	# alphas = np.ones(weights.shape)
	# alphas[:, 30:] = np.linspace(1, 0, 70)
	# ax.imshow(weights, alpha=alphas, **imshow_kwargs)

	# TODO need to maybe crop out just tissue regions --> probably too hard to do

	im = ax.imshow(
		masked_arr,
		norm=divnorm,
		extent=implot.get_extent(),
		cmap=cmap_new,
		alpha = heatmap_alpha,
		interpolation='none',
		zorder=10)

	if text_box_display:
		# set display string of predictions info
		# place a text box in upper left in axes coords
		props = dict(boxstyle='round', facecolor='white', alpha=1.0)
		ax.text(0.015, 0.98, text_box_display, transform=ax.transAxes, fontsize=16,
			verticalalignment='top', bbox=props)

	### Add in legend
	# https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
	import matplotlib.patches as mpatches
	# get the colors of the values, according to the 
	oc_lbls = get_params(model_path, ["outcome_labels"])['outcome_labels']
	# i.e. a sorted list of all values in data
	values = np.unique(masked_arr.ravel())
	# colormap used by imshow
	colors = [cmap_new(value) for value in values]
	# create a patch (proxy artist) for every color
	patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=oc_lbls[str(i)])) for i in range(len(values))[:-1]]
	# put those patched as legend-handles into the legend
	#f=plt.figure(0)
	# f=ax.get_figure()
	# f.legend(handles=patches, loc='upper left', bbox_to_anchor=(0.13, 0.44), borderaxespad=0.1, fontsize=14)
	ax.legend(handles=patches, loc='lower left', fontsize=16, framealpha=1.0, edgecolor='black')

	# plot the raw thumbnail
	def _savehmfig(label, bbox_inches='tight', **kwargs):
		plt.savefig(
			os.path.join(outdir, f'{hm.slide.name}-{label}.jpg'),
			bbox_inches=bbox_inches,
			**kwargs
		)

	thumb_kwargs = dict(roi_color='k', linewidth=2)
	hm.plot_thumbnail(show_roi=True, ax=axes[1], **thumb_kwargs)  # type: ignore
	#_savehmfig('raw')

	# check if outdir exists, else create it
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	plt.savefig(
		os.path.join(outdir, f'{hm.slide.name}-{oc}_overlaid.jpg'),
		bbox_inches='tight'
	)
	print(f"Saved {os.path.join(outdir, f'{hm.slide.name}-{oc}_overlaid.jpg')}")


# ============================================================================ #
# NOTE: really nice blog about visualization https://medium.com/geekculture/python-seaborn-statistical-data-visualization-in-plot-graph-f149f7a27c6e

####### ----- UMAP FUNCTIONS ----- ########
def label_umap(project, umap, model_path, label_name):
	'''Broad function to label UMAP with variety of labels. Edits the umap.data and assigns a new "label" column,
	which is then used by manual_umap_plot() to overlay matching colors on the UMAP plot. 
	Also will double-check labels and remove "" outcome value, if present.

	Label options include:
		- 'prediction': the logit class predictions given by the model for each tile.
		- 'misclassified': "correct" or "wrong" label depending on whether or not tile was predicted 
			accurately by the model. 
		- 'uncertainty': the uncertainty score for the tile (doesn't work with multicategorical models yet?)
		- label_name: any outcome column listed in the annotations file can be given as a string and will 
			be matched as the label to be overlaid.
	
	Args:
		project (class obj): Slideflow Project class object.
		umap (class obj): Slideflow's UMAP SlideMap class object. 
		model_path (str): Path to model.
		label name (str): desired label to overlay as color for each dot in UMAP projection.
		
	Returns:
		Modifies given umap object's data "label" column to contain labels to use when plotting.
		Modifies it in memory, returns no solid object.
	'''
	import seaborn as sns
	import matplotlib.pyplot as plt

    # use different strategy if label is just predictions
	if label_name == "prediction":
		print(label_name)
		if 'label' in umap.data.columns:
			umap.data.drop(columns='label')
		umap.data['label'] = umap.data["prediction"].values

	elif label_name == "misclassified":
		# get the slide mappings for the outcomes
		oc_label = get_params(model_path, ["outcomes"])['outcomes'][0]
		labels, unique = project.dataset().labels(oc_label)
		if "" in unique:
			new_labels = {k:(v-1) for (k,v) in labels.items()}
		while("" in unique):
			unique.remove("")
		# use new labels to label new "true" column
		umap.data['true'] = umap.data.slide.map(new_labels)
		umap.data['misclassified'] = np.where((umap.data["prediction"] == umap.data["true"]), "correct", "wrong")
		if 'label' in umap.data.columns:
			umap.data.drop(columns='label')
		umap.data['label'] = umap.data["misclassified"].values
    
	elif label_name == 'uncertainty' and umap.df.uncertainty:  # type: ignore
		# label with uncertainty
		uq_labels = np.stack(umap.data['uncertainty'].values)[:, 0]
		if 'label' in umap.data.columns:
			umap.data.drop(columns='label')
		umap.data['label'] = uq_labels
        
	else:
		# get labels
		labels, unique = project.dataset().labels(label_name)
		if "" in unique:
			new_labels = {k:(v-1) for (k,v) in labels.items()}
		while("" in unique):
			unique.remove("")
		unique_labels_dict = {k:v for k,v in enumerate(unique)}
		# now, use unique_labels_dict to label the new_labels from integer to string outcome value
		new_labels = {k:unique_labels_dict[v] for k,v in new_labels.items()}
		# label umap with adjusted labels
		umap.label_by_slide(new_labels)

# we are using umap.label_by_slide()
def manual_umap_plot(umap, title="test", plot_centroid=False, save_path=False, categorical=False):
	'''Plots 2D UMAP and legend corresponding to desired labels. 
	
	Args:
		umap (class obj): Slideflow UMAP class object.
		title (str): Title for UMAP plot.
		plot_centroid (bool): (not done) whether or not to plot centroid node.
		save_path (str): File path to save UMAP. If not provided, UMAP will not be saved.
		categorical (bool): Boolean for whether to treat the labels as categorical or not.

	Returns:
		Figure with UMAP.
		Plots an inline UMAP image. If save_path is given, will save said plot.

	'''
	# TODO: include ability to plot centroid over averages
	# TODO: maybe include heatmap cmap function here?
	# plotting
	import seaborn as sns
	import matplotlib.pyplot as plt

	# get the data from the UMAP
	plot_df = umap.data
	x = plot_df.x
	y = plot_df.y

	# parameters for UMAP plotting
	title = title
	xlim=(-0.05,1.05)
	ylim=(-0.05,1.05)

	# Make plot
	fig = plt.figure(figsize=(7.5, 4.5))
	ax = fig.add_subplot(111)

	if 'label' in plot_df.columns:
		labels = plot_df.label

        # Check for categorical labels
		if isinstance(categorical, dict):
			print("cmap provided, interpreting labels as categorical")
			hue=labels.astype('category')
			cmap = categorical
		elif (categorical is True
			or not pd.to_numeric(labels, errors='coerce').notnull().all()):
			print("Interpreting labels as categorical")
			hue=labels.astype('category')
			unique = list(labels.unique())
			unique.sort()

			# color palette
			if "wrong" in unique or "correct" in unique:
				sns_pal = sns.color_palette('hls', 12)
				cmap = {"correct": sns_pal[8], "wrong": sns_pal[0]}
			else: 
				if len(unique) >= 12:
					sns_pal = sns.color_palette("Paired", len(unique))
				else:
					sns_pal = sns.color_palette('hls', len(unique))
				cmap = {unique[i]: sns_pal[i] for i in range(len(unique))}
		else:
			print("Interpreting labels as continuous")
			hue=labels
			cmap=None

	# plot the UMAP
	umap_2d = sns.scatterplot(
		x=x,
		y=y,
		palette=cmap,
		ax=ax,
		hue=hue
	)
	# NOT DONE: "if you want to plot a second UMAP like centroid"

	# set other desired plot details
	ax.set_ylim(*((None, None) if not ylim else ylim))
	ax.set_xlim(*((None, None) if not xlim else xlim))
	#if 'hue' in scatter_kwargs:
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(
		loc='center left',
		bbox_to_anchor=(1, 0.5),
		ncol=1,
		#title=legend_title
		)
	umap_2d.set(xlabel="x", ylabel='y')
	if title:
		ax.set_title(title)

	if save_path:
		plt.savefig(save_path, bbox_inches='tight', dpi=300)
		print(f"Saved 2D UMAP to [green]{save_path}")

	return fig

def save_multi_image(filename):
    '''Utility function used to save multiple plots into one PDF.
    For as many figures have been created and can be accessed by plt, those figures will 
    all be saved into one PDF file. Useful for gathering all UMAPs for one model or 
	one slide's heatmaps together.

	Args:
		filename (str): save path for PDF.

	Returns:
		None. Simply saves a PDF.
	'''
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

# # alternative method
# from PIL import Image  # install by > python3 -m pip install --upgrade Pillow  # ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation
# images = [
#     Image.open("/Users/apple/Desktop/" + f)
#     for f in ["bbd.jpg", "bbd1.jpg", "bbd2.jpg"]
# ]
# pdf_path = "/Users/apple/Desktop/bbd1.pdf"
# images[0].save(
#     pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:]
# )

# TODO: Interactive UMAP functions - utilize James's code