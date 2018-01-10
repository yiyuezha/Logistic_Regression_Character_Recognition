import numpy as np 
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt




#Data partition and normailzation 
mnist = fetch_mldata('MNIST original')

#Preprocessing: normalize data
normailzed_mnist =  preprocessing.normalize(mnist.data, norm='l2')
train_img, test_img, train_label, test_label = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

'''
list_hog_fd = []
for feature in mnist.data:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')
'''

	
#Split date to five set by random choosing 
# set1   
# sl1  set1 label
tmp_set,set1,tmp_sl,sl1 = train_test_split(train_img, train_label, test_size=1/2.0, random_state=0)
set2,sl2 = tmp_set , tmp_sl

validations = [set1,set2]
val_labels = [sl1,sl2]

#Liblinear is for two class classification
#So I construct 10 models
#model 0 is to classify if it is 0 or not
#model 1 is to classify if it is 1 or not
#model 2 is to classify if it is 2 or not
#and so on.......
#For other solver on model is surfficient 
val_score = []

run_time = []
print ("here")
for j in range(2):
	start = time.time()
	val_set = validations[j]
	vsl = val_labels[j]
	train_set = []
	tsl = []
	pointer = False
	for k in range(2):
		if k!=j: 
			if pointer == False:
				train_set = validations[k]
				tsl = val_labels[k]
				pointer = True
				continue
			train_set = np.concatenate((train_set, validations[k]), axis=0)
			tsl = np.concatenate((tsl, val_labels[k]), axis=0)
	logreg = {}
	#Build the 10 models
	for m in range(10):
		new_tsl = []
		for l in tsl:
			if l == m: new_tsl.append(1)
			else: new_tsl.append(0)
		if j == 0: logreg[m] = LogisticRegression(penalty="l2", solver = 'liblinear')
		else: logreg[m] = LogisticRegression(penalty="l1", solver = 'liblinear')
		logreg[m].fit(train_set, new_tsl)
	print (val_score)

	#Pass validation set to 10 models and pick the one with highest confident score
	val_pred = []
	for sample in val_set:
		tmp_score =[]
		for m in range(10):
			res = logreg[m].decision_function(sample)
			tmp_score.append(sum(res))
		val_pred.append(tmp_score.index(max(tmp_score)))
	loc_score = 0
	for c in range(len(val_pred)):
		if val_pred[c]== vsl[c] : loc_score+=1
	loc_score /= len(val_pred)
	val_score.append(loc_score)
	end = time.time()	
	run_time.append(end-start)

print('L1' + ':' + str(val_score[1]))
print('L2' + ':' + str(val_score[0]))
print('Train Time: ', run_time)



#Best Model Setected for Test
#'newton-cg' : Convergence Warning: newton-cg failed to converge. Increase the number of iterations.
# 'liblinear' : For two classes Classification, not multiple classes, see project_lin
best_model = val_score.index(max(val_score))
logreg = {}
for m in range(10):
	new_train_l = []
	for l in train_label:
		if l == m: new_train_l.append(1)
		else: new_train_l.append(0)
	if best_model == 0: logreg[m] = LogisticRegression(penalty="l2", solver = 'liblinear')
	else: logreg[m] = LogisticRegression(penalty="l1", solver = 'liblinear')
	logreg[m].fit(train_img, new_train_l)

#Pass test set to 10 models and pick the one with highest confident score
test_pred = []
for sample in test_img:
	which = []
	for m in range(10):
		res = logreg[m].decision_function(sample)
		which.append(sum(res))
	test_pred.append(which.index(max(which)))
#Evaluate the 10 models
test_score = 0
for c in range(len(test_pred)):
	if test_pred[c]== test_label[c] : test_score+=1
test_score /= len(test_label)



#System Performance 
Performance = metrics.confusion_matrix(test_label, test_pred)
plt.figure(figsize=(9,9))
sns.heatmap(Performance, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(test_score)
plt.title(all_sample_title, size = 15);
plt.savefig('Liblinear_ConfusionSeabornCodementor.png')




#Check for the dataset
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_label[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.savefig('Liblinear_d.png')

#Check Miss Classificated Samples
index = 0
misclassifiedIndexes = []
for label, predict in zip(test_label, test_pred):
    if label != predict: 
        misclassifiedIndexes.append(index)
    index +=1
plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(test_img[badIndex], (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(test_pred[badIndex], test_label[badIndex]), fontsize = 15)
plt.savefig('Liblinear_miss.png')

print('L1' + ':' + str(val_score[1]))
print('L2' + ':' + str(val_score[0]))
print('Train Time: ', run_time)
print('liblinear test score' + ': ' + str(test_score))

		

