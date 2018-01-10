import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
import time


#Data partition and normailzation 
mnist = fetch_mldata('MNIST original')
#Preprossing: normalize data 
normailzed_mnist =  preprocessing.normalize(mnist.data, norm='l2')
train_img, test_img, train_label, test_label = train_test_split(normailzed_mnist, mnist.target, test_size=1/7.0, random_state=0)


solve = ['newton-cg','lbfgs', 'sag']
score = []


run_time = []
val_score = []
#Split date to five set by random choosing 
# set1   
# sl1  set1 label
tmp_set,set1,tmp_sl,sl1 = train_test_split(train_img, train_label, test_size=1/3.0, random_state=0)
tmp_set,set2,tmp_sl,sl2 = train_test_split(tmp_set, tmp_sl, test_size=1/2.0, random_state=0)
set3,sl3 = tmp_set , tmp_sl

validations = [set1,set2,set3]
val_labels = [sl1,sl2,sl3]

#Generate val set and train set
for j in range(3):
	start = time.time()
	val_set = validations[j]
	vsl = val_labels[j]
	train_set = []
	tsl = []
	pointer = False
	for k in range(3):
		if k!=j: 
			if pointer == False:
				train_set = validations[k]
				tsl = val_labels[k]
				pointer = True
				continue
			train_set = np.concatenate((train_set, validations[k]), axis=0)
			tsl = np.concatenate((tsl, val_labels[k]), axis=0)
	print (val_score)
	logisticRegr = LogisticRegression(penalty = 'l2',solver = solve[j])
	logisticRegr.fit(train_set, tsl)
	val_score .append(logisticRegr.score(val_set, vsl))
	end = time.time()
	run_time.append(end-start)
print(val_score)
print("Train Time: ",run_time)


#Best Model Setected for Test
#'newton-cg' : Convergence Warning: newton-cg failed to converge. Increase the number of iterations.
# 'liblinear' : For two classes Classification, not multiple classes

best_model = solve[run_time.index(min(run_time))]
print (best_model)
logisticRegr = LogisticRegression(penalty = 'l2',solver = best_model)
logisticRegr.fit(train_img, train_label)
# Make predictions on entire test data
test_pred = logisticRegr.predict(test_img)
#Compute for Test Score
test_score = logisticRegr.score(test_img, test_label)
print ('Test Score' + ": " + str(test_score))

#System Performance 
Performance = metrics.confusion_matrix(test_label, test_pred)
plt.figure(figsize=(9,9))
sns.heatmap(Performance, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(test_score)
plt.title(all_sample_title, size = 15);
plt.savefig(best_model + '_ConfusionSeabornCodementor.png')
#plt.show();



#Check for the dataset
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_label[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.savefig(best_model + '_d.png')

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
plt.savefig(best_model + '_miss.png')

