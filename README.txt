This is the famous Titanic dataset from Kaggle. Our goals are the following for this problems:

1. Taking care of the all the missing data of passengers and also making the data in proper format.
   Then apply various Machine learning and Deep learning algorithms such as kernel SVM, Random forest classifier, 
   Neural Networks etc. for predicting with good accuracy whether a passenger survived or not 
   for the test dataset based on the given inputs. This part is a classification problem.

2. In this part, we are creating some clusters of all the survivors based on the input variables. Also we are creating
   some histograms of survived passengers based on various input variables to understand which category of people were more likely to       survive. 

Predicted results are given below:

Kernel SVM Classifier:
(C = 1, gamma = 0.2, kernel = 'rbf')

Accuracy of trainingset (with k-fold cross validation) = 83.3932584269663 %
Accuracy of testset = 88.27751196172248 %

-------------------------------------------------------------

Random Forest Classifier:
(n_estimators = 200, criterion = entropy)

Accuracy of trainingset (with k-fold cross validation) = 81.71660424469414 %
Accuracy of testset = 81.10047846889952

------------------------------------------------------------

Artificial Neural Networks (ANN):
(with no. of neurons in the 1st hidden layer = 25, no. of neurons in the 2nd hidden layer = 25,
no. of neurons in the 3rd hidden layer = 25, no. of neurons in the 4th hidden layer = 25
batch_size = 20, nb_epoch = 480)

Accuracy of trainingset = 90.46%
Accuracy of testset = 86.60287081339713 %
Mean accuracy of trainingset (using k-fold cross validation) = 80.4731571674347 %
-----------------------------------------------

As you can see from the above, we achived good accuracy of results for testset data and training dataset both.
Kernel SVM and ANN, both of them have done a good job.

-------------------------------------------------

From 2nd part of results......following are observed:

1. Out of 494 survived passengers, around 340 passengers belong to Cluster1 (cluster number = 0 in the image) and they survived more
2. Passengers in the age group of 25-33 years survived more
3. Female passengers survived more as compared to male passengers
4. Passengers in first class and 3rd class survived more
5. Passengers with no Siblings/Spouses aboard or less no. of Siblings/Spouses aboard survived more
5. Passengers with no Parents/Children aboard or less no. of Parents/Children aboard survived more
6. Passengers with fare 50 dollars or less survived more
7. Passengers who embarked at Southampton port survived more



--------------Additional Information on the datasets-----------------------------------------------------

survival - Survival (0 = No; 1 = Yes)
class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
name - Name
sex - Sex
age - Age
sibsp - Number of Siblings/Spouses Aboard
parch - Number of Parents/Children Aboard
ticket - Ticket Number
fare - Passenger Fare
cabin - Cabin
embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat - Lifeboat (if survived)
body - Body number (if did not survive and body was recovered)




