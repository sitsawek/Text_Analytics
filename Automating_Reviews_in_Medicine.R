#Load data set from U.S. National Library of Medicine.
trials = read.csv("clinical_trial.csv",stringsAsFactors=FALSE)
str(trials)
summary(trials)
#Use nchar() to find how many charactors in longest abstract this data.
max(nchar(trials$abstract))
#How many search results provided no abstract?
table(nchar(trials$abstract)==0)
#Find the observation with the minimum number of characters in the title.
#What is the text of the title of this article?
which.min(nchar(trials$title))
trials$title[1258]
#Preparing the Corpus for title and abstract.
library(tm)
library(SnowballC)
corpus_title = Corpus(VectorSource(trials$title))
corpus_abstract = Corpus(VectorSource(trials$abstract))
#Covert both to lowercase.
corpus_title = tm_map(corpus_title,tolower)
corpus_abstract = tm_map(corpus_abstract,tolower)
#Remove punctuation of both.
corpus_title = tm_map(corpus_title,removePunctuation)
corpus_abstract = tm_map(corpus_abstract,removePunctuation)
#Remove English language stop words of both.
corpus_title = tm_map(corpus_title,removeWords,stopwords("english"))
corpus_abstract = tm_map(corpus_abstract,removeWords,stopwords("english"))
#Stem the words of both.
corpus_title = tm_map(corpus_title,stemDocument)
corpus_abstract = tm_map(corpus_abstract,stemDocument)
#Build a document term matrix of both.
dtm_title = DocumentTermMatrix(corpus_title)
dtm_abstract = DocumentTermMatrix(corpus_abstract)
#Limit dtm_title and dtm_abstract to terms with sparsenness of at most 95%
sparse_title = removeSparseTerms(dtm_title,0.95)
sparse_abstract = removeSparseTerms(dtm_abstract,0.95)
#Convert dtm_title and dtm_abstract to data frames.
dtm_title = as.data.frame(as.matrix(sparse_title))
dtm_abstract = as.data.frame(as.matrix(sparse_abstract))
sparse_abstract
mean(nchar(trials$title))
mean(nchar(trials$abstract))
#What is the most frequent word stem across all the abstracts?
which.max(colSums(dtm_abstract))
#Now combine dtm_title and dtm_abstract into a single data frame to make predictions.
colnames(dtm_title) = paste0("T", colnames(dtm_title))
colnames(dtm_abstract) = paste0("A", colnames(dtm_abstract))
#Using cbind(), combine dtm_title and dtm_abstract into a single data frame.
dtm = cbind(dtm_title,dtm_abstract)
#Add dependent variable "trial" to dtm.
dtm$trial = trials$trial
#Now that we have prepared our data frame.
library(caTools)
set.seed(144)
split = sample.split(dtm$trial,SplitRatio = 0.7)
train = subset(dtm,split == TRUE)
test = subset(dtm,split == FALSE)
#Baseline accuracy of training set.
table(train$trial)
(730)/(730+572)
#Now build CART model with all independent variable.
library(rpart)
library(rpart.plot)
mod_cart = rpart(trial~.,data = train ,method = "class")
prp(mod_cart)
#Let's predict in training set and find maximum of prediction.
pred_train = predict(mod_cart)
max(pred_train[,2])
#What is the training set accuracy of the CART model?
prob_train = pred_train[,2]
table(train$trial,prob_train >= 0.5)
(631+441)/(631+99+131+441)
#What is the training set sensitivity of the CART model?
#Sensitivity equal tp/(tp+fn)
(441)/(131+441)
#What is the training set specificity of the CART model?
#Specificity equal tn/(tn+fp)
(631)/(631+99)
#Evaluating the model on the testing set.
pred_test = predict(mod_cart,newdata = test)
prob_test = pred_test[,2]
#Find accuracy with testing set.
table(test$trial,prob_test >= 0.5)
(261+162)/(261+52+83+162)
#Using the ROCR package, what is the testing set AUC of the prediction model?
library(ROCR)
pred_rocr = prediction(prob_test,test$trial)
perf_rocr = performance(pred_rocr,"tpr","fpr")
plot(perf_rocr,colorize = TRUE)
performance(pred_rocr,"auc")@y.values
