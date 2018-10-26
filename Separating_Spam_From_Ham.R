#Load data set from publicly available dataset first described in the 2006 conference paper.
emails = read.csv("emails.csv",stringsAsFactors = FALSE)
which.min(nchar(emails$text))
#Preparing the Corpus.
corpus = Corpus(VectorSource(emails$text))
#Convert to lowercase.
library(tm)
library(SnowballC)
corpus = tm_map(corpus,tolower)
#Remove punctuation.
corpus = tm_map(corpus,removePunctuation)
#Remove stop words.
corpus = tm_map(corpus,removeWords,stopwords("english"))
#Conver stem words in corpus.
corpus = tm_map(corpus,stemDocument)
#Build document term matrix.
dtm = DocumentTermMatrix(corpus)
#Limit dtm to terms with sparsenness of at least 5%
spdtm = removeSparseTerms(dtm,0.95)
#Now let's convert the sparse matrix into a data frame.
email_sparse = as.data.frame(as.matrix(spdtm))
#Let's run the make.names function to make sure all of our words are appropiate variable names.
colnames(email_sparse) = make.names(colnames(email_sparse))
#ColSums() is an R function that returns the sum of values for each variable in our data frame.
#What is the word stem that shows up most frequently across all the emails in the dataset?
which.max(colSums(email_sparse))
#Add a variable called "spam" to email_sparse containing the email spam labels.
email_sparse$spam = emails$spam
#How many word stems appear at least 5000 times in the ham emails in the dataset?
ham = subset(email_sparse,spam==0)
table(colSums(ham)>=5000)        
sort(colSums(ham))  
#How many word stems appear at least 1000 times in the spam emails in the dataset?
#Not include spam variable we just add.
spam = subset(email_sparse,spam==1)
table(colSums(spam)>=1000)
sort(colSums(spam))  
#Building machine learning models.
#First convert the dependent variable to a factor.
email_sparse$spam = as.factor(email_sparse$spam)
#Split data to training set and testing set.
library(caTools)
set.seed(123)
split = sample.split(email_sparse$spam,SplitRatio = 0.7)
train = subset(email_sparse,split == TRUE)
test = subset(email_sparse,split == FALSE)
#Build logistic regression model.
mod_log = glm(spam~.,data = train,family = binomial)
#How many of the training set predicted probabilities from spamLog are less than 0.00001?
pred_log = predict(mod_log,type = "response")
sum(pred_log < 0.00001)
#How many of the training set predicted probabilities from spamLog are more than 0.99999?
sum(pred_log > 0.99999)
#How many of the training set predicted probabilities from spamLog are between 0.00001 and 0.99999?
nrow(train)-sum(pred_log < 0.00001)-sum(pred_log > 0.99999)
#How many variables are labeled as significant (at the p=0.05 level) in the logistic regression summary output?
summary(mod_log)
#What is the training set accuracy of mod_log, using a threshold of 0.5 for predictions?
table(train$spam,pred_log >= 0.5)
(3052+954)/(3052+0+4+954)
#What is the testing set accuracy of mod_log, using a threshold of 0.5 for predictions?
pt_log = predict(mod_log,newdata = test,type = "response")
table(test$spam,pt_log >= 0.5)
(1257+376)/(1257+51+34+376)
#What is the training set AUC of mod_log?
pred_rocr = prediction(pred_log,train$spam)
perf_rocr = performance(pred_rocr,"tpr","fpr")
plot(perf_rocr,colorize = TRUE)
performance(pred_rocr,"auc")@y.values
#What is the testing set AUC of mod_log?
pred_rocr = prediction(pt_log,test$spam)    
perf_rocr = performance(pred_rocr,"tpr","fpr")
plot(perf_rocr,colorize=TRUE)
performance(pred_rocr,"auc")@y.values
#Build CART model.
library(rpart)
library(rpart.plot)
mod_cart = rpart(spam~.,data = train,method = "class")
pred_cart = predict(mod_cart)
prob_cart = pred_cart[,2]
#What is the training set accuracy of mod_cart?
table(train$spam,prob_cart>=0.5)
(2885+894)/(2885+167+64+894)
#What is the testing set accuracy of mod_cart?
pt_cart = predict(mod_cart,newdata = test)
prt_cart = pt_cart[,2]
table(test$spam,prt_cart >= 0.5)
(1228+386)/(1228+80+24+386)
#What is the training set AUC of mod_cart?
pred_rocr = prediction(prob_cart,train$spam)
perf_rocr = performance(pred_rocr,"tpr","fpr")
plot(perf_rocr,colorize=TRUE)
performance(pred_rocr,"auc")@y.values
#What is the testing set AUC of mod_cart?
pred_rocr = prediction(prt_cart,test$spam)
perf_rocr = performance(pred_rocr,"tpr","fpr")
plot(perf_rocr,colorize = TRUE)
performance(pred_rocr,"auc")@y.values
#How many of the word stems "enron", "hou", "vinc", and "kaminski" appear in the CART tree? 
prp(mod_cart)
#Build random forest model.
set.seed(123)
library(randomForest)
mod_rf = randomForest(spam~.,data = train)
#What is the training set accuracy of mod_rf?
pred_rf = predict(mod_rf,type = "prob")
prob_rf = pred_rf[,2]
table(train$spam,prob_rf >= 0.5)
(3017+917)/(3017+35+41+917)
#What is the testing set accuracy of mod_rf?
pt_rf = predict(mod_rf,newdata = test,type= "prob")
prt_rf = pt_rf[,2]
table(test$spam,prt_rf >= 0.5)
(1290+385)/(1290+18+25+385)
#What is the training set AUC of mod_rf?
pred_rocr = prediction(prob_rf,train$spam)
perf_rocr = performance(pred_rocr,"tpr","fpr")
plot(perf_rocr,colorize = TRUE)
performance(pred_rocr,"auc")@y.values
#What is the testing set AUC of mod_rf?
pred_rocr = prediction(prt_rf,test$spam)
perf_rocr = performance(pred_rocr,"tpr","fpr")
plot(perf_rocr,colorize = TRUE)
performance(pred_rocr,"auc")@y.values
#Conclusions.
#Both CART and random forest had very similar accuracies on the training and testing sets. #However, logistic regression obtained nearly perfect accuracy and AUC on the training set. 
#And had far-from-perfect performance on the testing set.
#This is an indicator of overfitting.