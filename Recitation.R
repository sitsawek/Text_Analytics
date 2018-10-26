#Load data
emails = read.csv("energy_bids.csv",stringsAsFactors = FALSE)
str(emails)
emails$email[1]
#Sort of display
strwrap(emails$email[1])
#We can take a look at the value in the responsive.
emails$responsive[1]
strwrap(emails$email[2])
emails$responsive[2]
table(emails$responsive)
#Now construct and preprocess the corpus.
library(tm)
corpus = Corpus(VectorSource(emails$email))
#We can output the first email in the corpus.
strwrap(corpus[[1]])
#Now convert corpus to lowercase.
corpus = tm_map(corpus,tolower)
#Remove punctuation.
corpus = tm_map(corpus,removePunctuation)
#Remove stop words.
corpus = tm_map(corpus,removeWords,stopwords("english"))
#Stem the document.
corpus = tm_map(corpus,stemDocument)
strwrap(corpus[[1]])
#Now let's build the document term matrix for our corpus.
dtm = DocumentTermMatrix(corpus)
dtm
#We want to remove the terms that don't appear too often in our data set.
#And we going to have to determine the sparsity.
#We'll remove any term that doesn't appear in at least 3% of the documents.
dtm = removeSparseTerms(dtm, 0.97)
#Now build data frame.
labeled_terms = as.data.frame(as.matrix(dtm))
#We need to add outcome variable.
labeled_terms$responsive = emails$responsive
str(labeled_terms)
#Now spilt data to training set and testing set.
library(caTools)
set.seed(144)
split = sample.split(labeled_terms$responsive,SplitRatio = 0.7)
train = subset(labeled_terms,split == TRUE)
test = subset(labeled_terms,split == FALSE)
#Now build classification and regression tree.
library(rpart)
library(rpart.plot)
email_cart = rpart(responsive~.,data = train,method = "class")
prp(email_cart)
pred_cart = predict(email_cart,newdata = test)
pred_cart[1:10,]
# 0 that mean probability of non-responsive and 1 mean probability of responsive.
# In our case we want to extract the predicted probability of the document being responsive.
pred_prob = pred_cart[,2]
pred_prob
#We cutoffs of 0.5.
table(test$responsive,pred_prob >= 0.5)
#Find accuracy of model.
(195+25)/(195+20+17+25)
#Find accuracy of baseline.
table(test$responsive)
215/(215+42)
#Look at ROC curve we can understand the performance of our model at different cutoffs.
library(ROCR)
pred_rocr = prediction(pred_prob,test$responsive)
pred_rocr
perf_rocr = performance(pred_rocr,"tpr","fpr")
plot(perf_rocr,colorize =TRUE)
#We can see true positive rate around 70% and false positive rate 20%
#Now compute AUC value.
performance(pred_rocr,"auc")@y.values
