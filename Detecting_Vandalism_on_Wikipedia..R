#Load data from wikipedia.
wiki = read.csv("wiki.csv",stringsAsFactors = FALSE)
wiki$Vandal = as.factor(wiki$Vandal)
str(wiki)
table(wiki$Vandal)
#We need to convert our wiki to a corpus for pre-processing.
#Now to pre-process out text data that we can use the bag of words approach.
library(tm)
library(SnowballC)
corpusAdded = Corpus(VectorSource(wiki$Added))
#Now remove english stop words.
corpusAdded[[1]]$content
corpusAdded = tm_map(corpusAdded,removeWords,(stopwords("english")))
#Now stem the words.
corpusAdded = tm_map(corpusAdded,stemDocument)
#Build DocumentTermMatrix and call dtmAdded.
dtmAdded = DocumentTermMatrix(corpusAdded)
dtmAdded
#Now filter out sparse terms by keeping only terms that appear in 0.3%
sparseAdded = removeSparseTerms(dtmAdded,0.997)
sparseAdded
#Convert sparseAdded to a data frame called wordsAdded, and then prepend all the words with the letter A.
wordsAdded = as.data.frame(as.matrix(sparseAdded))
colnames(wordsAdded) = paste("A", colnames(wordsAdded))
#Now repeat all of the steps we've done so far to create a Removed bag-of-words dataframe.
corpusRemoved = Corpus(VectorSource(wiki$Removed))
#Now remove english stop words.
corpusRemoved[[1]]$content
corpusRemoved = tm_map(corpusRemoved,removeWords,(stopwords("english")))
#Now stem the words.
corpusRemoved = tm_map(corpusRemoved,stemDocument)
#Build DocumentTermMatrix and call dtmAdded.
dtmRemoved = DocumentTermMatrix(corpusRemoved)
dtmRemoved
#Now filter out sparse terms by keeping only terms that appear in 0.3%
sparseRemoved = removeSparseTerms(dtmRemoved,0.997)
sparseRemoved
#Removed bag-of-words to a data frame called wordsRemoved, and then prepend all the words with the letter R.
wordsRemoved = as.data.frame(as.matrix(sparseRemoved))
colnames(wordsRemoved) = paste("R", colnames(wordsRemoved))
#Combine the two data frames into a data frame called wikiWords.
wikiWords = cbind(wordsAdded,wordsRemoved)
#Then add the Vandal column.
wikiWords$Vandal = wiki$Vandal
#Now split data to training set and test set.
library(caTools)
set.seed(123)
split = sample.split(wikiWords$Vandal,SplitRatio = 0.7)
train = subset(wikiWords,split == TRUE)
test = subset(wikiWords,split == FALSE)
#Find accuracy of baseline in test set.
table(test$Vandal)
(618)/(618+545)
#Build CART model to predict vandal.
library(rpart)
library(rpart.plot)
mod_cart = rpart(Vandal~.,data = train,method = "class")
pred_cart = predict(mod_cart,newdata= test)
#If weu add the argument type="class" when making predictions, the output of predict will automatically use a threshold of 0.5
pred_cart
# 0 that mean probability of not-valdalism and 1 mean probability of vandalism.
# In our case we want to extract the predicted probability of the document being vandalism.
pred_prob = pred_cart[,2]
table(test$Vandal,pred_prob >= 0.5)
#Find accuracy of prediction CART.
(614+19)/(614+4+526+19)
prp(mod_cart)
#Now compare with baseline.
table(test$Vandal)
(618)/(618+545)
#CART can beats the baseline but not predictive.
#More specifically, the words themselves were not useful.
#We will try two techniques - identifying a key class of words, and counting words.
#We can search for the presence of a web address in the words added.
#By searching for "http" in the Added column.
#The grepl function returns TRUE if a string is found in another string, e.g.
grepl("cat","dogs and cats",fixed=TRUE) # TRUE
grepl("cat","dogs and rats",fixed=TRUE) # FALSE
#Create a copy of dataframe from the previous.
wikiWords2 = wikiWords
#Make a new column in wikiWords2 that is 1 if "http" was in Added.
wikiWords2$HTTP = ifelse(grepl("http",wiki$Added,fixed=TRUE), 1, 0)
table(wikiWords2$HTTP)
# Use that variable (do not recompute it with sample.split) to make new training and testing sets.
train2 = subset(wikiWords2, split==TRUE)
test2 = subset(wikiWords2, split==FALSE)
#Then create a new CART model using this new variable as one of the independent variables.
mod_cart2 = rpart(Vandal~.,data = train2,method = "class")
pred_cart2 = predict(mod_cart2,newdata = test2)
pred_prob2 = pred_cart2[,2]
table(test2$Vandal,pred_prob2 >= 0.5)
#Compute accuracy.
(605+64)/(605+13+481+64)
#Another possibility is that the number of words added and removed is predictive.
#Perhaps more so than the actual words themselves.
#Sum the rows of dtmAdded and dtmRemoved and add them as new variables in data frame wikiWords2.
wikiWords2$NumWordsAdded = rowSums(as.matrix(dtmAdded))
wikiWords2$NumWordsRemoved = rowSums(as.matrix(dtmRemoved))
mean(wikiWords2$NumWordsAdded)
#Split data on wikiWords2 to training set and testing set.
train3 = subset(wikiWords2,split == TRUE)
test3 = subset(wikiWords2,split == FALSE)
#Build new model.
mod_cart3 = rpart(Vandal~.,data=train3,method = "class")
pred_cart3 = predict(mod_cart3,newdata = test3)
pred_prob3 = pred_cart3[,2]
#Now compute accuracy of mod_cart3
table(test3$Vandal,pred_prob3 >= 0.5)
(514+248)/(514+104+297+248)
#Using Non-Textual Data.
wikiWords3 = wikiWords2
#Then add the two original variables Minor and Loggedin to this new data frame.
wikiWords3$Minor = wiki$Minor
wikiWords3$Loggedin = wiki$Loggedin
#Split data on wikiWords3 to training set and testing set.
train4 = subset(wikiWords3,split == TRUE)
test4 = subset(wikiWords3,split == FALSE)
#Build new model.
mod_cart4 = rpart(Vandal~.,data = train4,method = "class")
pred_cart4 = predict(mod_cart4,newdata = test4)
pred_prob4 = pred_cart4[,2]
#Now compute accuracy of mod_cart4
table(test4$Vandal,pred_prob4 >= 0.5)
(595+241)/(595+23+304+241)
#Plot CART tree to see how many splits.
prp(mod_cart4)
