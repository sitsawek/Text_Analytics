#Load data from tweets.
#We need to add extra argument when working on a text analytics problem.
#That the text is read in properly.
tweets = read.csv("tweets.csv",stringsAsFactors=FALSE)
str(tweets)
#We will see text of tweets and average sentiment score.
#We more interested in being able to detect the tweets which clear negative sentiment.
#Let's define new variable in our data and set tweets called negative.
tweets$negative = as.factor(tweets$Avg <= -1)
#This will set tweets$negative equal to TRUE if average sentiment score is less than or equal to negative 1.
#And will set tweets$negative equal to FALSE if average sentiment score greater than negative 1.
table(tweets$negative)
#Now to pre-process out text data that we can use the bag of words approach.
#By text mining.
library(tm)
library(SnowballC)
#We need to convert our tweets to a corpus for pre-processing.
corpus = Corpus(VectorSource(tweets$Tweet))
corpus
#Check documents match out tweets.
corpus[[1]]$content
#Now changing all text in our tweets to lower case.
corpus = tm_map(corpus,tolower)
#Now remove punctuation.
corpus = tm_map(corpus,removePunctuation)
#Now remove stop words for english language.
stopwords("english")[1:10]
#We'll remove all these english stop words but we'll also remove word apple.
#It probably won't be very useful in our prediction problem.
corpus = tm_map(corpus,removeWords,c("apple",stopwords("english")))
corpus[[1]]$content
#Now we can see significant fewer words only.
#Lastly we want to stem our document.
corpus = tm_map(corpus,stemDocument)
corpus[[1]]$content
#Extract the words frequencies to be use in our prediction.
frequencies = DocumentTermMatrix(corpus)
frequencies
#Let's see matrix look like.
inspect(frequencies[1000:1005,505:515])
#We can look at most popular term.
findFreqTerms(frequencies,lowfreq=20)
#We can see only 56 words at least 20 times in our tweets.
#This mean a lot terms that will be useless for our prediction model.
#One is computational more terms mean more independent variables.
#Let's remove some terms 
sparse = removeSparseTerms(frequencies,0.995)
#That means to only keep terms that appear in 0.5% or more of the tweets.
sparse
#This only about 9% of previous count of 3289.
#Now let's convert the sparse matrix into a data frame.
tweetsSparse = as.data.frame(as.matrix(sparse))
#Let's run the make.names function to make sure all of our words are appropiate variable names.
colnames(tweetsSparse) = make.names(colnames(tweetsSparse))
#Now add our dependent variable to this data set.
tweetsSparse$negative = tweets$negative
#Let's split data to into training set and testing set.
library(caTools)
set.seed(123)
split = sample.split(tweetsSparse$negative,SplitRatio = 0.7)
train = subset(tweetsSparse,split == TRUE)
test = subset(tweetsSparse,split == FALSE)
#Use CART to build a predictive model.
library(rpart)
library(rpart.plot)
tweet_cart = rpart(negative~.,data = train,method = "class")
prp(tweet_cart)
pred_cart = predict(tweet_cart,newdata = test,type = "class")
table(test$negative,pred_cart)
#Compute accuracy.
(294+18)/(294+6+37+18)
#Compare with baseline model.
table(test$negative)
300/(300+55)
#Try to build random forest model.
library(randomForest)
set.seed(123)
tweet_rf = randomForest(negative~.,data = train)
pred_rf = predict(tweet_rf,newdata = test)
table(test$negative,pred_rf)
#Compute accuracy.
(292+22)/(292+8+33+22)
#Try to build logistic regression model.
tweet_log = glm(negative~.,data = train,family = "binomial")
pred_log = predict(tweet_log,newdata = test,type = "response")
table(test$negative,pred_log)
#Compute accuracy.
(257+31)/(257+43+24+31)
