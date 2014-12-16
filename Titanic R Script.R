setwd("~/Desktop/Titanic Competition")
# Set working directory and import datafiles of Titanic:Machine Learning
# from disaster

# Jim Cen  16 Dec,2014

train <- read.csv("~/Desktop/Titanic Competition/train.csv")
View(train)
test <- read.csv("~/Desktop/Titanic Competition/test.csv")
View(test)

#take a quick look at the structure of the dataframe
str(train)

#let’s add our ‘everyone dies’ prediction to the test set dataframe
test$Survived <- rep(0, 418)

#We need to submit a csv file with the PassengerId as well as our 
#Survived predictions to Kaggle. So let’s extract those two columns 
#from the test dataframe, store them in a new container, and then send 
#it to an output file:
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "theyallperish.csv", row.names = FALSE)

#The disaster was famous for saving “women and children first.
#Take a look at the summary of this gender variable
summary(train$Sex)

#let’s expand the proportion table command we used last time to do a two-way 
#comparison on the number of males and females that survived
prop.table(table(train$Sex, train$Survived),1)

#Here we have begun with adding the ‘everyone dies’ prediction column as before
#We then altered that same column with 1’s for the subset of passengers where 
#the variable ‘Sex’ is equal to ‘female’.
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1

#A new prediction with all woman survive
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "AllWomanSurvive.csv", row.names = FALSE)

#Take a look at the age variable
summary(train$Age)

#let’s create a new variable, Child, to indicate whether the passenger is
#below the age of 18:
train$Child <- 0
train$Child[train$Age < 18] <- 1

# let’s try to find the number of survivors for the different subsets
aggregate(Survived ~ Child + Sex, data=train, FUN=sum)

#This simply looked at the length of the Survived vector for each subset 
#and output the result, the fact that any of them were 0’s or 1’s was irrelevant 
#for the length function.
aggregate(Survived ~ Child + Sex, data=train, FUN=length)

#Now we have the totals for each group of passengers, but really, we would like 
#to know the proportions again
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

#Let’s bin the fares into less than $10, between $10 and $20, $20 to $30 and more 
#than $30 and store it to a new variable:
train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '<10'

#Now let’s run a longer aggregate function to see if there’s anything interesting 
#to work with here:
aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

#While the majority of males, regardless of class or fare still don’t do so well, 
#we notice that most of the class 3 women who paid more than $20 for their ticket 
#actually also miss out on a lifeboat.
#Let’s make a new prediction based on the new insights
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] <- 0

#A new prediction with all woman survive except those in class 3 and fare greater than 20
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "WomanSurviveExceptClass3&Fare>20.csv", row.names = FALSE)

#Import rpart for ‘Recursive Partitioning and Regression Trees’ and 
#uses the CART decision tree algorithm.
library(rpart)

#You feed it the equation, headed up by the variable of interest and 
#followed by the variables used for prediction.
#If you wanted to predict a continuous variable, such as age, 
#you may use method=”anova”.In this case, we only want 0 and 1, so we set method="class"
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")

#Let’s examine the tree.
plot(fit)
text(fit)

#install graphics package for rpart:
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
library(rattle)
library(rpart.plot)
library(RColorBrewer)

#Let’s try rendering this tree a bit nicer with fancyRpartPlot.
fancyRpartPlot(fit)

#To make a prediction from this tree doesn’t require all the subsetting and overwriting 
#we did last lesson, it’s actually a lot easier.
Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "myfirstdtree.csv", row.names = FALSE)

#Unleash the rpart default parameter setting
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train,
             method="class", control=rpart.control(minsplit=2, cp=0))
fancyRpartPlot(fit)

#Since we obviously lack the Survived column in our test set, let’s create one full of 
#missing values (NAs) and then row bind the two datasets together:

train$Child <- NULL #reset these two extra columns to NULL in order to bind.
train$Fare2 <- NULL
test$Survived <- NA
combi <- rbind(train, test)

#We need to cast the name column back into a text string. To do this we use as.character.
combi$Name <- as.character(combi$Name)
combi$Name[1]

#We can easily use the function strsplit, which stands for string split, to break apart 
#our original name over these two symbols. 
strsplit(combi$Name[1], split='[,.]')

#Let’s try to dig into this new type of container by appending all those square brackets 
#to the original command:
strsplit(combi$Name[1], split='[,.]')[[1]]

#Let’s go a level deeper into the indexing mess and extract the title. It’s the second 
#item in this nested list, so let’s dig in to index number 2 of this new container:
strsplit(combi$Name[1], split='[,.]')[[1]][2]

#Use sapply function to apply to the whole name column
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

# We can use sub for this 
#(gsub would replace all spaces, poor ‘the Countess’ would look strange then though)
combi$Title <- sub(' ', '', combi$Title)

# Mademoiselle and Madame are pretty similar (so long as you don’t mind offending) 
#so let’s combine them into a single category:
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'

#Combine the rich fellas
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

#Our final step is to change the variable type back to a factor, as these are essentially
#categories that we have created:
combi$Title <- factor(combi$Title)

#eems reasonable to assume that a large family might have trouble tracking down little 
#Johnny as they all scramble to get off the sinking ship, so let’s combine the two variables 
#into a new one, FamilySize:
combi$FamilySize <- combi$SibSp + combi$Parch + 1

#So let’s first extract the passengers’ last names. This should be a pretty simple change 
#from the title extraction code we ran earlier, now we just want the first part of the 
#strsplit output:
combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})

#let’s convert the FamilySize variable temporarily to a string and combine it with the 
#Surname to get our new FamilyID variable
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")

#Given we were originally hypothesising that large families might have trouble 
#sticking together in the panic, let’s knock out any family size of two or less 
#and call it a “small” family. This would fix the Johnson problem too
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'

#all these one or two people groups is what we sought to avoid with the three person cut-off
#Let’s begin to clean this up:
famIDs <- data.frame(table(combi$FamilyID))

#Here we see again all those naughty families that didn’t work well with our assumptions, 
#so let’s subset this dataframe to show only those unexpectedly small FamilyID groups.
famIDs <- famIDs[famIDs$Freq <= 2,]

#We then need to overwrite any family IDs in our dataset for groups that were not correctly 
#identified and finally convert it to a factor:
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

#So let’s break them apart and do some predictions on our new fancy engineered variables:
train <- combi[1:891,]
test <- combi[892:1309,]
#Time to do our predictions! We have a bunch of new variables, so let’s send them to a new 
#decision tree. Last time the default complexity worked out pretty well, so let’s just grow 
#a tree with the vanilla controls and see what it can do:
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
             data=train, method="class")
fancyRpartPlot(fit)
Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "mythirddtree.csv", row.names = FALSE)

# Let’s pick up where we left off last lesson, and take a look at the combined dataframe’s age 
#variable to see what we’re up against:
summary(combi$Age)

# let’s grow a tree on the subset of the data with the age values available, and then replace 
#those that are missing:
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

#Because it’s so few observations and such a large majority boarded in Southampton, let’s just 
#replace those two with ‘S’. First we need to find out who they are though! We can use which for this:
which(combi$Embarked == '')

#This gives us the indexes of the blank fields. Then we simply replace those two, and
#encode it as a factor:
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)

#It’s only one passenger with a NA, so let’s find out which one it is and replace it with the median fare:
which(is.na(combi$Fare))
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

# To do this we’ll copy the FamilyID column to a new variable, FamilyID2, and then convert it from 
#a factor back into a character string with as.character(). We can then increase our cut-off to be 
#a “Small” family from 2 to 3 people. Then we just convert it back to a factor and we’re done:
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

# Split back into test and train sets
train <- combi[1:891,]
test <- combi[892:1309,]


#Install and load the package randomForest:
install.packages('randomForest')
library(randomForest)

#This makes your results reproducible next time you load the code up, otherwise you can get different
#classifications for each run.
set.seed(415)

#Run the model
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize +
                      FamilyID2, data=train, importance=TRUE, ntree=2000)

# Look at variable importance
varImpPlot(fit)

# Now let's make a prediction and write a submission file
Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)

#install the party package
install.packages('party')
library(party)

# Build condition inference tree Random Forest
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3)) 

# Now let's make a prediction and write a submission file
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "ciforest.csv", row.names = FALSE)



