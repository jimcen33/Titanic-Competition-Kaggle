#Titanic Survival Prediction
#Another approach to Deriving a Model 
#Organized by Jim Cen, Dec-08-2014
#Source from wehrley/wehrley.github.io on Github
#Reference: https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md

#Make the read.csv function as a re-usable custom funtion
readData <- function(path.name, file.name, column.types, missing.types) {
  read.csv( url( paste(path.name, file.name, sep="") ), 
            colClasses=column.types,
            na.strings=missing.types )
}

#Define each argument for this custom file reading function
Titanic.path <- "https://raw.github.com/wehrley/Kaggle_Titanic/master/"
train.data.file <- "train.csv"
test.data.file <- "test.csv"
missing.types <- c("NA", "")
train.column.types <- c('integer',   # PassengerId
                        'factor',    # Survived 
                        'factor',    # Pclass
                        'character', # Name
                        'factor',    # Sex
                        'numeric',   # Age
                        'integer',   # SibSp
                        'integer',   # Parch
                        'character', # Ticket
                        'numeric',   # Fare
                        'character', # Cabin
                        'factor'     # Embarked
)
test.column.types <- train.column.types[-2]     # # no Survived column in test.csv

#Much cleaner code for reading the csv files
train.raw <- readData(Titanic.path, train.data.file, 
                      train.column.types, missing.types)
df.train <- train.raw

test.raw <- readData(Titanic.path, test.data.file, 
                     test.column.types, missing.types)
df.infer <- test.raw  

#Let's start with a look at missing data in the training set. 
#I'll use the missmap function from the Amelia package to display those.

## map missing data by provided feature
require(Amelia)
missmap(df.train, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

#Putting some simple data visualization tools to work can take us 
#a long way toward understanding what might influence the outcome 
#we're trying to predict
barplot(table(df.train$Survived),
        names.arg = c("Perished", "Survived"),
        main="Survived (passenger fate)", col="black")
barplot(table(df.train$Pclass), 
        names.arg = c("first", "second", "third"),
        main="Pclass (passenger traveling class)", col="firebrick")
barplot(table(df.train$Sex), main="Sex (gender)", col="darkviolet")
hist(df.train$Age, main="Age", xlab = NULL, col="brown")
barplot(table(df.train$SibSp), main="SibSp (siblings + spouse aboard)", 
        col="darkblue")
barplot(table(df.train$Parch), main="Parch (parents + kids aboard)", 
        col="gray50")
hist(df.train$Fare, main="Fare (fee paid for ticket[s])", xlab = NULL, 
     col="darkgreen")
barplot(table(df.train$Embarked), 
        names.arg = c("Cherbourg", "Queenstown", "Southampton"),
        main="Embarked (port of embarkation)", col="sienna")

#The vcd package includes the mosaicplot function for creating those. 
#The following mosaic suggests that traveling class did influence the 
#odds of a passenger's survival.
mosaicplot(df.train$Pclass ~ df.train$Survived, 
           main="Passenger Fate by Traveling Class", shade=FALSE, 
           color=TRUE, xlab="Pclass", ylab="Survived")

mosaicplot(df.train$Sex ~ df.train$Survived, 
           main="Passenger Fate by Gender", shade=FALSE, color=TRUE, 
           xlab="Sex", ylab="Survived")

mosaicplot(df.train$Embarked ~ df.train$Survived, 
           main="Passenger Fate by Port of Embarkation",
           shade=FALSE, color=TRUE, xlab="Embarked", ylab="Survived")

#Take a look at Age vs Survived using boxplot
boxplot(df.train$Age ~ df.train$Survived, 
        main="Passenger Fate by Age",
        xlab="Survived", ylab="Age")

#Take a look at the correlogram
require(corrgram)
corrgram.data <- df.train
## change features of factor type to numeric type for inclusion on correlogram
corrgram.data$Survived <- as.numeric(corrgram.data$Survived)
corrgram.data$Pclass <- as.numeric(corrgram.data$Pclass)
corrgram.data$Embarked <- revalue(corrgram.data$Embarked, 
                                  c("C" = 1, "Q" = 2, "S" = 3))
## generate correlogram
corrgram.vars <- c("Survived", "Pclass", "Sex", "Age", 
                   "SibSp", "Parch", "Fare", "Embarked")
corrgram(corrgram.data[,corrgram.vars], order=FALSE, 
         lower.panel=panel.ellipse, upper.panel=panel.pie, 
         text.panel=panel.txt, main="Titanic Training Data")

#Time to tackle those missing ages. A common approach to this 
#type of situation is to replacing the missings with the average 
#of the available values. 
#Passengers in the upper classes (first and second) would tend to be wealthier,
#and in that period of U.S. history, acquiring wealth usually required a good 
#deal of time (no dot-com kings in their 20s were aboard the Titanic on her maiden voyage).
#There are no missing values in Pclass, so we could replace the missing age for, say, 
#a third class passenger with the average or median of the available ages for those 
#in Pclass="3". Doing so would be an improvement over assigning 29.7 to all Age missings.

## function for extracting honorific (i.e. title) from the Name feature
getTitle <- function(data) {
  title.dot.start <- regexpr("\\,[A-Z ]{1,20}\\.", data$Name, TRUE)
  title.comma.end <- title.dot.start 
  + attr(title.dot.start, "match.length")-1
  data$Title <- substr(data$Name, title.dot.start+2, title.comma.end-1)
  return (data$Title)
}   

#Let's fetch the titles, given them their own column in the df.train data frame, 
#and look at the uniques.
df.train$Title <- getTitle(df.train)
unique(df.train$Title)

#To identify the titles which have at least one record with an age missing,
#I'll use the bystats function from the Hmisc package.
options(digits=2)
require(Hmisc)
bystats(df.train$Age, df.train$Title, 
        fun=function(x)c(Mean=mean(x),Median=median(x)))
#                N Missing Mean Median
# Capt           1       0 70.0   70.0
# Col            2       0 58.0   58.0
# Don            1       0 40.0   40.0
# Dr             6       1 42.0   46.5
# Jonkheer       1       0 38.0   38.0
# Lady           1       0 48.0   48.0
# Major          2       0 48.5   48.5
# Master        36       4  4.6    3.5
# Miss         146      36 21.8   21.0
# Mlle           2       0 24.0   24.0
# Mme            1       0 24.0   24.0
# Mr           398     119 32.4   30.0
# Mrs          108      17 35.9   35.0
# Ms             1       0 28.0   28.0
# Rev            6       0 43.2   46.5
# Sir            1       0 49.0   49.0
# the Countess   1       0 33.0   33.0
# ALL          714     177 29.7   28.0

#Now I can assign the titles with at least one missing Age value to a list...

## list of titles with missing Age value(s) requiring imputation
titles.na.train <- c("Dr", "Master", "Mrs", "Miss", "Mr")

#then pass that list to the following custom function I created for imputing
#the missing ages:
imputeMedian <- function(impute.var, filter.var, var.levels) {
  for (v in var.levels) {
    impute.var[ which( filter.var == v)] <- impute(impute.var[ 
      which( filter.var == v)])
  }
  return (impute.var)
}

#After doing the age imputations, I check the Age data and find that 
#the function seems to have done its job.

df.train$Age <- imputeMedian(df.train$Age, df.train$Title, 
                               titles.na.train)
df.train$Age[which(df.train$Title=="Dr")]
#[1] 44.0 54.0 23.0 32.0 50.0 46.5 49.0
summary(df.train$Age)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.42   21.00   30.00   29.39   35.00   80.00


# It should be fine to replace those missings with "S", the most common value.
df.train$Embarked[which(is.na(df.train$Embarked))] <- 'S'

#While there are no missing Fare values, a summary does show at least one Fare=0...
summary(df.train$Fare)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.00    7.91   14.45   32.20   31.00  512.30

#(That exceptionally high fare of $512.30 suggests that some tickets were purchased 
#in groups. We'll address that later.) A zero fare might have been assigned to a baby. 
#However, a closer look at records where Fare = 0 suggests otherwise
subset(df.train, Fare < 7)[order(subset(df.train, Fare < 7)$Fare, 
                                 subset(df.train, Fare < 7)$Pclass), 
                           c("Age", "Title", "Pclass", "Fare")]

#The jump in fares from 0 to the 4-7 range suggests errors. I replaced the zero 
#Fare values with the median fare from the respective passenger class using 
#the imputMedian function introduced earlier.
## impute missings on Fare feature with median fare by Pclass
df.train$Fare[ which( df.train$Fare == 0 )] <- NA
df.train$Fare <- imputeMedian(df.train$Fare, df.train$Pclass, 
                              as.numeric(levels(df.train$Pclass)))

#Take a look at the title boxplot
df.train$Title <- factor(df.train$Title,
                         c("Capt","Col","Major","Sir","Lady","Rev",
                           "Dr","Don","Jonkheer","the Countess","Mrs",
                           "Ms","Mr","Mme","Mlle","Miss","Master"))
boxplot(df.train$Age ~ df.train$Title, 
        main="Passenger Age by Title", xlab="Title", ylab="Age")

#Change Title
## function for assigning a new title value to old title(s) 
changeTitles <- function(data, old.titles, new.title) {
  for (honorific in old.titles) {
    data$Title[ which( data$Title == honorific)] <- new.title
  }
  return (data$Title)
}
## Title consolidation
df.train$Title <- changeTitles(df.train, 
                               c("Capt", "Col", "Don", "Dr", 
                                 "Jonkheer", "Lady", "Major", 
                                 "Rev", "Sir"),
                               "Noble")
df.train$Title <- changeTitles(df.train, c("the Countess", "Ms"), 
                               "Mrs")
df.train$Title <- changeTitles(df.train, c("Mlle", "Mme"), "Miss")
df.train$Title <- as.factor(df.train$Title)

###Feature Engineering to the train and test datasets
require(plyr)     # for the revalue function 
require(stringr)  # for the str_sub function

## test a character as an EVEN single digit
isEven <- function(x) x %in% c("0","2","4","6","8") 
## test a character as an ODD single digit
isOdd <- function(x) x %in% c("1","3","5","7","9") 

## function to add features to training or test data frames
featureEngrg <- function(data) {
  ## Using Fate ILO Survived because term is shorter and just sounds good
  data$Fate <- data$Survived
  ## Revaluing Fate factor to ease assessment of confusion matrices later
  data$Fate <- revalue(data$Fate, c("1" = "Survived", "0" = "Perished"))
  ## Boat.dibs attempts to capture the "women and children first"
  ## policy in one feature.  Assuming all females plus males under 15
  ## got "dibs' on access to a lifeboat
  data$Boat.dibs <- "No"
  data$Boat.dibs[which(data$Sex == "female" | data$Age < 15)] <- "Yes"
  data$Boat.dibs <- as.factor(data$Boat.dibs)
  ## Family consolidates siblings and spouses (SibSp) plus
  ## parents and children (Parch) into one feature
  data$Family <- data$SibSp + data$Parch
  ## Fare.pp attempts to adjust group purchases by size of family
  data$Fare.pp <- data$Fare/(data$Family + 1)
  ## Giving the traveling class feature a new look
  data$Class <- data$Pclass
  data$Class <- revalue(data$Class, 
                        c("1"="First", "2"="Second", "3"="Third"))
  ## First character in Cabin number represents the Deck 
  data$Deck <- substring(data$Cabin, 1, 1)
  data$Deck[ which( is.na(data$Deck ))] <- "UNK"
  data$Deck <- as.factor(data$Deck)
  ## Odd-numbered cabins were reportedly on the port side of the ship
  ## Even-numbered cabins assigned Side="starboard"
  data$cabin.last.digit <- str_sub(data$Cabin, -1)
  data$Side <- "UNK"
  data$Side[which(isEven(data$cabin.last.digit))] <- "port"
  data$Side[which(isOdd(data$cabin.last.digit))] <- "starboard"
  data$Side <- as.factor(data$Side)
  data$cabin.last.digit <- NULL
  return (data)
}

## add remaining features to training data frame
df.train <- featureEngrg(df.train)

#Finish the data munging
train.keeps <- c("Fate", "Sex", "Boat.dibs", "Age", "Title", 
                 "Class", "Deck", "Side", "Fare", "Fare.pp", 
                 "Embarked", "Family")
df.train.munged <- df.train[train.keeps]

#Selecte an 80/20 split for training data and testing data. 
  
## split training data into train batch and test batch
set.seed(23)
training.rows <- createDataPartition(df.train.munged$Survived, 
                                     p = 0.8, list = FALSE)
train.batch <- df.train.munged[training.rows, ]
test.batch <- df.train.munged[-training.rows, ]

#By setting the argument family to binomial with a logit link, 
#I'm asking glm( ) to produce a logistic regression.
Titanic.logit.1 <- glm(Fate ~ Sex + Class + Age + Family + Embarked + Fare, 
                       data = train.batch, family=binomial("logit")
anova(Titanic.logit.1, test="Chisq")

#Second model involved the Fare.pp feature
Titanic.logit.2 <- glm(Fate ~ Sex + Class + Age + Family + Embarked + Fare.pp,                        
                         data = train.batch, family=binomial("logit"))
anova(Titanic.logit.2, test="Chisq")

#Third Model without Fare
glm(Fate ~ Sex + Class + Age + Family + Embarked, 
    data = train.batch, family=binomial("logit")

#All of these things I want -- 3x 10-fold CV, estimation of class probabilities, 
#metrics from twoClassSummary -- can be passed through the trainControl function.

## Define control function to handle optional arguments for train function
## Models to be assessed based on largest absolute area under ROC curve
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)

#Tune the model sans Fare
set.seed(35)
glm.tune.1 <- train(Fate ~ Sex + Class + Age + Family + Embarked,
                    data = train.batch,
                    method = "glm",
                    metric = "ROC",
                    trControl = cv.ctrl)
glm.tune.1
summary(glm.tune.1)

#Tune the model with Embarked=="s"
set.seed(35)
glm.tune.2 <- train(Fate ~ Sex + Class + Age + Family + I(Embarked=="S"),
                      data = train.batch, method = "glm",
                      metric = "ROC", trControl = cv.ctrl)
summary(glm.tune.2)

#Tune the model with Title
set.seed(35)
glm.tune.3 <- train(Fate ~ Sex + Class + Title + Age 
                      + Family + I(Embarked=="S"), 
                      data = train.batch, method = "glm",
                      metric = "ROC", trControl = cv.ctrl)
summary(glm.tune.3)

#Drop Age from the formula and collapse the titles “Miss” and “Mrs” and 
#leave a duo of Title-related factors which should represent the “women 
#and children first” theme well.

set.seed(35)
glm.tune.4 <- train(Fate ~ Class + I(Title=="Mr") + I(Title=="Noble") 
                      + Age + Family + I(Embarked=="S"), 
                      data = train.batch, method = "glm",
                      metric = "ROC", trControl = cv.ctrl)
summary(glm.tune.4)

#“grown men in the lower decks had it tough” is such a gross understatement 
#that I hesitated to put it in type. A feature reflecting those third-class 
#men might make a further dent in that residual deviance. Indeed, it does...
set.seed(35)
glm.tune.5 <- train(Fate ~ Class + I(Title=="Mr") + I(Title=="Noble") 
                      + Age + Family + I(Embarked=="S") 
                      + I(Title=="Mr"&Class=="Third"), 
                      data = train.batch, 
                      method = "glm", metric = "ROC", 
                      trControl = cv.ctrl)
summary(glm.tune.5)

#For purpose of illustration, I'll simply take the logistic regression model 
#formula from glm.tune.1 and pass it through train() for each of three other 
#model types, with one new twist: tuning variables specific to each model.

#Create a data frame with these three variables as column names and one row
#per tuning variable combination, and you're good to go. 
#Here is just one example of a tuning grid for ada:
## note the dot preceding each variable
ada.grid <- expand.grid(.iter = c(50, 100),
                        .maxdepth = c(4, 8),
                        .nu = c(0.1, 1))

#Specify method=”ada” and tuneGrid=ada.grid in train, and away we go...
set.seed(35)
ada.tune <- train(Fate ~ Sex + Class + Age + Family + Embarked, 
                  data = train.batch,
                  method = "ada",
                  metric = "ROC",
                  tuneGrid = ada.grid,
                  trControl = cv.ctrl)
ada.tune
plot(ada.tune)

#Time to give the popular Random Forest (RF) model a shot at the Titanic challenge. 
rf.grid <- data.frame(.mtry = c(2, 3))
set.seed(35)
rf.tune <- train(Fate ~ Sex + Class + Age + Family + Embarked, 
                 data = train.batch,
                 method = "rf",
                 metric = "ROC",
                 tuneGrid = rf.grid,
                 trControl = cv.ctrl)
rf.tune

#And finally, we'll fit a support vector machine (SVM) model to the Titanic data.
set.seed(35)
svm.tune <- train(Fate ~ Sex + Class + Age + Family + Embarked, 
                  data = train.batch,
                  method = "svmRadial",
                  tuneLength = 9,
                  preProcess = c("center", "scale"),
                  metric = "ROC",
                  trControl = cv.ctrl)
svm.tune

#MODEL Evaluation

## Logistic regression model
glm.pred <- predict(glm.tune.5, test.batch)
confusionMatrix(glm.pred, test.batch$Fate)

## Boosted model
ada.pred <- predict(ada.tune, test.batch)
confusionMatrix(ada.pred, test.batch$Fate)

## Random Forest model
rf.pred <- predict(rf.tune, test.batch)
confusionMatrix(rf.pred, test.batch$Fate)

## SVM model 
svm.pred <- predict(svm.tune, test.batch)
confusionMatrix(svm.pred, test.batch$Fate)

## Logistic regression model (BLACK curve)
glm.probs <- predict(glm.tune.5, test.batch, type = "prob")
glm.ROC <- roc(response = test.batch$Fate,
               predictor = glm.probs$Survived,
               levels = levels(test.batch$Fate))
plot(glm.ROC, type="S")   
## Area under the curve: 0.8609 
## Boosted model (GREEN curve)
ada.probs <- predict(ada.tune, test.batch, type = "prob")
ada.ROC <- roc(response = test.batch$Fate,
               predictor = ada.probs$Survived,
               levels = levels(test.batch$Fate))
plot(ada.ROC, add=TRUE, col="green")    
## Area under the curve: 0.8759
## Random Forest model (RED curve)
rf.probs <- predict(rf.tune, test.batch, type = "prob")
rf.ROC <- roc(response = test.batch$Fate,
              predictor = rf.probs$Survived,
              levels = levels(test.batch$Fate))
plot(rf.ROC, add=TRUE, col="red") 
## Area under the curve: 0.8713
## SVM model (BLUE curve)
svm.probs <- predict(svm.tune, test.batch, type = "prob")
svm.ROC <- roc(response = test.batch$Fate,
               predictor = svm.probs$Survived,
               levels = levels(test.batch$Fate))
plot(svm.ROC, add=TRUE, col="blue")
## Area under the curve: 0.8077


#The following R script uses caret function resamples to collect 
#the resampling results, then calls the dotplot function to create
#a visualization of the resampling distributions.
cv.values <- resamples(list(Logit = glm.tune.5, Ada = ada.tune, 
                            RF = rf.tune, SVM = svm.tune))
dotplot(cv.values, metric = "ROC")

#Cast Your Votes
# get titles
df.infer$Title <- getTitle(df.infer)

# impute missing Age values
df.infer$Title <- changeTitles(df.infer, c("Dona", "Ms"), "Mrs")
titles.na.test <- c("Master", "Mrs", "Miss", "Mr")
df.infer$Age <- imputeMedian(df.infer$Age, df.infer$Title, titles.na.test)

# consolidate titles
df.infer$Title <- changeTitles(df.infer, c("Col", "Dr", "Rev"), "Noble")
df.infer$Title <- changeTitles(df.infer, c("Mlle", "Mme"), "Miss")
df.infer$Title <- as.factor(df.infer$Title)

# impute missing fares
df.infer$Fare[ which( df.infer$Fare == 0)] <- NA
df.infer$Fare <- imputeMedian(df.infer$Fare, df.infer$Pclass, 
                              as.numeric(levels(df.infer$Pclass)))
# add the other features
df.infer <- featureEngrg(df.infer)

# data prepped for casting predictions
test.keeps <- train.keeps[-1]
pred.these <- df.infer[test.keeps]

# use the logistic regression model to generate predictions
Survived <- predict(glm.tune.5, newdata = pred.these)

# reformat predictions to 0 or 1 and link to PassengerId in a data frame
Survived <- revalue(Survived, c("Survived" = 1, "Perished" = 0))
predictions <- as.data.frame(Survived)
predictions$PassengerId <- df.infer$PassengerId

# write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="Titanic_predictions.csv", row.names=FALSE, quote=FALSE)

#If you must know, the logistic regression model scored 0.77990 on Kaggle –
#roughly middle of the pack on the leaderboard (as of late January 2014). 
#I have submitted better scoring models; my best thus far, at 0.79426, 
#trails just 17 percent of the 1150+ participants on the leaderboard.



