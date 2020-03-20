library(ggplot2)
library(cowplot)
library(randomForest)

## NOTE: The data used in this demo comes from the UCI machine learning
## repository.
## http://archive.ics.uci.edu/ml/index.php
## Specifically, this is the heart disease data set.
## http://archive.ics.uci.edu/ml/datasets/Heart+Disease

url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

data <- read.csv(url, header=FALSE)

#####################################
##
## Reformat the data so that it is 
## 1) Easy to use (add nice column names)
## 2) Interpreted correctly by randomForest..
##
#####################################
head(data) # you see data, but no column names

colnames(data) <- c(
  "age",
  "sex",# 0 = female, 1 = male
  "cp", # chest pain 
          # 1 = typical angina, 
          # 2 = atypical angina, 
          # 3 = non-anginal pain, 
          # 4 = asymptomatic
  "trestbps", # resting blood pressure (in mm Hg)
  "chol", # serum cholestoral in mg/dl
  "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
  "restecg", # resting electrocardiographic results
          # 1 = normal
          # 2 = having ST-T wave abnormality
          # 3 = showing probable or definite left ventricular hypertrophy
  "thalach", # maximum heart rate achieved
  "exang",   # exercise induced angina, 1 = yes, 0 = no
  "oldpeak", # ST depression induced by exercise relative to rest
  "slope", # the slope of the peak exercise ST segment 
          # 1 = upsloping 
          # 2 = flat 
          # 3 = downsloping 
  "ca", # number of major vessels (0-3) colored by fluoroscopy
  "thal", # this is short of thalium heart scan
          # 3 = normal (no cold spots)
          # 6 = fixed defect (cold spots during rest and exercise)
          # 7 = reversible defect (when cold spots only appear during exercise)
  "hd" # (the predicted attribute) - diagnosis of heart disease 
          # 0 if less than or equal to 50% diameter narrowing
          # 1 if greater than 50% diameter narrowing
  )

head(data) # now we have data and column names

str(data) # this shows that we need to tell R which columns contain factors
          # it also shows us that there are some missing values. There are "?"s
          # in the dataset.

## First, replace "?"s with NAs.
data[data == "?"] <- NA

## Now add factors for variables that are factors and clean up the factors
## that had missing data...
data[data$sex == 0,]$sex <- "F"
data[data$sex == 1,]$sex <- "M"
data$sex <- as.factor(data$sex)

data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)

data$ca <- as.integer(data$ca) # since this column had "?"s in it (which
                               # we have since converted to NAs) R thinks that
                               # the levels for the factor are strings, but
                               # we know they are integers, so we'll first
                               # convert the strings to integiers...
data$ca <- as.factor(data$ca)  # ...then convert the integers to factor levels

data$thal <- as.integer(data$thal) # "thal" also had "?"s in it.
data$thal <- as.factor(data$thal)

## This next line replaces 0 and 1 with "Healthy" and "Unhealthy"
data$hd <- ifelse(test=data$hd == 0, yes="Healthy", no="Unhealthy")
data$hd <- as.factor(data$hd) # Now convert to a factor

str(data) ## this shows that the correct columns are factors and we've replaced
  ## "?"s with NAs because "?" no longer appears in the list of factors
  ## for "ca" and "thal"


#####################################
##
## Now we are ready to build a random forest.
##
#####################################
set.seed(42)

## NOTE: For most machine learning methods, you need to divide the data
## manually into a "training" set and a "test" set. This allows you to train 
## the method using the training data, and then test it on data it was not
## originally trained on. 
##
## In contrast, Random Forests split the data into "training" and "test" sets 
## for you. This is because Random Forests use bootstrapped
## data, and thus, not every sample is used to build every tree. The 
## "training" dataset is the bootstrapped data and the "test" dataset is
## the remaining samples. The remaining samples are called
## the "Out-Of-Bag" (OOB) data.

## impute any missing values in the training set using proximities
data.imputed <- rfImpute(hd ~ ., data = data, iter=6)
## NOTE: iter = the number of iterations to run. Breiman says 4 to 6 iterations
## is usually good enough. With this dataset, when we set iter=6, OOB-error
## bounces around between 17% and 18%. When we set iter=20, 
# set.seed(42)
# data.imputed <- rfImpute(hd ~ ., data = data, iter=20)
## we get values a little better and a little worse, so doing more 
## iterations doesn't improve the situation.
##
## NOTE: If you really want to micromanage how rfImpute(),
## you can change the number of trees it makes (the default is 300) and the
## number of variables that it will consider at each step.

## Now we are ready to build a random forest.

## NOTE: If the thing we're trying to predict (in this case it is 
## whether or not someone has heart disease) is a continuous number 
## (i.e. "weight" or "height"), then by default, randomForest() will set 
## "mtry", the number of variables to consider at each step, 
## to the total number of variables divided by 3 (rounded down), or to 1 
## (if the division results in a value less than 1).
## If the thing we're trying to predict is a "factor" (i.e. either "yes/no"
## or "ranked"), then randomForest() will set mtry to 
## the square root of the number of variables (rounded down to the next
## integer value).

## In this example, "hd", the thing we are trying to predict, is a factor and
## there are 13 variables. So by default, randomForest() will set 
## mtry = sqrt(13) = 3.6 rounded down = 3
## Also, by default random forest generates 500 trees (NOTE: rfImpute() only
## generates 300 tress by default)
model <- randomForest(hd ~ ., data=data.imputed, proximity=TRUE)

## RandomForest returns all kinds of things
model # gives us an overview of the call, along with...
      # 1) The OOB error rate for the forest with ntree trees. 
      #    In this case ntree=500 by default
      # 2) The confusion matrix for the forest with ntree trees.
      #    The confusion matrix is laid out like this:
#          
#                Healthy                      Unhealthy
#          --------------------------------------------------------------
# Healthy  | Number of healthy people   | Number of healthy people      |
#          | correctly called "healthy" | incorectly called "unhealthy" |
#          | by the forest.             | by the forest                 |
#          --------------------------------------------------------------
# Unhealthy| Number of unhealthy people | Number of unhealthy peole     |
#          | incorrectly called         | correctly called "unhealthy"  |
#          | "healthy" by the forest    | by the forest                 |
#          --------------------------------------------------------------

## Now check to see if the random forest is actually big enough...
## Up to a point, the more trees in the forest, the better. You can tell when
## you've made enough when the OOB no longer improves.
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "Healthy", "Unhealthy"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
    model$err.rate[,"Healthy"], 
    model$err.rate[,"Unhealthy"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))
# ggsave("oob_error_rate_500_trees.pdf")

## Blue line = The error rate specifically for calling "Unheathly" patients that
## are OOB.
##
## Green line = The overall OOB error rate.
##
## Red line = The error rate specifically for calling "Healthy" patients 
## that are OOB.

## NOTE: After building a random forest with 500 tress, the graph does not make 
## it clear that the OOB-error has settled on a value or, if we added more 
## trees, it would continue to decrease.
## So we do the whole thing again, but this time add more trees.

model <- randomForest(hd ~ ., data=data.imputed, ntree=1000, proximity=TRUE)
model

oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "Healthy", "Unhealthy"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
    model$err.rate[,"Healthy"], 
    model$err.rate[,"Unhealthy"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))
# ggsave("oob_error_rate_1000_trees.pdf")

## After building a random forest with 1,000 trees, we get the same OOB-error
## 16.5% and we can see convergence in the graph. So we could have gotten
## away with only 500 trees, but we wouldn't have been sure that number
## was enough.


## If we want to compare this random forest to others with different values for
## mtry (to control how many variables are considered at each step)...
oob.values <- vector(length=10)
for(i in 1:10) {
  temp.model <- randomForest(hd ~ ., data=data.imputed, mtry=i, ntree=1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values
## find the minimum error
min(oob.values)
## find the optimal value for mtry...
which(oob.values == min(oob.values))
## create a model for proximities using the best value for mtry
model <- randomForest(hd ~ ., 
                      data=data.imputed,
                      ntree=1000, 
                      proximity=TRUE, 
                      mtry=which(oob.values == min(oob.values)))

## Now let's create an MDS-plot to show how the samples are related to each 
## other.
##
## Start by converting the proximity matrix into a distance matrix.
distance.matrix <- as.dist(1-model$proximity)

mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)

## calculate the percentage of variation that each MDS axis accounts for...
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

## now make a fancy looking plot that shows the MDS axes and the variation:
mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
  X=mds.values[,1],
  Y=mds.values[,2],
  Status=data.imputed$hd)

ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) + 
  geom_text(aes(color=Status)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using (1 - Random Forest Proximities)")
# ggsave(file="random_forest_mds_plot.pdf")
