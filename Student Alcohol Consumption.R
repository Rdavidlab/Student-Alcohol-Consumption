##Install and Load packages
if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)
if (!require(randomForest)) install.packages('randomForest')
library(randomForest)
if (!require(kknn)) install.packages('kknn')
library(kknn)
if (!require(plyr)) install.packages('plyr')
library(plyr) # to combine dataset
if (!require(dplyr)) install.packages('dplyr')
library(dplyr)
if (!require(caret)) install.packages('caret')
library(caret) # to evaluate model
if (!require(Boruta)) install.packages('Boruta')
library(Boruta) #variable selection to improve model

#Preprocessing data to make it more manageable
#These lines will depend on where you store the dataset.
#You might have to set file directory to your desktop or download tab to load it. 
d1 <- read.csv("datasets_251_561_student-mat.csv")
#add new column class Math
d1<-data.frame(Class="Math",d1)
d2 <- read.csv("datasets_251_561_student-por.csv")
#add new column class Math
d2<-data.frame(Class="Port",d2)
#BIND
d3<-rbind.fill(d1,d2)

#create binary data, FAIL or NOT (dichotomic approach)
d3$failures<-ifelse(d3$failures==0,0,1)%>%
  as.factor

#Initial data exploration 
#It is a good practice to do this to know the variables characteristic and names
sum(is.na(d3))
#Check data structure
str(d3)
#Check variable names 
names(d3)
#Top 6 data
head(d3)

#Random forest model 
set.seed(123,sample.kind = "Rounding")
# if using R 3.5 or earlier, use `set.seed(123)` instead
rf0 = randomForest(failures ~., # The model
                   data=d3, # The dataset
                   mtry=2) # Hyperparameter mtry

#Which variable make the model became better?
##The bor function will help us decide which variable to use, so that our model is maximised. If the variable 
##will maximise our model, it will return the value as 'confirmed'.If not, it will either return a 'tentative' or
##'rejected'.
set.seed(123,sample.kind = "Rounding")
bor = bor = Boruta(failures ~., data=d3)

bor$finalDecision[which(bor$finalDecision=="confirmed")]
# The purpose of the boruta package is to know the variable importance towards our dependent variable, which is 
#students failures.

set.seed(123,sample.kind = "Rounding")
# Using the variables selected by Boruta algorithm
rf_bor <- randomForest(failures ~ Class + age + Medu + Fedu + Mjob + guardian + studytime + paid + higher + famrel + Dalc + absences + G1 + G2 + G3,
                       data = d3)

# Checking the model
varImpPlot(rf_bor)

#Boxplot of G1,G2,G3 against failures
ggplot(d3) +
  aes(x = failures, y = G1) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

ggplot(d3) +
  aes(x = failures, y = G2) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

ggplot(d3) +
  aes(x = failures, y = G3) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()


# Checking correlation between G1, G2, G3
pairs(d3[,c("G1","G2","G3")])


# Joining the 3 variables by average grade
#G1,G2,and G3 are related to each other with very high correlation, so we can combine them into one. 
d4 <- mutate(d3, avg_Grade = (d3$G1 + d3$G2 + d3$G3)/3)
d4 <- select(d4, !c(G1,G2,G3))

#we will try to see if our model accuracy is better
# In this case, its nothing better...
set.seed(123,sample.kind = "Rounding")
rf_bor2 <- randomForest(failures ~ Class + age + Medu + Fedu + Mjob + guardian + studytime + paid + higher + famrel + Dalc + absences + avg_Grade,
                        data = d4)


#checking for error rate:
rf_bor
rf_bor2


ggplot(d3) +
  aes(x = failures, y = age) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

ggplot(d3) +
  aes(x = failures, y = absences) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

ggplot(d3) +
  aes(x = failures, y = famrel) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

#Separating data into test and training set
set.seed(123,sample.kind = "Rounding")
idTrain = createDataPartition(d3$failures,
                              p = 0.7,
                              list = FALSE)

train = d3[idTrain,] # 70%
test = d3[-idTrain,] # 30%

#Finalize our model
set.seed(123,sample.kind = "Rounding")
rf_final <- randomForest(failures ~ Class + age + Medu + Fedu + Mjob + guardian + studytime + paid + higher + famrel + Dalc + absences + G1 + G2 + G3, data = train)

rf_eval <- confusionMatrix(test$failures,
                           predict(rf_final,
                                   test))

rf_eval



#2nd Model:KNN

results = list(k = rep(0,100), Accur = rep(0,100))

for (i in 2:101){
  my_kknn <- kknn(failures ~ Class + age + Medu + Fedu + Mjob + guardian + studytime + paid + higher + famrel + Dalc + absences + G1 + G2 + G3, train, test, k = i)
  
  kknn_eval <- confusionMatrix(test$failures, my_kknn$fitted.values)
  
  results$k[i] = i
  results$Accur[i] = kknn_eval$overall[1]
  
}

# K vs Accuracy
plot(results$k[-1], results$Accur[-1], type="l",
     xlab="k", ylab="Accuracy")

best.K = results$k[which.max(results$Accur)]

#input the best value of k (best.k) into the model
best_kknn <- kknn(failures ~ Class + age + Medu + Fedu + Mjob + guardian + studytime + paid + higher + famrel + Dalc + absences + G1 + G2 + G3, train, test, k = best.K)

(best_kknn_eval <- confusionMatrix(test$failures, best_kknn$fitted.values))

#Results
rf_eval #random forest results
best_kknn_eval #knn results 


#Citation: 
# P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

#Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

#BiBTeX citation:
#  @misc{Dua:2019 ,
#   author = "Dua, Dheeru and Graff, Casey",
#   year = "2017",
#   title = "{UCI} Machine Learning Repository",
#   url = "http://archive.ics.uci.edu/ml",
#   institution = "University of California, Irvine, School of Information and Computer Sciences" }


print("Operating System:")
version