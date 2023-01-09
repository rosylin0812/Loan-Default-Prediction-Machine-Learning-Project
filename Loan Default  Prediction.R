# load data
loan.df <- read.csv("C:/Users/16787/Desktop/df.csv")
# drop X column
loan.df <- loan.df[-1]

library(tidyverse)
library(repr)
######################## split the dataset #############################

# Training / Test datasets
set.seed(1) 
train.index <- sample(c(1:dim(loan.df)[1]), 0.8*nrow(loan.df[1]))
train.df <- loan.df[train.index,]
valid.df <- loan.df[-train.index,]

########################################################################
####################### logistic regression ############################
########################################################################

options(scipen = 999)
loan.glm<- glm(TARGET~., data=loan.df, family="binomial")
summary(loan.glm)
reg.pred <- predict(loan.glm, valid.df, type='response')
reg.pred.class <- factor(ifelse(reg.pred > 0.5, 1, 0), levels=c('1', '0'))
confusionMatrix(reg.pred.class, factor(valid.df$TARGET, levels=c('1', '0')), positive='1')

##########################################################################
########### stepwise feature selection to logistic regression ############
##########################################################################
# bidirection

library(StepReg)
formula = TARGET ~ .
glm.step <- stepwiseLogit(formula, data = train.df,  include = NULL, 
                          selection = "bidirection",
                          select="SL",
                          sle=0.001,
                          sls=0.001,
                          sigMethod="Rao")
glm.step
# fit to logistic regression 
reg <- glm(TARGET ~ DAYS_BIRTH+ NAME_EDUCATION_TYPE_Higher.education+ 
             REGION_RATING_CLIENT_W_CITY+ CODE_GENDER_M+ FLAG_OWN_CAR_Y+ 
             DAYS_ID_PUBLISH + NAME_CONTRACT_TYPE_Cash.loans +REG_CITY_NOT_WORK_CITY+    
             NAME_INCOME_TYPE_State.servant+ NAME_FAMILY_STATUS_Married+ DAYS_REGISTRATION+
             NAME_INCOME_TYPE_Working+ AMT_CREDIT+ NAME_EDUCATION_TYPE_Incomplete.higher+
             REG_CITY_NOT_LIVE_CITY+ FLAG_PHONE+ FLAG_WORK_PHONE+ HOUR_APPR_PROCESS_START+ 
             NAME_INCOME_TYPE_Unemployed+ NAME_EDUCATION_TYPE_Secondary...secondary.special+
             FLAG_OWN_REALTY_N+ NAME_HOUSING_TYPE_House...apartment,data=train.df, family="binomial")
summary(reg) 

# performance
reg.pred <- predict(reg, valid.df, type='response')
reg.pred.class <- factor(ifelse(reg.pred > 0.5, 1, 0), levels=c('1', '0'))
confusionMatrix(reg.pred.class, factor(valid.df$TARGET, levels=c('1', '0')), positive='1')



########################################################################
##############  Regression with Principle Components ###################
########################################################################
library(corrplot)
plot.new(); dev.off()
corrplot(cor(loan.df))

### PCA on all 64 variables WITH normalization ###
pcs.cor <- prcomp(na.omit(train.df), scale. = T) # scale. = T is for normalization
summary(pcs.cor) 

# plot distribution of priciple components
PoV.cor <- pcs.cor$sdev^2 / sum(pcs.cor$sdev^2)
barplot(PoV.cor, xlab = "Principal Components", ylab = "Proportion of Variance Explained",ylim=c(0,0.3))

# take first 40 variables from PCA because PC40 the cumulative proportion is closed to 0.90
pcs.data<-data.frame(pcs.cor$x[,1:40], train.df$TARGET) 

pcs.data

# fit logistic regression
reg <- glm(train.df.TARGET ~ .,data=pcs.data, family="binomial")
summary(reg)
reg.pred <- predict(reg, pcs.data, type='response')
reg.pred.class <- factor(ifelse(reg.pred > 0.5, 1, 0), levels=c('1', '0'))
library(caret)
confusionMatrix(reg.pred.class, factor(valid.df$TARGET, levels=c('1', '0')), positive='1')

########################################################################
############################  Decision Tree   ##########################
########################################################################

library(rpart)
library(rpart.plot)
library(rattle)
class.tree <- rpart(TARGET ~ ., data=train.df, method='class',
                         control=rpart.control(maxdepth=6))
summary(class.tree)

fancyRpartPlot(class.tree)
rpart.rules(class.tree, extra = 4, cover = TRUE)

classification.tree.pred <- predict(classification.tree, valid.df)
classification.tree.pred
accuracy3<-accuracy(classification.tree.pred, valid.df$medv)

########################################################################
############################  Neural Network  ##########################
########################################################################
library(neuralnet)
library(NeuralNetTools)
library(nnet)
library(caret)
#install.packages("nnet")
# run neural network
nn <- neuralnet(TARGET ~ ., 
                data=train.df, hidden=c(2,3,4),learningrate=0.01,stepmax=1e6)# highest

# neural network results
plotnet(nn)
neuralweights(nn)

# neural network performance
nn.pred <- neuralnet::compute(nn, valid.df)
nn.class <- ifelse(nn.pred$net.result > 0.1, 1, 0)
#nn.class
confusionMatrix(as.factor(nn.class), as.factor(valid.df$TARGET))