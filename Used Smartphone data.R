##### *****************Used Mobile devices dataset****************
library(dplyr)
library(tidyverse)
library(lessR)
library(tidyr)
library(corrplot)
used_device_data <- read.csv("C:/Users/deeps/OneDrive/Documents/WEBSTER/Analytics Practicum/Used smartphone/used_device_data.csv")
dim(used_device_data)
class(used_device_data)
head(used_device_data)
names(used_device_data)
View(used_device_data)

# We can calculate how many NAs there are in each variable by using the map() in the purrr package
library(purrr)
used_device_data %>% 
  map(is.na) %>%
  map(sum)

####Handling missing values NA's.
##install.packages("VIM")
library(VIM)
head(used_device_data)
used_device_data_imp <- kNN(used_device_data, imp_var = FALSE)
#checking for missing values
missing_values <- sum(is.na(used_device_data_imp))
missing_values
head(used_device_data_imp)

#checking for zero values
zero_values <- sum(used_device_data_imp == 0, na.rm = TRUE)
zero_values
###check for zero values column wise
colSums(used_device_data_imp==0)
str(used_device_data_imp)
  summary(used_device_data_imp)

###**************Exploratory Data Analysis *********************
###Visualizing the distribution of each variable.

library(ggplot2)
ggplot(data=used_device_data_imp) + geom_bar(mapping = aes(x = os), fill = "brown")
ggplot(data=used_device_data_imp) + geom_bar(mapping = aes(y = device_brand),fill= "brown")
hist(used_device_data_imp$battery, main = "battery distribution", xlab = "battery", col = "skyblue", border = "black")
ggplot(data=used_device_data_imp) + geom_boxplot(mapping = aes(x = battery))  ##Outliers
ggplot(data=used_device_data_imp) + geom_bar(mapping = aes(x = X5g), fill = "brown")
ggplot(data=used_device_data_imp) + geom_bar(mapping = aes(x = X4g), fill = "brown")
hist(used_device_data_imp$screen_size, main = "screen size distribution", xlab = "screen_size", col = "skyblue", border = "black")
ggplot(data=used_device_data_imp) + geom_boxplot(mapping = aes(x = screen_size))  ##Outliers
head(used_device_data_imp)

hist(used_device_data_imp$front_camera_mp, main = "front camera distribution", xlab = "front_camera_mp", col = "skyblue", border = "black")
ggplot(data=used_device_data_imp) + geom_boxplot(mapping = aes(x = front_camera_mp))
hist(used_device_data_imp$rear_camera_mp, main = "rear camera distribution", xlab = "rear camera", col = "skyblue", border = "black")
ggplot(data=used_device_data_imp) + geom_boxplot(mapping = aes(x = rear_camera_mp))
hist(used_device_data_imp$internal_memory, main = "internal memory distribution", xlab = "internalmemory", col = "skyblue", border = "black")
ggplot(data=used_device_data_imp) + geom_boxplot(mapping = aes(x = internal_memory)) ##Outliers
hist(used_device_data_imp$weight, main = "weight distribution", xlab = "weight", col = "skyblue", border = "black")
ggplot(data=used_device_data_imp) + geom_boxplot(mapping = aes(x = weight)) ##Outliers

hist(used_device_data_imp$ram, main = "ram distribution", xlab = "ram", col = "skyblue", border = "black")
hist(used_device_data_imp$release_year, main = "release_year distribution", xlab = "release_year", col = "skyblue", border = "black")
hist(used_device_data_imp$days_used, main = "days_used distribution", xlab = "days_used", col = "skyblue", border = "black")

#### Distribution of Normalized new and used price 
ggplot(used_device_data_imp, aes(x = normalized_new_price)) +  geom_density(fill = "skyblue", color = "black") +
  labs(title = "Density Plot: normalized_new_price", x = "normalized_new_price", y = "Density")
ggplot(used_device_data_imp, aes(x = normalized_used_price)) +  geom_density(fill = "skyblue", color = "black") +
  labs(title = "Density Plot: normalized_used_price", x = "normalized_used_price", y = "Density")

# Create a basic correlation plot
correlation_matrix <- cor(used_device_data_imp[,-c(1,2,4,5,12)])
corrplot(correlation_matrix, method = "number")
##In numeric columns, 'normalized_used_price' have strong correlation with normalized_new_price >0.8, battery >0.6,screen_size >0.6,
#front_camera_mp >0.5,rear_camera_mp >0.5

####VIF
vif_values <- car::vif(lm(normalized_used_price ~ ., data = used_device_data_imp))
vif_values

####Checking multicollinearity
library("olsrr")
mymodel <-lm(normalized_used_price~., data=used_device_data_imp)
ols_vif_tol(mymodel)
##Device_brandApple, osios has high multicollinearity which is obvious because device brand "apple" only launches iOS operating system.

###*********Data Analysis***************
##install.packages("ggplot2")
library(ggplot2)
###When ram is more than 4 normalized used price will be higher.
plot(used_device_data_imp$ram, used_device_data_imp$normalized_used_price, main = "Scatter Plot: RAM vs. normalized_used_price", xlab = "RAM", ylab = "normalized_used_price", col = "black")
### Used mobile devices price goes up with the increase in phone screen size.
plot(used_device_data_imp$screen_size, used_device_data_imp$normalized_used_price, main = "screen_size vs. normalized_used_price Distribution", xlab = "screen size", col = "black", border = "black")
### 
plot(used_device_data_imp$rear_camera_mp, used_device_data_imp$normalized_used_price, main = "Plot: rear_camera_mp vs. normalized_used_price", xlab = "rear_camera_mp", ylab = "normalized_used_price", col = "black")
## varies in wide range when Front camera is >5 tends to stay in higher used price.
plot(used_device_data_imp$front_camera_mp, used_device_data_imp$normalized_used_price, main = "Scatter Plot: front_camera_mp vs. normalized_used_price", xlab = "front_camera_mp", ylab = "normalized_used_price", col = "black")
#### 4G enabled phones has higher camera mp and prices tends to be little higher.
ggplot(used_device_data_imp,aes(x=front_camera_mp,y=normalized_used_price,col=X4g))+geom_point()
ggplot(used_device_data_imp,aes(x=rear_camera_mp,y=normalized_used_price,col=X4g))+geom_point()
ggplot(used_device_data_imp,aes(x=os,y=normalized_used_price))+geom_point()
### 2019and 2020 have 5g enabled phones and their prices are relatively higher than non-5g enabled phones.
ggplot(used_device_data_imp,aes(x=release_year,y=normalized_used_price,col=X5g))+geom_point()
### Price is slightly increasing with increase in years but old phones are not 4g enabled.
ggplot(used_device_data_imp,aes(x=release_year,y=normalized_used_price,col=X4g))+geom_point()
ggplot(used_device_data_imp,aes(x=X4g,y=normalized_used_price))+geom_point()
ggplot(used_device_data_imp,aes(x=X5g,y=normalized_used_price))+geom_point()

###Days used and difference in price 
#obivious downward trend in price with increase in phone used days and 4g enabled phones have comparatively higher price even if they have been used for larger days.
ggplot(used_device_data_imp,aes(x=days_used,y=normalized_used_price,col=X4g))+geom_point() 

###*********Data Analysis***************
###Feature Selection - Dimension reduction
#### When performed Boruta in no dummy variables it showed all brandS are are important. After creating dummy variables it was found that 16 attributes are unimportant.
#install.packages("Boruta")
library(Boruta)
set.seed(123)
# !!!!!!!takes around 6 second!!!!!!
featureslctn<-Boruta(normalized_used_price~.-normalized_new_price, data=used_device_data_imp, doTrace=0)
print(featureslctn)
featureslctn$ImpHistory[1:6, 1:14]

plot(featureslctn, xlab="", xaxt="n")
lz<-lapply(1:ncol(featureslctn$ImpHistory), function(i)
  featureslctn$ImpHistory[is.finite(featureslctn$ImpHistory[, i]), i])
names(lz)<-colnames(featureslctn$ImpHistory)
lb<-sort(sapply(lz, median))
axis(side=1, las=2, labels=names(lb), at=1:ncol(featureslctn$ImpHistory), cex.axis=0.6, font = 4)

library(plotly)
df_long <- tidyr::gather(as.data.frame(featureslctn$ImpHistory), feature, measurement)

plot_ly(df_long, y = ~ measurement, color = ~feature, type = "box") %>% layout(title="Box-and-whisker Plots across all 13 Features (ALS Data)",
  xaxis = list(title="Features"), yaxis = list(title="Importance"), showlegend=F)

###*************DATA PREPARATION**********************

###** Converting Release year variable into categorical because they represent year and are not numbers.
used_device_data_imp$release_year <- factor(used_device_data_imp$release_year)
str(used_device_data_imp)
###****Creating new column as %Price drop 
Percent_PriceDrop <- (used_device_data_imp$normalized_new_price - used_device_data_imp$normalized_used_price)/used_device_data_imp$normalized_new_price *100
summary(Percent_PriceDrop)
used_device_data_imp$Percent_PriceDrop <- Percent_PriceDrop
summary(used_device_data_imp$Percent_PriceDrop)

#Percentage drop 
meanprice = mean(used_device_data_imp$Percent_PriceDrop)
meanprice
CAT.Used_Price <- filter(used_device_data_imp, Percent_PriceDrop>(mean(used_device_data_imp$Percent_PriceDrop)))
used_device_data_imp<- used_device_data_imp %>% mutate(CAT.Used_Price = case_when(
  Percent_PriceDrop> meanprice ~ 1,
  Percent_PriceDrop<= meanprice ~ 0
))
tail(used_device_data_imp)
used_device_data_imp$CAT.Used_Price <- factor(used_device_data_imp$CAT.Used_Price,levels = c("1", "0"))
summary(used_device_data_imp)
###Here, Class 1 is high percentage drop in price means low priced used phone which is value for money
#and Class 0 is low percentage drop in price means high priced phone which is not feasible to business. 

library(tidyverse)
library(fastDummies)

#Creating new dataframe with name "used_device_data.df" for dummies for KNN algorithm.
used_device_data.df <- used_device_data_imp %>%  dummy_cols(select_columns=c('os'), remove_selected_columns=TRUE, remove_first_dummy=FALSE)
used_device_data.df <- used_device_data.df %>%  dummy_cols(select_columns=c('X4g'), remove_selected_columns=TRUE, remove_first_dummy=FALSE)
used_device_data.df <- used_device_data.df %>%  dummy_cols(select_columns=c('X5g'), remove_selected_columns=TRUE, remove_first_dummy=FALSE)
used_device_data.df <- used_device_data.df %>%  dummy_cols(select_columns=c('device_brand'), remove_selected_columns=TRUE, remove_first_dummy=FALSE)
head(used_device_data.df)

####****************Data Partitioning********************
set.seed(1234)
#Data preparation: creating random training and test datasets
##Partition the data randomly into training (60%), validation (20%), holdout (20%)
# randomly sample 40% of the row IDs for training
train.rows <- sample(rownames(used_device_data_imp), nrow(used_device_data_imp)*0.60)

# sample 20% of the row IDs into the validation set, drawing only from records not already in the training set
# use setdiff() to find records not already in the training set
valid.rows <- sample(setdiff(rownames(used_device_data_imp), train.rows),
                     nrow(used_device_data_imp)*0.20)

# assign the remaining 20% row IDs serve as holdout
holdout.rows <- setdiff(rownames(used_device_data_imp), union(train.rows, valid.rows))

train_usedmobile <- used_device_data_imp[train.rows, ]
valid_usedmobile <- used_device_data_imp[valid.rows, ]
holdout_usedmobile <- used_device_data_imp[holdout.rows, ]

###For dummies
traindummy <- used_device_data.df[train.rows,]
validdummy <- used_device_data.df[valid.rows, ]
holdoutdummy <- used_device_data.df[holdout.rows, ]

#### Fitting Multiple linear regression model 
used_device.lm_model <- lm(normalized_used_price~.-(CAT.Used_Price+Percent_PriceDrop) ,data = train_usedmobile)
summary(used_device.lm_model)

##model performance on validation
predvalid_lmmodel <- predict(used_device.lm_model, valid_usedmobile)
library(forecast)
accuracy(predvalid_lmmodel,valid_usedmobile$normalized_used_price)
cor(predvalid_lmmodel,valid_usedmobile$normalized_used_price)

##Improving model performance
###By Stepwise regression
backward=step(used_device.lm_model,direction = "backward")
summary(backward)
predvalid_stepmodel <- predict(backward, valid_usedmobile)
## Compute validation accuracy for Stepwise BackwaRD Regression model.
library(forecast)
accuracy(predvalid_stepmodel,valid_usedmobile$normalized_used_price)
cor(predvalid_stepmodel,valid_usedmobile$normalized_used_price)

##********Training a Regression Tree Model On the Data
library(rpart)
used_price.rpart<-rpart(normalized_used_price~.-(CAT.Used_Price+Percent_PriceDrop),data = train_usedmobile)
used_price.rpart

# Visualizing Regression Trees
library(rpart.plot)
rpart.plot(used_price.rpart, digits=3)
rpart.plot(used_price.rpart, digits = 4, fallen.leaves = T, type=3, extra=101)
library(rattle)
fancyRpartPlot(used_price.rpart, cex = 0.8)
###Evaluating performance on validation dataset
library(caret)
pred.rpart_used<- predict(used_price.rpart,valid_usedmobile)
accuracy(pred.rpart_used,valid_usedmobile$normalized_used_price)
cor(pred.rpart_used,valid_usedmobile$normalized_used_price)

#####************************ Classification Model *****************************
###Here, 1 is high percentage price drop as low price of used phone and "0" as low percent price drop that means it is high priced phone. 
##**Logistic Regression
trControl <- caret::trainControl(method="cv", number=5, allowParallel=TRUE)
logit.regression <- caret::train(CAT.Used_Price~.-(Percent_PriceDrop+normalized_used_price+normalized_new_price),
                                 data = train_usedmobile, trControl=trControl,method="glm", family="binomial")
logit.regression
summary(logit.regression)

logitreg.pred <- predict(logit.regression, valid_usedmobile, type="prob")
logit.reg.pred <- ifelse(logitreg.pred[,1] >0.5, 1, 0)
confusionMatrix(as.factor(logit.reg.pred), as.factor(valid_usedmobile$CAT.Used_Price), positive = "1")

##**Training a Decision Tree Model On the Data*
# install.packages("C50")
library(C50)
set.seed(1234)
C_usedmobile_model<-C5.0(train_usedmobile[,-c(14,15,16,17)], train_usedmobile$CAT.Used_Price)
C_usedmobile_model
summary(C_usedmobile_model)
plot(C_usedmobile_model)

# Evaluating Decision Tree Model Performance
C_usedmobile_pred<-predict(C_usedmobile_model, valid_usedmobile[ ,-c(14,15,16,17)])  # removing the last 2 columns and used price
library(caret)
confusionMatrix(table(C_usedmobile_pred, valid_usedmobile$CAT.Used_Price), positive = "1")

## Step Model improvement: Trial Option
set.seed(1234)
C_usedmobile_model7<-C5.0(train_usedmobile[,-c(14,15,16,17)], train_usedmobile$CAT.Used_Price, trials=7) 
C_usedmobile_model7
plot(C_usedmobile_model7, type="simple")
C_usedmobile_model7_pred7 <- predict(C_usedmobile_model7, valid_usedmobile[ ,-c(14,15,16,17)])
confusionMatrix(table(C_usedmobile_model7_pred7, valid_usedmobile$CAT.Used_Price), positive = "1")

##****Run Naive Bayes Model***
library(e1071)
usedmobile.nb <- naiveBayes(CAT.Used_Price~.-(Percent_PriceDrop+normalized_used_price+normalized_new_price),data = train_usedmobile)
usedmobile.nb
# evaluate on validation dataset
pred_NB <- predict(usedmobile.nb, valid_usedmobile)
confusionMatrix(data= as.factor(pred_NB), as.factor(valid_usedmobile$CAT.Used_Price), positive = "1")

###******KNN MODEL*****
library(gmodels)
library(class)
###Used dummy variable dataframe.
Knn_valid_pred<-knn(train=traindummy[,-c(10,11,12,13)], test=validdummy[,-c(10,11,12,13)], cl=traindummy$CAT.Used_Price, k=5)
# Step: Evaluating model performance on validation set
confusionMatrix(table(Knn_valid_pred, validdummy$CAT.Used_Price), positive = "1")

library(e1071)
set.seed(123)
knntuning = tune.knn(x= traindummy, y = traindummy$CAT.Used_Price, k = 1:30)
knntuning ##k=20
plot(knntuning)
summary(knntuning)
Knn_valid_pred.20<-knn(train=traindummy[,-c(10,11,12,13)], test=validdummy[,-c(10,11,12,13)], cl=traindummy$CAT.Used_Price, k=20)
confusionMatrix(table(Knn_valid_pred.20, validdummy$CAT.Used_Price), positive = "1")

### Model Evaluation on Holdout set
###**********Selected stepwise backward Model Evaluation on Holdout********
## Compute holdout accuracy on prediction set for Stepwise BackwaRD Regression model.
stepwise.lmholdpred<- predict(backward, holdout_usedmobile ,type = "response")
accuracy(stepwise.lmholdpred, holdout_usedmobile$normalized_used_price)

###**********Selected Classification Model i.e. Logistic model Evaluation on Holdout********
## Selected model was Logistic model, will evaluate its performance on Holdout
logit.reg.predhold <- predict(logit.regression, holdout_usedmobile)
confusionMatrix(logit.reg.predhold, holdout_usedmobile$CAT.Used_Price,positive = "1")

###*****Unsupervised Model**** 

# Data Preparation for clustering.
used_device_data_df <- used_device_data_imp %>%  dummy_cols(select_columns=c('os'), remove_selected_columns=TRUE, remove_first_dummy=TRUE)
used_device_data_df <- used_device_data_df %>%  dummy_cols(select_columns=c('X4g'), remove_selected_columns=TRUE, remove_first_dummy=TRUE)
used_device_data_df <- used_device_data_df %>%  dummy_cols(select_columns=c('X5g'), remove_selected_columns=TRUE, remove_first_dummy=TRUE)
used_device_data_df <- used_device_data_df %>%  dummy_cols(select_columns=c('device_brand'), remove_selected_columns=TRUE, remove_first_dummy=TRUE)
str(used_device_data_df)
used_device_data.cluster <- used_device_data_df[,c(1,2,3,4,5,6,7,9,10,17,18)]
used_device_data.cluster <- scale(used_device_data.cluster)

library(dplyr)
k <- 4
#used_device_data.cluster <- used_device_data_df[,c(1,2,3,4,5,6,7,9,10,17,18)]
#used_device_data.cluster <- scale(used_device_data.cluster)
#elbow curve
wss <- numeric(10)
for (i in 1:10) {
  wss[i] <- sum(kmeans(used_device_data.cluster, centers = i)$withinss)
}
plot(1:10, wss, type = "b", xlab = "Number of Clusters", ylab = "Within-cluster Sum of Squares (WSS)", main = "Elbow Curve")

###*************************
library(stats)
set.seed(321)
km<-kmeans(used_device_data.cluster, 4)
km$size
require(cluster)
dis = dist(used_device_data.cluster)
sil = silhouette(km$cluster, dis)
summary(sil)
plot(sil, col=c(1:length(km$size)))
#plot(sil, col=c(1:length(diz_clussters$size)), border = NA)
factoextra::fviz_silhouette(sil, label=T, palette = "jco", ggtheme = theme_classic())
km$centers

par(mfrow=c(1,1), mar=c(4, 4, 4, 2))
myColors <- c("darkblue", "red", "green", "brown", "pink", "purple", "yellow","skyblue","orange","blue","gray")
barplot(t(km$centers), beside = TRUE, xlab="cluster", 
        ylab="value", col = myColors)
legend("bottomleft", ncol=2, legend = c("screen_size", "rear_camera_mp", "front_camera_mp", "internal_memory", "ram", "battery", "weight","days_used","normalized_used_price","X4g_yes","X5g_yes"), fill = myColors)

kmt <- as.data.frame(t(km$centers))
rowNames <- rownames(kmt)
colnames(kmt) <- paste0("Cluster",c(1:4))
library(plotly)
plot_ly(kmt, x = rownames(kmt), y = ~Cluster1, type = 'bar', name = 'Cluster1') %>% 
  add_trace(y = ~Cluster2, name = 'Cluster2') %>% 
  add_trace(y = ~Cluster3, name = 'Cluster3') %>%   add_trace(y = ~Cluster4, name = 'Cluster4') %>%  
  layout(title="Explicating Derived Cluster Labels",
         yaxis = list(title = 'Cluster Centers'), barmode = 'group')
