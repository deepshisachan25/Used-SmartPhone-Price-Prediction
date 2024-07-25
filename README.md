# Used-SmartPhone-Price-Prediction
## Introduction
This project focuses on how much used mobile phones cost, with an emphasis on all the different factors (variables) that influence consumer decision-making. In today's dynamic smartphone market, pricing strategies are not solely determined by basic features like screen size or camera quality. Lots of different features affect how much a new or used phone costs. Figuring out what these factors are and how they're connected can give us useful information for both the people selling phones and the people buying them. Ultimately, the project aims to assist both providers and consumers in making informed decisions about pricing strategies and purchasing choices within the used mobile phone industry. This project aims to gain insights into the pricing dynamics of used mobile phones by examining a wide range of variables that influence consumer decisions.

## Business Goal
Here, our business goal is for businesses who sell used mobile phones at competitive prices. The business wants to generate higher revenue by displaying good deals to customers.
•	The business goal of this project is to find out what combination of factors really decide how much the used mobile device will cost. This might help businesses in setting up the prices of used mobile devices. The project aims to understand how these factors affect the pricing of used phones.
•	Classify if the percent drop in price is high or low of the used mobile devices. This will allow businesses to provide consumers with information around good deals (cheaper prices) vs costly premium options.
•	Business aims to identify groups of used mobile devices with similar price determinants. By grouping used mobile devices based on shared or similar features and pricing dynamics, businesses can optimize pricing strategies, inventory management, and purchasing decisions.
With accurate price predictions, used mobile device businesses and dealers can set prices that are attractive to customers. Also, businesses can make good decisions on buying devices from third party and reselling purposes based on High and Low price of used mobile devices. 

## Analytics Goal
Here, the analytical goals are:
•	Predict the prices of used mobile devices and identify different predictors that impact the pricing of used mobile devices.
•	Classify if the percentage drop in price of the used mobile devices is high or low. This will allow businesses to provide consumers with information around good deals (cheaper prices) vs costly premium options. 
•	Identify different combination of features using clustering methods in used mobile devices that could have impact on used device price. In clustering analysis, identifying features that drive the price of used mobile devices can help group similar devices together based on their price determinants. 
To achieve these goals, we will create regression model which will predict the price of used mobile devices and will create another classification model which will classify the price of used smartphone as low and high prices which will be derived from percentage change between normalized_new_price and normalized_used_price variable and will name it as “CAT_Used_Price”. CAT_Used_Price variable is for devices whose percentage change in used mobile devices is high or low. Important class will be high percentage drop (1) because business aim is to find cheaper phones with good features so that more and more users get attracted to their store/websites seeing the cheaper phones. 
By building predictive models such as regression or classification models, we can help business make smarter decisions on how to set the prices of mobile devices while ensuring how to generate higher revenue and what type of phones to buy from third party so that they will likely sell most. (as businesses can be a purchaser too if they are buying used phones from third party). 
Data Preprocessing step includes- 
Attributes defined:
•	device_brand: Name of manufacturing brand. This variable tells the name of the manufacturing brand of mobile device like honor, etc.
•	os: This OS variable tells the operating system of phones on which the device runs.
•	screen_size: This variable explains the Size of the screen of mobile phones in centimeters. 
•	4g: This variable tells whether 4G is available or not in mobile devices which means 4th generation of broadband network technology which is succeeding 3G and preceding 5G. 
•	5g: This variable tells whether 5G is available or not in mobile device.
•	front_camera_mp: This variable tells the resolution of the front camera in megapixels.
•	rear_camera_mp: This variable tells the resolution of the rear/ back camera in megapixels.
•	internal_memory: This variable explains the amount of internal memory (ROM) available in Giga Byte.
•	ram: This variable explains the amount of RAM present in GB.
•	battery: Energy capacity of the device battery in mAh.
•	weight: This variable explains the weight of the device (mobile device) in grams.
•	release_year: This variable tells in which Year the smartphone model was released in the market.
•	days_used: This variable tells the Number of days the used/refurbished device has been used.
•	normalized_new_price: This variable tells the normalized price of a new device of the same model. 
•	normalized_used_price (TARGET): This target variable tells the normalized price of the used/refurbished device.

## DATA EXPLORATION
Used Smartphone dataset is imported on R for data analysis. Exploring and analysis the data is a crucial step to uncover patterns within the dataset. In data exploration we look for number of rows and columns in the dataset. We found the number of rows and columns the data frame has is 3454 rows and 15 variables. Using names () function we can find the names of the variable in the dataset.  
Handling Missing values: There were some missing values in a few columns like rear_camera_mp, front_camera__mp, internal_memory, ram, battery, weight. 
We have handled NA values using advanced methods. Used the kNN() function from the VIM package to impute the missing values using the KNN method. It will find the observations that are most similar and have the values and use these cases to make the imputation.  
![image](https://github.com/user-attachments/assets/24dfc373-dc52-479b-863c-028566dbc5dd)
![image](https://github.com/user-attachments/assets/dd3cf272-1157-4bf4-ac3d-0ee08a14c379)

Summary statistics of used smartphone dataset is given below. It shows the mean, median and quartile range of each variable. Based on summary statistics, we can identify the screen size of mobile devices in our dataset are between 5.08cm to 30.71cm. Some mobile devices have a minimum 0.08 MP rear camera and there are some devices whose back camera is 48MP. Similarly, we can identify some mobile devices do not have front camera as their minimum is 0 and some mobile devices in our dataset have 32MP front camera. Based on the summary statistics, we can interpret the variables range and can also detect the outliers. The structure and summary statistics of the dataset is shown below that shows the mean, median, min, max value of each varaiable. It also tells which column is categorical in nature, which is numerical.
![image](https://github.com/user-attachments/assets/66bcc2b2-ea19-4f81-94ab-48cba1d84198)

## Exploratory Data Analysis
Now we will visualize the distribution of each variable. 
Below OS plot shows the operating system of the mobile devices. Most of our data is of Android operating system which means in our data many mobile devices are Android and comparatively less of iOS, Windows, and others. Here, windows and others show that our data has few tablets records in dataset.
![image](https://github.com/user-attachments/assets/9fec5075-6653-4ddb-8fe4-832088249214)
![image](https://github.com/user-attachments/assets/bc4e7c4f-d9a3-4dbc-a60a-6a35268ed7a5)
In Above plot, there are all the mobile device brands. Will convert this categorical variable into dummy variable during data preparation for modelling. Most of our smartphone’s data is of Samsung, Huawei, Others and LG brand. 
![image](https://github.com/user-attachments/assets/59a05d99-f416-48e4-8556-1c5d87626b52)
![image](https://github.com/user-attachments/assets/578b3409-2532-4bde-8a7e-0a0880956fa5)
This battery plot is right skewed which means most of our data falls on right side which is between 2000 to 10000 range of battery mAh. 
  

![image](https://github.com/user-attachments/assets/2d5243eb-a659-4127-b98c-4494d08f6529)

![image](https://github.com/user-attachments/assets/80d1874a-4624-4548-a708-4458d79079b8)

Here we can see in plots, X5g column we have most of the smartphone’s data without 5G connection. Only 5% of the smart phones in our data have 5G connections rest are No 5G compatible. And in X4g column, we have most of the smartphone’s data with 4G connection. Almost 70% of the smart phones in our data have 4G connections rest are with No 4G.
![image](https://github.com/user-attachments/assets/26e2e537-9d0d-4963-aa71-be32ecad4a13)
Only 4.4% of phones are both 4G and 5G compatible devices. All the mobile devices that are 5G compatible are also 4G compatible devices. 

Screen size distribution shows the size of the mobile phone screens. We can see most of our screen sizes fall in the range of 5 to 20 cm and few records have screen sizes above 20cm. In the real world, mobile screens range below 20cm. So here we are considering all the sizes above 20cm as Tablet. 

![image](https://github.com/user-attachments/assets/f1bac501-2036-4517-9796-1521a5f1804c)

Here, we can see many records have 0 MP which means there are some mobile devices that do not have front camera features in them. Few phones have cameras whose megapixel is above 20mp which we can consider here as an outlier which we will handle later. 

![image](https://github.com/user-attachments/assets/7d456d15-53be-4bb7-a4d7-29dd35f42951)

The above rear camera MP column shows some outliers. In this dataset, we have very few records (almost 5) which have rear_camera_mp above 40 megapixels. So, we are considering them as an outlier in the current scenario and will remove them. 

![image](https://github.com/user-attachments/assets/abe0c907-9c10-41a5-81a2-655ce61359f8)

![image](https://github.com/user-attachments/assets/909406f3-bd93-446e-a8ab-f24b2d7b881b)

The internal memory column shows how much built in internal memory mobile device has. Here we have most of the records with 0-100 GB. And here can see potential outliers in our dataset. Here, Internal memory 1024 is the outlier for our current analysis as most of our data is for internal memory below 520. 

![image](https://github.com/user-attachments/assets/a9638a81-ab5b-4ccb-b851-4da41d4a2b1a)
In Fig. 1.20, the weight column shows the weight of our mobile device. Most of our smartphone’s weight lies in the range of 100-200 grams. Fig 1.20 shows there are potential outliers in our dataset which we will handle by normalizing our data using z-score normalization method.
![image](https://github.com/user-attachments/assets/09c22ef1-0070-4f5d-ba11-ef299049313e)

![image](https://github.com/user-attachments/assets/ffe2034e-f291-4cbc-a4d7-127159ad99b8)
 
Above Fig shows the ram of mobile devices in our dataset. The above distribution shows that most of our data is of 4 GB ram. There are some outliers in data with Ram = 12GB.

![image](https://github.com/user-attachments/assets/56d4f45b-03a0-4cd3-9039-59be7a07495d)
Fig  shows the distribution of release year column which tells the year when mobile devices were released. Most of our data of mobile devices is of release_year 2014 which is 10 years back. This is a categorical variable. We will convert release_years numerical column into categorical column.

![image](https://github.com/user-attachments/assets/794c99a5-4a95-4d82-b882-e6fec3efe93e)

Here in Figure 1.23 we can see that most of the used mobile devices were used for around 500 to 1000 days. 
Fig. 1.24 below shows the density plot of normalized new price. We can clearly see that normalized new prices are uniformly distributed.

![image](https://github.com/user-attachments/assets/b0af7207-696d-4060-ae84-6d43e9ab8220)

Fig. 1.25 shows the density plot of normalized used price. Which is the target variable. We can clearly see that normalized used price is uniformly distributed but it is slightly left skewed.

![image](https://github.com/user-attachments/assets/67eded8c-9b69-4e94-9c48-e01cd53af6b9)

## PREDICTOR ANALYSIS AND RELEVANCY
Out of 15 variables, we are taking “Normalized_Used_Price” as the outcome/ target variables which is dependent on other predictor, which will predict the price of used mobile devices based on the different features of used smartphones. We will build regression and classification model which will predict the prices of new data of used mobile devices and classify them as High Price and Low price based on percentage drop in price from new price. 
We will then check for correlation between our variables using correlation matrix. Below correlation plot shows the correlation of each variable with other variables. 
•	We can see that variables like battery and weight are highly correlated with screen size. Which means with an increase in size of the phone weight and battery will rise. 
•	We can also see that normalized used prices are strongly correlated with screen size, rear_camera_mp, front_camera_mp, ram, battery. Which tells that the price of phone will increase with an increase in phone camera resolution, screen size, and RAM.
•	Similarly, normalized new price is strongly correlated with rear_camera_mp, ram, battery and normalized used price. And normalized new price is moderately correlated with screen size, front_camera_mp, battery.
•	Days used are negatively correlated with used price which is obvious it tells the trend in price with an increase in phone used days.

![image](https://github.com/user-attachments/assets/baebb967-3e30-4537-8b73-08a691c00d6b)
There is no multicollinearity between our variables.
Using Variance Inflation Factor, we can find that there is only high collinearity in device brand apple, and os_iOS which is obvious because apple brand launches their devices in iOS operating systems. Other than this no collinearity was found.


![image](https://github.com/user-attachments/assets/84d169b5-cc76-46f1-8593-9f0e7ca73035)

![image](https://github.com/user-attachments/assets/a6b10b9c-b344-42ce-8622-a21f7674121f)

These two plots show that 4G enabled phones have higher camera both rear and front mega pixel and prices of used mobile devices tend to be little higher when compared with No 4G enabled phones.

![image](https://github.com/user-attachments/assets/5eb1b221-c28b-41de-be60-6e0e43b29a5a)
Fig 3.6 represents that mobile devices that have iOS operating systems have higher price of used mobile devices. And Android phones range at different prices starting from 2.5 to 6 and few android devices have the highest price when compared with iOS operating systems phones.

![image](https://github.com/user-attachments/assets/be727605-4b0d-4481-b59d-e9f8a83343ca)
Figure above shows that prices of used mobile devices are lower that are not 4G compatible. Many mobile devices were 4G compatible from year 2015 and their prices were higher when compared to non 4G enabled devices. 
Figure below shows that downward trend in price of used mobile devices with increase in phone used days and we can clearly tell 4g enabled phones have comparatively higher price even if they have been used for longer days.

![image](https://github.com/user-attachments/assets/6a108e3f-7ec8-496e-a28e-f2be3d3f360c)

## Dimension Reduction
Dimension Reduction is done to preserve the most important information in the data while reducing its dimensionality. We have used the Boruta features selection method, and it was found all variables are important in our dataset and no variable deemed unimportant. So, no dimension reduction is required.
Using Boruta feature selection we found the top features in our dataset that contribute more to normalized used price are rear_camera_mp, weight, screen_size, internal_memory, battery, ram, front_camera_mp, days_used, etc. 
Based on important features we can make clusters of mobile devices into groups with high, medium, & low features phones.

![image](https://github.com/user-attachments/assets/fd24de79-3c91-4003-a5fe-02b73b467677)

## DATA PREPARATION
We have converted release year into categorical variable. We created a new variable named “Percent_PriceDrop” which tells the percentage drop in price of used mobile phones when compared from normalized new price. We will create a new variable named “CAT.Used_Price which will be Outcome Variable for Classification Models with Class 1 (High % Price drop) and 0 (Low % Price Price) which is derived from “Percent_PriceDrop” variable. Classes are defined based on the mean value of Percent_PriceDrop. The value above average of the Percent_PriceDrop will categorize as High % Price drop then price of used devices will be low price which is value for money (1) and below & equal to mean value is Low % Price drop which means their price is high (0) which is not value for money for customer. Here, the mean of percent price drop is 16.425. 
  
We are using the mean value as the threshold that classifies into High and low price of used mobile devices. Here, the maximum price drop (percentage change in price from new to used) was 57.33%. The 3rd quartile value is 20.4997 which means there was almost a 20.5% drop in price from the new price and mean is 16.425 which tells that average of price drop is 16.425%. This value makes more sense to me as used mobile devices with less than this price drop seems much closer to new phone price and hence will be classified as high-priced mobile devices.

## DATA PARTITIONING
Partitioning is done so that we can build a model which will predict the price of used mobile devices on new data. Data partitioning is essential for evaluating the performance of machine learning models. 
Using R, we are going to train the machine learning model, after fitting model on the training dataset, the model is evaluated on the validation dataset. This step helps to fine-tune the model & access its performance. Holdout dataset is reserved to assess the final performance of the model after training and validation is done. It is used to evaluate how well the model performs on new, unseen data. It gives an estimate of how well the model will work in real-life situations.
We have randomly sampled the data and divided dataset in Training, Validation and Holdout dataset: 60% of rows for training, 20% of rows for validation and remaining 20% for holdout datasets. 
This step is done so that our model gets train on 60% of records. More data allows the model to learn better representations of the underlying patterns and relationships in the data.

# SUPERVISED MODEL
## REGRESSION MODELS
For predicting the price of used mobile devices, we are going to build regression models to predict how much will the used mobile device cost. With the accurate price predictions of used mobile devices, businesses and dealers can set prices that are attractive to customers while ensuring their profits.

•	Linear regression model
•	Stepwise regression model
•	Regression tree model

### Multiple linear regression model
Using lm() function will build linear regression model on training data and will measure its performance on valid dataset. The linear regression model in used smartphones price prediction is to predict the price of a used mobile devices based on various features or attributes of the devices. Regression analysis is a statistical method used to model the relationship between a dependent variable (the price of the used mobile devices) and one or more independent variables (the features of the mobile devices). They can capture linear relationships between the features of the smartphone and its price.
I used all the important variables as input in the model. Regression model goodness of fit is R square= 0.8419 and we can see the significant variables of multiple regression model that contribute most to used price like release year, weight, battery, ram, normalized new price, etc. When I tested model performance on validation data it gave low error and maximum correlation between actual and predicted value.
The goal of using a regression model in used mobile devices price prediction is to provide accurate and reliable estimates of the prices of used mobile devices based on their features, helping businesses in making informed decisions in the secondary smartphone market. 

### Stepwise Regression Model
The goal of using stepwise regression in used smartphone price prediction is to automatically select a subset of relevant features (independent variables) from a larger pool of potential predictors while improving the predictive accuracy of the model.
 
Below are the predictors that are important in our model. We can see variables like rear camera mp, front camera mp, screen_size, os, X5g, battery, RAM, etc. are helpful for business in deciding which feature leads to increase or decrease in the price of used smartphones. 
When model performance was checked on validation dataset it performed better with low error. 

### Regression Tree 
Regression tree is a type of decision tree which is used to predict the price of a used mobile device based on its features or attributes. We can identify the most important features for predicting mobile devices prices. By examining the splits in the tree and the corresponding feature important scores, we can gain insights into which features are most influential in determining the price of a used mobile devices.
The goal of using a regression tree model in used smartphone price prediction is to build an interpretable and accurate model that can effectively predict the price of a used mobile devices based on its features, providing valuable insights for user, sellers, and market analysts in the secondary smartphone market. 
We will compare the model performance on validation dataset and choose the best performing model on validation and will do prediction on new holdout unseen data.

## CLASSIFIERS MODELS
Then, we will build models to classify the price of used mobile devices as High or Low. Given the characteristics of the predictors and target attributes, we’ve opted for the following classification algorithms to construct the classification models. Will fit the model and choose the best model among all of them. For predicting Categorical Used_price, we are going to take CAT.Used_Price as an outcome variable and will deselect Normalized_used_price variable in our predictor classifier models. If Percentage Price drop of used mobile devices is high, then mobile devices will be low price compared to new phones then more customers will buy because of cheaper used phone in this case business will generate high revenue because of high volume of sale. 
Here, Class 1 represents a high percentage drop in the price of a used mobile device. Positive Class will be High (1) because our / business aim is to find cheaper phones with good features so that more and more users get attracted to their store/websites seeing the cheaper phones. Based on these classification model, Businesses can target Low Price used mobile devices so that they can make marketing strategies to attract more customers by making good deals and displaying the best mobile features on its websites in cheaper prices. 
•	Logistic Regression
•	Naïve Bayes
•	KNN 
•	Decision Tree

### Logistic Regression
Using the logistic regression model, we are going to build a model to classify the price of used mobile devices as High or Low. Here, Positive class is 1 which means business aim is to identify “Low priced phones”. Logistic models can capture the non-linear relationship between predictors and used mobile devices’ price. Positive coefficient on predictors is associated with a higher probability of price of used mobile devices as low. 
When we kept all predictors except percent_dropprice and normalized_used_price as an input, model gave accuracy of 75.8% when checked its performance on validation dataset and we can see the significant variables of logistic model like screen_size, weight, rear and front camera mp, ram, release_year and normalized_new_price.
Goal: This model will be built to classify prices of used devices as High or Low based on a percentage drop in price. From the model output, can identify the significant variables of this model by identifying odds associated with each predictor. This model will find the probability of a used mobile device price and based on threshold value will classify mobile as High or Low percentage drop. Here we have taken threshold as 0.5. Based on this classification model, will suggest Businesses to targets those used mobile devices which have Low Price so that they can make marketing strategies to attract more customers by making good deals and displaying the best mobile features on its websites in cheaper prices. We have checked the model performance on validation dataset and got accuracy as-Validation accuracy = 0.6913 and sensitivity as 0.6916 which tells the true positive rate.

### DECISION TREE
A decision tree is a supervised machine learning algorithm used for both classification and regression tasks. Here, a decision tree will create a series of decision rules based on the dataset's features to classify the prices of used mobile devices as High or Low based on a percent drop in price. Variables that are contributing most are screen_size, normalized_new_price, weight, os, front_camera_mp, rear_camera_mp, battery, X4g, internal_memory. Using C5.0 Decision Tree Algorithm we are going to build a decision tree and our aim is to find splits in the data to reduce the entropy. The output model can be used to assign a class to new unclassified data items. 
The model performance on validation dataset gave accuracy of 67.83%. To improve the model performance will use the Trial option with trial =7 and with trial option it gave 70% accuracy. The trails option specifies the number of boosting iterations. 
Validation accuracy = 70%

### NAÏVE BAYES
Using the naive bayes model, will build a model to classify the price of used mobile devices as High or Low % price drop. Naïve bayes estimates the conditional probability individually for each predictor against each class as High or Low. For naïve bayes, we have assumed that each predictor like battery, device brand, RAM is conditionally independent of each other. Each predictor is given equal weightage and importance in the model. It is assumed that each predictor is not correlated to other variables.
This model gives the conditional probability of each predictor against classes for example. Here, A-priori probability shows the probability of used mobile devices prices on each class High or Low. The conditional probability of mobile devices being 4G given that their price is High is 0.6016 which means there are 60.16% chances (which is assumed by model based on conditional probability) that price of 4G compatible phones will be higher. 
The goal is to select features that are relevant and informative for predicting the price of a used mobile device. Not all features may be important. 
Validation accuracy = 66.09%

### KNN Model
Using KNN model, we will be classifying the price of the used mobile device as high-priced phone or low-price phone. CAT.Used_Price variable will be used as an outcome variable for classifying which is derived from the percentage drop in price of a used smartphone. K-NN will assign new data according to certain similarity or distance measures from its neighbor. To classify the used mobile devices, we will use predict function on datasets (validation) by training the model on train dataset. The performance of the model on validation dataset when used k= 5 nearest neighbors, it gave an accuracy of 0.6014 and sensitivity= 0.5418.
To find the best k-nearest neighbour will use knn tunning method to get best parameters.

![image](https://github.com/user-attachments/assets/1e96c348-ee53-4c74-938b-80d00fa7a091)

Using the KNN tuning method, we got the best k value = 20 as it gave low error and classified new data with an accuracy of 61.3%.

 ![image](https://github.com/user-attachments/assets/90f2465d-1293-4185-b8c1-fb327609f67f)
Validation accuracy= 61.3%

Using all these classification model, will check the performance of models on validation dataset and will choose the best performing classification model out of all, to classify the price of used mobile devices (on holdout datasets) and will use this model in real world to classify new used mobile devices as Highly priced or low price so that businesses can make decisions on selling used mobile devices by displaying good cheaper deals on their websites.

## MODEL PERFORMANCE
Validation Accuracy
We have checked model performance on validation dataset for both classification and regression model where Used_price was outcome variable for regression model and for classification model we have used Cat.Used_Price which was derived from Percent_PriceDrop variable. In classification models, the Logistic regression is the best model among all classification models because it gave the highest value of Sensitivity, Specificity and Accuracy of the Logistic regression model. If we are getting higher value for accuracy and sensitivity, then it is preferable to use logistics model.

![image](https://github.com/user-attachments/assets/f851dc39-19cc-4f93-8b10-b7e2d54bb1a8)

Decision: Seeing the performance of the model, we can tell the logistic model is working good in classifying used phone as High priced or Low price used mobile device as model gives high sensitivity comparatively i.e., high percentage drop in price (which means low price of used mobile devices) more accurately. 
So will use logistic regression model to fit in holdout/test data for classifying new data. 
For predicting price of used mobile device (Outcome Variable), Model Evaluation is done on validation dataset based on the error like MAE and RMSE. Seeing the accuracy measures of Stepwise Regression model like MAE & RMSE which is comparatively lower than other model, so stepwise regression model was chosen as it works best for predicting the price of used device on validation set.

![image](https://github.com/user-attachments/assets/b430cf3b-41a8-4fc4-b3d6-1b5ea32cfc09)
Since our best performing model is a stepwise regression model and will fit this model on new, 
unseen data i.e., holdout dataset. Here, we can see that stepwise regression models are showing a comparatively low degree of error which means it will give correct price prediction of used mobile devices with less error.


#### Holdout Accuracy of Selected Models
![image](https://github.com/user-attachments/assets/d22d3ec7-aac8-4146-9afa-7b84086041eb)

Model performance for Logistic model is done based on the confusion matrix for the accuracy of model, sensitivity, specificity, etc. which was highest. Sensitivity shows the model can accurately predict by 62.46% that a phone will be a low price means percentage drop in price will be higher (1). This will help businesses in classifying High and low price used devices so that they can make marketing strategies/deals on used phone so that more customers purchase a mobile device. So, this information will be highly useful to business as they can advertise and target the customers by displaying good deals which will result in more customers buying the cheap priced phones from the store, so in this case business will generate high volume of sales. 

![image](https://github.com/user-attachments/assets/fa22e5db-85bb-4d8f-965c-8085726add82)

It was found that the stepwise regression model is performing better on holdout dataset as well as it gave lower RMSE value and strong correlation between predicted and actual value on holdout set. Using this model, we can predict the price of used mobile devices. This regression model will help businesses in setting/deciding the prices of used mobile devices on their websites so that mobile device users can check and compare websites from other websites. Here business goal is to sell cheaper used mobile devices so that more and more customers visit their store/website and purchase used mobile devices, this will generate higher sales.

## MODEL EVALUATION 
Model evaluation is done on a separate dataset from the one used for fitting the model. When a model is trained on the same data it's tested on, it may perform well on that specific dataset but might not generalize well to new data. So, the model performance was checked on new unseen data i.e. holdout dataset. Using confusion matrix of Logistic Model, found model gave an accuracy of 0.6749 on holdout set for classification. The model accurately predicts Positive class (1) (Sensitivity of 0.6246) which means it can identify mobile devices that have a high percentage drop in price with an accuracy of 62.46%. 
Logistic Regression & Linear Regression is easy to explain to business as well because the output of these models is a mathematical equation and the coefficients for each of the variables provide the size and impact (positive and negative) on the outcome variables. 

# UNSUPERVISED MODEL: CLUSTERING
Using the Unsupervised model clustering method will identify different combinations of features in used mobile devices that could have an impact on the price of used phones. Using k-means clustering will cluster used mobile devices based on different combinations of features that drive the price of the used phones so that businesses will get insight into it. To find the optimal number of k-clusters will use an elbow method to determine the best k value of a k-means clustering using elbow curve. Clustering insight can help business provide better guidance to customers on how choosing (not choosing) certain features can vary prices of the used mobile devices. Can give customer valuable feedback.
![image](https://github.com/user-attachments/assets/6ba2d942-dd37-4f13-aa58-3aa7dd7564ce)

K= 4 clusters will be the optimal no. of clusters for clustering in used mobile devices features. The below plot shows how different mobile devices features are clustered into Cluster 1, 2, 3, 4. Here cluster 2 are high features phone because its screen_size, front camera, RAM, internal memory are high, devices are 4G &5G compatible and are being used for less days and they have high price of used phone, cluster 1 tells the mobile devices that have medium phone features with 4G connectivity and their used price is average and Cluster 3 is low feature phone with no 4G & 5G connectivity, with high internal memory of used devices and will likely have very low price of used mobile devices. Cluster 4 is a very low feature phone with no 4G & 5G connectivity and has low price of used mobile devices and mobile devices have been used for many days.
![image](https://github.com/user-attachments/assets/2670a2c9-272e-407c-a417-678a090074f1)
![image](https://github.com/user-attachments/assets/e993d603-3cef-40ae-93b8-bda5630c2dae)
There are 4 clusters with high, medium, and low price of used mobile devices.
Cluster 1 is an average features phone with a medium price of the used mobile devices and with 4G connectivity, has higher rear camera, is being used for lesser days but comparatively more days than cluster 2.
Cluster 2 is a high phone features phone with 4G and 5G enabled phone, bigger screen size, high front camera, phone is being used for lesser days, and the price of used devices is higher than other features phones.
Cluster 3 contains mobile devices will low features like they have less screen size, low rear camera, are neither 4G nor 5G enabled phones, and low RAM (low features phones) but with high internal memory have been used for longer days but not longer than cluster 4 devices and their price is comparatively least.
Cluster 4 is a low feature phone with no 4G & 5G connectivity and has a low price but not lower than Cluster 3 devices and mobile devices have been used for longer days.
All 4 clusters show the mobile device features like low, medium and high phone features vary the price of the used phones.

## CONCLUSION
The goal of this project is to uncover the relationship between the pricing of used mobile devices with different phone features. By analyzing these relationships, businesses can find out what combination of factors really decide how much the used mobile device will cost. This will help businesses in making decisions on setting prices of used mobile devices and classifying used mobile devices as High and Low price for selling purposes. Based on this classification, business can focus on low priced phones and display good deals on their websites so that more customers visit their website or store and buy a used device. Business aim was to generate high revenue by attracting more and more customers to make a purchase from their stores/website by providing competitive prices. Through this project, we seek to empower businesses to make informed pricing decisions and enhance their competitiveness in the used mobile phone industry. By understanding the complex interplay of factors driving device pricing, businesses can better cater to consumer preferences and drive sustainable growth in a rapidly evolving market.
For analytics, datasets have been partitioned into training, validation, and holdout set. Model is fitted on the training dataset, then its performance was evaluated on validation set and chose the best performing models (i.e., Logistic Regression model) from classification and Stepwise Regression from regression models and did prediction on holdout unseen dataset. Will then deploy these models for usage in the real world on new (unseen data) so that businesses can make decisions on setting the prices of used devices for selling purposes.
Utilizing the unsupervised k-means clustering model, we aim to categorize used mobile devices according to their distinct features, thereby allowing business to gain further insight into what combination of feature drive price. From clustering model, we found three combinations of features which have high, medium, and low prices respectively. Like, Cluster 2 contains used mobile devices with high RAM, greater screen size, high megapixel of front and rear camera and with 4G and 5G enabled mobile devices they have higher price of used mobile devices. By considering these features in clustering analysis, businesses can group similar devices together based on their price determinants, allowing for more targeted analysis and decision-making in the used mobile device market. This insight can help business provide better guidance to customers on how choosing (not choosing) certain features can vary prices of the used mobile devices.
### Recommendation: 
•	Target selling high percentage drop in price of phones as it will sell faster and increase the net sales of the company.
•	If businesses are purchasing devices from third party, then they should buy less such mobile devices that have low percentage drop in price because it will take longer to sell that mobile device in the market. And focus more on high depreciation phones.
•	Provide additional deals to acquire more customers.


