################################################################
# Full setup for downloading libraries and the kaggle database #
################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
install.packages(c("devtools"))
devtools::install_github("ldurazo/kaggler")

library(readr)
library(kaggler)
library(tidyverse)
library(caret)
library(rpart)
library(Hmisc)
library(reshape2)

kgl_auth(creds_file = 'kaggle.json')
response <- kgl_datasets_download_all(owner_dataset = "samuelcortinhas/credit-card-approval-clean-data")

download.file(response[["url"]], "data/temp.zip", mode="wb")
unzip_result <- unzip("data/temp.zip", exdir = "data/", overwrite = TRUE)

filename <- "data/clean_dataset.csv"
data <- read.csv(filename)

# Data preparation

# Trying to label-encode all of the data points
data <- data %>%
  mutate(Industry = factor(Industry), 
         Ethnicity = factor(Ethnicity), 
         Citizen = factor(Citizen))

# Saving all the factor levels to a data frame.
dataFactorLevels <- data.frame(levels(data$Industry), 
                               c(levels(data$Ethnicity), replicate(9, NA)),
                               c(levels(data$Citizen), replicate(11, NA)))

# Converting the factors into actual integers
data <- data %>%
  mutate(Industry = as.integer(Industry), 
         Ethnicity = as.integer(Ethnicity), 
         Citizen = as.integer(Citizen))

# Data Exploration

head(data)
length(data[,0:1]) # Length of the data set
is.na(data) %>% sum() # Checking for null values

# Creating a histogram for every variable
data %>% 
    select(Gender, Age, Debt, Married, BankCustomer, YearsEmployed, PriorDefault, 
           Employed, CreditScore, DriversLicense, Approved) %>%
    gather(key = "variable", value = "value") %>%
    ggplot(aes(value)) +
      geom_histogram(binwidth = 0.5) +
      facet_wrap(~variable, scales = 'free')

# Creating a correlation heatmap
data %>% 
  select(Gender, Age, Debt, Married, BankCustomer, YearsEmployed, PriorDefault, 
          Employed, CreditScore, DriversLicense, Income, Approved) %>%
  cor() %>%
  round(2) %>%
  melt() %>%
  ggplot(aes(x=Var1, y=Var2, fill=value)) + 
    geom_tile(size = 2) +
    geom_text(aes(Var2, Var1, label = value), size = 3) +
    scale_fill_gradient2(high = "#FFDAA6",
          limit = c(-1,1), name="Correlation") +
    theme(axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          panel.background = element_blank(),
          axis.text.x = element_text(size=8, angle=270, hjust = 0, colour="black"),
          axis.text.y = element_text(size=8, colour="black")) +
    ggtitle("Correlation Heatmap")

# MinMaxScaling all variables
process <- preProcess(data, method = c('range'))
data <- predict(process, data)

# Using dimensional reduction to plot the data points
prcomp_data <- prcomp(data[, 1:16])
pca_data <- data.frame(
  PC1 = prcomp_data$x[, 1],
  PC2 = prcomp_data$x[, 2]
)

ggplot(pca_data, aes(x = PC1, y = PC2)) +
  geom_point()

# From this, we can clearly see some potential clusters, so now, we use a clustering
# algorithm to boost our potential model.
k <- kmeans(data, centers = 8)
data$Cluster <- k$cluster
pca_data$cluster <- k$cluster
ggplot(pca_data, aes(x = PC1, y = PC2, col = cluster)) +
  geom_point() +
  scale_color_gradientn(colors = rainbow(5))


# Model training

# Splitting the data into training and test sets
train_indices <- createDataPartition(data$Gender, times = 1, p = 0.8, list = FALSE)
train_set <- data[train_indices, ]
test_set <- data[-train_indices, ]

# Creating a logistic regression model (because the predicted value is either 0 or 1)
train_set$Approved <- factor(train_set$Approved)
test_set$Approved <- factor(test_set$Approved)
glm_fit <- train(Approved ~ ., method = 'glm', data = train_set)
glm_accuracy <- confusionMatrix(predict(glm_fit, test_set), 
                                test_set$Approved)$overall["Accuracy"]

knn_fit <- train(Approved ~ ., method = 'knn', 
                 data = train_set, tuneGrid = data.frame(k = seq(9, 71, 2)))
knn_accuracy <- confusionMatrix(predict(knn_fit, test_set, type = 'raw'), 
                                test_set$Approved)$overall["Accuracy"]