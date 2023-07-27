# usedCar_price_prediction
Project Overview
The goal of this project is to predict the prices of used cars using several different machine learning models. The dataset used for this analysis is a CSV file containing information about used cars, including various features such as make, year, odometer reading, and price. The project aims to compare the performance of different models and identify the most accurate model for price prediction.

Step 1: Importing Necessary Libraries
In this step, all the required Python libraries are imported, including NumPy, Pandas, Matplotlib, Seaborn, and various machine learning models from scikit-learn and other specialized libraries.

Step 2: Loading the Dataset
The dataset, named 'craigslistVehicles.csv', is loaded into a Pandas DataFrame for further analysis. Unnecessary columns are dropped, and missing values are handled accordingly.

Step 3: Exploratory Data Analysis (EDA)
EDA is performed to gain insights into the dataset. Data cleaning and preprocessing are carried out, including encoding categorical features using label encoding. The correlation between features is analyzed, and a profiling report is generated to summarize the data.

Step 4: Preparing to Train the Models
The dataset is split into training and testing sets (80-20 split). For boosting models, an additional split (70-30) is performed. The data is scaled using StandardScaler to match the requirements of some models.

Step 5: Training Models and Testing for All Features
In this step, 15 different machine learning models are trained and tested on the dataset. The models include Linear Regression, Support Vector Machines (SVM), MLPRegressor, Stochastic Gradient Descent (SGD), Decision Tree Regressor, Random Forest, XGBoost (XGB), Gradient Boosting Regressor, Ridge Regressor, and AdaBoost Regressor. For some models, hyperparameter tuning is performed using GridSearchCV. The models' accuracy is evaluated using metrics like R-squared (R2), relative error (d), and root mean squared error (RMSE) for both training and testing datasets.

Step 6: Models Comparison
The performance of each model is compared based on R2, relative error, and RMSE criteria. Graphs are plotted to visualize the comparison of these metrics for each model on the train and test datasets.

Step 7: Prediction
Finally, the best-performing models (Linear Regression and Ridge Regressor) are used to predict the prices of used cars on the test dataset.

Please note that the code provided here is a part of the complete project, and further details, as well as the full implementation, may be available in the complete project files.
