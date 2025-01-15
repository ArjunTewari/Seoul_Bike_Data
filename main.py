from taipy import Gui
import os
# SEOUL BIKE DATA
#

# We'll see how well we can predict the demand using various atmospheric conditions.

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv("SeoulBikeData.csv", index_col = "Date", parse_dates = True)
df = pd.DataFrame(data)


# In[16]:


from scipy.stats.mstats import winsorize

df.select_dtypes("object").columns.astype("category")
numerical_col = df.select_dtypes("number").columns

for col in numerical_col:
    df[col] = winsorize(df[col], limits = [0.1, 0.1])


# Select datatype returns a dataframe of columns either numerical or categorical.

# We will now find the correlation between numerical columns and see..

# In[17]:


df.select_dtypes("number").corr()["Rented Bike Count"].sort_values(ascending=False)


# Splitting the data into test, train and val

# In[18]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, shuffle=False, test_size=0.2)
train, val = train_test_split(train, shuffle=False, test_size=0.25)


# In[19]:


X_train = train.drop(columns=["Rented Bike Count"])  # Features
X_test = test.drop(columns=["Rented Bike Count"])
X_val = train.drop(columns=["Rented Bike Count"])

y_train = train["Rented Bike Count"]  # Target
y_val = train["Rented Bike Count"]
y_test = test["Rented Bike Count"]


# You should fit the scaler on the training set only and then apply it to the train, validation, and test sets to avoid data leakage

# In[25]:


from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_x.fit(X_train.select_dtypes("number"))
X_train_scaled = scaler_x.transform(X_train.select_dtypes("number"))
X_test_scaled = scaler_x.transform(X_test.select_dtypes("number"))
X_val_scaled = scaler_x.transform(X_val.select_dtypes("number"))

scaler_y = StandardScaler()
y_df = pd.DataFrame(y_train)
scaler_y.fit(y_df.select_dtypes("number"))


# Our data is ready for model usage, we will use histgradientRegressor, I did not encoded because it can handle categrical values.

# In[29]:


from sklearn.ensemble import HistGradientBoostingRegressor
model_1 = HistGradientBoostingRegressor(random_state=42)
model_1.fit(X_train_scaled,y_train)


# In[30]:


predictions = model_1.predict(X_test_scaled)
y_predicted = (predictions.reshape(-1,1)).flatten()
# type(y_predicted)


# Metrics like MAPE, MAE, RMSE, and R² accept both NumPy arrays and pandas Series. However, NumPy arrays are generally preferred because they are more lightweight and universally supported across libraries.

y_actual = np.array(y_test)
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score
mape = mean_absolute_percentage_error(y_actual, y_predicted)
rmse = root_mean_squared_error(y_actual, y_predicted)
r2 = r2_score(y_actual, y_predicted)
# mape, rmse, r2

def prepare_chart_data(y_actual, y_predicted):
    data = pd.DataFrame({"Index":range(len(y_actual)), "Actual":y_actual.flatten(), "Predicted":y_predicted.flatten()})
    return data

#Now we will use GridSearch to find best parameters or in other words, Fine tune the model to the given dataset.



from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

tcsv = TimeSeriesSplit(n_splits=5)

param_grid = {
    'learning_rate': [0.01, 0.05, 0.07, 0.1],      # Few learning rates
    'max_iter': [100, 200, 300, 400, 500, 600],            # Number of iterations
    'max_depth':  [3, 5, 7, 10, 15],                   # Max depth
    # 'min_samples_leaf': [10, 20, 30, 40],          # Control for overfitting
    # 'l2_regularization': [0.0, 0.1, 0.5, 1.0],     # Regularization strength
}

grid_search = GridSearchCV(
    estimator=HistGradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  # Metric for evaluation
    cv=tcsv,                              # Cross-validation folds
    verbose=2,
    n_jobs=-1                          # Use all cores
)

# grid_search.fit(X_train_scaled, y_train)

# print("Best Parameters:", grid_search.best_params_)



import joblib

#Save the model
# joblib.dump(grid_search.best_estimator_, 'best_model.joblib')


# Load the model later
loaded_model = joblib.load('best_model.joblib')# Try other encodings if needed

predictions_new = loaded_model.predict(X_test_scaled)
y_predicted_new = predictions_new.reshape(-1,1).flatten()




from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score
new_mape = mean_absolute_percentage_error(y_actual, y_predicted_new)
new_rmse = root_mean_squared_error(y_actual, y_predicted_new)
new_r2 = r2_score(y_actual, y_predicted_new)

page = """

<|text-center|
<h1> Predicting Seoul Bike Demand </h1>

<h4> We will predict the demand of bike based on various atmospheric conditions. I used HistGradientRegressor and tried to fine-tune using GridSearchCV </h4>

<a href = "https://github.com/ArjunTewari/Seoul_Bike_Data.git">Link to GitHub repository</a>

<h6> Here is a sample of the dataset, It is a kaggle dataset. </h6>

<|{df.head()}|table|>

<h6> Here is a graph showing predicted vs actual value of model before fine tuning </h6>
 
<|{prepare_chart_data(y_actual, y_predicted)}|chart|type=line|x=Index|y[1]=Actual|y[2]=Predicted|title=Actual vs Predicted|>

<|layout|columns = 1 1 1|
# RMSE : <|metric|value={rmse:.2f}|>
# MAPE : <|metric|value={mape:.2f}%|>
# R² : <|metric|value={r2:.2f}|>
|>

<h6> After hyperparameter optimization </h6>

<|{prepare_chart_data(y_actual, y_predicted_new)}|chart|type=line|x=Index|y[1]=Actual|y[2]=Predicted|title=Actual vs Predicted|>

<|layout|columns = 1 1 1|
# RMSE : <|metric|value={new_rmse:.2f}|>
# MAPE : <|metric|value={new_mape:.2f}%|>
# R² : <|metric|value={new_r2:.2f}|>
|>

"""


if __name__== "__main__":
    app = Gui(page)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)


