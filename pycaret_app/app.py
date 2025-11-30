import pandas as pd 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,mean_absolute_error, r2_score

df = pd.read_csv('shopping1_behavior.csv')


y = df['Purchase Amount (USD)'] 
x = df.drop('Purchase Amount (USD)', axis=1)

num_cols = ['Customer ID', 'Age','Review Rating','Previous Purchases']
cat_cols = [col for col in x.columns if col not in num_cols]


num_transform = Pipeline([('imputer',SimpleImputer(strategy='median'))])
cat_transform = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([('num',num_transform , num_cols),('cat',cat_transform,cat_cols)])

model = Pipeline([('prep',preprocessor),('linear',LinearRegression())])

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

model.fit(x_train , y_train)
y_pred = model.predict(x_test)

print(mean_squared_error(y_test , y_pred))
print(mean_absolute_error(y_test , y_pred))
print(r2_score(y_test , y_pred))



















