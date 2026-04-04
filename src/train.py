import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data/california-housing.csv')

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('median_house_value', axis=1), 
    df['median_house_value'], 
    test_size=0.2,
    random_state=31415
)

X_test.iloc[:5].to_csv('data/sample_input.csv', index=False)

numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),('model', LinearRegression())])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

joblib.dump(pipeline, 'models/model.pkl')

loaded_pipeline = joblib.load('models/model.pkl')
loaded_pipeline.predict(X_test.iloc[[0]])