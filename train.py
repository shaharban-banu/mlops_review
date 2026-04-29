import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import mlflow
import pickle

# mlflow.set_experiment("review-iris")
# mlflow.set_tracking_uri("file:./mlruns")

df=pd.read_csv("data/iris.csv")
x=df.drop(columns=['Species','Id'])
y=df['Species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# with mlflow.start_run():
model=RandomForestClassifier()
model.fit(x_train,y_train)
pred=model.predict(x_test)
acc=accuracy_score(y_test,pred)
#mlflow.log_metric("accuracy",acc)
print("accuracy",acc)

with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print("model trained and mlflow run")