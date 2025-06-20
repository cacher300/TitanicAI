import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

train_df = pd.read_csv('train.csv')

train_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].mean())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

train_df['Sex'] = LabelEncoder().fit_transform(train_df['Sex'])
train_df['Embarked'] = LabelEncoder().fit_transform(train_df['Embarked'])

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features].values
y = train_df['Survived'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {acc * 100:.2f}%')

test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

test_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')
test_df['Age'] = test_df['Age'].fillna(train_df['Age'].mean())
test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].mean())
test_df['Embarked'] = test_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Sex'] = LabelEncoder().fit_transform(test_df['Sex'])
test_df['Embarked'] = LabelEncoder().fit_transform(test_df['Embarked'])

X_test = test_df[features].values
X_test = scaler.transform(X_test)

predictions = model.predict(X_test)

submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
submission.to_csv('submission.csv', index=False)
