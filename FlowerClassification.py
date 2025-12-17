import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('IRIS.csv')
print(df.head())
print(df.describe())
print(df.info())

print(df['species'].value_counts())
print(df.isnull().sum())

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
df['sepal_length'].hist()
plt.title('Sepal Length')

plt.subplot(2, 2, 2)
df['sepal_width'].hist()
plt.title('Sepal Width')

plt.subplot(2, 2, 3)
df['petal_length'].hist()
plt.title('Petal Length')

plt.subplot(2, 2, 4)
df['petal_width'].hist()
plt.title('Petal Width')

plt.tight_layout()
plt.show()

colors= ['red','orange','blue']
species= ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

plt.figure(figsize=(12, 12))

for i in range(3):
    x = df[df['species'] == species[i]]
    plt.subplot(2, 2, 1)
    plt.scatter(x['sepal_length'], x['sepal_width'], c=colors[i], label=species[i])
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.scatter(x['petal_length'], x['petal_width'], c=colors[i], label=species[i])
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.scatter(x['sepal_length'], x['petal_length'], c=colors[i], label=species[i])
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.scatter(x['sepal_width'], x['petal_width'], c=colors[i], label=species[i])
    plt.xlabel('Sepal Width')
    plt.ylabel('Petal Width')
    plt.legend()

plt.tight_layout()
plt.show()
    
corr = df.drop(columns=['species']).corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot = True, ax= ax, cmap = 'coolwarm')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
print(df.head())

from sklearn.model_selection import train_test_split
X= df.drop(columns=['species'])
Y = df['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test, y_test) * 100)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test, y_test) * 100)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test, y_test) * 100)