# Data argumentation
%%capture
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from unidecode import unidecode
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

train = pd.read_csv('../input/fakenewsvortexbsb/train_df.csv', sep=';', error_bad_lines=False, quoting=3);

# First four registries
train.head(4) 

# Number of registries in dataset
train.shape

# Dataset columns
array(['index', 'manchete', 'Class'], dtype=object)

# Dataset sample
train["manchete"][80]
train["Class"][80]

# Different types of labels
train['Class'].unique()

# Testing dataset sample
example = train["manchete"][1]
print(unidecode(example)) 

# Applying libraries, unidecode aplies unicode and re removes punctuation 
letters_only=re.sub("[^a-zA-Z]"," ",unidecode(example)) 
print(letters_only)

# Convert entire text to lowercase
lower_case=letters_only.lower()
words=lower_case.split()

# Stop words portuguese list
print (stopwords.words("portuguese"))

# Build an array with these stop words
stop = stopwords.words("portuguese")

# It is not possible to apply unicode in a list, so we travel through the array applying in each registry
lista_stop = [unidecode(x) for x in stop]  
print (lista_stop)

print(words)

# Filter of non-"stop words" words in the text
words=[w for w in words if not w in lista_stop] 
print(words)

# Function review_to_words transforms what has been changed before
def review_to_words(raw_review):
    raw_review = unidecode(raw_review)
    raw_review.lstrip('Jovem Pan')
    letters_only=re.sub("[^a-zA-Z]"," ",raw_review)
    words=letters_only.lower().split()
    meaningful_words=[w for w in words if not w in lista_stop]
    return(' '.join(meaningful_words))
  
# Tests function
clean_review=review_to_words(train['manchete'][1])
print(clean_review)

# Gets data size to use in the next for
num_reviews=train['manchete'].size
print (num_reviews)

# Loop to apply transformations in each registry of column manifestacao_clean of the dataset
clean_train_review=[]
for i in range(0,num_reviews):
    clean_train_review.append(review_to_words(train['manchete'][i]))

# Configure parameters of WordtoVec/Tokenization and create object
vectorizer=CountVectorizer(analyzer='word',tokenizer=None,preprocessor = None, stop_words = None,max_features = 7000)

# Apply WordtoVec 
train_data_features=vectorizer.fit_transform(clean_train_review)

# Apply data structure numpy array 
train_data_features=train_data_features.toarray()
train_data_features.shape
train_data_features[1]

# Vocabulary of the most important words of all the requisitions
vcab=vectorizer.get_feature_names()
print(vcab)
train_y = train["Class"]

# Data split in train and validation
X_train, X_test, y_train, y_test = train_test_split(train_data_features, train_y, test_size=0.25, random_state=42)

# Creation of object according to specified hyperparameter model
model = KNeighborsClassifier(n_neighbors=3)

# Training model
%time model = model.fit( X_train, y_train )

# Prediction of test data with training model
result = model.predict(X_test)

# Data validation: absolute result accuracy 
accuracy_score(y_test, result)

# Function to test of recall and f1 score of model. It's important to base yourself not only in accuracy and precision
print (classification_report(y_test, result))

confusion_matrix(result, y_test, labels=y_train.unique())
array = confusion_matrix(result, y_test, labels=y_train.unique())

# Normalization of values  
array = array.astype('float') / array.sum(axis=1)[:, np.newaxis]

# Create a data frame based on the graphic
df_cm = pd.DataFrame(array, index = y_train.unique(), 
                  columns = y_train.unique())
plt.figure(figsize = (10,7)) 
sn.heatmap(df_cm, annot=True, cmap=sn.light_palette((210, 90, 60), input="husl"))

# Test your models on the following:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# Using the model
def prevendo_noticias(string, model):
    to_array=[]
    to_array.append(review_to_words(string))
    sample_final=vectorizer.transform(to_array)
    sample_final=sample_final.toarray()
    result = model.predict(sample_final)
    if  result[0] == 1:
        label = 'Fake News'
    else:
        label = 'Verdadeira'
        
    return label, string
  
prevendo_noticias('Aras: decisão do STF não deveria valer para casos concluídos', model)
prevendo_noticias('Bolsonaro pessoalmente incendêia a amazonia e mata as girafas', model)
prevendo_noticias('Jornalista joga água benta em Temer e ele admite que impeachment foi golpe', model)
  
# Creating a submission

# Creating an object that corresponds to specified hyperparameter model
model_final = KNeighborsClassifier(n_neighbors=3)

# Training model
%time model_final = model_final.fit( train_data_features, train_y )
test = pd.read_csv('../input/fakenewsvortexbsb/sample_submission.csv', sep=';', error_bad_lines=False, quoting=3);
test.head(5)
num_reviews, = test['Manchete'].shape
print(num_reviews)
clean_test_review=[]
for i in range(0,num_reviews):
    clean_test_review.append(review_to_words(test['Manchete'][i]))
test_data_features = vectorizer.transform(clean_test_review)
test_data_features=test_data_features.toarray()
result_test = model_final.predict(test_data_features)

# Creating dataframe with results to submit
minha_sub = pd.DataFrame({'index': test.index, 'Category': result_test})
minha_sub.to_csv('submission.csv', index=False)
