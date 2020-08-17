from py_text_misclass import Misclass

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

path = 'labelled_data.txt'
data = pd.read_csv(path, header=None, names=['text', 'label'], sep='\t', encoding = 'unicode_escape')
label_values = data['label'].unique()

# Creates list of label values
binary_labels = [0, 1]
# Checks if current label values match the values in list binary_list
if (all(i in binary_labels for i in label_values)) == False:
    # If they do not match, assigns the current label values to another column "label_name"
    # So the original label values aren't lost
    data['label_name'] = data['label']
    # Maps 0 and 1 values to original label values
    data['label'] = data.label.map({label_values.item(0): 0, label_values.item(1): 1})

a = data.text
b = data.label

vect_alltokens = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')  # include token pattern to include 1 char words
vect_alltokens.fit(a)
all_word_count = vect_alltokens.transform(a)
wc_matrix2 = (pd.DataFrame(all_word_count.toarray(), columns=vect_alltokens.get_feature_names()))

vect = CountVectorizer(stop_words='english', min_df=3)
wc = vect.fit_transform(a)
wc_matrix = (pd.DataFrame(wc.toarray(), columns=vect.get_feature_names()))

tfidf_transformer = TfidfTransformer()
# fit and transform lines above equivalent to ...
tfidf = tfidf_transformer.fit_transform(wc)
tfidf_matrix = (pd.DataFrame(tfidf.toarray(), columns=vect.get_feature_names()))

nb = MultinomialNB()

# fitting finalise model after cross validation on entire dataset
nb.fit(tfidf, b)

# produces a numpy.ndarray of predicted labels
c = nb.predict(tfidf)
c = pd.Series(c, name="pred_label")

# creating a instance of the misclass class
mc = Misclass()
# calling the build function of the misclass class and passing required data
mc.build(a,b,c, wc_matrix,tfidf_matrix, wc_matrix2,"First Iteration","sms data")

