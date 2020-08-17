import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from .misclass import Misclass

class User_Input_Misclass(object):

    # Prompts user to enter path to data
    def input_data_path(self):
        print("To enter the path to the data to be processed, enter the relative path to the data from where the library is run")
        print("e.g. filename.txt if data file is in the same folder that the library is run, data/filename.txt if in a folder called data")
        datafile_path = input('Enter relative path to data?: ')
        return datafile_path

    # Reads in file from the specified relative path location to pandas DataFrame and sets column names
    def read_data(self, path):
        data = pd.read_csv(path, header=None, names=['text', 'label'], sep='\t', encoding='unicode_escape')
        return data

    # Function to identify class label values
    def get_label_values(self, label_column):
        # Identifies unique values in specified column
        label_values = label_column.unique()
        return label_values

    # Function to set class labels to 0 and 1 values if not already
    def set_label(self, data):
        # Calls function to identify unique values in column
        label_values = self.get_label_values(data['label'])
        # Creates list of label values
        binary_labels = [0, 1]
        # Checks if current label values match the values in list binary_list
        if (all(i in binary_labels for i in label_values)) == False:
            # If they do not match, assigns the current label values to another column "label_name"
            # So the original label values aren't lost
            data['label_name'] = data['label']
            # Maps 0 and 1 values to original label values
            data['label'] = data.label.map({label_values.item(0): 0, label_values.item(1): 1})
        return data

    # Function to tokenize text of instances, call stemming and lemmatise functions if selected as input parameter
    def tokenize(self, text, stem):
        text = text.lower()
        punctuation = set(string.punctuation)
        text = ''.join(tkn for tkn in text if tkn not in punctuation)
        text = ' '.join(text for text in text.split() if len(text) >= 2)
        tokens = word_tokenize(text)

        if stem == 'S':
            tokens = self.stem(tokens)
        if stem == 'L':
            tokens = self.lemmatize(tokens)
        return tokens

    # Function to stem tokens
    def stem(self, data):
        stemmed_tokens = []
        for i in data:
            stemmed_tokens.append(PorterStemmer().stem(i))
        return stemmed_tokens

    # Function to lemmatize tokens
    def lemmatize(self, data):
        lemmatized_tokens = []
        for i in data:
            lemmatized_tokens.append(WordNetLemmatizer().lemmatize(i))
        return lemmatized_tokens

    # Prompt the user to enter value for stopwords parameter of the CountVectorizer
    def input_stopword_selection(self):
        try:
            sw = input('Remove Stopwords? Y/N: ')
            if sw == 'N':
                stopword = None
            elif sw == 'Y':
                sw_option = input('Stopwords, from Text File (F) or CountVectorizer (C) list? (default: C) F or C:  ')
                if sw_option == 'F':
                    with open("stopwords.txt", "r") as f:
                        sw_option = [i for line in f for i in line.rstrip('\n').split(',')]
                    stopword = frozenset(sw_option)
                elif sw_option == 'C':
                    stopword = 'english'
        except ValueError:
            print('Invalid Entry')
        return stopword

    # Prompts the user to enter a value for stemming or lemmatisation
    def input_stem_selection(self):
        stem = input('Stem(S), Lemmitisation(L), or None? (default:N) S, L, N: ')
        if stem not in ['S', 'L', 'N']:
            stem = 'N'
        return stem

    # Prompts the user to enter a value for the n_gram parameter of the CountVectorizer
    def input_ngrams_selection(self):
        no_of_ngrams = int(input('Enter number of n grams? (default:1) 1,2,3,4: '))
        if no_of_ngrams not in [1, 2, 3, 4]:
            no_of_ngrams = 1
        return no_of_ngrams

    # Prompts the user to enter a value for the min_df parameter of the CountVectorizer
    def input_min_df_selection(self):
        no_of_min_df = int(input('Enter number of min df? (default:1) 1-10 : '))
        if no_of_min_df not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            no_of_min_df = 1
        return no_of_min_df

    # Prompts the user to enter value for the max_df parameter of the CountVectorizer
    def input_max_df_selection(self):
        no_of_max_df = float(input('Enter number of max df?(default:1.0) 0.5-1.0: '))
        if no_of_max_df <= 0.01 or no_of_max_df >= 1.0:
            no_of_max_df = 1.0
        return no_of_max_df

    # Prompts the user to select a classifier from the list
    def input_model_selection(self):
        model_selection = input('Select a classifier...(default:NB)|LR|KNN|DT|NB|SVM|: ')
        if model_selection not in ['LR', 'KNN', 'DT', 'NB', 'SVM']:
            model_selection = 'NB'
        return model_selection

    # Assign dataframe column to X (series)
    def get_data_X(self, data):
        X = data.text
        return X

    # Assign dataframe column to y (series)
    def get_data_y(self, data):
        y = data.label
        return y

    # Create, fit and transform the training data using CountVectorizer object with no pre-processing parameters to get all tokens
    # and convert all_word_count sparse matrix to dense matrix and return
    def get_all_tokens_wc(self, X):
        # CountVectorizer (with the default parameters)
        vect_alltokens = CountVectorizer(
            token_pattern='(?u)\\b\\w+\\b')  # include token pattern to include 1 char words
        vect_alltokens.fit(X)
        all_word_count = vect_alltokens.transform(X)
        all_word_count_matrix = (pd.DataFrame(all_word_count.toarray(), columns=vect_alltokens.get_feature_names()))
        print("all_word_count_matrix")
        print(all_word_count_matrix)
        return all_word_count_matrix

    # Create CountVectorizer object
    def get_vect(self, stopword, no_of_ngrams, no_of_min_df, no_of_max_df, records, stem):

        vect = CountVectorizer(tokenizer=(lambda records: self.tokenize(records, stem)),
                               stop_words=stopword, ngram_range=(1, no_of_ngrams),
                               min_df=no_of_min_df, max_df=no_of_max_df,
                               vocabulary = None)
        return vect

    # Fit and transform the training data using CountVectorizer object
    def get_word_count(self, vect, X):
        vect.fit(X)
        word_count = vect.transform(X)
        return word_count

    # Transform word_count sparse matrix to dense matrix
    def get_word_count_matrix(self, vect, word_count):
        word_count_matrix = (pd.DataFrame(word_count.toarray(), columns=vect.get_feature_names()))
        print("word_count_matrix")
        print(word_count_matrix)
        return word_count_matrix

    # Fit and transform the sparse word_count matrix to tf-idf matrix using TfidfTransformer object
    def get_tfidf(self, word_count):
        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(word_count)
        return tfidf

    # Transform tf-idf sparse matrix to dense matrix
    def get_tfidf_matrix(self, tfidf, vect):
        tfidf_matrix = (pd.DataFrame(tfidf.toarray(), columns=vect.get_feature_names()))
        return tfidf_matrix

    # Run selected classifier model
    def run_classification(self, model_selection, tfidf, y):
        models = []
        models.append(('NB', MultinomialNB()))
        models.append(('DT', DecisionTreeClassifier()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('LR', LogisticRegression()))
        models.append(('SVM', SVC()))

        for key, model_name in models:
            if model_selection == key:
                # instantiate the model (using the default parameters)
                model = model_name
        # fitting model
        model.fit(tfidf, y)
        # produces a numpy.ndarray of predicted labels
        y_pred_label = model.predict(tfidf)
        return y_pred_label

    # Convert numpy.ndarray to a pandas series
    def convert_to_series(self, col, name):
        new_series = pd.Series(col, name=name)
        return new_series

    # Get model description from inputted values
    def get_model_desc(self, stopword, stem, model_selection, no_of_ngrams, no_of_min_df, no_of_max_df):
        if stopword == None:
            stopword_value = "No stop words removed"
        elif stopword == 'english':
            stopword_value = "CountVectorizer's stop words removed"
        else:
            stopword_value = "Custom list of stop words removed"

        if stem == 'L':
            stem_value = "Lemmitisation applied"
        elif stem == 'S':
            stem_value = "Stemming applied"
        else:
            stem_value = "No lemmitisation or stemming applied"

        model_desc = "Classifier: " + model_selection + ", Pre-processing: " + stopword_value + ", " + stem_value + ", No of N-grams - " + str(
            no_of_ngrams) + ", Min df - " + str(no_of_min_df) + ", Max df - " + str(no_of_max_df)
        return model_desc

    def build(self):
        print()
        print("Welcome to Text Misclassification Analysis")
        print("Please enter valid input (or default values will be used)")
        print()
        path = self.input_data_path()
        data = self.read_data(path)
        #data = self.set_label(data)
        X = self.get_data_X(data)
        y = self.get_data_y(data)
        all_word_count_matrix = self.get_all_tokens_wc(X)

        stopword = self.input_stopword_selection()
        stem = self.input_stem_selection()
        no_of_ngrams = self.input_ngrams_selection()
        no_of_min_df = self.input_min_df_selection()
        no_of_max_df = self.input_max_df_selection()

        vect = self.get_vect(stopword, no_of_ngrams, no_of_min_df, no_of_max_df, X, stem)
        word_count = self.get_word_count(vect, X)
        word_count_matrix = self.get_word_count_matrix(vect, word_count)
        tfidf = self.get_tfidf(word_count)
        tfidf_matrix = self.get_tfidf_matrix(tfidf, vect)

        model_selection = self.input_model_selection()
        y_pred_label = self.run_classification(model_selection, tfidf, y)
        y_pred_label = self.convert_to_series(y_pred_label, "pred_label")
        model_desc = self.get_model_desc(stopword, stem, model_selection, no_of_ngrams, no_of_min_df, no_of_max_df)

        mc = Misclass()
        mc.build(X, y, y_pred_label, word_count_matrix, tfidf_matrix, all_word_count_matrix, model_desc, path)


