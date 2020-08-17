import pandas as pd
import nltk
import re
import gc
from nltk import word_tokenize
import string
import json
from string import punctuation
from spellchecker import SpellChecker
from .visualisations import build_word_cloud, build_word_cloud2, build_single_top_tfidf_tokens_bar_chart

# #######################################################################
# Functions for Feature Statistics

# Sentence count of each instance and add count value to column of dataframe
def calc_sentence_count(df):
    df['sentences'] = df['text'].str.count(r'[?|!|.]+[ ]')
    return df

# Word count of each instance and add count value to column of dataframe
def calc_word_count(df):
    df['words'] = df['text'].str.split().map(len)
    return df

# Unique word count of each instance and add count value to column of dataframe
def calc_unique_words_count(df):
    # Split text into new series
    text_tokens = df['text'].str.lower().str.split()
    # Get number of unique words
    #how does apply(set)work
    df['unique_tokens'] = text_tokens.apply(set).apply(len)
    return df

# Char count of each instance and add count value to column of dataframe
def calc_char_count(df):
    df['chars'] = df['text'].map(len)  # including spaces
    return df

# Uppercase chars count of each instance and add count value to column of dataframe
def calc_uppercase_char_count(df):
    df['uppercase'] = df['text'].str.count(r'[A-Z]')
    return df

# Lovercase chars count of each instance and add count value to column of dataframe
def calc_lowercase_char_count(df):
    df['lowercase'] = df['text'].str.count(r'[a-z]')
    return df

# Numbers count of each instance and add count value to column of dataframe
def calc_number_count(df):
    df['numbers'] = df['text'].str.count(r'[0-9]')
    return df

# Non-alphanumeric chars count of each instance and add count value to column of dataframe
def calc_non_alphanumeric_char_count(df):
    df['non_alphanumeric'] = df['text'].str.count(r'[^a-zA-Z0-9\_]')
    return df

# Punctuation count of each instance and add count value to column of dataframe
def calc_punctuation_count(df):
    # ???????????
    # check for many of the functions
    #is this a data series or dataframe passed??
    instance_tokens = []
    for index, row in df.iterrows():
        text = row['text'].lower()
        punctuation = set(string.punctuation)
        text = ''.join(tkn for tkn in text if tkn in punctuation)
        punctuation_count = len(text)
        instance_tokens.append((text, punctuation_count))
    new_df = pd.DataFrame(instance_tokens, columns=['punctuations','punctuation'])
    df = pd.concat([df, new_df], axis=1)
    return df

# Space count of each instance and add count value to column of dataframe
def calc_space_count(df):
    df['space'] = df['text'].str.count(r'[ ]')
    return df

# Function to identify and count the number of misspelt tokens in each instance
# and adds columns to dataframe to hold this information
def calc_mispellings_count(df):
    instance_tokens = []
    spell = SpellChecker()
    for index, row in df.iterrows():
        text = row['text'].lower()
        punctuation = set(string.punctuation)
        text = ''.join(tkn for tkn in text if tkn not in punctuation)
        tokens = word_tokenize(text)
        # identifies misspellings
        misspelled = spell.unknown(tokens)
        # counts number of misspellings per instance
        no_of_misspellings = misspelled.__len__()
        instance_tokens.append((tokens, misspelled, no_of_misspellings))
    new_df = pd.DataFrame(instance_tokens, columns=['tokens', 'misspelled', 'misspellings'])
    df = pd.concat([df, new_df], axis=1)
    return df

# Function to identify the parts of speech of each word for each instance
# and then count per part of speech
def calc_pos_count(df, pos):
    instance_tokens = []
    if pos == 'nouns':
        part_of_speech = ['NN', 'NNS', 'NNP', 'NNPS']
    elif pos == 'verbs':
        part_of_speech = ['VB', 'VBD', 'VBZ', 'VBG', 'VBP', 'VBN']
    elif pos == 'adverbs':
        part_of_speech = ['RB', 'RBR', 'RBS', 'WRB']
    elif pos == 'adjectives':
        part_of_speech = ['JJ', 'JJR', 'JJS']

    for index, row in df.iterrows():
        text = row['text'].lower()
        text = word_tokenize(text)
        # Identifies the part of speech for each word of the instance
        pos_token = nltk.pos_tag(text)
        count = 0
        for word, tag in pos_token:
            # Checks if the tag is within the relevant part of speech list
            if tag in part_of_speech:
                count += 1
        instance_tokens.append((count))
    column_name = pos
    # Adds column and count values to dataframe
    new_df = pd.DataFrame(instance_tokens, columns=[column_name])
    df = pd.concat([df, new_df], axis=1)
    return df


# Function to add the classification outcome to each group based on label and
# predicted label values
def get_classifications(dataframe_series, y, y_pred_label, labels):
    df = pd.DataFrame()
    for index, value in labels.items():
        group_classification = []
        for index2, value2 in labels.items():
            #avg = get_count(dataframe_series[(y == value) & (y_pred_label == value2)])
            if value == value2:
                classification = "T" + str(value)
            elif value != value2:
                classification = "T" + str(value2) + "F" + str(value)
            group_classification.append(classification)
            new_series = pd.Series(group_classification, name=value)
        df = pd.concat([df, new_series], axis=1)
    labels_as_list = labels.tolist()
    df.set_index([labels_as_list], inplace=True)
    df.name = "classification"
    return df

def get_count(df_series):
    count = int(df_series.count())
    return count

def get_sum(df_series):
    sum = df_series.sum()
    return sum

# Sum specified column and divide by number of instances in dataframe
# to get average word_count per instance
def get_feature_statistics_count(dataframe_series, y, y_pred_label, labels):
    all_averages = []

    all = get_count(dataframe_series)
    all_averages.append(("Whole Dataset", all))

    for index, value in labels.items():

        avg = get_count(dataframe_series[(y == value)])
        classification = "Class " + str(value)
        all_averages.append((classification, avg))

        avg2 = get_count(dataframe_series[(y == value) & (y_pred_label == value)])
        classification1 = "True " + str(value)
        all_averages.append((classification1, avg2))

        avg3 = get_count(dataframe_series[(y != value) & (y_pred_label == value)])
        classification2 = "False " + str(value)
        all_averages.append((classification2, avg3))

        avg4 = get_count(dataframe_series[(y == value) & (y_pred_label != value)])
        classification3 = "Misclassified Class " + str(value)
        all_averages.append((classification3, avg4))

        all_averages_df = pd.DataFrame(all_averages, columns=['label', 'count'])
        all_averages_df.set_index('label', inplace=True)

        #all_averages_df = pd.Series(all_averages, name='count')

    return all_averages_df

# Sum specified column and divide by number of instances in dataframe
# to get average word_count per instance
def get_feature_statistics_count2(dataframe_series, y, y_pred_label, labels):
    all_averages = []

    all = get_count(dataframe_series)
    all_averages.append(("Whole Dataset", all))

    for index, value in labels.items():
        avg = get_count(dataframe_series[(y == value)])
        classification = "Class " + str(value)
        all_averages.append((classification, avg))

    for index, value in labels.items():
        avg2 = get_count(dataframe_series[(y == value) & (y_pred_label == value)])
        classification1 = "True " + str(value)
        all_averages.append((classification1, avg2))

    for index, value in labels.items():
        avg3 = get_count(dataframe_series[(y != value) & (y_pred_label == value)])
        classification2 = "False " + str(value)
        all_averages.append((classification2, avg3))

    for index, value in labels.items():
        avg4 = get_count(dataframe_series[(y == value) & (y_pred_label != value)])
        classification3 = "Misclassified Class " + str(value)
        all_averages.append((classification3, avg4))

        all_averages_df = pd.DataFrame(all_averages, columns=['label', 'count'])
        all_averages_df.set_index('label', inplace=True)

        #all_averages_df = pd.Series(all_averages, name='count')

    return all_averages_df

# Sum specified column and divide by number of instances in dataframe
# to get average word_count per instance
def get_feature_statistics_count3(dataframe_series, y, y_pred_label, labels, name):
    df = pd.DataFrame()
    for index, value in labels.items():
        count = []
        for index2, value2 in labels.items():
            avg = get_count(dataframe_series[(y == value2) & (y_pred_label == value)])
            count.append(avg)
            new_series = pd.Series(count, name=value)
        df = pd.concat([df, new_series], axis=1)
    labels_as_list = labels.tolist()
    df.set_index([labels_as_list], inplace=True)
    df.name = name
    return df

def get_avg(df_series):
    column_total = df_series.sum()
    avg = round(column_total / df_series.count(),3)
    return avg



# Sum specified column and divide by number of instances in dataframe
# to get average word_count per instance
def get_feature_statistics_avg(dataframe_series, column_name, y, y_pred_label, labels):

    all_averages = []

    all = get_avg(dataframe_series)
    all_averages.append(("Whole Dataset", all))

    for index, value in labels.items():
        avg = get_avg(dataframe_series[(y == value)])
        classification = "Class " + str(value)
        all_averages.append((classification, avg))

        avg2 = get_avg(dataframe_series[(y == value) & (y_pred_label == value)])
        classification1 = "True " + str(value)
        all_averages.append((classification1, avg2))

        avg3 = get_avg(dataframe_series[(y != value) & (y_pred_label == value)])
        classification2 = "False " + str(value)
        all_averages.append((classification2, avg3))

        avg4 = get_avg(dataframe_series[(y == value) & (y_pred_label != value)])
        classification3 = "Misclassified Class " + str(value)
        all_averages.append((classification3, avg4))

        #for index2, value2 in labels.items():
            #avg2 = get_avg(dataframe_series[(y == value)& (y_pred_label == value2)])
            #classification = "T " + value + " F " + value2
            #all_averages.append((classification, avg2))

        all_averages_df = pd.DataFrame(all_averages, columns=['label', column_name])
        all_averages_df.set_index('label', inplace=True)

        #all_averages_df = pd.Series(all_averages, name=column_name)
    return all_averages_df

# Sum specified column and divide by number of instances in dataframe
# to get average word_count per instance
def get_feature_statistics_avg2(dataframe_series, column_name, y, y_pred_label, labels):

    all_averages = []

    all = get_avg(dataframe_series)
    all_averages.append(("Whole Dataset", all))

    for index, value in labels.items():
        avg = get_avg(dataframe_series[(y == value)])
        classification = "Class " + str(value)
        all_averages.append((classification, avg))

    for index, value in labels.items():
        avg2 = get_avg(dataframe_series[(y == value) & (y_pred_label == value)])
        classification1 = "True " + str(value)
        all_averages.append((classification1, avg2))

    for index, value in labels.items():
        avg3 = get_avg(dataframe_series[(y != value) & (y_pred_label == value)])
        classification2 = "False " + str(value)
        all_averages.append((classification2, avg3))

    for index, value in labels.items():
        avg4 = get_avg(dataframe_series[(y == value) & (y_pred_label != value)])
        classification3 = "Misclassified Class " + str(value)
        all_averages.append((classification3, avg4))

        all_averages_df = pd.DataFrame(all_averages, columns=['label', column_name])
        all_averages_df.set_index('label', inplace=True)

    return all_averages_df

# Sum specified column and divide by number of instances in dataframe
# to get average word_count per instance
def get_feature_statistics_avg3(dataframe_series, y, y_pred_label, labels, name):
    df = pd.DataFrame()
    for index, value in labels.items():
        count = []
        for index2, value2 in labels.items():
            avg = get_avg(dataframe_series[(y == value2) & (y_pred_label == value)])
            count.append(avg)
            new_series = pd.Series(count, name=value)
        df = pd.concat([df, new_series], axis=1)
    labels_as_list = labels.tolist()
    df.set_index([labels_as_list], inplace=True)
    df.name = name
    return df


# #######################################################################
# Functions for Feature Analysis

def build_whole_ds_feature_analysis(df, df2, y, labels):

    df2_count = build_token_count(df2, "All")
    token_class_count = build_token_count_by_class(df, y, labels)
    print("token_class_count")
    print(token_class_count)

    rows = ""
    for key, value in token_class_count.iteritems():
        per_token = ""
        per_token += """**%s ** by class: """ % (key)
        if key in df2_count['token_name'].values:
            font = "font2"
        else:
            font = "font3"

        for index2, feature_count in value.iteritems():
            if index2 != 'total':
                per_token += """%s (%s) """ % (index2, int(feature_count))
            else:
                total = int(feature_count)

        rows +="""<span class="%s" title="%s">%s (%s), </span>"""%(font, per_token, key, total)
    table = rows

    return table


# Function to build dataframe of terms frequencies
def build_token_count_by_class(df, y, labels):
    token_count = []
    #df = pd.concat([df, y], axis=1)
    df_new = pd.DataFrame()
    for col in df.columns:
        count = []
        for index, value in labels.items():
            df2 = df[(y == value)]
            sum = df2[col].sum()
            count.append(sum)
            new_series = pd.Series(count, name=col)
        df_new = pd.concat([df_new, new_series], axis=1)
    labels_as_list = labels.tolist()
    df_new.set_index([labels_as_list], inplace=True)

    df_new.loc['total', :] = df_new.sum(axis=0)

    df_new = df_new.sort_values(by='total', axis=1, ascending=False)

    return df_new

#zero_token_count = build_token_count(word_count_matrix[(y == 0)], "Class Zero tokens")

# Function to build dataframe of terms frequencies
def build_token_count(df, label):
    token_count = []
    # For each column in dataframe sum values i.e. total token occurrences
    for col in df.columns:
        column_total = df[col].sum()
        # Remove columns totaling 0 as token doesn't appear
        if column_total > 0:
            token_count.append((col, column_total))
        token_count_df = pd.DataFrame(token_count, columns=['token_name', 'count'])
        # To sort by frequency
        token_count_df = token_count_df.sort_values(by='count', ascending=False)
        token_count_df.label = label
    return token_count_df

# Function to build dataframe of terms frequencies
def build_token_count_with_POS(df, label):
    token_count = []
    # For each column in dataframe sum values i.e. total token occurrences
    for col in df.columns:
        col_pos = nltk.pos_tag([col])
        for word, tag in col_pos:
            col_pos = tag

        if col_pos in ['NN', 'NNS', 'NNP', 'NNPS']:
            pos = "Noun"
        elif col_pos in ['VB', 'VBD', 'VBZ', 'VBG', 'VBP', 'VBN']:
            pos = "Verb"
        elif col_pos in ['RB', 'RBR', 'RBS', 'WRB']:
            pos = "Adv."
        elif col_pos in ['JJ', 'JJR', 'JJS']:
            pos = "Adj."
        else:
            pos = "pos"

        column_total = df[col].sum()
        # Remove columns totaling 0 as token doesn't appear
        if column_total > 0:
            token_count.append((col, column_total, pos))

        token_count_df = pd.DataFrame(token_count, columns=['token_name', 'count', 'pos'])
        # To sort by frequency
        token_count_df = token_count_df.sort_values(by='count', ascending=False)
        token_count_df.label = label
    return token_count_df

# First find the shared tokens by comparing tokens in one dataframe to tokens of another
def build_whole_class_token_counts_as_json(labels, df, y):

    for index, value in labels.items():

        token_list_df = build_token_count_with_POS(df[(y == value)], value)
        token_list_df.name = value
        print("token_list_df")
        print(token_list_df)
        print(type(token_list_df))

        file_name = value
        build_word_cloud2(token_list_df, file_name, 300, 60)

        path = "data/feature_analysis_" + str(value) + ".json"

        token_list_df.to_json(path, orient='records') #, orient='index'
        del [[token_list_df]]
        gc.collect()

        #class_features.append((value, token_list_df))

# First find the shared tokens by comparing tokens in one dataframe to tokens of another
def build_whole_class_token_counts_as_json2(labels, df, y, y_pred_label):

    for index, value in labels.items():
        label = "True_" + str(value)
        token_list_df = build_token_count_with_POS(df[(y == value) & (y_pred_label == value)], label)
        token_list_df.name = label
        print("token_list_df")
        print(token_list_df)
        print(type(token_list_df))
        #file_name = label
        #build_word_cloud2(token_list_df, file_name, 300, 60)
        path = "data/feature_analysis_" + label + ".json"
        token_list_df.to_json(path, orient='records')
        del [[token_list_df]]
        gc.collect()

    for index, value in labels.items():
        label = "False_" + str(value)
        token_list_df = build_token_count_with_POS(df[(y != value) & (y_pred_label == value)], label)
        token_list_df.name = label
        print("token_list_df")
        print(token_list_df)
        print(type(token_list_df))
        #file_name = label
        #build_word_cloud2(token_list_df, file_name, 300, 60)
        path = "data/feature_analysis_" + label + ".json"
        token_list_df.to_json(path, orient='records')
        del [[token_list_df]]
        gc.collect()

    for index, value in labels.items():
        label = "Misclassified_" + str(value)
        token_list_df = build_token_count_with_POS(df[(y == value) & (y_pred_label != value)], label)
        token_list_df.name = label
        print("token_list_df")
        print(token_list_df)
        print(type(token_list_df))
        #file_name = label
        #build_word_cloud2(token_list_df, file_name, 300, 60)
        path = "data/feature_analysis_" + str(label) + ".json"
        token_list_df.to_json(path, orient='records') #, orient='index'

        del [[token_list_df]]
        gc.collect()

    #DELETE DATAFRAMEs
        #class_features.append((value, token_list_df))


def build_whole_class_feature_analysis(labels):

    tag1 = """<table>"""

    complete_tag = ""
#  <div id="div1" class="linked-div" style="display: none">&nbsp;One</div>
# <table class="table_alternate_colour" id="%s" style="display:none">
    for index, value in labels.items():

        path = "data/feature_analysis_" + str(value) + ".json"

        with open(path) as json_file:
            data = json.load(json_file)

        tag2 = """
         <div >
                <tr id="%s" class="linked-div" style="display: none">
                     <td>                     
                   <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
       Class %s Feature Analysis 
      </th>
      </tr>    
         <tr>
            <td>
                <img src="images/%s_wordcloud.png" alt="Most common tokens in %s class">
            </td>
         </tr>
             <tr>
               <td>"""%(str(value), str(value), str(value), str(value))

        rows = ""

        for row in data:

            for i in row.values():
                token = row['token_name']
                count = row['count']
            rows += """%s (%s) """ % (str(token), str(count))

        tag3 = """
            </td>
            </tr>
        </table>
         </td>
        </tr>

            </div>
            """

        complete_tag += tag2 + rows + tag3

    tag4 = """</table>"""

    table = tag1 + complete_tag + tag4

    return table

# First find the shared tokens by comparing tokens in one dataframe to tokens of another
def build_feature_analysis(labels, df, y):

    class_features = []

    for index, value in labels.items():
        token_list_df = build_token_count(df[(y == value)], value)
        class_features.append((value, token_list_df))
        del [[token_list_df]]
        gc.collect()
        token_list_df = pd.DataFrame()

    print("class_features")
    print(class_features)
    return class_features



# #######################################################################
# Functions for TFIDF Analysis

# Function to calculate the highest averaging tfidf words/tokens
# for each classification group as passed in as dataframe argument
# for all or the specified part of speech
def top_tfidf_tokens(df, pos, label):
    if pos != 'all':
        if pos == 'nouns':
            part_of_speech = ['NN', 'NNS', 'NNP', 'NNPS']
        elif pos == 'verbs':
            part_of_speech = ['VB', 'VBD', 'VBZ', 'VBG', 'VBP', 'VBN']
        elif pos == 'adverbs':
            part_of_speech = ['RB', 'RBR', 'RBS', 'WRB']
        elif pos == 'adjectives':
            part_of_speech = ['JJ', 'JJR', 'JJS']

        pos_reg_ex = re.compile(r'\b(?:%s)\b' % '|'.join(part_of_speech))
        df = df.filter(regex=pos_reg_ex)

    tfidf_avg = []
    for col in df.columns:
        # Sum column of tfidf values
        column_avg = df[col].mean()
        # Add token name and tfidf value to list
        tfidf_avg.append((col, column_avg))
        tfidf_avg_df = pd.DataFrame(tfidf_avg, columns=['token', 'tfidf'])
        # Sort by tfidf value and return top 50
        top_tfidf_df = tfidf_avg_df.sort_values(by='tfidf', ascending=False)[:50]
        top_tfidf_df.label = label

    return top_tfidf_df

def build_top_tfidf(tn_tfidf_matrix, tp_tfidf_matrix, fn_tfidf_matrix, fp_tfidf_matrix, pos):

    tn_top_tfidf_tokens_pos = top_tfidf_tokens(tn_tfidf_matrix,pos)
    tn_top_tfidf_tokens_pos.label = "True Negatives"
    tp_top_tfidf_tokens_pos = top_tfidf_tokens(tp_tfidf_matrix,pos)
    tp_top_tfidf_tokens_pos.label = "True Positives"
    fn_top_tfidf_tokens_pos = top_tfidf_tokens(fn_tfidf_matrix,pos)
    fn_top_tfidf_tokens_pos.label = "False Negatives "
    fp_top_tfidf_tokens_pos = top_tfidf_tokens(fp_tfidf_matrix,pos)
    fp_top_tfidf_tokens_pos.label = "False Positives"

    tfidf_by_pos_lists = build_list(tn_top_tfidf_tokens_pos, tp_top_tfidf_tokens_pos, fn_top_tfidf_tokens_pos, fp_top_tfidf_tokens_pos)

    del [[tn_tfidf_matrix, tp_tfidf_matrix, fn_tfidf_matrix, fp_tfidf_matrix]]
    gc.collect()

    return tfidf_by_pos_lists

def build_tfidf_analysis(labels, data_matrix, y, y_pred_label):
    for index, value in labels.items():
        # Per Class
        df = data_matrix[(y == value)]
        label = 'all_' + value
        # Call function to build the top average tfidf values for each classification group for each part of speech
        tfidf_by_pos_all = top_tfidf_tokens(df, 'all', label)

        #tfidf_by_pos_nouns = top_tfidf_tokens(df, 'nouns')
        #tfidf_by_pos_verbs = top_tfidf_tokens(df, 'verbs')
        #tfidf_by_pos_adjs = top_tfidf_tokens(df, 'adjectives')
        #tfidf_by_pos_advs = top_tfidf_tokens(df, 'adverbs')

        # Plot top 50 tfidf value words results of each group for comparison
        build_single_top_tfidf_tokens_bar_chart(tfidf_by_pos_all, label)
        #build_single_top_tfidf_tokens_bar_chart(tfidf_by_pos_nouns, "nouns")
        #build_single_top_tfidf_tokens_bar_chart(tfidf_by_pos_verbs, "verbs")
        #build_single_top_tfidf_tokens_bar_chart(tfidf_by_pos_adjs, "adjs")
        #build_single_top_tfidf_tokens_bar_chart(tfidf_by_pos_advs, "advs")

        del [[df]]
        gc.collect()

        # Per True
        df2 = data_matrix[(y == value) & (y_pred_label == value)]
        label = 'all_true_' + value
        tfidf_by_pos_all2 = top_tfidf_tokens(df2, 'all', label)
        build_single_top_tfidf_tokens_bar_chart(tfidf_by_pos_all2, label)
        del [[df2]]
        gc.collect()

        # Per False
        df3 = data_matrix[(y != value) & (y_pred_label == value)]
        label = 'all_false_' + value
        tfidf_by_pos_all3 = top_tfidf_tokens(df3, 'all', label)
        build_single_top_tfidf_tokens_bar_chart(tfidf_by_pos_all3, label)
        del [[df3]]
        gc.collect()

        #Per Misclassifed
        df4 = data_matrix[(y == value) & (y_pred_label != value)]
        label = 'all_misclassified_' + value
        tfidf_by_pos_all4 = top_tfidf_tokens(df4, 'all', label)
        build_single_top_tfidf_tokens_bar_chart(tfidf_by_pos_all4, label)
        del [[df4]]
        gc.collect()



def add_pos_to_colname(df):
    colname = df.columns.get_values()
    pos_token = nltk.pos_tag(colname)
    pos_list = list()
    for x, y in pos_token:
        pos_list.append(x + "-" + y)
    df.columns = [pos_list]

def build_list(df,df2,df3,df4):
    df_list = list()
    df_list.append(df)
    df_list.append(df2)
    df_list.append(df3)
    df_list.append(df4)
    return df_list

def build_misclassified_feature_analysis(labels, data_matrix, feature_group1, feature_group2):

    tag1 = """<table>"""
    complete_tag = ""

    for index, value in labels.items():
        path = "data/feature_analysis_" + str(feature_group1) + "_" + str(value) + ".json"
        print("path")
        print(path)
        with open(path) as json_file:
            data = json.load(json_file)

        data_df = pd.DataFrame.from_dict(data, orient='columns')

        # For class
        path2 = "data/feature_analysis_" + str(value) + ".json"
        # For true class
        # path2 = "data/feature_analysis_" + str(feature_group2) + "_" + str(value) + ".json"
        print("path2")
        print(path2)
        with open(path2) as json_file:
            data2 = json.load(json_file)
        data2_df = pd.DataFrame.from_dict(data2, orient='columns')

        if data_df.empty:

            tag2 = """
                <div >
                <tr id="%s" class="linked-div" style="display: none">
                     <td>                     
                   <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
      %s %s Feature Analysis 
      </th>
      </tr>    
            <tr>
      <td>
      </br>
    NO %s %s INSTANCES
    </br>
    </td>
      </tr>
      </table> 
                 </td>
                </tr>
    
                    </div>
                    """% (str(value), str(feature_group1), str(value), str(feature_group1).upper(), str(value).upper())
            complete_tag += tag2

        else:
            if feature_group1 == "False":
                df = data_matrix[(data_matrix['label'] != value) & (data_matrix['pred_label'] == value)]
                print("false instances")
                print(df)
            else:
                df = data_matrix[(data_matrix['label'] == value) & (data_matrix['pred_label'] != value)]
                print("instances")
                print(df)

            misclassifications_table = build_misclassifications_list(feature_group1, df, value)

            shared_tokens1 = data_df[data_df['token_name'].isin(data2_df['token_name'])]
            shared_tokens1 = shared_tokens1.sort_values(by='token_name')
            print("shared_tokens1")
            print(shared_tokens1)

            shared_tokens2 = data2_df[data2_df['token_name'].isin(data_df['token_name'])]
            shared_tokens2 = shared_tokens2.sort_values(by='token_name')
            print("shared_tokens2")
            print(shared_tokens2)

            shared_tokens = pd.concat([shared_tokens1.reset_index(drop=True), shared_tokens2.reset_index(drop=True)], axis=1)
            print("shared_tokens after concat")
            print(shared_tokens)
            shared_tokens.columns = ['count_1', 'pos','token_name', 'count_2', 'pos_2', 'token_name_2']
            print("shared_tokens after column rename")
            print(shared_tokens)
            shared_tokens = shared_tokens.drop('token_name_2', 1)
            shared_tokens = shared_tokens.drop('pos_2', 1)
            print("shared_tokens after columns dropped")
            print(shared_tokens)
            shared_tokens = shared_tokens.sort_values(by='count_1', ascending=False)

            # Next find the unique terms by checking tokens of dataframe do not appear
            # in the previously identified shared terms dataframe
            unique_tokens_1 = data_df[~data_df['token_name'].isin(shared_tokens['token_name'])]
            unique_tokens_2 = data2_df[~data2_df['token_name'].isin(shared_tokens['token_name'])]

            file_name1 = "unique_" + feature_group1 + "_" + value
            file_name2 = feature_group1 + "_unique_class_" + value
            file_name3 = "shared_" + feature_group1 + "_and_class_" + value

            tfidf_file_name1 = "tfidf_all_" + feature_group1 + "_" + value + "_barchart"
            tfidf_file_name2 = "tfidf_all_" + value + "_barchart"

            build_word_cloud2(unique_tokens_1, file_name1, 300, 60)
            build_word_cloud2(unique_tokens_2, file_name2, 300, 60)
            build_word_cloud2(shared_tokens, file_name3, 300, 60)

            print("shared_tokens")
            print(shared_tokens)
            print("unique_tokens_1")
            print(unique_tokens_1)
            print("unique_tokens_2")
            print(unique_tokens_2)

            class0_unique_tokens = build_content_dataframe_to_string(unique_tokens_1)
            common_tokens = build_content_dataframe_to_string2(shared_tokens)
            class1_unique_tokens = build_content_dataframe_to_string(unique_tokens_2)

            tag2 = """
                <div >
                <tr id="%s" class="linked-div" style="display: none">
                     <td>                     
                   <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
     %s %s Feature Analysis 
      </th>
      </tr>
      
            <tr>
      <td>
     <table class="features">
      <tr class="features">        
        <td class="features" colspan="3"><h2><a id="top"></a>%s %s instances listed below </h2></td>
      </tr>
      <tr class="features">        
        <td class="features"><h2>Tokens unique to %s %s instances</h2></td> 
        <td class="features"><h2>Shared tokens in %s %s and class %s instances</h2></td>
        <td class="features"><h2>Tokens unique to class %s instances</h2></td>
      </tr>
      <tr class="features">
        <td class="features1"><img src="images/%s_wordcloud.png"><p>%s <b>(# of tokens: ???)</b></p> </td>
        <td class="features2"><img src="images/%s_wordcloud.png"><p>%s <b>(# of tokens: ???)</b></p></td>
        <td class="features3"><img src="images/%s_wordcloud.png"><p>%s <b>(# of tokens: ???)</b></td>
      </tr>
      <tr class="features">
        <td colspan=3>
        <table>
           <tr>
             <td><img src="images/%s.png"></td>
             <td><img src="images/%s.png"></td>
           </tr>
        </table>
        </td>
      </tr>
      <tr class="table_alternate_colour">
        <td colspan="3">%s</td>
      </tr>
    </table> 
    </td>
      </tr>
      </table> 
                 </td>
                </tr>  
                    </div>    
                    """%(str(value), str(feature_group1), str(value), str(feature_group1), str(value), str(feature_group1), str(value), str(feature_group1), str(value), str(value), str(value), file_name1, class0_unique_tokens, file_name3, common_tokens, file_name2, class1_unique_tokens, tfidf_file_name1, tfidf_file_name2, misclassifications_table)
            complete_tag += tag2

    tag4 = """</table>"""

    table = tag1 + complete_tag + tag4

    return table


# First find the shared tokens by comparing tokens in one dataframe to tokens of another
def build_feature_analysis_binary(df1, df2):
    shared_tokens1 = df1[df1['token_name'].isin(df2['token_name'])]
    shared_tokens1 = shared_tokens1.sort_values(by='token_name')

    shared_tokens2 = df2[df2['token_name'].isin(df1['token_name'])]
    shared_tokens2 = shared_tokens2.sort_values(by='token_name')

    shared_tokens = pd.concat([shared_tokens1.reset_index(drop=True), shared_tokens2.reset_index(drop=True)], axis=1)
    shared_tokens.columns = ['token_name', 'count_1', 'token_name_2', 'count_2']
    shared_tokens = shared_tokens.drop('token_name_2', 1)
    shared_tokens = shared_tokens.sort_values(by='count_1', ascending=False)

    # Next find the unique terms by checking tokens of dataframe do not appear
    # in the previously identified shared terms dataframe
    unique_tokens_1 = df1[~df1['token_name'].isin(shared_tokens['token_name'])]
    unique_tokens_2 = df2[~df2['token_name'].isin(shared_tokens['token_name'])]

    return shared_tokens, unique_tokens_1, unique_tokens_2

def build_feature_analysis_word_clouds(df, df_name, df2, df2_name, df3, df3_name):

    build_word_cloud(df2, df2_name, 300, 60)
    build_word_cloud(df, df_name, 300, 60)
    build_word_cloud(df3, df3_name, 300, 60)


def build_content_dataframe_to_string(df):
    rows = []
    for index, row in df.iterrows():
        token = row['token_name']
        count = row['count']
        count = str(count)
        rows.append("""%s(%s) """ % (token, count))
    result = ''.join(rows)
    return result


def build_content_dataframe_to_string2(df):
    rows = []
    for index, row in df.iterrows():
        token = row['token_name']
        count1 = row['count_1']
        count2 = row['count_2']
        rows.append("""%s(%s, %s) """ % (token, count1, count2))
    result = ''.join(rows)
    return result

def build_misclassifications_list(feature_group1, misclassifications, value):
    tag1 = """<br><table class="table_alternate_colour">"""
    rows = ""
    rows += """<tr>
                  <th colspan=3><h2> <a id="fn"></a> %s Instances for Class %s </h2>  <a href="#top" class="anchor">Back to top</a></th>
               </tr>""" % (feature_group1, value)

    for index, row in misclassifications.iterrows():
        text = row['text']
        true_label = row['label']
        pred_label = row['pred_label']

        rows +="""<tr>
                        <td>Label %s</td>
                        <td>False %s</td>
                        <td>%s</td>
                    </tr>""" % (true_label, pred_label, text)
        temp_label = pred_label
    tag3 = """</table>"""
    table = tag1 + rows + tag3
    return table

"""
# Sum word_count column and divide by number of instances in dataframe
# to get average word_count per instance
def get_avg_num_words(dataframe):
    total_words = dataframe['words'].sum()
    avg_num_words = total_words/dataframe['words'].count()
    return avg_num_words

# Sum char_count column and divide by number of instances in dataframe
# to get average char_count per instance
def get_avg_num_chars(dataframe):
    total_chars = dataframe['chars'].sum()
    avg_char_count = total_chars/dataframe['chars'].count()
    return avg_char_count

# Sum misspellings column and divide by number of instances in dataframe
# to get average misspellings per instance
def get_avg_num_misspellings(df):
    total_num_misspellings = df['misspellings'].sum()
    avg_num_misspellings = total_num_misspellings/df['misspellings'].count()
    return avg_num_misspellings

# Sum specified pos column and divide by number of instances in dataframe
# to get average specified pos per instance
def get_avg_num_pos(df, column_name):
    pos_total = df[column_name].sum()
    avg_pos = pos_total/df[column_name].count()
    return avg_pos
"""
"""
# #######################################################################
# Functions for TFIDF Analysis

# Function to calculate the highest averaging tfidf words/tokens
# for each classification group as passed in as dataframe argument
# for all or the specified part of speech
def top_tfidf_tokens(df, pos):
    if pos != 'all':
        if pos == 'nouns':
            part_of_speech = ['NN', 'NNS', 'NNP', 'NNPS']
        elif pos == 'verbs':
            part_of_speech = ['VB', 'VBD', 'VBZ', 'VBG', 'VBP', 'VBN']
        elif pos == 'adverbs':
            part_of_speech = ['RB', 'RBR', 'RBS', 'WRB']
        elif pos == 'adjectives':
            part_of_speech = ['JJ', 'JJR', 'JJS']

        pos_reg_ex = re.compile(r'\b(?:%s)\b' % '|'.join(part_of_speech))
        df = df.filter(regex=pos_reg_ex)

    tfidf_avg = []
    for col in df.columns:
        # Sum column of tfidf values
        column_avg = df[col].mean()
        # Add token name and tfidf value to list
        tfidf_avg.append((col, column_avg))
        tfidf_avg_df = pd.DataFrame(tfidf_avg, columns=['token', 'tfidf'])
        # Sort by tfidf value and return top 50
        top_tfidf_df = tfidf_avg_df.sort_values(by='tfidf', ascending=False)[:50]

    return top_tfidf_df

def build_top_tfidf(tn_tfidf_matrix, tp_tfidf_matrix, fn_tfidf_matrix, fp_tfidf_matrix, pos):

    tn_top_tfidf_tokens_pos = top_tfidf_tokens(tn_tfidf_matrix,pos)
    tn_top_tfidf_tokens_pos.label = "True Negatives"
    tp_top_tfidf_tokens_pos = top_tfidf_tokens(tp_tfidf_matrix,pos)
    tp_top_tfidf_tokens_pos.label = "True Positives"
    fn_top_tfidf_tokens_pos = top_tfidf_tokens(fn_tfidf_matrix,pos)
    fn_top_tfidf_tokens_pos.label = "False Negatives "
    fp_top_tfidf_tokens_pos = top_tfidf_tokens(fp_tfidf_matrix,pos)
    fp_top_tfidf_tokens_pos.label = "False Positives"

    tfidf_by_pos_lists = build_list(tn_top_tfidf_tokens_pos, tp_top_tfidf_tokens_pos, fn_top_tfidf_tokens_pos, fp_top_tfidf_tokens_pos)

    del [[tn_tfidf_matrix, tp_tfidf_matrix, fn_tfidf_matrix, fp_tfidf_matrix]]
    gc.collect()

    return tfidf_by_pos_lists


def add_pos_to_colname(df):
    colname = df.columns.get_values()
    pos_token = nltk.pos_tag(colname)
    pos_list = list()
    for x, y in pos_token:
        pos_list.append(x + "-" + y)
    df.columns = [pos_list]

def build_list(df,df2,df3,df4):
    df_list = list()
    df_list.append(df)
    df_list.append(df2)
    df_list.append(df3)
    df_list.append(df4)
    return df_list





  <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        <a id="top"></a> %s Feature Analysis 
      </th>
      </tr>
            <tr>
      <td>
     <table class="features">
      <tr class="features">        
        <td class="features"><a id="tn"></a><h2> Tokens unique to %s </h2></td>
        <td class="features"><h2>%s tokens shared with TN</h2></td>
        <td class="features"><h2> Tokens unique to TN </h2> <a href="#top" class="anchor">Back to top</a></td>
      </tr>
      <tr class="features">
        <td class="features1"><img src="images/%s.png"><p>%s <b>(# of tokens: %s)</b></p> </td>
        <td class="features2"><img src="images/%s.png"><p>%s <b>(# of tokens: %s)</b></p></td>
        <td class="features3"><img src="images/%s.png"><p>%s <b>(# of tokens: %s)</b></td>
      </tr>
    </table> 
    </td>
      </tr>
      </table>  









"""