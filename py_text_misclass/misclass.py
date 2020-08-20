import pandas as pd
import nltk
from sklearn import metrics
import time

from .analysis_content import calc_sentence_count, calc_word_count, calc_unique_words_count, calc_char_count, \
    calc_punctuation_count, calc_uppercase_char_count, calc_lowercase_char_count, \
    calc_non_alphanumeric_char_count, calc_number_count, calc_space_count, calc_mispellings_count, get_classifications, \
    get_feature_statistics_avg, get_feature_statistics_avg2, get_feature_statistics_avg3, \
    get_feature_statistics_count, get_feature_statistics_count2, get_feature_statistics_count3, \
    calc_pos_count, \
    build_token_count, build_feature_analysis_binary, build_whole_class_token_counts_as_json, \
    build_whole_class_token_counts_as_json2, build_whole_class_feature_analysis, \
    build_feature_analysis_word_clouds, build_whole_ds_feature_analysis, build_misclassified_feature_analysis, \
    top_tfidf_tokens, add_pos_to_colname, build_top_tfidf, build_tfidf_analysis

from .page_content import build_index_content, build_statistics_content, build_pos_content, \
    build_whole_ds_feature_analysis_content, build_feature_analysis_content, \
    build_misclassified_feature_analysis_content
# build_misclassified_features_analysis_content, \
# build_tfidf_summary_content, build_misclassified_features_content, build_classification_features_analysis_content, \
# build_class_features_analysis_content
from .build import create_directory, create_image_directory, create_page, open_website, copy_css, copy_js_folder, \
    move_images
from .visualisations import build_top_tfidf_tokens_bar_chart


class Misclass(object):

    def start_timer(self):
        start_time = time.process_time()
        return start_time

    def end_timer(self, start_time):
        elapsed_time = round((time.process_time() - start_time), 3)
        return elapsed_time

    def initial_message(self):
        print()
        print("Please wait a few moments....")
        print("The HTML Text Misclassification Analysis report will open in your browser....")

    def get_record_count(self, word_count_matrix):
        # Count number of records
        record_count = word_count_matrix.shape[0]
        return record_count

    def get_num_tokens(self, word_count_matrix):
        # Count number of (columns) tokens in word count matrix
        num_tokens = word_count_matrix.shape[1]
        return num_tokens

    def get_all_num_tokens(self, all_word_count_matrix):
        # Count number of (columns) tokens in word count matrix before any pre-processing
        all_num_tokens = all_word_count_matrix.shape[1]
        return all_num_tokens

    def get_label_count(self, y, record_count):
        # Find class label distribution
        label_count = y.value_counts()
        # Find percentage class label distribution
        label_list = []
        for index, value in label_count.items():
            label_percentage = round(100 / record_count * label_count[index], 2)
            label = " <b>Class " + str(index) + ': </b>' + str(value) + " (" + str(label_percentage) + '%) | '
            label_list.append(label)
            label_list_str = ''
            for s in label_list:
                label_list_str += s
        return label_list_str

    def get_labels(self, y):
        labels = y.unique()
        labels = pd.Series(labels, name="labels")
        return labels

    def get_data_matrix(self, X, y, y_pred_label):
        # Concat predicted label to text and label dataframe
        data_matrix = pd.concat([X, y, y_pred_label], axis=1)
        return data_matrix

    def get_null_accuracy(self, y):
        # Calculate null accuracy
        null_accuracy = float(y.value_counts().head(1) / len(y))
        return null_accuracy

    def get_accuracy(self, y, y_pred_label):
        # Calculate classification accuracy
        accuracy = round(metrics.accuracy_score(y, y_pred_label), 4)
        return accuracy

    def get_misclassification_rate(self, y, y_pred_label):
        # Calculate misclassification rate
        misclassification_rate = round(1 - metrics.accuracy_score(y, y_pred_label), 4)
        return misclassification_rate

    def get_confusion_matrix(self, y, y_pred_label, labels):

        # Using Scikit-Learns confusion matrix metric using predicted and class label values
        confusion_matrix = metrics.confusion_matrix(y, y_pred_label, labels=labels)
        confusion_matrix = pd.DataFrame(data=confusion_matrix[0:, 0:], index=labels, columns=labels)
        return confusion_matrix

    def get_classification_report(self, y, y_pred_label, labels):

        # ???
        report = metrics.classification_report(y, y_pred_label, labels=labels, output_dict=True)
        report = pd.DataFrame(report).transpose()
        report_index = list(report.index.values)
        drop_list = []
        for i in report_index:
            if i not in labels.values:
                # drop_list = ['micro avg', 'macro avg', 'weighted avg']
                drop_list.append(i)
        report.drop(drop_list, inplace=True)
        return report

    def get_f1_score(self, classification_report):
        # Calculate F1 Score
        f1_scores = round(classification_report['f1-score'], 3)
        # Find percentage class label distribution
        label_list = []
        for index, value in f1_scores.items():
            label = " <b>Class " + str(index) + ": </b>" + str(value) + " | "
            label_list.append(label)
            label_list_str = ''
            for s in label_list:
                label_list_str += s
        return label_list_str

    def get_misclassifications(self, data_matrix):
        classifications = []
        for index, row in data_matrix.iterrows():
            classification_type = True
            if row['label'] != row['pred_label']:
                classification_type = False
                row['pred_label']
            elif row['label'] == row['pred_label']:
                classification_type = True
                row['pred_label']
            classifications.append((classification_type))
        new_series = pd.Series(classifications, name="classifications")
        data_matrix = pd.concat([data_matrix, new_series], axis=1)

        misclassifications = data_matrix[data_matrix.classifications == False]
        misclassifications = misclassifications.sort_values(by=['pred_label'])
        return misclassifications

    def build(self, X, y, y_pred_label, word_count_matrix, tfidf_matrix, all_word_count_matrix, model_desc, path):

        word_count_matrix.to_csv('data/word_count_matrix.csv')
        tfidf_matrix.to_csv('data/tfidf_matrix.csv')
        all_word_count_matrix.to_csv('data/all_word_count_matrix.csv')

        start_time = self.start_timer()
        self.initial_message()
        record_count = self.get_record_count(word_count_matrix)
        num_tokens = self.get_num_tokens(word_count_matrix)
        all_num_tokens = self.get_all_num_tokens(all_word_count_matrix)
        label_count = self.get_label_count(y, record_count)
        labels = self.get_labels(y)
        data_matrix = self.get_data_matrix(X, y, y_pred_label)
        confusion_matrix = self.get_confusion_matrix(y, y_pred_label, labels)
        classification_report = self.get_classification_report(y, y_pred_label, labels)
        null_accuracy = self.get_null_accuracy(y)
        accuracy = self.get_accuracy(y, y_pred_label)
        misclassification_rate = self.get_misclassification_rate(y, y_pred_label)
        f1_score = self.get_f1_score(classification_report)
        misclassifications = self.get_misclassifications(data_matrix)

        # Number of words
        # Number of chars
        # ***Number of uppercase chars
        # ***Number of lowercase chars
        # ***Number of non-alphabetic characters
        # ***Number of numbers
        # ***Number of sentences
        # ***Number of punctuation
        # Number of misspellings
        # Number of nouns
        # Number of verbs
        # Number of adjectives
        # Number of adverbs
        # ***Additional POS

        # Per whole dataset before pre-processing
        # Per whole dataset after pre-processing
        # Per class label (before or after pre-processing ????)
        # Per false predictions to a particular class (precision), i.e., all false class 1 (after pre-processing as these are the tokens classification are based on)
        # Per false predictions of a particular class (recall), i.e., all class 1s falsely predicted (after pre-processing as these are the tokens classification are based on)

        # ************************************************************
        # Feature Statistics
        # ************************************************************

        # call function to compute word count per instance for dataframe
        data_matrix = calc_word_count(data_matrix)
        # call function to compute char count per instance for dataframe
        data_matrix = calc_char_count(data_matrix)
        # call function to compute char count per instance for dataframe
        data_matrix = calc_sentence_count(data_matrix)
        # call function to compute char count per instance for dataframe
        data_matrix = calc_unique_words_count(data_matrix)
        # call function to compute char count per instance for dataframe
        data_matrix = calc_punctuation_count(data_matrix)
        # call function to compute char count per instance for dataframe
        data_matrix = calc_uppercase_char_count(data_matrix)
        # call function to compute char count per instance for dataframe
        data_matrix = calc_lowercase_char_count(data_matrix)
        # call function to compute char count per instance for dataframe
        data_matrix = calc_non_alphanumeric_char_count(data_matrix)
        # call function to compute char count per instance for dataframe
        data_matrix = calc_number_count(data_matrix)
        # call function to compute char count per instance for dataframe
        data_matrix = calc_space_count(data_matrix)
        # call function to compute mispellings per instance for dataframe
        data_matrix = calc_mispellings_count(data_matrix)

        # Top Feature Statistics table
        ds_record_count = get_feature_statistics_count(data_matrix['label'], y, y_pred_label, labels)
        avg_sentences = get_feature_statistics_avg(data_matrix['sentences'], "sentences", y, y_pred_label, labels)
        avg_words = get_feature_statistics_avg(data_matrix['words'], "words", y, y_pred_label, labels)
        avg_unique_tokens = get_feature_statistics_avg(data_matrix['unique_tokens'], "unique_tokens", y, y_pred_label,
                                                       labels)
        avg_chars = get_feature_statistics_avg(data_matrix['chars'], "chars", y, y_pred_label, labels)
        avg_uppercase = get_feature_statistics_avg(data_matrix['uppercase'], "uppercase", y, y_pred_label, labels)
        avg_lowercase = get_feature_statistics_avg(data_matrix['lowercase'], "lowercase", y, y_pred_label, labels)
        avg_numbers = get_feature_statistics_avg(data_matrix['numbers'], "numbers", y, y_pred_label, labels)
        avg_non_alphanumeric = get_feature_statistics_avg(data_matrix['non_alphanumeric'], "non_alphanumeric", y,
                                                          y_pred_label, labels)
        avg_punctuation = get_feature_statistics_avg(data_matrix['punctuation'], "punctuation", y, y_pred_label, labels)
        avg_space = get_feature_statistics_avg(data_matrix['space'], "space", y, y_pred_label, labels)
        avg_misspellings = get_feature_statistics_avg(data_matrix['misspellings'], "misspellings", y, y_pred_label,
                                                      labels)
        #  Joining Top Feature Statistics table
        feature_statistics_df = pd.concat([ds_record_count, avg_sentences, avg_words, avg_unique_tokens, avg_chars,
                                           avg_uppercase, avg_lowercase, avg_numbers, avg_non_alphanumeric,
                                           avg_punctuation,
                                           avg_space, avg_misspellings], axis=1)

        # Second Feature Statistics table
        ds_record_count2 = get_feature_statistics_count2(data_matrix['label'], y, y_pred_label, labels)
        avg_sentences2 = get_feature_statistics_avg2(data_matrix['sentences'], "sentences", y, y_pred_label, labels)
        avg_words2 = get_feature_statistics_avg2(data_matrix['words'], "words", y, y_pred_label, labels)
        avg_unique_tokens2 = get_feature_statistics_avg2(data_matrix['unique_tokens'], "unique_tokens", y, y_pred_label,
                                                         labels)
        avg_chars2 = get_feature_statistics_avg2(data_matrix['chars'], "chars", y, y_pred_label, labels)
        avg_uppercase2 = get_feature_statistics_avg2(data_matrix['uppercase'], "uppercase", y, y_pred_label, labels)
        avg_lowercase2 = get_feature_statistics_avg2(data_matrix['lowercase'], "lowercase", y, y_pred_label, labels)
        avg_numbers2 = get_feature_statistics_avg2(data_matrix['numbers'], "numbers", y, y_pred_label, labels)
        avg_non_alphanumeric2 = get_feature_statistics_avg2(data_matrix['non_alphanumeric'], "non_alphanumeric", y,
                                                            y_pred_label, labels)
        avg_punctuation2 = get_feature_statistics_avg2(data_matrix['punctuation'], "punctuation", y, y_pred_label,
                                                       labels)
        avg_space2 = get_feature_statistics_avg2(data_matrix['space'], "space", y, y_pred_label, labels)
        avg_misspellings2 = get_feature_statistics_avg2(data_matrix['misspellings'], "misspellings", y, y_pred_label,
                                                        labels)
        #  Joining Second Feature Statistics table
        feature_statistics_df2 = pd.concat(
            [ds_record_count2, avg_sentences2, avg_words2, avg_unique_tokens2, avg_chars2,
             avg_uppercase2, avg_lowercase2, avg_numbers2, avg_non_alphanumeric2, avg_punctuation2,
             avg_space2, avg_misspellings2], axis=1)

        classifications = get_classifications(data_matrix['label'], y, y_pred_label, labels)
        ds_record_count3 = get_feature_statistics_count3(data_matrix['label'], y, y_pred_label, labels, "record count")
        avg_sentences3 = get_feature_statistics_avg3(data_matrix['sentences'], y, y_pred_label, labels, "sentences")
        avg_words3 = get_feature_statistics_avg3(data_matrix['words'], y, y_pred_label, labels, "words")
        avg_unique_tokens3 = get_feature_statistics_avg3(data_matrix['unique_tokens'], y, y_pred_label, labels,
                                                         "unique_tokens")
        avg_chars3 = get_feature_statistics_avg3(data_matrix['chars'], y, y_pred_label, labels, "chars")
        avg_uppercase3 = get_feature_statistics_avg3(data_matrix['uppercase'], y, y_pred_label, labels, "uppercase")
        avg_lowercase3 = get_feature_statistics_avg3(data_matrix['lowercase'], y, y_pred_label, labels, "lowercase")
        avg_numbers3 = get_feature_statistics_avg3(data_matrix['numbers'], y, y_pred_label, labels, "numbers")
        avg_non_alphanumeric3 = get_feature_statistics_avg3(data_matrix['non_alphanumeric'], y, y_pred_label, labels,
                                                            "non_alphanumeric")
        avg_punctuation3 = get_feature_statistics_avg3(data_matrix['punctuation'], y, y_pred_label, labels,
                                                       "punctuation")
        avg_space3 = get_feature_statistics_avg3(data_matrix['space'], y, y_pred_label, labels, "space")
        avg_misspellings3 = get_feature_statistics_avg3(data_matrix['misspellings'], y, y_pred_label, labels,
                                                        "misspellings")

        print("* POS *")
        print(nltk.help.upenn_tagset())

        # ***Add additional POS
        # call function to compute specified parts of speech per instance for dataframe
        data_matrix = calc_pos_count(data_matrix, 'nouns')
        data_matrix = calc_pos_count(data_matrix, 'verbs')
        data_matrix = calc_pos_count(data_matrix, 'adverbs')
        data_matrix = calc_pos_count(data_matrix, 'adjectives')

        # Top POS Analysis table
        # ds_record_count = get_feature_statistics_count(data_matrix['label'], y, y_pred_label, labels)
        avg_nouns = get_feature_statistics_avg(data_matrix['nouns'], "nouns", y, y_pred_label, labels)
        avg_verbs = get_feature_statistics_avg(data_matrix['verbs'], "verbs", y, y_pred_label, labels)
        avg_adverbs = get_feature_statistics_avg(data_matrix['adverbs'], "adverbs", y, y_pred_label, labels)
        avg_adjectives = get_feature_statistics_avg(data_matrix['adjectives'], "adjectives", y, y_pred_label,
                                                    labels)
        #  Joining Top POS Analysis table
        POS_df = pd.concat([ds_record_count, avg_nouns, avg_verbs, avg_adverbs, avg_adjectives], axis=1)

        # Second POS Analysis table
        # ds_record_count2 = get_feature_statistics_count2(data_matrix['label'], y, y_pred_label, labels)
        avg_nouns2 = get_feature_statistics_avg2(data_matrix['nouns'], "nouns", y, y_pred_label, labels)
        avg_verbs2 = get_feature_statistics_avg2(data_matrix['verbs'], "verbs", y, y_pred_label, labels)
        avg_adverbs2 = get_feature_statistics_avg2(data_matrix['adverbs'], "adverbs", y, y_pred_label, labels)
        avg_adjectives2 = get_feature_statistics_avg2(data_matrix['adjectives'], "adjectives", y, y_pred_label, labels)
        #  Joining Second POS Analysis table
        POS_df2 = pd.concat([ds_record_count2, avg_nouns2, avg_verbs2, avg_adverbs2, avg_adjectives2], axis=1)

        # classifications = get_classifications(data_matrix['label'], y, y_pred_label, labels)
        # ds_record_count3 = get_feature_statistics_count3(data_matrix['label'], y, y_pred_label, labels,"record count")
        avg_nouns3 = get_feature_statistics_avg3(data_matrix['nouns'], y, y_pred_label, labels, "nouns")
        avg_verbs3 = get_feature_statistics_avg3(data_matrix['verbs'], y, y_pred_label, labels, "verbs")
        avg_adverbs3 = get_feature_statistics_avg3(data_matrix['adverbs'], y, y_pred_label, labels, "adverbs")
        avg_adjectives3 = get_feature_statistics_avg3(data_matrix['adjectives'], y, y_pred_label, labels, "adjectives")

        data_matrix.to_csv('data/data_matrix_multi.csv')

        time_elapsed = self.end_timer(start_time)

        # ************************************************************
        # Misclassification Feature Analysis - Intersect-Union
        # ************************************************************
        # all_token_count = build_token_count(all_word_count_matrix, "All")  # all features/tokens before any pre-processing
        # word_token_count = build_token_count(word_count_matrix, "All")
        whole_ds_feature_analysis = build_whole_ds_feature_analysis(all_word_count_matrix, word_count_matrix, y, labels)
        title = "Feature Analysis"

        build_whole_class_token_counts_as_json(labels, word_count_matrix, y)
        class_feature_analysis = build_whole_class_feature_analysis(labels)

        build_whole_class_token_counts_as_json2(labels, word_count_matrix, y, y_pred_label)

        # "feature_analysis_", "feature_analysis_True_", "feature_analysis_False", "feature_analysis_Misclassified_
        # false_feature_analysis = build_misclassified_feature_analysis(labels, data_matrix, "False", "True")
        false_feature_analysis = build_misclassified_feature_analysis(labels, data_matrix, "False", "")
        # misclassified_feature_analysis = build_misclassified_feature_analysis(labels, data_matrix, "Misclassified", "True")
        misclassified_feature_analysis = build_misclassified_feature_analysis(labels, data_matrix, "Misclassified", "")

        # zero_token_count = build_token_count(word_count_matrix[(y == 0)], "Class Zero tokens")
        # one_token_count = build_token_count(word_count_matrix[(y == 1)], " Class One tokens")
        # tn_token_count = build_token_count(word_count_matrix[(y == 0) & (y_pred_label == 0)], "TN")
        # fn_token_count = build_token_count(word_count_matrix[(y == 1) & (y_pred_label == 0)], "FN")
        # tp_token_count = build_token_count(word_count_matrix[(y == 1) & (y_pred_label == 1)], "TP")
        # fp_token_count = build_token_count(word_count_matrix[(y == 0) & (y_pred_label == 1)], "FP")

        # fn_tokens_top_50 = build_token_count(word_count_matrix[(y == 1) & (y_pred_label == 0)], "False Negatives").head(
        # 50)

        # fp_tokens_top_50 = build_token_count(word_count_matrix[(y == 0) & (y_pred_label == 1)], "False Positives").head(
        # 50)

        # to delete a dataframe from memory
        # del [[zero_wc_matrix,one_wc_matrix,tn_wc_matrix,tp_wc_matrix,fn_wc_matrix,fp_wc_matrix]]
        # gc.collect()

        # *************************************************
        # CLASSIFICATION FEATURE ANALYSIS - 4 Word clouds
        # *************************************************
        # Create Word Clouds for classification groups
        # build_word_cloud(tn_token_count, "true_negatives", 400, 100)
        # build_word_cloud(fn_token_count, "false_negatives", 400, 100)
        # build_word_cloud(fp_token_count, "false_positives", 400, 100)
        # build_word_cloud(tp_token_count, "true_positives", 400, 100)

        # *************************************************
        # ANALYSIS OF FALSE NEGATIVES FEATURES
        # *************************************************

        """
        # *****   COMPARING FN and TN   *****
        common_tokens_tn_fn, fn_tokens_not_in_tn, tn_tokens_not_in_fn = build_feature_analysis(
                                                                           fn_token_count, tn_token_count)

        common_tokens_tn_fn_count = common_tokens_tn_fn.shape[0]
        fn_tokens_not_in_tn_count = fn_tokens_not_in_tn.shape[0]
        tn_tokens_not_in_fn_count = tn_tokens_not_in_fn.shape[0]

        build_feature_analysis_word_clouds(common_tokens_tn_fn, "FN_common_tokens_in_TN",
                                           fn_tokens_not_in_tn, "FN_tokens_not_in_TN",
                                           tn_tokens_not_in_fn, "FN_TN_tokens_not_in")

        # *****   COMPARING FN and TP   *****
        common_tokens_tp_fn, fn_tokens_not_in_tp, tp_tokens_not_in_fn = build_feature_analysis(
                                                                           fn_token_count, tp_token_count)

        common_tokens_tp_fn_count = common_tokens_tp_fn.shape[0]
        fn_tokens_not_in_tp_count = fn_tokens_not_in_tp.shape[0]
        tp_tokens_not_in_fn_count = tp_tokens_not_in_fn.shape[0]

        build_feature_analysis_word_clouds(common_tokens_tp_fn, "FN_common_tokens_in_TP",
                                           fn_tokens_not_in_tp, "FN_tokens_not_in_TP",
                                           tp_tokens_not_in_fn, "FN_TP_tokens_not_in")

        # *****   COMPARING FN and Zero   *****")
        common_tokens_zero_fn, fn_tokens_not_in_zero, zero_tokens_not_in_fn = build_feature_analysis(
                                                                                 fn_token_count, zero_token_count)

        common_tokens_zero_fn_count = common_tokens_zero_fn.shape[0]
        fn_tokens_not_in_zero_count = fn_tokens_not_in_zero.shape[0]
        zero_tokens_not_in_fn_count = zero_tokens_not_in_fn.shape[0]

        build_feature_analysis_word_clouds(common_tokens_zero_fn, "FN_common_tokens_in_Zero",
                                           fn_tokens_not_in_zero, "FN_tokens_not_in_Zero",
                                           zero_tokens_not_in_fn, "FN_Zero_tokens_not_in")

        # *****   COMPARING FN and One   *****
        common_tokens_one_fn, fn_tokens_not_in_one, one_tokens_not_in_fn = build_feature_analysis(
                                                                              fn_token_count, one_token_count)

        common_tokens_one_fn_count = common_tokens_one_fn.shape[0]
        fn_tokens_not_in_one_count = fn_tokens_not_in_one.shape[0]
        one_tokens_not_in_fn_count = one_tokens_not_in_fn.shape[0]

        build_feature_analysis_word_clouds(common_tokens_one_fn, "FN_common_tokens_in_One",
                                           fn_tokens_not_in_one, "FN_tokens_not_in_One",
                                           one_tokens_not_in_fn, "FN_One_tokens_not_in")

        # *****   COMPARING FN and All   *****")
        common_tokens_all_fn, fn_tokens_not_in_all, all_tokens_not_in_fn = build_feature_analysis(
                                                                                          fn_token_count, all_token_count)

        common_tokens_all_fn_count = common_tokens_all_fn.shape[0]
        fn_tokens_not_in_all_count = fn_tokens_not_in_all.shape[0]
        all_tokens_not_in_fn_count = all_tokens_not_in_fn.shape[0]

        build_feature_analysis_word_clouds(common_tokens_all_fn, "FN_common_tokens_in_All",
                                           fn_tokens_not_in_all, "FN_tokens_not_in_All",
                                           all_tokens_not_in_fn, "FN_All_tokens_not_in")

        # *************************************************
        # ANALYSIS OF FALSE POSITIVES FEATURES
        # *************************************************
        # *****   COMPARING FP and TN   *****
        common_tokens_tn_fp, fp_tokens_not_in_tn, tn_tokens_not_in_fp = build_feature_analysis(fp_token_count,
                                                                                               tn_token_count)

        common_tokens_tn_fp_count = common_tokens_tn_fp.shape[0]
        fp_tokens_not_in_tn_count = fp_tokens_not_in_tn.shape[0]
        tn_tokens_not_in_fp_count = tn_tokens_not_in_fp.shape[0]

        build_feature_analysis_word_clouds(common_tokens_tn_fp, "FP_common_tokens_in_TN",
                                           fp_tokens_not_in_tn, "FP_tokens_not_in_TN",
                                           tn_tokens_not_in_fp, "FP_TN_tokens_not_in")

        # *****   COMPARING FP and TP   *****
        common_tokens_tp_fp, fp_tokens_not_in_tp, tp_tokens_not_in_fp = build_feature_analysis(
                                                                           fp_token_count, tp_token_count)

        common_tokens_tp_fp_count = common_tokens_tp_fp.shape[0]
        fp_tokens_not_in_tp_count = fp_tokens_not_in_tp.shape[0]
        tp_tokens_not_in_fp_count = tp_tokens_not_in_fp.shape[0]

        build_feature_analysis_word_clouds(common_tokens_tp_fp, "FP_common_tokens_in_TP",
                                           fp_tokens_not_in_tp, "FP_tokens_not_in_TP",
                                           tn_tokens_not_in_fp, "FP_TP_tokens_not_in")

        # *****   COMPARING FP and Zero   *****
        common_tokens_zero_fp, fp_tokens_not_in_zero, zero_tokens_not_in_fp = build_feature_analysis(
                                                                                 fp_token_count, zero_token_count)

        common_tokens_zero_fp_count = common_tokens_zero_fp.shape[0]
        fp_tokens_not_in_zero_count = fp_tokens_not_in_zero.shape[0]
        zero_tokens_not_in_fp_count = zero_tokens_not_in_fp.shape[0]

        build_feature_analysis_word_clouds(common_tokens_zero_fp, "FP_common_tokens_in_Zero",
                                           fp_tokens_not_in_zero, "FP_tokens_not_in_Zero",
                                           zero_tokens_not_in_fp, "FP_Zero_tokens_not_in")

        # *****   COMPARING FP and One   *****
        common_tokens_one_fp, fp_tokens_not_in_one, one_tokens_not_in_fp = build_feature_analysis(
                                                                              fp_token_count, one_token_count)

        common_tokens_one_fp_count = common_tokens_one_fp.shape[0]
        fp_tokens_not_in_one_count = fp_tokens_not_in_one.shape[0]
        one_tokens_not_in_fp_count = one_tokens_not_in_fp.shape[0]

        build_feature_analysis_word_clouds(common_tokens_one_fp, "FP_common_tokens_in_One",
                                           fp_tokens_not_in_one, "FP_tokens_not_in_One",
                                           one_tokens_not_in_fp, "FP_One_tokens_not_in")

        # *****   COMPARING FP and All   *****
        common_tokens_all_fp, fp_tokens_not_in_all, all_tokens_not_in_fp = build_feature_analysis(
                                                                                          fp_token_count, all_token_count)

        common_tokens_all_fp_count = common_tokens_all_fp.shape[0]
        fp_tokens_not_in_all_count = fp_tokens_not_in_all.shape[0]
        all_tokens_not_in_fp_count = all_tokens_not_in_fp.shape[0]

        build_feature_analysis_word_clouds(common_tokens_all_fp, "FP_common_tokens_in_All",
                                           fp_tokens_not_in_all, "FP_tokens_not_in_All",
                                           all_tokens_not_in_fp, "FP_All_tokens_not_in")

        # *************************************************
        # COMPARING CLASS 0 VS CLASS 1 FEATURE ANALYSIS
        # *************************************************
        common_tokens_class0_class1, class0_tokens_not_in_class1, class1_tokens_not_in_class0 = build_feature_analysis(
                                                                                    zero_token_count, one_token_count)

        common_tokens_class0_class1_count = common_tokens_class0_class1.shape[0]
        class0_tokens_not_in_class1_count = class0_tokens_not_in_class1.shape[0]
        class1_tokens_not_in_class0_count = class1_tokens_not_in_class0.shape[0]

        build_feature_analysis_word_clouds(common_tokens_class0_class1, "common_tokens_class0_class1",
                                           class0_tokens_not_in_class1, "class0_tokens_not_in_class1",
                                           class1_tokens_not_in_class0, "class1_tokens_not_in_class0")
        """

        # *************************************************
        # TF-IDF Evaluation
        # *************************************************
        # Add part of speech to column names of tfidf matrix
        add_pos_to_colname(tfidf_matrix)

        build_tfidf_analysis(labels, tfidf_matrix, y, y_pred_label)

        """
        # Filter tfidf matrix to classification group
        tn_tfidf_matrix = tfidf_matrix[(y == 0) & (y_pred_label == 0)]
        tp_tfidf_matrix = tfidf_matrix[(y == 1) & (y_pred_label == 1)]
        fn_tfidf_matrix = tfidf_matrix[(y == 1) & (y_pred_label == 0)]
        fp_tfidf_matrix = tfidf_matrix[(y == 0) & (y_pred_label == 1)]

        fn_top_tfidf_tokens = top_tfidf_tokens(fn_tfidf_matrix, 'all')

        fp_top_tfidf_tokens = top_tfidf_tokens(fp_tfidf_matrix, 'all')

        # Call function to build the top average tfidf values for each classification group for each part of speech
        tfidf_by_pos_all = build_top_tfidf(tn_tfidf_matrix, tp_tfidf_matrix, fn_tfidf_matrix, fp_tfidf_matrix,
                                                 'all')
        tfidf_by_pos_nouns = build_top_tfidf(tn_tfidf_matrix, tp_tfidf_matrix, fn_tfidf_matrix, fp_tfidf_matrix,
                                                    'nouns')
        tfidf_by_pos_verbs = build_top_tfidf(tn_tfidf_matrix, tp_tfidf_matrix, fn_tfidf_matrix, fp_tfidf_matrix,
                                                    'verbs')
        tfidf_by_pos_adjs = build_top_tfidf(tn_tfidf_matrix, tp_tfidf_matrix, fn_tfidf_matrix, fp_tfidf_matrix,
                                                   'adjectives')
        tfidf_by_pos_advs = build_top_tfidf(tn_tfidf_matrix, tp_tfidf_matrix, fn_tfidf_matrix, fp_tfidf_matrix,
                                                   'adverbs')

        # Plot top 50 tfidf value words results of each group for comparison
        build_top_tfidf_tokens_bar_chart(tfidf_by_pos_all, "all")
        build_top_tfidf_tokens_bar_chart(tfidf_by_pos_nouns, "nouns")
        build_top_tfidf_tokens_bar_chart(tfidf_by_pos_verbs, "verbs")
        build_top_tfidf_tokens_bar_chart(tfidf_by_pos_adjs, "adjs")
        build_top_tfidf_tokens_bar_chart(tfidf_by_pos_advs, "advs")

        """

        # *************************************************
        # Creating content for webpages
        # *************************************************

        index_content = build_index_content(model_desc, time_elapsed, record_count, label_count,
                                            all_num_tokens, num_tokens, null_accuracy,
                                            accuracy, misclassification_rate, confusion_matrix, classification_report,
                                            f1_score, misclassifications)

        statistics_content = build_statistics_content(feature_statistics_df, feature_statistics_df2, classifications,
                                                      ds_record_count3, avg_sentences3,
                                                      avg_words3, avg_unique_tokens3, avg_chars3, avg_uppercase3,
                                                      avg_lowercase3,
                                                      avg_numbers3, avg_non_alphanumeric3, avg_punctuation3, avg_space3,
                                                      avg_misspellings3)

        pos_content = build_pos_content(POS_df, POS_df2, classifications, ds_record_count3,
                                        avg_nouns3, avg_verbs3, avg_adverbs3, avg_adjectives3)

        whole_ds_feature_analysis_content = build_whole_ds_feature_analysis_content(whole_ds_feature_analysis)

        feature_analysis_content = build_feature_analysis_content(class_feature_analysis, labels)

        false_feature_analysis_content = build_misclassified_feature_analysis_content(false_feature_analysis, labels)
        misclassified_feature_analysis_content = build_misclassified_feature_analysis_content(
            misclassified_feature_analysis, labels)

        """


        fp_feature_analysis_content = build_misclassified_features_analysis_content("FP", fp_tokens_not_in_tn,
                                                                                    fp_tokens_not_in_tn_count,
                                                                                    common_tokens_tn_fp,
                                                                                    common_tokens_tn_fp_count,
                                                                                    tn_tokens_not_in_fp,
                                                                                    tn_tokens_not_in_fp_count,
                                                                                    fp_tokens_not_in_tp,
                                                                                    fp_tokens_not_in_tp_count,
                                                                                    common_tokens_tp_fp,
                                                                                    common_tokens_tp_fp_count,
                                                                                    tp_tokens_not_in_fp,
                                                                                    tp_tokens_not_in_fp_count,
                                                                                    fp_tokens_not_in_zero,
                                                                                    fp_tokens_not_in_zero_count,
                                                                                    common_tokens_zero_fp,
                                                                                    common_tokens_zero_fp_count,
                                                                                    zero_tokens_not_in_fp,
                                                                                    zero_tokens_not_in_fp_count,
                                                                                    fp_tokens_not_in_one,
                                                                                    fp_tokens_not_in_one_count,
                                                                                    common_tokens_one_fp,
                                                                                    common_tokens_one_fp_count,
                                                                                    one_tokens_not_in_fp,
                                                                                    one_tokens_not_in_fp_count,
                                                                                    fp_tokens_not_in_all,
                                                                                    fp_tokens_not_in_all_count,
                                                                                    common_tokens_all_fp,
                                                                                    common_tokens_all_fp_count,
                                                                                    all_tokens_not_in_fp,
                                                                                    all_tokens_not_in_fp_count)

        classification_features_content = build_classification_features_analysis_content()

        class_features_content = build_class_features_analysis_content(class0_tokens_not_in_class1,
                                                                       class0_tokens_not_in_class1_count,
                                                                       common_tokens_class0_class1,
                                                                       common_tokens_class0_class1_count,
                                                                       class1_tokens_not_in_class0,
                                                                       class1_tokens_not_in_class0_count)
        """
        # ******************************************************************************************
        # Building webpages
        # ******************************************************************************************

        directory = create_directory()
        create_page(directory, "index", path, index_content)
        create_page(directory, "feature_statistics", path, statistics_content)
        create_page(directory, "pos_analysis", path, pos_content)
        create_page(directory, "feature_insight", path, whole_ds_feature_analysis_content)
        create_page(directory, "feature_analysis", path, feature_analysis_content)
        create_page(directory, "false_feature_analysis", path, false_feature_analysis_content)
        create_page(directory, "misclassified_feature_analysis", path, misclassified_feature_analysis_content)
        move_images(directory)
        copy_css(directory)
        copy_js_folder(directory)
        open_website(directory)

