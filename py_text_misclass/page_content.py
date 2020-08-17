import requests
from urllib import parse
import os

def build_index_content(model_desc, time_elapsed, recordCount, label_count, no_of_all_tokens,no_of_tokens,null_accuracy,accuracy, misclassification_rate,
                        confusion_matrix, classification_report, f1_score, misclassifications):
    misclassifications_table = build_misclassifications_list(misclassifications)
    confusion_matrix_table = build_confusion_matrix(confusion_matrix, classification_report, accuracy)
    table = """
 <table class="maincontenttb">
     <tr class="maincontenttb">
      <th class="maincontenttb">
        <a id="top"></a>Text Classification Performance
      </th>
      </tr> 
       <tr class="maincontenttb">
       <td class="maincontenttb">  
       <table>
           <tr>
              <td colspan=3><p> <b> Model Description: </b> %s</p></td>
              <td><p> <b> Time to run processes (excluding building pages): </b> %s</p></td>
           </tr>
       </table>
      </td>     
      </tr>   
      <tr>
      <td>      
      <table>
      <tr>
      <td>      
      <table class="table_alternate_colour">
      <tr>
        <td><p><b>Number of records:</b> %s</p></td>
        <td><p><b>Number of unique tokens in data set:</b> %s</p></td>
        <td><p><b>Number of unique tokens after pre-processing:</b> %s</p></td>
      </tr>
      <tr>
        <td colspan=3><p><b>Class Counts: </b>%s</p></td>
      </tr>
      <tr>
        <td><p><b>Null Accuracy:</b> %s</p></td>
        <td><p><b>Classification Accuracy:</b> %s</p></td>
        <td><p><b>Misclassification Rate / Classification Error: </b>%s</p></td>
      </tr>
      <tr>
      <td colspan=3><p><b>F1 Score: </b> %s</p></td>
      </tr>
      </table>     
      </td>     
      </tr>
      <tr>
      <td>
         %s   
      </td>
      </tr>
      <tr>
      <td>
         %s   
      </td>
      </tr>
    </table> 
     </td>     
      </tr> 
    </table> 
    <br>
<br/>   
      """%(model_desc, time_elapsed, recordCount, no_of_all_tokens, no_of_tokens, label_count, null_accuracy, accuracy, misclassification_rate, f1_score, confusion_matrix_table, misclassifications_table)

    content = table
    return content


def build_statistics_content(feature_statistics_df, feature_statistics_df2, classifications, ds_record_count3, avg_sentences3,
                                                      avg_words3, avg_unique_tokens3, avg_chars3, avg_uppercase3, avg_lowercase3,
                                                      avg_numbers3, avg_non_alphanumeric3, avg_punctuation3, avg_space3, avg_misspellings3):
    feature_statistics_table = build_feature_statistics_table1(feature_statistics_df)
    feature_statistics_table2 = build_feature_statistics_table1(feature_statistics_df2)
    feature_statistics_table3 = build_feature_statistics_table2(classifications, ds_record_count3, avg_words3, avg_unique_tokens3)
    feature_statistics_table4 = build_feature_statistics_table2(classifications, avg_chars3, avg_uppercase3, avg_lowercase3)
    feature_statistics_table5 = build_feature_statistics_table2(classifications, avg_numbers3, avg_non_alphanumeric3, avg_punctuation3)
    feature_statistics_table6 = build_feature_statistics_table2(classifications, avg_sentences3, avg_space3, avg_misspellings3)
    table1 = """
     <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        Feature Statistics 
      </th>
      </tr>
    <tr>
      <td> %s</td>
    </tr>
    <tr>
      <td> %s</td>
    </tr>
    <tr>
      <td> TITLE ??</td>
    </tr>
    <tr>
      <td>%s</td>
    </tr>
    <tr>
      <td>%s</td>
    </tr>
    <tr>
      <td>%s</td>
    </tr>
    <tr>
      <td>%s</td>
    </tr>
      <th class="maincontenttb">
        Feature Statistic Scatter plots
      </th>
    </tr>
    <tr>
     <td>
      <table>
    <tr>
    <td><img src="images/words_misspellings.png" alt="Words vs Misspellings"></td>
    </tr>
    <tr>
    <td><img src="images/nouns_verbs.png" alt="Nouns vs Verbs"></td>
    </tr>
    <tr>
    <td><img src="images/adjectives_adverbs.png" alt="Adjectives vs Adverbs"></td>
    </tr>
    </table>  
     </td>
    </tr>
    </table>    

    """ % (feature_statistics_table, feature_statistics_table2, feature_statistics_table3, feature_statistics_table4,
           feature_statistics_table5,feature_statistics_table6)
    content = table1
    return content

def build_pos_content(POS_df, POS_df2, classifications, ds_record_count3,
                                               avg_nouns3, avg_verbs3, avg_adverbs3, avg_adjectives3):
    pos_analysis_table = build_feature_statistics_table1(POS_df)
    pos_analysis_table2 = build_feature_statistics_table1(POS_df2)
    pos_analysis_table3 = build_feature_statistics_table2(classifications, ds_record_count3, avg_nouns3, avg_verbs3)
    pos_analysis_table4 = build_feature_statistics_table2(classifications, ds_record_count3, avg_adverbs3, avg_adjectives3)
    table1 = """
     <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        Part of Speech Statistics & Analysis
      </th>
      </tr>
    <tr>
      <td> %s</td>
    </tr>
    <tr>
      <td> %s</td>
    </tr>
    <tr>
      <td> TITLE ??</td>
    </tr>
    <tr>
      <td>%s</td>
    </tr>
    <tr>
      <td>%s</td>
    </tr>
    </table>    

    """ % (pos_analysis_table, pos_analysis_table2, pos_analysis_table3, pos_analysis_table4)
    content = table1
    return content


def build_whole_ds_feature_analysis_content(whole_ds_feature_analysis):

    table1 = """
   <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        Analysis of tokens - Tokens before and after pre-processing
      </th>
      </tr>
      <tr class="maincontenttb">
      <td class="maincontenttb">
      <p>
        </br>The tokens below are all of the tokens contained in the data set. The <span class="font2">GREEN</span> tokens are those used in the classification model and the <span class="font3">BLUE</span> tokens are those removed by pre-processing.</br></br>
        Hover over each token for a class breakdown.
      </p>
      </td>
      </tr>
      
      
      <tr>
      <td>
         <table class="features">
        <tr class="features">
          <td class="features"><h2>%s</h2></td>
      </tr>
      </table>
      </td>
  </tr>
      </table>
    """%(whole_ds_feature_analysis)
    content = table1
    return content


def build_feature_analysis_content(class_feature_analysis, labels):
    tag1 = """
  <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        <a id="top"></a> Feature Analysis per Class Label
      </th>
      </tr>
      <tr>
        <td>
      <table class="features">
           <tr class="features">
          <td colspan=3 class="features">
           <section class="sub_nav">
           <ul>"""
    #                <a href="#" title="div1">Show DIV 1</a>
    #                    <a href="#" onclick="myFunction('%s'); return false; "><h2>%s</h2></a>
    rows = ""
    for index, value in labels.items():
        value = str(value)
        rows += """
               <li>
               <a href="#" title="%s"><h2>%s</h2></a>
               </li>""" % (value, value)

        tag2 = """
             </ul>
           </section>       
           </td>
      </tr>
    </table> 
       </td>
      </tr>

       <tr>
      <td>
     <table class="features">
       <tr>
        <td>
           %s
        </td>
      </tr>
    </table> 

    </td>
      </tr>
      </table>  
    """ % (class_feature_analysis)

    content = tag1 + rows + tag2
    return content


def build_misclassified_feature_analysis_content(misclassified_feature_analysis, labels):
    tag1 = """
  <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        Feature Analysis 
      </th>
      </tr>
      
      <tr>
        <td>
      <table class="features">
           <tr class="features">
          <td colspan=3 class="features">
           <section class="sub_nav">
           <ul>"""
    rows = ""
    for index, value in labels.items():
        value = str(value)
        rows += """
               <li>
               <a href="#" title="%s"><h2>%s</h2></a>
               </li>""" % (value, value)
        tag2 = """
             </ul>
           </section>       
           </td>
      </tr>
    </table> 
       </td>
      </tr>

       <tr>
      <td>
     <table class="features">
       <tr>
        <td>
           %s
        </td>
      </tr>
    </table> 
    
    </td>
      </tr>
      </table>  
    """% (misclassified_feature_analysis)

    content = tag1 + rows + tag2
    return content

def build_confusion_matrix(df, df2, accuracy):

    number = df.shape[0]
    number = number + 2

    precisions = round(df2['precision'],3)
    recalls = round(df2['recall'],3)

    tag1 = """
      <table class="bordertable">  
       <tr class="bordertable">
       <th class="bordertable" colspan=%d><h2>Confusion Matrix</h2></th>
      </tr>
       <tr class="bordertable">    
       <td class="bordertable"></td>
       <td class="bordertable" colspan=%d><h3>PREDICTED</h3></td>
      </tr>
    """% (df.shape[0]+3, number)
    tag2 = """<tr>
                 <td class="bordertable" rowspan=%d><div class="vertical-text"><h3>ACTUAL</h3></div></td>
                 <td class="bordertable"></td>""" %(number)

    for col in df.columns:
        tag2 += """<td class="bordertable"><h2>%s </h2></td>"""%(col)
    tag2 += """<td class="bordertable"><h3>RECALL</h3></td>
      </tr>"""
    rows = ""
    for index, row in df.iterrows():
        rows += """<tr class="bordertable">"""
        rows += """<td class="bordertable"><h2>%s </h2></td>""" % (index)
        print("index")
        print(index)
        print(type(index))
        recall = recalls.loc[str(index)]

        for index2, column in row.iteritems():
            if index == index2:
                bgcolor = "bordertable1"
            else:
                bgcolor = "bordertable2"
            rows += """<td class="%s"><a href="#fp"><p>%s </p></a></td>"""%(bgcolor, column)
        rows += """<td class="bordertable"><p>%s </p></td>""" %(str(recall))
        rows += """</tr>"""

    tag3 = ""
    tag3 += """<tr>
              <td class="bordertable"><h3>PRECISION</h3></td>"""
    for index, precision in precisions.items():
        tag3 +="""<td class="bordertable"><p>%s </p></td>""" % (str(precision))
    tag3 += """ <td class="bordertable"><b>Accuracy</b> <br>%s</td>
               </tr>
              </table>"""%(accuracy)
    table = tag1 + tag2 + rows + tag3
    return table

def build_misclassifications_list(misclassifications):
    tag1 = """<br><table class="table_alternate_colour">
    <tr>
         <th colspan=3><h2>MISCLASSIFIED INSTANCES </h2></th>
    </tr>"""
    rows = ""
    temp_label = ""

    for index, row in misclassifications.iterrows():
        text = row['text']
        true_label = row['label']
        pred_label = row['pred_label']
        pred_label_upper = str(pred_label).upper()
        if temp_label != pred_label:
            rows += """<tr>
                           <td colspan=3><h2> FALSE %s </h2>  <a href="#top" class="anchor">Back to top</a></td>
                        </tr>""" % (pred_label_upper)
        rows +="""<tr>
                        <td>Label %s</td>
                        <td>False %s</td>
                        <td>%s</td>
                    </tr>""" % (true_label, pred_label, text)
        temp_label = pred_label
    tag3 = """</table>"""
    table = tag1 + rows + tag3
    return table

def build_feature_statistics_table1(df):
    no_of_columns = len(df.columns) + 1
    tag1 = """<br><table class="table_alternate_colour">
          <tr>
        <th colspan="%d">The table contains the average per instance for various instance groups. Full text of instances before pre-processing used.</th>
      </tr> """%(no_of_columns)
    tag2 = """<tr>
           <th></th>"""
    for col in df.columns:
        tag2 += """<th><h2>%s </h2></th>"""%(col.capitalize())
    tag2 += """</tr>"""
    rows = ""
    for index, row in df.iterrows():
    #for row in df.itertuples():
        rows += """<tr>"""
        rows += """<td><b>%s </b></td>""" %(index)#row.Index
        for index2, column in row.items():# iteritems
            rows += """<td><p>%s </p></td>"""%(column)
        rows += """</tr>"""
    tag3 = """</table>"""
    table = tag1 + tag2 + rows + tag3
    return table


def build_feature_statistics_table2(df, df2, df3, df4):

    tag1 = """<br><br><table class="features_table">
                <tr>
                    <td>Legion: </td> 
                    <td><div class="font1">%s</div></td> 
                    <td><div class="font2">%s</div></td> 
                    <td><div class="font3">%s</div></td> 
                    <td><div class="font4">%s</div></td> 
                </tr>
              </table>
              <br>
           <table class="features_table">
                <tr>
                    <td></td>"""%(df.name.capitalize(),df2.name.capitalize(),df3.name.capitalize(),df4.name.capitalize())
    tag2 = ""
    for col in df.columns:
        tag2 += """<td><h2>%s </h2></td>"""%(col)
    tag2 += """</tr>"""
    rows = ""
    #https: // stackoverflow.com / questions / 24709557 / pandas - how - could - i - iterate - two - dataframes - which - have - exactly - same - format
    #for i in range(0, len(df_one.index)):
    #    for j in range(0, len(df_one.columns)):
    #        print(df_one.values[i, j], df_two.values[i, j], i, j)

    for i in range(0, len(df.index)):
        rows += """<tr>"""
        rows += """<td><h2>%s</h2></td>"""%(df.index[i])
        for j in range(0, len(df.columns)):
            rows += """<td>
            <table class="features_table2">
                <tr>
                    <td><div class="font1"> %s </div></td> 
                    <td><div class="font2"> %s </div></td> 
                </tr>
                <tr>
                    <td><div class="font3"> %s </div></td> 
                    <td><div class="font4"> %s </div></td> 
                </tr>
            </table>
            </td>"""%(df.values[i, j], df2.values[i, j], df3.values[i, j], df4.values[i, j])
        rows += """</tr>"""
    tag3 = """</tr>
              </table>"""
    table = tag1 + tag2 + rows + tag3
    return table


def build_feature_statistics_table2ONE_DF(df):

    tag1 = """<table class="table_alternate_colour">
                <tr>
                    <td></td>"""
    tag2 = ""
    for col in df.columns:
        tag2 += """<td><h2>%s </h2></td>"""%(col)
    tag2 += """</tr>"""
    rows = ""
    for index, row in df.iterrows():
        rows += """<tr>"""
        rows += """<td>%s</td>""" %(index)
        for index2, column in row.iteritems():
            rows += """<td><p>%s </p></td>"""%(column)
        rows += """</tr>"""

    tag3 = """</tr>
              </table>"""
    table = tag1 + tag2 + rows + tag3
    return table


def build_tfidf_summary_content():
    table1 = """
      <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        <a id="top"></a> TFIDF Values for Parts of Speech
      </th>
      </tr>
      <tr>
      <td>
     <table>
      <tr>
        <td>  TFIDF Values for <a href="#nouns">Nouns</a> | <a href="#verbs">Verbs</a> | <a href="#adjectives">Adjectives</a> | <a href="#adverbs">Adverbs</a> | </td>
      </tr> 
     <tr>
        <td><h4>  Top 50 TF-IDF tokens for each classification group - All Parts of Speech</h4></td>
      </tr> 
      <tr>
        <td><img src="images/tfidf_all_barchart.png" alt="TFIDF"></td>
      </tr>
      <tr>
        <td><a id="nouns"></a> <h4>Top 50 TF-IDF NOUN tokens for each classification group </h4> <a href="#top" class="anchor">Back to top</a></td>
      </tr>
      <tr>
        <td><img src="images/tfidf_nouns_barchart.png" alt="Nouns"></td>
      </tr>
       <tr>
        <td><a id="verbs"></a> <h4>Top 50 TF-IDF VERB tokens for each classification group </h4><a href="#top" class="anchor">Back to top</a></td>
      </tr>
      <tr>
        <td><img src="images/tfidf_verbs_barchart.png" alt="Verbs"></td>
      </tr>
      <tr>
        <td><a id="adjectives"></a> <h4>Top 50 TF-IDF ADJECTIVE tokens for each classification group </h4><a href="#top" class="anchor">Back to top</a></td>
      </tr>
      <tr>
        <td><img src="images/tfidf_adjs_barchart.png" alt="Adjectives"></td>
      </tr>
      <tr>
        <td><a id="adverbs"></a> <h4>Top 50 TF-IDF ADVERB tokens for each classification group </h4><a href="#top" class="anchor">Back to top</a></td>
      </tr>
      <tr>
        <td><img src="images/tfidf_advs_barchart.png" alt="Adverbs"></td>
      </tr>
      </table>
            </td>
       </tr>
      </table>  
      <br/>   
    """
    content = table1
    return content

def build_misclassified_features_content(fn_tokens_top_50,fn_top_mean_feats,fp_tokens_top_50,fp_top_mean_feats):

    fn_tokens_top_50 = build_content_to_list_string(fn_tokens_top_50,'token_name','count')
    fn_top_mean_feats = build_content_to_list_string(fn_top_mean_feats,'token','tfidf')
    fp_tokens_top_50 = build_content_to_list_string(fp_tokens_top_50,'token_name','count')
    fp_top_mean_feats = build_content_to_list_string(fp_top_mean_feats,'token','tfidf')

    table1 = """
   <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        Significant Tokens within FN and FP instances 
      </th>
      </tr>
      <tr>
      <td>
         <table class="features">
        <tr class="features">
          <td colspan=2 class="features"><h2>False Negatives</h2></td>
          <td colspan=2 class="features"><h2>False Positives</h2></td>
      </tr>
      <tr class="features">
          <td class="features"><h2>50 Top Word Count</h2></td>
          <td class="features"><h2>50 Top TFIDF tokens</h2></td>
          <td class="features"><h2>50 Top Word Count</h2></td>
          <td class="features"><h2>50 Top TFIDF tokens</h2></td>
      </tr>
      <tr class="features">
          <td class="features3">%s</td>
          <td class="features3">%s</td>
          <td class="features2">%s</td>
          <td class="features2">%s</td>
      </tr>
      </table>
      </td>
  </tr>
      </table>
    """%(fn_tokens_top_50, fn_top_mean_feats,fp_tokens_top_50,fp_top_mean_feats)
    content = table1
    return content

def build_classification_features_analysis_content():
    table1 = """
  <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        Tokens per Classification  
      </th>
      </tr>
      <tr>
      <td>
     <table>
      <tr>
        <td><h4>TRUE NEGATIVES</h4><img src="images/true_negatives_wordcloud.png" alt="True Negatives"></td><td><h4>FALSE POSITIVES</h4><img src="images/false_positives_wordcloud.png" alt="False Positives"></td>
      </tr>
      <tr>
        <td><h4>FALSE NEGATIVES</h4><img src="images/false_negatives_wordcloud.png" alt="False Negatives"></td><td><h4>TRUE POSITIVES</h4><img src="images/true_positives_wordcloud.png" alt="True Positives"></td>
      </tr>
      </table>  
      </td>
       </tr>
      </table>  
     <br/>   
    """
    content = table1
    return content

def build_class_features_analysis_content(class0_tokens_not_in_class1,class0_tokens_not_in_class1_count, common_tokens_class0_class1,common_tokens_class0_class1_count, class1_tokens_not_in_class0,class1_tokens_not_in_class0_count):

    class0_tokens_not_in_class1 = build_content_dataframe_to_string(class0_tokens_not_in_class1)
    common_tokens_class0_class1 = build_content_dataframe_to_string2(common_tokens_class0_class1)
    class1_tokens_not_in_class0 = build_content_dataframe_to_string(class1_tokens_not_in_class0)

    table1 = """
  <table class="maincontenttb">
      <tr class="maincontenttb">
      <th class="maincontenttb">
        Tokens per Class Label: Class 0 & Class 1 - Intersect & Union
      </th>
      </tr>
       <tr>
      <td>
     <table class="features">
      <tr class="features">        
        <td class="features"><a id="tn"></a><h2>Tokens unique to Class 0 </h2></td>
        <td class="features"><h2>Tokens in both Class 0 and Class 1</h2></td>
        <td class="features"><h2> Tokens unique Class 1 </h2></td>
      </tr>
      <tr class="features">
        <td class="features1"><img src="images/class0_tokens_not_in_class1_wordcloud.png"><p>%s <b> (# of tokens: %s)</b></p> </td>
        <td class="features2"><img src="images/common_tokens_class0_class1_wordcloud.png"><p>%s <b> (# of tokens: %s)</b></p></td>
        <td class="features3"><img src="images/class1_tokens_not_in_class0_wordcloud.png"><p>%s <b> (# of tokens: %s)</b></p></td>
      </tr>
     </table> 
    </td>
      </tr>
      </table>  
    """%(class0_tokens_not_in_class1,class0_tokens_not_in_class1_count, common_tokens_class0_class1,common_tokens_class0_class1_count, class1_tokens_not_in_class0,class1_tokens_not_in_class0_count)
    content = table1
    return content

def build_content_dataframe_series(title, df):
    tag1 = """<br><table class="strippedtb">
             <tr class="strippedtb">
               <th class="strippedtb"><h2> %s</h2>  <a href="#top" class="anchor">Back to top</a></th>
             </tr>
    """% (title)
    rows = []
    for i, x in df.iteritems():
        rows.append("""
           <tr class="strippedtb"`>
               <td class="strippedtb">%s</td>
           </tr>"""%(x))
    result = ''.join(rows)
    tag2 = """</table><br>"""
    table = tag1 + result + tag2
    return table

def build_content_dataframe_to_string(df):
     rows = []
     for index, row in df.iterrows():
         token = row['token_name']
         count = row['count']
         count = str(count)
         rows.append("""%s(%s) """%(token,count))
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


def build_content_to_list_string(df,feature1,feature2):
    feature1 = feature1
    feature2 = feature2
    rows = []
    for index, row in df.iterrows():
        token = row[feature1]
        count = row[feature2]
        count = str(count)
        rows.append("""%s(%s) </br>""" % (token, count))
    result = ''.join(rows)
    return result

