from matplotlib import pyplot as plt
from wordcloud import (WordCloud, get_single_color_func)
import numpy as np
import pandas as pd
import os
from PIL import Image
from PIL import ImageFont, ImageDraw
#from .analysis_content import build_token_count, build_token_count_by_class


# Function to add jitter to scatter plot points
def add_jitter(val):
    stdev = .02*(max(val)-min(val))
    jitter = val + np.random.randn(len(val)) * stdev
    return jitter

# Function to plot scatter plots
def build_scatter_plot(df,x_axis,y_axis):
    scatter_x = df[x_axis]
    scatter_y = df[y_axis]
    group = df['classifications']
    cdict = {'C - TN': 'green', 'C - TP': 'blue','FN': 'yellow', 'FP': 'red'}
    markers = {"C - TN": "+", "C - TP": "x", "FP": "o", "FN": "s"}
    fig, ax = plt.subplots()
    for g in np.unique(group):
        ix = (group == g)
        ax.scatter(add_jitter(scatter_x[ix]), add_jitter(scatter_y[ix]), c=cdict[g], marker=markers[g],label=g, alpha=0.7)
        x_label = x_axis.capitalize() + " - count per record"
        y_label = y_axis.capitalize() + " - count per record"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    ax.legend()
    file_name = "images/" + x_axis + "_"+ y_axis + ".png"
    plt.savefig(os.path.join(file_name), pad_inches = 0, bbox_inches = 'tight')
    plt.close()

# Function create word cloud from term frequencies
def build_word_cloud(df, filename, size, max_words):

    # Convert dataframes to dictionarys for word clouds
    if 'count_1' in df.columns:
        terms = dict(zip(df['token_name'], df['count_1']))
    else:
        terms = dict(zip(df['token_name'], df['count']))

    if terms:
        wordcloud = WordCloud(width=size, height=size, stopwords='False', max_words=max_words).generate_from_frequencies(terms)
        plt.figure(figsize=(20, 20))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        file_name = "images/" + filename + "_wordcloud.png"
        wordcloud.to_file(os.path.join(file_name))
        plt.close()
    else:
        draw_no_tokens_img(filename)

class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)

# Function create word cloud from term frequencies
def build_word_cloud2(df, filename, size, max_words):

    default_colour = 'grey'
    colour_of_word = {
        'red': [],
        'blue': [],
        'green': [],
        'yellow': []
    }

    for index, row in df.iterrows():
        if row['pos'] == 'Noun':
            colour_of_word['red'].append(row['token_name'])
        elif row['pos'] == 'Verb':
            colour_of_word['blue'].append(row['token_name'])
        elif row['pos'] == 'Adv.':
            colour_of_word['green'].append(row['token_name'])
        elif row['pos'] == 'Adj.':
            colour_of_word['yellow'].append(row['token_name'])

    # Convert dataframes to dictionarys for word clouds
    if 'count_1' in df.columns:
        terms = dict(zip(df['token_name'], df['count_1']))
    else:
        terms = dict(zip(df['token_name'], df['count']))

    if terms:
        wordcloud = WordCloud(width=size, height=size, stopwords='False', max_words=max_words, collocations=False).generate_from_frequencies(terms)

        # Create a color function with multiple tones
        grouped_color_func = SimpleGroupedColorFunc(colour_of_word, default_colour)

        # Apply our color function
        wordcloud.recolor(color_func=grouped_color_func)

        plt.figure(figsize=(20, 20))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        file_name = "images/" + str(filename) + "_wordcloud.png"
        wordcloud.to_file(os.path.join(file_name))
        plt.close()
    else:
        draw_no_tokens_img(filename)

# Function to create image if no terms are in list
def draw_no_tokens_img(filename):
    img = Image.new('RGB', (300, 300))
    draw = ImageDraw.Draw(img)
    font_path = "data/arial.ttf"
    font = ImageFont.truetype(font_path, 35)
    draw.text((50, 120), "NO TOKENS", (32, 147, 53), font=font)
    file_name = "images/" + filename + "_wordcloud.png"
    img.save(file_name)
    plt.close()

# Code for bar chart found on https://buhrmann.github.io/tfidf-analysis.html
# Function for plotting multiple bar charts
def build_top_tfidf_tokens_bar_chart(dfs, filename):
    fig = plt.figure(figsize=(11, 9))
    ypos = np.arange(len(dfs[0]))
    for key, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), key + 1)
        ax.barh(ypos, df.tfidf, color='darkslategray', align='center')
        ax.set_title(str(df.label))
        ax.set_yticks(ypos)
        ax.set_yticklabels(df.token)
        ax.invert_yaxis()
        ax.set_frame_on(False)
        ax.set_xlabel("Average TF-IDF Values")
        ax.set_ylim([-1, ypos[-1] + 1])
        plt.subplots_adjust(wspace=1)
    file_name = "images/tfidf_" + filename + "_barchart.png"
    plt.savefig(os.path.join(file_name), pad_inches=0, bbox_inches='tight')
    plt.close()

# Code for bar chart found on https://buhrmann.github.io/tfidf-analysis.html
# Function for plotting multiple bar charts

def build_single_top_tfidf_tokens_bar_chart(df, filename):
    fig = plt.figure(figsize=(4, 9))
    ypos = np.arange(len(df))
    #for key, df in enumerate(dfs):

    ax = plt.subplot()
    ax.barh(ypos, df.tfidf, color='darkslategray', align='center')
    ax.set_title(str(df.label))
    ax.set_yticks(ypos)
    ax.set_yticklabels(df.token)
    ax.invert_yaxis()
    ax.set_frame_on(False)
    ax.set_xlabel("Average TF-IDF Values")
    ax.set_ylim([-1, ypos[-1] + 1])
    plt.subplots_adjust(wspace=1)
    file_name = "images/tfidf_" + filename + "_barchart.png"
    plt.savefig(os.path.join(file_name), pad_inches=0, bbox_inches='tight')
    plt.close()

