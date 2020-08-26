import os
import shutil
import datetime
from webbrowser import open_new_tab

# function to create folder
def create_directory():
    # find date - time
    date_time = datetime.datetime.today().strftime("%d%m%y-%H%M")
    # creates folder path
    _dir = os.path.join(date_time)
    # create directory, if it does not exist
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir


# function to build each HTML page
def create_page(directory, pg_name, datafile,main_content):
    filename = directory + '/' + pg_name + '.html'
    f = open(filename, 'w', encoding='utf-8')
    wrapper = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Text Misclassification Analysis</title>
    <style>
        @import 'css/all.css';
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script> 
    </head>
    <body>
	<!-- header -->
	<header>
	  	<section class="header"><a href="index.html"><h1><i class="fa fa-cog"></i>  Text Misclassification Analysis</h1></a></section>
	</header>
	 <!-- main content section of website - divided into 2 sections -->
     <main>
     <!-- section 1 - sidebar -->
         <section class="side_nav">
            <ul>
              <li>
			   	<a href="index.html"><h2><i class="fa fa-cog"></i> Home</h2></a>
			  </li>
			  <li>
			  	<a href="feature_statistics.html"><h2><i class="fa fa-cogs"></i> Feature Statistics</h2></a>
			  </li>
			  <li>
			  	<a href="pos_analysis.html"><h2><i class="fa fa-comment"></i> POS Analysis</h2></a>
			  </li>
			  <li>
			  	<a href="feature_insight.html"><h2><i class="fa fa-comment"></i> Feature Insight</h2></a>
			  </li>
			  <li>
			   	<a href="feature_analysis.html"><h2><i class="fa fa-table"></i> Class Feature Analysis</h2></a>
			  </li>
			   <li>
			   	<a href="false_feature_analysis.html"><h2><i class="fa fa-table"></i> False n Feature Analysis</h2></a>
			  </li>
			  <li>
			   	<a href="misclassified_feature_analysis.html"><h2><i class="fa fa-table"></i> Misclassified n Feature Analysis</h2></a>
			  </li>
			  <li>
			   	<a href="tfidf_summary.html"><h2><i class="fa fa-bar-chart"></i> TFIDF Insights</h2></a>
			  </li>
			  <li>
			   	<a href="misclassified_features.html"><h2><i class="fa fa-list-ul"></i> FN & FP Tokens </h2></a>
			  </li>
			  <li>
			   	<a href="classification_analysis.html"><h2><i class="fa fa-cloud"></i> Tokens per Classification</h2></a>
			  </li>
			  <li>
			   	<a href="class_label_analysis.html"><h2><i class="fa fa-columns"></i> Tokens per Class Label</h2></a>
			  </li>
	      </ul>
         </section>
     <!-- section 2 - main content -->
         <section class="wide"> 
              <table class="title">
      <tr>
        <th class="title">Dataset:    %s</th>
      </tr>
      <tr>
        <td class="title">%s</td>
      </tr>
      </table>  
		 </section>
	 </main>
 <script src="js/scripts.js"></script> 
     </body>   
     </html>"""

    whole = wrapper % (datafile, main_content)
    f.write(whole, )
    f.close()

def create_image_directory():
    # create image folder if it does not already exist
    filename = 'images'
    if not os.path.exists(filename):
        os.makedirs(filename)

def create_data_directory():
    # create image folder if it does not already exist
    filename = 'data'
    if not os.path.exists(filename):
        os.makedirs(filename)

# function to move images into
def move_images(directory):
    source = 'images/'
    curr_dir = os.path.dirname(__file__)
    dest1 = directory + '/images'
    os.mkdir(dest1)
    files = os.listdir(source)
    for f in files:
        shutil.move(source + f, dest1)

# function to create css folder and copy css file into dynamically created folder
def copy_css(directory):
    curr_dir = os.path.dirname(__file__)
    source = curr_dir + '\css\\'
    dest1 = directory + '\css'
    os.mkdir(dest1)

    # create css folder
    filename = 'css'
    if not os.path.exists(dest1):
        os.makedirs(filename)
    # copy css file
    files = os.listdir(source)
    for f in files:
        shutil.copy(source + f, dest1)

# function to create css folder and copy css file into dynamically created folder
def copy_js_folder(directory):
    curr_dir = os.path.dirname(__file__)
    source = curr_dir + '\js\\'
    dest1 = directory + '\js'
    os.mkdir(dest1)

    # create css folder
    filename = 'js'
    if not os.path.exists(dest1):
        os.makedirs(filename)
    # copy css file
    files = os.listdir(source)
    for f in files:
        shutil.copy(source + f, dest1)

# Function to open index page in browser window
def open_website(directory):
    cwd = os.getcwd()
    filename = "%s\%s\index.html"%(cwd, directory)
    open_new_tab(filename)
