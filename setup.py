from distutils.core import setup

setup(
  name = 'py_text_misclass',
  version = '0.0.1',
  description = 'A text misclassification analysis package.',
  author = 'lauramcloughlin',
  author_email = 'b00092153@student.itb.ie',
  url = 'https://github.com/lauramcloughlin/py_text_misclass',
  download_url = 'https://github.com/lauramcloughlin/py_git_test/archive/1.4.12.tar.gz',
  keywords = ['py_text_misclass', 'misclassification', 'classification', 'text'],
  packages = ['py_text_misclass'],
  classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
  install_requires=[
        'scikit-learn',
        'nltk ',
        'pandas',
        'numpy ',
        'matplotlib ',
        'wordcloud ',
        'pyspellchecker ',
        'pillow'
    ],
  package_data={
        # Include any *.css files found in the 'css' subdirectory
        # of package:
        '': ['css/*.css', 'js/*.js'],
   }
)
