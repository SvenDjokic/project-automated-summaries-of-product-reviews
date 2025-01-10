# project-automated-summaries-of-product-reviews

**Introduction**

This project provides a comprehensive solution for analyzing customer reviews to enhance product and service offerings. It encompasses sentiment analysis, product category clustering, and generative AI for summarizing reviews into insightful articles.

**Features**

1) Sentiment Analysis: Classifies customer reviews into positive or negative using the DistilBERT model.

Preprocessing includes:
- Converted text to lowercase
- Removed special characters and stopwords
- Tokenized text and performed lemmatization using NLTK’s WordNetLemmatizer
- Applied TF-IDF Vectorization to transform text into numerical data

2) Product Category Clustering: Utilizes K-Means clustering to group over 100 product categories into 8 meta-categories, optimizing the Elbow Method to determine the optimal number of clusters.

Preprocessing includes:
- Converted text to lowercase
- Removed special characters and stopwords
- Tokenized text and performed lemmatization using NLTK’s WordNetLemmatizer
- Applied TF-IDF Vectorization to transform text into numerical data

3) Review Summarization: Employs generative AI to create articles recommending top products for each category.

**Dependencies**

This project utilizes several Python libraries for data manipulation, analysis, natural language processing, machine learning, and visualization. Below is a summary of the key libraries and their roles:

- pandas: Provides data structures and functions needed to manipulate structured data seamlessly.
- numpy: Offers support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
- re: Enables operations involving regular expressions, facilitating pattern matching and text processing tasks.
- os: Supplies a way of using operating system-dependent functionality, such as reading or writing to the file system.
- nltk: The Natural Language Toolkit is employed for tasks in natural language processing, including tokenization, stopword removal, and lemmatization.
- word_tokenize: Splits text into individual words or tokens.
- stopwords: Provides a list of common words (e.g., ‘and’, ‘the’) that can be excluded from text analysis.
- wordnet: A lexical database for the English language, useful for finding synonyms, antonyms, and understanding word relationships.
- WordNetLemmatizer: Reduces words to their base or root form (e.g., ‘running’ to ‘run’).
- sklearn: A machine learning library featuring various tools for data analysis and modeling.
- TfidfVectorizer: Converts a collection of raw documents into a matrix of TF-IDF features, reflecting the importance of words in the documents.
- ENGLISH_STOP_WORDS: A predefined list of common English stopwords provided by scikit-learn.
- classification_report: Generates a text report showing the main classification metrics.
- confusion_matrix: Computes a confusion matrix to evaluate the accuracy of a classification.
- KMeans: Implements the K-Means clustering algorithm for partitioning data into clusters.
- transformers: A library by Hugging Face that provides general-purpose architectures for natural language understanding and generation.
- AutoModelForCausalLM: Automatically selects and initializes a model for causal language modeling.
- AutoTokenizer: Automatically selects and initializes a tokenizer corresponding to a chosen model.
- pipeline: Eases the creation of end-to-end inference pipelines for various tasks like text generation and classification.
- seaborn: A data visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
- matplotlib: A comprehensive library for creating static, animated, and interactive visualizations in Python.