# Gen AI Recipe Recommendation System

## Description

This project implements a recipe recommendation system. 

## What it does? 

Given a list of ingredients provided by the user, the system suggests relevant recipes. It leverages Natural Language Processing (NLP) techniques for text cleaning and feature extraction, Topic Modeling (LDA) to understand recipe themes, and TF-IDF based similarity scoring to rank recipes matching the user's query. The system is built using Python and various data science libraries.

## Data

* **Source:** The system was initially trained on a dataset of approximately 125,000 recipes sourced from raw JSON files (`recipes_raw_nosource_*.json`). 

The final recommendation model utilizes pre-processed data stored in CSV files (`recipes_tagged_*.csv`, `tokenized_text.csv`)

* **Features:**
    * Original: Title, Ingredients, Instructions, Picture Link.
    * Generated: Cooking Time, Veg/Non-Veg classification, Cuisine Type, Ingredient Count, Cleaned/Combined Text, Topic-based Tags/Keywords.

## Process Overview

The project follows these main steps:

1.  **Data Loading and Cleaning (`Adding Data And Cleaning.ipynb`):**
    * Reads raw recipe data from JSON files.
    * Performs extensive cleaning: drops rows with missing ingredients/instructions, removes overly short recipes, eliminates punctuation, numbers, excess whitespace, and standard English stop words.
    * Enriches the dataset by adding features like estimated cooking time, veg/non-veg labels, and cuisine type using tools like TextBlob and potentially LLMs (Transformers library imported).
    * Tokenizes and lemmatizes text data using libraries like NLTK and SpaCy.

2.  **Modeling and Recommendation (`Recipe_Recommendation_Final.ipynb`):**
    * Loads the pre-processed and tagged recipe data.
    * Applies TF-IDF vectorization to recipe titles, the main combined text (ingredients + instructions), and generated tags/keywords.
    * Uses Latent Dirichlet Allocation (LDA) for topic modeling to discover latent themes within the recipes.
    * Generates relevant keywords and tags for each recipe, potentially using methods involving POS tagging and graph-based ranking (like PageRank) on ingredient co-occurrence networks.
    * Implements the core `Search_Recipes` function which takes a user's ingredient query.
    * Calculates recipe relevance using a weighted cosine similarity score between the TF-IDF vector of the query and the TF-IDF vectors of the recipes. The score is a weighted average across title (40%), text (40%), and tags/categories (20%)
    * Supports ranked ingredient queries, where the order of ingredients in the query influences the weighting.
    * Returns a ranked list of the most similar recipes based on the calculated scores.

## How to Use

1.  **Environment Setup:** Ensure all required dependencies are installed (see Dependencies section).
2.  **Data:** Make sure the pre-processed data files (`recipes_tagged_*.csv`, `tokenized_text.csv`) are accessible, for instance, by mounting Google Drive as shown in the `Recipe_Recommendation_Final.ipynb` notebook.
3.  **Run Notebook:** Execute the cells in the `Recipe_Recommendation_Final.ipynb` notebook.
4.  **Search:** Call the `Search_Recipes` function with your desired ingredients:
    ```python
    # Example Search
    query = ['chicken', 'tomato', 'onion']
    Search_Recipes(query, query_ranked=False, recipe_range=(0, 5)) # Get top 5 recipes

    # Example Ranked Search (order matters)
    ranked_query = ['cinnamon', 'cream', 'banana']
    Search_Recipes(ranked_query, query_ranked=True, recipe_range=(0, 3)) # Get top 3 recipes
    ```
    * `query`: A list of ingredient strings.
    * `query_ranked`: Set to `True` if the order of ingredients should influence the ranking.
    * `recipe_range`: A tuple specifying the range of ranked results to return (e.g., `(0, 5)` for the top 5).

## Dependencies

* Python 3.x
* **Core Libraries:**
    * `pandas`
    * `numpy`
    * `nltk` (requires 'stopwords', 'wordnet' downloads) 
    * `spacy` (requires `en_core_web_sm` model) 
    * `scikit-learn` (for `TfidfVectorizer`, `cosine_similarity`, `LDA`, etc.)
    * `gensim`
    * `networkx` (used for keyword generation) 
* **Other Libraries Used:**
    * `matplotlib`, `seaborn` (for plotting)
    * `tabulate`
    * `google-colab` (if running in Google Colaboratory) 
    * `textblob` (used in data cleaning/feature generation) 
    * `transformers`, `torch`, `tensorflow` (potentially used for LLM-based feature generation in cleaning phase) 
    * Standard libraries: `re`, `string`, `itertools`, `functools`, `multiprocessing`, `os`, `pickle`, `json`, `zipfile`, `time`, `pathlib`

*(Note: The output in `Recipe_Recommendation_Final.ipynb` indicated potential issues installing `en_core_web_md` and `en_core_web_lg` spacy models, but `en_core_web_sm` was successfully loaded and used)*.

