# Yelp Reviews Sentiment Analysis
# Analyzed over 260,000 Yelp tips to identify ideal markets to open a new restaurant based on cuisine.
# Utilized NLTK (VADER) for sentiment analysis, LDA for topic modeling, and geopandas to identify metro areas.
# Dataset available for academic use only.

import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import geopandas as gpd
from shapely.geometry import Point

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load the data
tips_df = pd.read_csv("/Users/satya/Desktop/YELP DATA/Yelp Food TIPS.csv")
tips_df.dropna(subset=['text'], inplace=True)  # Drop rows with missing 'text'

# Create GeoDataFrame for spatial analysis
geometry = [Point(xy) for xy in zip(tips_df['longitude'], tips_df['latitude'])]
geo_df = gpd.GeoDataFrame(tips_df, geometry=geometry)

# Load metro area shapefile
metro_areas = gpd.read_file('/Users/satya/Downloads/cb_2023_us_cbsa_5m/cb_2023_us_cbsa_5m.shp')
merged = gpd.sjoin(geo_df, metro_areas, predicate='intersects')
unique_metro_names = merged['NAMELSAD'].unique()

# Merge metro data
tips_df = tips_df.merge(
    merged[['geometry', 'NAMELSAD']], 
    left_index=True, 
    right_index=True, 
    how='left'
)
tips_df.rename(columns={'NAMELSAD': 'metro'}, inplace=True)
tips_df['metro'].fillna('Unknown Metro Area', inplace=True)

# Exclude certain metro areas
excluded_metros = [
    'Truckee-Grass Valley, CA', 'Reading, PA', 'Trenton-Princeton, NJ',
    'Unknown Metro Area', 'Atlantic City-Hammonton, NJ', 
    'New York-Newark-Jersey City, NY-NJ'
]
tips_df = tips_df[~tips_df['metro'].isin(excluded_metros)]

# Identify top cuisines
specific_cuisines = {'Mexican', 'Italian', 'Chinese', 'Japanese', 'Cajun/Creole', 
                     'Barbeque', 'Southern', 'Thai', 'Mediterranean', 'Vietnamese'}

# Extract cuisine from categories
def find_cuisine(categories):
    categories_list = categories.split(',')
    for category in categories_list:
        category = category.strip()
        if category in specific_cuisines:
            return category
    return None  

tips_df['matched_cuisine'] = tips_df['categories'].apply(find_cuisine)
tips_df.dropna(subset=['matched_cuisine'], inplace=True)

# Split text into sentences for analysis
tips_df['sentences'] = tips_df['text'].str.replace(r'[.!?]', '!', regex=True).str.split('!')
expanded_df = tips_df.explode('sentences')

# Sentiment analysis using VADER
sia = SentimentIntensityAnalyzer()
expanded_df['pos'] = expanded_df['sentences'].apply(lambda x: sia.polarity_scores(x)['pos'])
expanded_df['neg'] = expanded_df['sentences'].apply(lambda x: sia.polarity_scores(x)['neg'])
expanded_df['neutral'] = expanded_df['sentences'].apply(lambda x: sia.polarity_scores(x)['neu'])
expanded_df['compound'] = expanded_df['sentences'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Sentiment metrics for each cuisine for top US metro areas
final_results = []
for cuisine in specific_cuisines:
    cuisine_data = expanded_df[expanded_df['matched_cuisine'] == cuisine]
    sentiment_summary = cuisine_data.groupby('metro').agg(
        total_sentences=('sentences', 'count'),  
        unique_businesses=('business_id', 'nunique'),  
        total_tips=('text', 'nunique'),  
        sum_positive=('pos', 'sum'),  
        sum_negative=('neg', 'sum'),  
        sum_neutral=('neutral', 'sum'),  
        sentiment_variance=('compound', 'var')  
    ).reset_index()
    sentiment_summary['total_sentiment'] = (
            sentiment_summary['sum_positive'] + 
            sentiment_summary['sum_negative'] + 
            sentiment_summary['sum_neutral']
    )
    sentiment_summary['percent_positive'] = (sentiment_summary['sum_positive'] / sentiment_summary['total_sentiment']) * 100
    sentiment_summary['percent_negative'] = (sentiment_summary['sum_negative'] / sentiment_summary['total_sentiment']) * 100
    sentiment_summary['matched_cuisine'] = cuisine
    final_results.append(sentiment_summary)

# Export results for visualization
final_df = pd.concat(final_results, ignore_index=True)
final_df.to_csv('Yelp_Tips_Sentiment_Analysis.csv', index=False)

# Topic Modeling with LDA on negative reviews
def preprocess_text(text):
    text = text.lower()
    return re.sub(f'[{re.escape(string.punctuation)}]', '', text)

neg_reviews = expanded_df[expanded_df["neg"] > 0.4]
neg_reviews['cleaned_sentence'] = neg_reviews['sentences'].apply(preprocess_text)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(neg_reviews['cleaned_sentence'])

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(tfidf_matrix)

def display_topics(model, feature_names, num_top_words=5):
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

display_topics(lda, tfidf.get_feature_names_out())

# Quality control - checking if overall tip sentiment aligns with star rating
expanded_df['pos_text'] = expanded_df['text'].apply(lambda x: sia.polarity_scores(x)['pos'])
expanded_df['neg_text'] = expanded_df['text'].apply(lambda x: sia.polarity_scores(x)['neg'])
expanded_df['neutral_text'] = expanded_df['text'].apply(lambda x: sia.polarity_scores(x)['neu'])
expanded_df['compound_text'] = expanded_df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

mismatch_neg = expanded_df[(expanded_df['neg_text'] > 0.3) & (expanded_df['stars'] > 3)]
print(mismatch_neg[['name', 'metro', 'pos', 'neg', 'neutral', 'sentences', 'stars']].sample(10))
