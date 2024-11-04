import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Load the Yelp data
tips_df = pd.read_csv("Yelp_Food_TIPS.csv")
print("Tips Columns:", tips_df.columns)

# Create GeoDataFrame from tips data
geometry = [Point(xy) for xy in zip(tips_df['longitude'], tips_df['latitude'])]
geo_df = gpd.GeoDataFrame(tips_df, geometry=geometry)

# Load metropolitan areas shapefile
metro_areas = gpd.read_file('/Users/satya/Downloads/cb_2023_us_cbsa_5m/cb_2023_us_cbsa_5m.shp')

# Filter only points within the metropolitan areas
merged = gpd.sjoin(geo_df, metro_areas, predicate='intersects')
unique_names = merged['NAMELSAD'].unique()

# Merge metro area information with tips data
tips_df = tips_df.merge(merged[['geometry', 'NAME']], left_index=True, right_index=True, how='left')
tips_df.rename(columns={'NAME': 'metro'}, inplace=True)
tips_df['metro'].fillna('Unknown Metro Area', inplace=True)

# Display unique metro areas
print(tips_df['metro'].unique())
print(f"Original shape: {geo_df.shape}")  # Before merge
print(f"New shape: {tips_df.shape}")       # After merge

# Load the tips data with metro area information
tips_df = pd.read_csv("Yelp Tips with Metro.csv")
missing_tips = tips_df['text'].isna().sum()
tips_df = tips_df.dropna(subset=['text'])  # Drop rows without text

# List of metro areas to filter out due to small sample size
excluded_metros = [
    'Truckee-Grass Valley, CA', 
    'Reading, PA', 
    'Trenton-Princeton, NJ', 
    'Unknown Metro Area', 
    'Atlantic City-Hammonton, NJ', 
    'New York-Newark-Jersey City, NY-NJ'
]

# Filter tips by metro area
tips_with_metro = tips_df[~tips_df['metro'].isin(excluded_metros)]
print("Major City Tips:", tips_with_metro.shape)

# Define specific cuisines
specific_cuisines = {'Mexican', 'Italian', 'Chinese', 'Japanese', 'Cajun/Creole', 'Barbeque', 'Southern', 'Thai', 'Mediterranean', 'Vietnamese'}

# Function to find matching cuisine
def find_cuisine(categories):
    categories_list = categories.split(',')
    for category in categories_list:
        category = category.strip()  
        if category in specific_cuisines: 
            return category 
    return None  

# Apply the function to find matched cuisine
tips_with_metro['matched_cuisine'] = tips_with_metro['categories'].apply(find_cuisine)
major_city_expanded = tips_with_metro.dropna(subset=['matched_cuisine'])

# Calculate top tags
top_tags = major_city_expanded['matched_cuisine'].value_counts().reset_index(name='Number of tips').rename(columns={'index': 'tags'}).head(10)
print(top_tags)

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
major_city_expanded['pos'] = major_city_expanded['sentences'].apply(lambda x: sia.polarity_scores(x)['pos'])
major_city_expanded['neg'] = major_city_expanded['sentences'].apply(lambda x: sia.polarity_scores(x)['neg'])
major_city_expanded['neutral'] = major_city_expanded['sentences'].apply(lambda x: sia.polarity_scores(x)['neu'])
major_city_expanded['compound'] = major_city_expanded['sentences'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Aggregate sentiment scores by cuisine
final_results = []
for cuisine in specific_cuisines:
    print(f"Sentiment scores for {cuisine}:")
    cuisine_data = major_city_expanded[major_city_expanded['matched_cuisine'] == cuisine]

    # Group by metro and calculate sentiment statistics
    sentence_sentiments = cuisine_data.groupby('metro').agg(
        total_sentences=('text', 'count'),  
        unique_businesses=('business_id', 'nunique'),  
        total_tips=('text', 'nunique'),  
        sum_positive=('pos', 'sum'),  
        sum_negative=('neg', 'sum'),  
        sum_neutral=('neutral', 'sum'),  
        sentiment_variance=('compound', 'var')  
    ).reset_index()

    sentence_sentiments['total_sentiment'] = (
        sentence_sentiments['sum_positive'] + 
        sentence_sentiments['sum_negative'] + 
        sentence_sentiments['sum_neutral']
    )
    
    sentence_sentiments['percent_positive'] = (sentence_sentiments['sum_positive'] / sentence_sentiments['total_sentiment']) * 100
    sentence_sentiments['percent_negative'] = (sentence_sentiments['sum_negative'] / sentence_sentiments['total_sentiment']) * 100
    sentence_sentiments['matched_cuisine'] = cuisine

    final_results.append(sentence_sentiments[['metro', 'total_tips', 'total_sentences', 'unique_businesses', 
                                               'percent_positive', 'percent_negative', 'sentiment_variance', 'matched_cuisine']])

# Create final dataframe with results
final_results_df = pd.concat(final_results, ignore_index=True)

# Save the final results to CSV
final_results_df.to_csv('Yelp_Tips_Sentiment_Analysis.csv', index=False)
