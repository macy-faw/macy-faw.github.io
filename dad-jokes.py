#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:22:22 2024

@author: macyfaw
"""

#%% Importing libraries and organizing the data
# Importing Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import markovify
import nltk
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Downloading data and converting all to lowercase
jokes = pd.read_csv("dad_jokes_cleaned.csv")
jokes['joke'] = jokes['joke'].str.lower()


# Initializing the TF-IDF vectorizer and removing english stop words
vectorizer = TfidfVectorizer(stop_words='english')


# Fit the model on the jokes
X = vectorizer.fit_transform(jokes['joke'])

# Get feature names (words) and their corresponding scores
feature_names = vectorizer.get_feature_names_out()

# Convert to a DataFrame to check top keywords for each joke
tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)

# Show the top keywords for each joke (you can adjust how many keywords to show)
print(tfidf_df.head())


#%% Word Frequency
# Calculating word frequency across all jokes
word_frequencies = X.sum(axis = 0).A1
word_freq_df = pd.DataFrame(list(zip(feature_names, word_frequencies)), columns=['Word', 'Frequency'])
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

#%% Topic Modeling with LDA
lda = LatentDirichletAllocation(n_components=12, random_state=42)
lda.fit(X)


#%% Examining LDA chosen topics
# Number of words to display per topic
num_words = 20

# Get the top words for each topic
for topic_idx, topic in enumerate(lda.components_):
    # Get the indices of the top words in this topic (sorted by probability)
    top_words_idx = topic.argsort()[-num_words:][::-1]
    
    # Get the actual words corresponding to these indices
    top_words = [feature_names[i] for i in top_words_idx]
    
    # Print the topic number and its top words
    print(f"Topic {topic_idx}: {' '.join(top_words)}")
    
#%%
# Examining using word cloud
from wordcloud import WordCloud

# Generate word clouds for each topic
for topic_idx, topic in enumerate(lda.components_):
    plt.figure(figsize=(8, 6))
    word_freq = {feature_names[i]: topic[i] for i in topic.argsort()[-num_words:][::-1]}
    
    # Generate and display the word cloud
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Topic {topic_idx}")
    plt.show()



#%% Rule-based approach to make sure things are actually grouped by topics
# Define lists of keywords for each category
animal_keywords = ['dog', 'cat', 'bear', 'fish', 'lion', 'elephant', 'zebra', 'horse', 'bird', 
                   'tiger', 'parrot', 'penguin', 'dogs', 'toad', 'bison', 'cats', 'buffalo', 'bark', 'cow', 
                   'dragons', 'kangaroo', 'turkey', 'cheetahs', 'sheep', 'lamb', 'bee', 'chicken', 'ant',
                   'monkey', 'zoo', 'zoology', 'zookeeper', 'wombat', 'wombats', 'werewolf', 'warthog',
                   'wasps', 'wasp', 'turtle', 'turtles', 'rabbit', 'rabbits', 'oyster', 'owl', 'mosquito',
                   'mosquitoes', 'fox', 'foxes', 'flamingo', 'flamingoes', 'flamingos' ,'elephant', 'donkey',
                   'cardinal']

food_keywords = ['apple', 'banana', 'cheese', 'cake', 'pizza', 'cookie', 'burger', 'soup', 'bread', 'hotdog', 
                 'apples', 'carrot', 'orange', 'milk', 'nacho', 'noodle', 'noodles', 'impasta', 'pumpkin',
                 'melons', 'cantaloupe', 'cooked', 'cook', 'cooking', 'bake', 'steak', 'corn',
                 'doughnut', 'donut', 'walnut', 'walnuts', 'waffle', 'waffles', 'veggie', 'veggies', 'tofu',
                 'sandwich', 'sandwhich', 'sandwitch', 'rigatoni', 'popsicle', 'popcorn', 'pie', 'pepperoni',
                 'olive', 'hamburger', 'hamburgers', 'cheeseburger', 'halapeno', 'gummy', 'cucumber', 'tomato',
                 'pepper', 'cookie', 'cookies']


family_keywords = ['dad', 'mom', 'son', 'daughter', 'family', 'kids', 'wife', 'husband', 'parents', 'father', 'mother',
                   'wedding', 'weddings','house', 'home', 'clean', 'cleaning', 'aunt', 'uncle', 'cousin', 'grandma', 'grandpa',
                   'triplets', 'twins', 'relative', 'relatives', 'relationship', 'relationsheep',
                   'gran', 'grampa', 'grams', 'granddaughter', 'grandfather', 'grandmother', 'grandmothers', 'grandparents',
                   'grandson', 'grandsons', 'granny']

school_work_keywords = ['alphabet', 'book', 'reading', 'pencil', 'writing', 'write', 'read', 'wrote', 'newspaper', 
                   'magazine', 'minus', 'math', 'chemistry', 'study', 'studying', 'principal',
                   'teacher', 'student', 'computer', 'boss', 'algebra', 'geometry', 'calculus', 'written',
                   'vocabulary', 'teachers', 'studies', 'studied']


weather_keywords = ['igloos', 'cold', 'hot', 'snow', 'rain', 'raining', 'snowing', 'rains', 'snows',
                    'storm', 'stormy', 'storming', 'storms', 'hurricane', 'hurricanes', 'earthquale',
                    'earthquakes', 'landslide', 'snowman', 'gravity', 'tornado', 'lightning']

outdoor_keywords = ['tree', 'field', 'mud', 'muddy', 'fire','trees', 'scarecrow', 'farmer', 'farm']

sport_keywords = ['ball', 'football', 'baseball', 'soccer', 'run', 'running', 'hunt', 'hunting', 'fish', 'fishing',
                  'boat', 'boats', 'boating', 'ski', 'skiing', 'dance', 'wrestle', 'wrestlers', 'walking',
                  'yoga', 'volleyball', 'olympics']

time_keywords = ['clock', 'time', 'day', 'date', 'hour', 'january', 'february', 'march', 'april', 'june', 'july',
                 'august', 'september', 'october', 'november', 'monday', 'tuesday', 'wednesday',
                 'thursday', 'friday', 'saturday', 'sunday', 'month', 'weekday', 'weekend', 'years', 'yearly',
                 'mondays', 'tuesdays', 'wednesdays', 'thursdays', 'fridays', 'saturdays', 'sundays',
                 'summer', 'fall', 'autumn', 'winter', 'spring']

appearance_keywords = ['hair', 'hairy', 'face', 'facial', 'diet', 'haircut', 'dressed', 'shoes', 'closet', 'dress',
                       'shirt', 'pants', 'ears', 'eyes', 'arms', 'legs', 'arm', 'leg', 'shoe',
                       'nose', 'toes', 'foot', 'head', 'wardrobe', 'turtlenecks']


tired_keywords = ['tired', 'sleeping', 'sleep', 'sleepy', 'exhaust', 'exhausted', 'yawn', 'yawns']

music_keywords = ['dance', 'boogie', 'piano', 'guitar', 'sing', 'sings', 'singer',
                  'keyboard', 'banjo', 'saxophone', 'ukulele', 'opera', 'guitarist', 'concert',
                  'concerts']

mail_keywords = ['mail', 'postage', 'stamp', 'stamps', 'stamped', 'mailed', 'post', 'postmaster', 'shipment',
                 'usps', 'parcel', 'parcels', 'mailing', 'envelope', 'envelopes', 'courier']

holiday_keywords = ['thanksgiving', 'christmas', 'easter', 'halloween', 'santa', 'elf', 'elves', 'holiday',
                    'spooky', 'boo', 'ornaments']




#%% Assign jokes to a cateogry based on defined categories 

def assign_category(joke):
    # Define categories and their respective keyword lists
    categories = {
        "Animal": animal_keywords,
        "Food": food_keywords,
        "Family": family_keywords,
        "School_work": school_work_keywords,
        "Weather": weather_keywords,
        "Outdoor": outdoor_keywords,
        "Sport": sport_keywords,
        "Time": time_keywords,
        "Clothes/Appearance": appearance_keywords,
        "Tired": tired_keywords,
        "Music": music_keywords,
        "Mail": mail_keywords,
        "Holiday": holiday_keywords
    }
    
    # Iterate over the categories and check for keyword matches in the joke
    for category, keywords in categories.items():
        if any(keyword in joke for keyword in keywords):
            return category  # Return the first matching category
    
    return "Undetermined"  # Return "undetermined" if no match found


# Apply the function to each joke and assign categories
jokes['category'] = jokes['joke'].apply(assign_category)

category_counts = jokes['category'].value_counts()

# Print the category counts
print(category_counts)


# Save the dataset with the assigned categories to a new file
jokes.to_csv("dad_jokes_with_categories.csv", index=False)

# Optional: Print the first few rows to verify
print(jokes.head())



#%% Streamlit for Interactive Display
# Load the jokes dataset with the assigned categories
jokes = pd.read_csv("dad_jokes_with_categories.csv")

# Create a list of available categories
categories = jokes['category'].unique().tolist()

# Set Streamlit page configuration (optional)
st.set_page_config(page_title="Dad Jokes",layout="centered")

# Add custom title and header with styling
st.markdown("""
    <h1 style='color:#4775d1; text-align:center;'>Welcome to the Dad Jokes App</h1>
    <h2 style='color:#d279a6; text-align:center;'>Select a category to hear a funny joke!</h2>
""", unsafe_allow_html=True)


# Dropdown for category selection
selected_category = st.selectbox("Choose a category:", categories)

# Filter jokes based on the selected category
filtered_jokes = jokes[jokes['category'] == selected_category]

# Function to get random joke
def get_random_joke():
    return filtered_jokes.sample(1).iloc[0]['joke'] if not filtered_jokes.empty else "No jokes found for this category."

# Initialize session_state for storing the joke if it doesn't exist
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = selected_category
    # Set initial joke when page loads
    st.session_state.joke = get_random_joke()  

# Check if the category has changed
if st.session_state.selected_category != selected_category:
    st.session_state.selected_category = selected_category
    # Update joke for the new category
    st.session_state.joke = get_random_joke()  

# Show the current joke in a text box
st.text_area(f"Random {selected_category.capitalize()} Joke", st.session_state.joke, height=150, max_chars=500)

# Optional: Add a button for interactivity
if st.button("Get Another Joke!"):
    st.session_state.joke = get_random_joke()  # Update joke in session_state
   
#%% Open terminal and run following line
# python -m streamlit run dad-jokes.py






