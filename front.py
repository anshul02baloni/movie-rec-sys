import streamlit as st
import pandas as pd
from backend import MovieRecommender  # Import the MovieRecommender class from backend.py

# Path
dataset_path = "IMDB_Top250Engmovies2_OMDB_Detailed.csv"
# recommender system
recommender = MovieRecommender(dataset_path)
st.title("Movie Recommendation System") # Title
st.write(""" entert tho Movie title 
""")
movie_title = st.text_input("Enter Movie Name", "") # Title name or Movie Name

# Check if the movie title is entered
if movie_title:
    # Get movie recommendations
    recommendations = recommender.recommend(movie_title)

    # If recommendations are available
    if recommendations and isinstance(recommendations, list):
        st.subheader(f"Top 10 Movies Similar to '{movie_title}':")
        for movie in recommendations:
            # Display the movie title
            st.write(f"**{movie}**")
    else:
        st.error(f"No recommendations found for '{movie_title}'. Please check the movie title.")
else:
    st.write("Enter a movie title above to get started!")
