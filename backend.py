
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
class MovieRecommender:
    def __init__(self, dataset_path):
        dataset_path='IMDB_Top250Engmovies2_OMDB_Detailed.csv'
        self.df = pd.read_csv(dataset_path)
        self.cosine_sim = None
        self.indices = None
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess the dataset to prepare it for recommendations."""
        self.df = self.df[['Title', 'Director', 'Actors', 'Plot', 'Genre']]
        self.df['Plot'] = self.df['Plot'].str.replace('[^\w\s]', '', regex=True)  # Remove punctuation

        # Extract keywords using RAKE
        self.df['Key_words'] = ''
        r = Rake()
        for index, row in self.df.iterrows():
            if pd.notnull(row['Plot']):
                r.extract_keywords_from_text(row['Plot'])
                key_words_dict_scores = r.get_word_degrees()
                self.df.at[index, 'Key_words'] = list(key_words_dict_scores.keys())

        # Split and clean columns
        self.df['Genre'] = self.df['Genre'].map(lambda x: x.split(','))
        self.df['Actors'] = self.df['Actors'].map(lambda x: x.split(',')[:3])
        self.df['Director'] = self.df['Director'].map(lambda x: x.split(','))

        for index, row in self.df.iterrows():
            row['Genre'] = [x.lower().replace(' ', '') for x in row['Genre']]
            row['Actors'] = [x.lower().replace(' ', '') for x in row['Actors']]
            row['Director'] = [x.lower().replace(' ', '') for x in row['Director']]

        # Combine into 'Bag_of_words'
        self.df['Bag_of_words'] = ''
        columns = ['Genre', 'Director', 'Actors', 'Key_words']
        for index, row in self.df.iterrows():
            words = ''
            for col in columns:
                words += ' '.join(row[col]) + ' '
            self.df.at[index, 'Bag_of_words'] = words.strip()

        # Generate the count matrix
        count = CountVectorizer()
        count_matrix = count.fit_transform(self.df['Bag_of_words'])

        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)

        # Create indices for titles
        self.indices = pd.Series(self.df['Title'])

    def recommend(self, title):
        """Generate a list of recommended movies for a given title."""
        if title not in self.indices.values:
            return ["Movie not found in the dataset!"]
        
        idx = self.indices[self.indices == title].index[0]
        score_series = pd.Series(self.cosine_sim[idx]).sort_values(ascending=False)
        top_10_indices = list(score_series.iloc[1:11].index)
        recommended_movies = [list(self.df['Title'])[i] for i in top_10_indices]
        return recommended_movies