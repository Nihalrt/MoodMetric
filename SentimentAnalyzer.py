import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import os

# Download required NLTK resources (uncomment the following two lines on the first run)
# nltk.download('vader_lexicon')
# nltk.download('punkt')

class SentimentAnalyzer:
    """
    A class for analyzing sentiment of text using the VADER sentiment analysis tool.
    """
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a given text.

        Args:
            text (str): The input text to analyze.

        Returns:
            str: The sentiment label ('Extremely Positive', 'Very Positive', 'Positive',
                 'Extremely Negative', 'Very Negative', 'Negative', or 'Neutral').
        """
        sentiment_scores = self.sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.3:
            return 'Extremely Positive'
        elif compound_score >= 0.1:
            return 'Very Positive'
        elif compound_score > 0:
            return 'Positive'
        elif compound_score <= -0.3:
            return 'Extremely Negative'
        elif compound_score <= -0.1:
            return 'Very Negative'
        elif compound_score < 0:
            return 'Negative'
        else:
            return 'Neutral'


class SocialMediaAnalyzer:
    """
    A class for analyzing social media texts using sentiment analysis.
    """
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer

    def analyze_texts(self, texts):
        """
        Analyze a list of texts and return sentiment results for each text.

        Args:
            texts (list): A list of text strings to analyze.

        Returns:
            list: A list of tuples, where each tuple contains the text and its sentiment label.
        """
        results = []
        for text in texts:
            sentiment = self.sentiment_analyzer.analyze_sentiment(text)
            results.append((text, sentiment))
        return results


class SocialMediaApp:
    """
    A social media sentiment analysis application.
    """
    def __init__(self):
        self.analyzer = SocialMediaAnalyzer(SentimentAnalyzer())

    def load_texts(self, file_path):
        """
        Load text data from a file.

        Args:
            file_path (str): The path to the file containing social media texts.

        Returns:
            list: A list of text strings.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")

        with open(file_path, 'r') as file:
            texts = file.readlines()
        texts = [text.strip() for text in texts if text.strip()]
        return texts

    def display_sentiment_results(self, sentiment_results):
        """
        Display sentiment analysis results to the console.

        Args:
            sentiment_results (list): A list of tuples, where each tuple contains the text and its sentiment label.
        """
        print("Sentiment Analysis Results:")
        for text, sentiment in sentiment_results:
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}")
            print("-" * 20)

    def plot_sentiment_distribution(self, sentiment_results):
        """
        Plot the distribution of sentiment labels.

        Args:
            sentiment_results (list): A list of tuples, where each tuple contains the text and its sentiment label.
        """
        sentiment_counts = {}
        for _, sentiment in sentiment_results:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        labels = sentiment_counts.keys()
        counts = sentiment_counts.values()

        fig, ax = plt.subplots()
        ax.bar(labels, counts)

        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.set_title('Sentiment Distribution')

        plt.show()

    def save_sentiment_results(self, sentiment_results, output_file):
        """
        Save sentiment analysis results to a text file.

        Args:
            sentiment_results (list): A list of tuples, where each tuple contains the text and its sentiment label.
            output_file (str): The path to the output file.
        """
        with open(output_file, 'w') as file:
            for text, sentiment in sentiment_results:
                file.write(f"Text: {text}\n")
                file.write(f"Sentiment: {sentiment}\n")
                file.write("-" * 20 + "\n")

        print(f"Sentiment results saved to {output_file}")

    def run(self):
        file_path = 'social_media_texts.txt'
        output_file = 'sentiment_results.txt'

        texts = self.load_texts(file_path)

        sentiment_results = self.analyzer.analyze_texts(texts)

        self.display_sentiment_results(sentiment_results)
        self.plot_sentiment_distribution(sentiment_results)
        self.save_sentiment_results(sentiment_results, output_file)


if __name__ == '__main__':
    nltk.download('vader_lexicon')
    nltk.download('punkt')

    app = SocialMediaApp()
    app.run()
