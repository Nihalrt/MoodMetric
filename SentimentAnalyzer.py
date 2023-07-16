import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download required NLTK resources (uncomment the following two lines on the first run)
# nltk.download('vader_lexicon')
# nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
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
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer

    def analyze_texts(self, texts):
        results = []
        for text in texts:
            sentiment = self.sentiment_analyzer.analyze_sentiment(text)
            results.append((text, sentiment))
        return results


class SocialMediaApp:
    def __init__(self):
        self.analyzer = SocialMediaAnalyzer(SentimentAnalyzer())

    def load_texts(self, file_path):
        with open(file_path, 'r') as file:
            texts = file.readlines()
        texts = [text.strip() for text in texts if text.strip()]
        return texts

    def display_sentiment_results(self, sentiment_results):
        print("Sentiment Analysis Results:")
        for text, sentiment in sentiment_results:
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}")
            print("-" * 20)

    def plot_sentiment_distribution(self, sentiment_results):
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
