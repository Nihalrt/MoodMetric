# MoodMetrc

## Overview

A Python-based application that leverages a sentiment analysis tool to analyze the sentiment of social media text data. It provides insights into whether text content is positive, negative, neutral, or extremely positive/negative. The project includes three main components: a sentiment analysis module, a social media text analyzer, and a user-friendly application to run the analysis.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features

### Sentiment Analysis

The project employs the sentiment analysis tool to categorize text data into the following sentiment labels:

- Extremely Positive
- Very Positive
- Positive
- Extremely Negative
- Very Negative
- Negative
- Neutral

### Social Media Text Analysis

The `SocialMediaAnalyzer` class can analyze a list of social media texts and provide sentiment results for each text, making it suitable for batch processing.

### User-Friendly Application

The `SocialMediaApp` class serves as an easy-to-use interface for running sentiment analysis on social media texts. It allows users to load data from a file, view sentiment results, visualize the sentiment distribution, and save results to an output file.

## Usage

To use the Social Media Sentiment Analyzer, follow these steps:

1. Install the required Python dependencies (NLTK and Matplotlib) using `pip`:

```bash
pip install nltk matplotlib
