from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import ndiff
import re

# Function to read the documents
def read_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to calculate cosine similarity
def cosine_similarity_texts(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score

# Function to calculate Jaccard similarity
def jaccard_similarity(text1, text2):
    set1 = set(re.findall(r'\w+', text1.lower()))
    set2 = set(re.findall(r'\w+', text2.lower()))
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_score = len(intersection) / len(union) if union else 0
    return jaccard_score

# Function to highlight differences line by line
def line_diff(text1, text2):
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    diff = list(ndiff(text1_lines, text2_lines))
    return '\n'.join(diff)

# Main function
def compare_documents(file1, file2):
    text1 = read_document(file1)
    text2 = read_document(file2)

    # Cosine Similarity
    cosine_score = cosine_similarity_texts(text1, text2)
    print(f"Cosine Similarity Score: {cosine_score:.2f}")

    # Jaccard Similarity
    jaccard_score = jaccard_similarity(text1, text2)
    print(f"Jaccard Similarity Score: {jaccard_score:.2f}")

    # Line-by-line difference
    print("\nLine-by-Line Differences:")
    print(line_diff(text1, text2))

# Example usage
file1 = 'document1.txt'
file2 = 'document2.txt'
compare_documents(file1, file2)
