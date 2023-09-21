import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Read Moby Dick file
with open('moby_dick.txt', 'r') as file:
    text = file.read()

# Tokenization and stopwords filtering
doc = nlp(text)
tokens = [token.text for token in doc if not token.is_stop]

# Parts-of-Speech (POS) tagging
pos_tags = [token.pos_ for token in doc]

# POS frequency
pos_freq = {}
for tag in pos_tags:
    pos_freq[tag] = pos_freq.get(tag, 0) + 1
top_pos = sorted(pos_freq.items(), key=lambda x: x[1], reverse=True)[:5]

# Lemmatization
lemmatized_tokens = [token.lemma_ for token in doc][:20]

# Plotting frequency distribution
pos_labels, pos_counts = zip(*top_pos)
plt.bar(pos_labels, pos_counts)
plt.xlabel('POS')
plt.ylabel('Frequency')
plt.title('POS Frequency Distribution')
plt.show()

# Print the results
print("Top 5 POS and their frequency:")
for pos, freq in top_pos:
    print(f"{pos}: {freq}")

print("\nLemmatized tokens:")
print(lemmatized_tokens)