import os
from transformers import pipeline
from collections import Counter

predict = pipeline('sentiment-analysis')
neg_files = os.listdir('./test/neg')

lines = []
for neg in neg_files[:2]:
    with open(os.path.join('./test/neg', neg), 'r') as f:
        lines += f

predictions = predict(lines)
counter = Counter()
for prediction in predictions:
    counter[prediction['label']] += 1

print(counter)
