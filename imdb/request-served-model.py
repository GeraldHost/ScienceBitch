import requests
import os
from collections import Counter

neg_path = os.path.join('test/neg')
text_file_names = os.listdir(neg_path)

def get_neg_reviews(text_file_names):
    print("[*] reading neg files")
    for file_name in text_file_names:
        file_path = os.path.join(neg_path, file_name)
        f = open(file_path, 'r')
        yield [f.read()]
        f.close()

def get_data_sets(data):
    n = 32
    for i in range(0, len(data), n):
        yield data[i:i+n]

def request_prediction(instances):
    r = requests.post('http://localhost:8501/v1/models/my_model:predict', json={"instances": instances})
    print("Status %s" % r.status_code)
    return r.json()

neg_reviews = list(get_neg_reviews(text_file_names))
data_set = list(get_data_sets(neg_reviews))

results = []
for ds in data_set:
    print("[*] running data set: %s" % len(results))
    ret = request_prediction(ds)
    results.append(ret['predictions'])

c = Counter()
print("[*] counting results")
for res in results:
    for r in res:
        if round(r[0]) == 1:
            c['one'] += 1
        elif round(r[0]) == 0:
            c['zero'] += 1
        else: 
            c['wft'] += 1

print(c)
