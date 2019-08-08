import numpy as np

path = "/Users/polybahn/Desktop/ConvE/data/full_score.npy"
scores = np.load(path, allow_pickle=True).item()
scores_final = dict()
for key, val in scores.items():
    new_key = (key[0], key[-1])
    old_score = .0
    if new_key in scores_final:
        old_score = scores_final[new_key]
    scores_final[new_key] = max(old_score, val)

print(len(scores))
print(len(scores_final))
