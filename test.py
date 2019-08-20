import numpy as np

path = "/Users/polybahn/Desktop/ConvE/data/full_score.npy"
scores = np.load(path, allow_pickle=True).item()
scores_final = dict()
for key, val in scores.items():
    print(key)
    print(val)
    new_key = (key[0], key[-1])
    old_score = .0
    if new_key in scores_final:
        old_score = scores_final[new_key]
    scores_final[new_key] = max(old_score, val)

print(len(scores))
print(len(scores_final))



# Test tf.stack
# import tensorflow as tf
# x = tf.constant([[1, 4]])
# y = tf.constant([[2, 5], [1, 8]])
# z = tf.constant([[3, 6]])
# print(tf.concat([x, y, z], axis=0))# [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)