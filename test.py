import numpy as np

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b) / ((np.linalg.norm(a))* np.linalg.norm(b))

x= cosine_similarity([1,2],[3,4])
print(x)