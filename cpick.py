import pickle
with open('../lstm/train/lemma','rb') as f:
    obj = pickle.load(f)
with open('../lstm/train/lemma2','wb') as f:
    pickle.dump(obj,f,protocol=2)
