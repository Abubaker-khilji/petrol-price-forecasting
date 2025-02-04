import logging

def save_pickle(obj, filename):
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    import pickle
    with open(filename, "rb") as f:
        return pickle.load(f)