


def to_cat(LIST, df): 
    for col in LIST: 
        df[col] = df[col].astype('category')

def maj_vote(p1, p2, p3): 
    import numpy as np
    import pandas as pd
    pred_ens = np.stack([p1, p2, p3], axis = 1)
    preds = pd.DataFrame(data=pred_ens, columns=["p1", "p2","p3"])
    preds['maj_vote'] = preds.mode(axis=1)[0]
    return preds['maj_vote']