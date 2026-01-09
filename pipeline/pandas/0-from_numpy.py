#!/usr/bin/env python3

def from_numpy(seed):
    return pd.DataFrame(seed, columns=list(string.ascii_uppercase[:seed.shape[1]]))
