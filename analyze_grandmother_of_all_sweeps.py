import pandas as pd
import sys

if len(sys.argv) > 1:
    training_results = pd.read_csv(sys.argv[1],delimiter="\t", comment="#")
else:
    training_results = None
if len(sys.argv) > 2:
    bounds = pd.read_csv(sys.argv[2],delimiter="\t", comment="#")
else:
    bounds = None


