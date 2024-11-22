import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

def load(file: str) -> pd.DataFrame:
    