
# tests/test_preprocessing.py
import pandas as pd
from src.preprocessing import identify_skewed_columns, log_transform

def test_identify_skewed_columns():
    df = pd.DataFrame({'a':[1,2,3],'b':[0,0,1000],'c':[1,1,1]})
    sk = identify_skewed_columns(df, threshold=1.0, exclude=['c'])
    assert 'b' in sk

def test_log_transform():
    df = pd.DataFrame({'x':[0,1,9]})
    df2 = log_transform(df.copy(), ['x'])
    assert 'x_log' in df2.columns
    assert df2['x_log'].iloc[0] == 0.0
