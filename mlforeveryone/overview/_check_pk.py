import pandas as pd
import numpy as np

def check_pk(dtf, pk):
    unique_pk, len_dtf = dtf[pk].nunique(), len(dtf)
    check = "unique "+pk+": "+str(unique_pk)+"  |  len dtf: "+str(len_dtf)
    if unique_pk == len_dtf:
        msg = "OK!!!  "+check
        print(msg)
    else:
        msg = "WARNING!!!  "+check
        ERROR = dtf.groupby(pk).size().reset_index(name="count").sort_values(by="count", ascending=False)
        print(msg)
        print("Example: ", pk, "==", ERROR.iloc[0,0])