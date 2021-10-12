import numpy as np
import pandas as pd
f=open("Model.csv","w")

X = np.array([
                    [5, 3],
                  [10, 15],
                  [15, 12],
                  [24, 10],
                  [30, 30],
                  [85, 70],
                  [71, 80],
                [60, 78],
                  [70, 55],
                  [80, 91], ])
intanceText=np.array(["ONE",
                  "TWO",
                  "THREE",
                  "FOUR",
                  "FIVE",
                  "SIX",
                  "SEVEN",
                  "EIGHT",
                  "NINE",
                  "TEN",])

for i in range(0,len(X)):
    a=""
    a+=str(intanceText[i])
    a+=";"
    a+=str(X[i])
    a+=";"
    a+=str(i)
    f.write(a)
    f.write("\n")

f.close()

