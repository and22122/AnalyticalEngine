from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

class model():
    def __init__(self, model):
        if model == None:
            self.model = "unknown"
        else:
            self.model = model
        
        filename = input("What file do you want to use data from?")

        self.data = []

        with open(filename, "r") as file:
            for line in file:
                self.data.push(line.split(','))
        
    
    def process(self):
        print("The default model is programmed only to display data.Displaying...\n")
        for values in self.data:
            print(values.join(", "))

