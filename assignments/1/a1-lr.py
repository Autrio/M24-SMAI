import argparse
from models.linearRegression.linearRegression import *
from models.linearRegression.GIF import *

parser = argparse.ArgumentParser(prog="A1",description="argument parser for KNN")

parser.add_argument("-q","--question",type=int,help="""
                    Choose which subpart of KNN to execute
                    """,default=False)
args  = parser.parse_args()

path1 = "./data/external/linreg.csv"
path2 = "./data/external/regularisation.csv"

LR = 0.01
epochs = 10000

reg = LinearRegressor(LR=LR,epochs=epochs)

def Q1():
    reg.load_data(path1)
    reg.run(1)

def Q2():
    reg.load_data(path1)
    reg.run(6)

def Q3():
    reg.load_data(path1)
    reg.GIFplots(1,path = './assignemnts/1/figures/1')
    convertGIF(deg=1)
    reg.GIFplots(2,path = './assignemnts/1/figures/2')
    convertGIF(deg=2)
    reg.GIFplots(3,path = './assignemnts/1/figures/3')
    convertGIF(deg=3)
    reg.GIFplots(4,path = './assignemnts/1/figures/4')
    convertGIF(deg=4)
    reg.GIFplots(5,path = './assignemnts/1/figures/5')
    convertGIF(deg=5)

def Q4():
    reg.load_data(path2)
    reg.reg_param = 30
    reg.reg_type = 'L2'
    reg.run(20)

if __name__ == "__main__":
    if(args.question==1):
        Q1()
    elif(args.question==2):
        Q2()
    elif(args.question==3):
        Q3()
    elif(args.question==4):
        Q4()
    else:
        pass