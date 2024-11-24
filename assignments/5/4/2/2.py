import models.RNN.ocr as ocr 
import warnings
from icecream import ic

warnings.filterwarnings("ignore")

path = "./data/interim/5/4/nltk"

train_loader,val_loader,char_set,max_len = ocr.loader(path,ratio=0.8,batch_size=32,percentage=0.1)  
ic(max_len)

model = ocr.model(n_hidden=256,max_length=max_len,char_set=char_set,lr=0.001)

ocr.train(model,train_loader,val_loader,epochs=10)