import deepchem as dc
import numpy as np
#matrix with 100 samples and 10 features each
x=np.random.rand(100,10)
#randomly assigning binary labels
y=np.random.randint(0,2,size=(100,))
# creating a dataset that is deepchem specific obj
dataset=dc.data.NumpyDataset(x,y)
model=dc.models.MultitaskClassifier(n_tasks=1,n_features=10)
model.fit(dataset)
preds=model.predict(dataset)
print(preds.shape)
