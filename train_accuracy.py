import torch
import numpy as np
from cnn.cnn_architecture import *
from cnn.validation_accuracy import *
from sklearn.metrics import accuracy_score
from cnn.train_data import *

# prediction for training set
with torch.no_grad():
    output = model(train_x.cuda())
    
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
accuracy_score(train_y, predictions)