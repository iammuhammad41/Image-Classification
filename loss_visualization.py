import pandas as pd
import matplotlib.pyplot as plt
from cnn.validation_data import *
from cnn.train_function import *

# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()