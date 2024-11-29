from cnn.dataset import *
from cnn.test_prediction import *
# replacing the label with prediction
sample_submission['label'] = predictions
sample_submission.head()