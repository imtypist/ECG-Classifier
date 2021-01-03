# ECG Classifier - Class Project for Statistical Learning 2020

### Data

You can obtain datasets from [StatLearning SJTU 2020](https://www.kaggle.com/c/statlearning-sjtu-2020/).

### Environment

- Python 3.5+
  - tensorflow
  - sklearn
  - pywt
  - seaborn
  - biosppy
  - scipy

### Run it

1. Place the `ecg.py` in the dataset directory
2. Run the python code with one parameter

``` bash
# using CNN
$ python ecg.py cnn
# using SVM
$ python ecg.py svm
# using kNN
$ python ecg.py knn
```

3. You will get a trained model and the classification result in `.csv` format.