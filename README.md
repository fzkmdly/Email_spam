# Email_spam

This project can use the following libraries for data handling, preprocessing, and model selection in an email spam classification workflow.

## Core imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
```

## Preprocessing libraries

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
```

## Model selection and evaluation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
```