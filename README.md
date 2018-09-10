# poketto
Pocketto means the Yon-jigen poketto of Doraemon.

Hopefully to be a collection of machine learning toolkits.

## Install
```
python setup.py install
```
OR
```
pip install poketto
```

## Functions

The modules are the follows:

1.do metrics (now support binary classification)

```
import poketto.metrics as mc

y = np.random.randint(0, 2, 10000) # or your real data
pred = np.random.rand(10000) # or your real data

metric = mc.BinaryMetrics(y, pred)

# print the format result to the stdout
print(metric)

# get the result metrics
result = metric.metrics
print(result["auc"])
print(result["logloss"])
print(result["mse"])
print(result["ks"])
print(result["opt_cut"])
print(result["average_precision"])
print(result["accuracy"])
print(result["precision"])
print(result["recall"])

# plot the metrics plot
metric.plot(path_dir="your/path", title="your own title prefix")
```

2.do eda (now support features and labels distribution for classification problem)

```
import poketto.eda as eda

bunch = load_boston()
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
y = pd.Series(np.random.randint(0, 2, size=len(X)))
numeric_cols = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"]
category_cols = ["CHAS", "RAD"]

my_eda = eda.Eda(X=X, y=y, numeric_cols=numeric_cols, category_cols=category_cols)

result = my_eda.features_distribution(plot=True)
result = my_eda.target_distribution(plot=True)
```
