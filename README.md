# poketto
Pocketto means the Yon-jigen poketto of Doraemon.

Hopefully to be a collection of machine learning toolkits.

## Install
```
python setup.py install
```

The modules are the follows:

1.do metrics (now support binary classification)

```
from poketto.metrics import binary_metrics as bin

y = np.random.randint(0, 2, 10000) # or your real data
pred = np.random.rand(10000) # or your real data

metric = bin.BinaryMetrics(y, pred)

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
