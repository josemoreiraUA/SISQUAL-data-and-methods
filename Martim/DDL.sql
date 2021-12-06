IF OBJECT_ID('Martim.models', 'U') IS NULL
CREATE TABLE  Martim.models(
model_type varchar(500),
model_pickle varbinary(500),
model_metrics varchar(500),
best_metric varchar(500),
model_train_time real,
);