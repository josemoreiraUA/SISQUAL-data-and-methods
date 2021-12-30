DROP TABLE MARTIM.models
IF OBJECT_ID('Martim.models', 'U') IS NULL
CREATE TABLE  Martim.models(
ts datetime,
tienda int,
model_type varchar(500),
model_params varchar(max),
model_pickle varbinary(max),
model_metrics varchar(max),
best_metric real,
model_train_time real,
train_size int
);