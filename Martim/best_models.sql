DECLARE @result1 TABLE (loja int)
Declare @Id int
DECLARE @best_models TABLE (
ts datetime,
tienda int,
model_type varchar(500),
model_params varchar(max),
model_pickle varbinary(max),
model_metrics varchar(max),
best_metric real,
model_train_time real,
train_size int
)

INSERT INTO @result1
SELECT DISTINCT tienda from Martim.models;

While EXISTS(SELECT * From @result1)
Begin

    Select Top 1 @Id = loja From @result1
	Insert into @best_models
	SELECT top 1 * from Martim.models
	where tienda=@Id
	order by best_metric desc;

    Delete @result1 Where loja = @Id

End

SELECT * from @best_models
