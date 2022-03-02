from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fast_forecaster import preprocessors as pp

TARGET = 'n_clients' ### this is a univariate model
class Pipe:
    def __init__(self):
        self.forecast_pipeline = Pipeline(
            [
                ('Dropper', pp.DropColumns(TARGET)),
                ('NDArrayer', pp.TransformDFtoNDArray()),
                ('Scaler', StandardScaler()),
                ('Windower', pp.SplitInWindows()),
            ]
        )

    def get_pipeline(self) -> Pipeline:
        return self.forecast_pipeline