from sklearn import linear_model
import constants
import pandas as pd

class CompetitionRun(object):
    def __init__(self, model):
        self.model = model

    def train(self):
        training_filename = constants.training_data_csv
        training_input, training_output = self._read_training_data_vectorized(training_filename)
        self._fit_model(training_input, training_output)

    def _read_training_data_vectorized(self, training_filename):
        file_contents = pd.read_csv(training_filename, index_col=constants.ID_COLUMN)
        feature_columns, target_column = file_contents.columns[:-1], file_contents.columns[-1]
        return file_contents[feature_columns], file_contents[target_column]

    def _fit_model(self, training_input, training_output):
        self.model.fit(training_input, training_output)

    def test(self):
        testing_filename = constants.testing_data_csv
        testing_input = self._read_testing_data_vectorized(testing_filename)
        predicted_output = self._predict_model(testing_input)
        print(predicted_output)

    def _read_testing_data_vectorized(self, testing_filename):
        file_contents = pd.read_csv(testing_filename, index_col=constants.ID_COLUMN)
        feature_columns= file_contents.columns
        return file_contents[feature_columns]

    def _predict_model(self, testing_input):
        return self.model.predict(testing_input)

if __name__ == '__main__':
    model = linear_model.LogisticRegression()
    run = CompetitionRun(model)
    run.train()
    results = run.test()
    output_mapping = 