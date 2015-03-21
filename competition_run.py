from sklearn import ensemble
import pandas as pd
import copy
import constants
from sklearn import cross_validation

class AbstractRun(object): #TODO: Drop abstract class, make IO a separate class which is composed
    def __init__(self, model):
        self.model = model

    def _read_training_data_vectorized(self, training_filename):
        file_contents = pd.read_csv(training_filename, index_col=constants.ID_COLUMN)
        feature_columns, target_column = file_contents.columns[:-1], file_contents.columns[-1]
        return file_contents[feature_columns], file_contents[target_column]

class CrossValidationRun(AbstractRun):
    def run_cross_validation(self):
        training_filename = constants.training_data_csv
        training_input, training_output = self._read_training_data_vectorized(training_filename)
        cv_train_in, cv_test_in, cv_train_out, cv_test_out = cross_validation.train_test_split(training_input, training_output, test_size=0.7, random_state=0)
        self.model.fit(cv_train_in, cv_train_out)
        predictions = self.model.predict(cv_test_in)
        correct = sum([1 for predicted, gold in zip(predictions,cv_test_out) if predicted == gold])
        percent_correct = (1.0*correct/len(predictions)) * 100
        print('Percent correct is {0}%'.format(percent_correct))


    def _split_list(self, L, k):
        buckets = [list() for i in range(k)]
        for element in L:
            key = hash(element) % k
            buckets[key].append(element)
        return buckets

class CompetitionRun(AbstractRun):
    def train(self):
        training_filename = constants.training_data_csv
        training_input, training_output = self._read_training_data_vectorized(training_filename)
        self._fit_model(training_input, training_output)

    def _fit_model(self, training_input, training_output):
        self.model.fit(training_input, training_output)

    def test(self):
        testing_filename = constants.testing_data_csv
        testing_input = self._read_testing_data_vectorized(testing_filename)
        predicted_output = self._predict_model(testing_input)
        self._write_output_to_file(testing_input.index, predicted_output)

    def _read_testing_data_vectorized(self, testing_filename):
        file_contents = pd.read_csv(testing_filename, index_col=constants.ID_COLUMN)
        feature_columns= file_contents.columns
        return file_contents[feature_columns]

    def _predict_model(self, testing_input):
        return self.model.predict(testing_input)

    def _write_output_to_file(self, index, predicted_output):
        f_out = open(constants.output_filename, 'w')
        f_out.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
        for id, predicted_class in zip(index, predicted_output):
            row = [str(id)] + self._vectorize_output_class(predicted_class)
            f_out.write(','.join(row) + '\n')
        f_out.close()

    def _vectorize_output_class(self, predicted_class):
        classes = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
        return ['0.0' if c != predicted_class else '1.0' for c in classes]

if __name__ == '__main__':
    model = ensemble.RandomForestClassifier(n_estimators=100)
    cv = CrossValidationRun(model)
    cv.run_cross_validation()
    # run = CompetitionRun(model)
    # run.train()
    # run.test()