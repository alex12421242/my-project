import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from your_script import (
    load_csv,
    load_excel,
    validate_data,
    scale_numeric_features,
    encode_categorical,
    get_categorical_columns,
    calculate_statistics,
    train_regression_model,
    train_classification_model,
    save_analysis_result,
    save_model_parameters
)
import os
import statsmodels.api as sm  # Для декомпозиции временных рядов

# Реализация функции анализа временного ряда
def analyze_time_series(df, date_col, numeric_col, period=None):
    """
    Анализ временного ряда с использованием сезонной декомпозиции.
    Удаляет пропуски перед декомпозицией.
    """
    series = df.set_index(date_col)[numeric_col]
    series = series.dropna()
    decomposition = sm.tsa.seasonal_decompose(series, model='additive', period=period)
    return decomposition

class TestYourScript(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'id': [1, 2, 2, 4],
            'numeric_col': [1.0, 2.5, np.nan, 4.0],
            'category': ['A', 'B', 'A', 'C'],
            'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04'])
        })

    def test_load_csv_success(self):
        test_csv = 'test.csv'
        self.df.to_csv(test_csv, index=False)
        df_loaded = load_csv(test_csv)
        self.assertIsInstance(df_loaded, pd.DataFrame)
        os.remove(test_csv)

    def test_load_csv_failure(self):
        df_none = load_csv('nonexistent.csv')
        self.assertIsNone(df_none)

    def test_validate_data_drop(self):
        df_valid, report = validate_data(self.df.copy(), key_columns=['id'])
        self.assertIn('duplicates', report)
        self.assertEqual(report['duplicates'], 1)

    def test_scale_numeric_features_minmax(self):
        scaled = scale_numeric_features(self.df.copy(), ['numeric_col'], method='minmax')
        self.assertAlmostEqual(scaled['numeric_col'].min(), 0)
        self.assertAlmostEqual(scaled['numeric_col'].max(), 1)

    def test_encode_categorical_onehot(self):
        df_enc = encode_categorical(self.df.copy(), ['category'], method='onehot')
        self.assertIn('category_B', df_enc.columns)
        self.assertIn('category_C', df_enc.columns)

    def test_get_categorical_columns(self):
        cats = get_categorical_columns(self.df, threshold=1.0)
        self.assertIn('category', cats)

    def test_calculate_statistics(self):
        stats = calculate_statistics(self.df, ['numeric_col'])
        self.assertIn('mean', stats['numeric_col'])

    def test_analyze_time_series(self):
        result = analyze_time_series(self.df.copy(), 'date', 'numeric_col', period=1)
        self.assertIsNotNone(result)
        # Проверка наличия атрибута 'trend'
        self.assertTrue(hasattr(result, 'trend'))

    def test_train_regression_model(self):
        df = pd.DataFrame({
            'feat1': [1, 2, 3, 4],
            'feat2': [10, 20, 30, 40],
            'target': [2, 4, 6, 8]
        })
        model = train_regression_model(df, ['feat1', 'feat2'], 'target')
        self.assertIsNotNone(model)

    def test_train_classification_model(self):
        df = pd.DataFrame({
            'feat1': [1, 2, 3, 4],
            'feat2': [10, 20, 30, 40],
            'type': [0, 1, 0, 1]
        })
        model = train_classification_model(df, ['feat1', 'feat2'], 'type')
        self.assertIsNotNone(model)

    def test_save_analysis_result_and_model_parameters(self):
        df = pd.DataFrame({'col': [1, 2]})
        save_analysis_result('test_table', df)
        save_model_parameters(LinearRegression(), 'test_model_params', ['feat1', 'feat2'])

if __name__ == '__main__':
    unittest.main()