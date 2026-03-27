import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DataQualityAgent:
    """Агент для детекции и исправления проблем качества данных."""

    def __init__(self):
        pass

    def detect_issues(self, df: pd.DataFrame) -> dict:
        """
        Возвращает отчёт о проблемах:
        - missing: словарь с количеством пропусков по колонкам
        - duplicates: количество полных дубликатов строк
        - outliers: словарь с количеством выбросов по числовым колонкам (IQR)
        - imbalance: словарь с распределением классов (есть ли колонка 'label')
        """
        report = {}

        # 1. Пропущенные значения
        missing = df.isnull().sum().to_dict()
        report['missing'] = {col: int(val) for col, val in missing.items() if val > 0}

        # 2. Дубликаты строк
        report['duplicates'] = int(df.duplicated().sum())

        # 3. Выбросы (IQR) для числовых колонок
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers_count > 0:
                outliers[col] = int(outliers_count)
        report['outliers'] = outliers

        # 4. Дисбаланс классов (если есть колонка 'label')
        if 'label' in df.columns:
            class_counts = df['label'].value_counts().to_dict()
            total = len(df)
            imbalance = {str(cls): {'count': cnt, 'ratio': cnt/total} for cls, cnt in class_counts.items()}
            report['imbalance'] = imbalance
        else:
            report['imbalance'] = None

        return report

    def fix(self, df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        """
        Применяет стратегии очистки:
        - strategy['missing']: 'drop', 'median' (для числовых + mode для строковых), 'constant'
        - strategy['duplicates']: 'drop' или 'keep_first'
        - strategy['outliers']: 'clip_iqr' (ограничение), 'drop'
        """
        df_clean = df.copy()

        # 1. Обработка пропусков
        if 'missing' in strategy:
            missing_method = strategy['missing']
            if missing_method == 'drop':
                df_clean.dropna(inplace=True)
            elif missing_method == 'median':
                for col in df_clean.select_dtypes(include=[np.number]).columns:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                for col in df_clean.select_dtypes(include=['object']).columns:
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col].fillna(mode_val[0], inplace=True)
                    else:
                        df_clean[col].fillna('unknown', inplace=True)
            elif missing_method == 'constant':
                df_clean.fillna('missing', inplace=True)
            else:
                raise ValueError(f"Неизвестный метод обработки пропусков: {missing_method}")

        # 2. Обработка дубликатов
        if 'duplicates' in strategy:
            dup_method = strategy['duplicates']
            if dup_method == 'drop':
                df_clean.drop_duplicates(inplace=True)
            elif dup_method == 'keep_first':
                df_clean.drop_duplicates(keep='first', inplace=True)
            else:
                raise ValueError(f"Неизвестный метод обработки дубликатов: {dup_method}")

        # 3. Обработка выбросов
        if 'outliers' in strategy:
            outlier_method = strategy['outliers']
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            if outlier_method == 'clip_iqr':
                for col in numeric_cols:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df_clean[col] = df_clean[col].clip(lower, upper)
            elif outlier_method == 'drop':
                for col in numeric_cols:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    mask = (df_clean[col] >= lower) & (df_clean[col] <= upper)
                    df_clean = df_clean[mask]
            else:
                raise ValueError(f"Неизвестный метод обработки выбросов: {outlier_method}")

        return df_clean

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        """Сравнивает метрики качества до и после."""
        report_before = self.detect_issues(df_before)
        report_after = self.detect_issues(df_after)

        comparison = {
            'Metric': ['Missing values (total)', 'Duplicate rows', 'Outliers (total)'],
            'Before': [sum(report_before['missing'].values()), report_before['duplicates'], sum(report_before['outliers'].values())],
            'After': [sum(report_after['missing'].values()), report_after['duplicates'], sum(report_after['outliers'].values())]
        }

        if report_before['imbalance'] is not None:
            ratios_before = [v['ratio'] for v in report_before['imbalance'].values()]
            ratios_after = [v['ratio'] for v in report_after['imbalance'].values()] if report_after['imbalance'] else [0]
            comparison['Metric'].append('Class imbalance (min ratio)')
            comparison['Before'].append(min(ratios_before) if ratios_before else 0)
            comparison['After'].append(min(ratios_after) if ratios_after else 0)

        return pd.DataFrame(comparison)
