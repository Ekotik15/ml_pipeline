import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Union

class ActiveLearningAgent:
    """
    Агент для активного обучения на текстовых данных.
    """
    def __init__(self, model='logreg', vectorizer=None):
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=5000, stop_words='english')
        self.is_fitted = False

    def _preprocess(self, texts):
        if self.is_fitted:
            return self.vectorizer.transform(texts)
        else:
            return self.vectorizer.fit_transform(texts)

    def fit(self, labeled_df: pd.DataFrame, text_col='content', label_col='label'):
        X = self._preprocess(labeled_df[text_col].astype(str))
        y = labeled_df[label_col].astype(int)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, df: pd.DataFrame, text_col='content'):
        X = self._preprocess(df[text_col].astype(str))
        return self.model.predict_proba(X)

    def query(self, pool_df: pd.DataFrame, strategy: str = 'entropy', batch_size: int = 20):
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")
        proba = self.predict_proba(pool_df)
        if proba.shape[1] == 2:
            proba_pos = proba[:, 1]
        else:
            proba_pos = proba

        if strategy == 'entropy':
            uncertainty = 1 - np.abs(proba_pos - 0.5) * 2
        elif strategy == 'margin':
            margin = np.abs(proba_pos - 0.5) * 2
            uncertainty = 1 - margin
        elif strategy == 'random':
            uncertainty = np.random.random(len(pool_df))
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")

        top_indices = np.argsort(uncertainty)[-batch_size:][::-1]
        return top_indices.tolist()

    def evaluate(self, labeled_df: pd.DataFrame, test_df: pd.DataFrame, text_col='content', label_col='label'):
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")
        X_test = self._preprocess(test_df[text_col].astype(str))
        y_true = test_df[label_col].astype(int)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        return {'accuracy': acc, 'f1': f1}

    def run_cycle(self, labeled_df: pd.DataFrame, pool_df: pd.DataFrame, test_df: pd.DataFrame,
                  strategy: str = 'entropy', n_iterations: int = 5, batch_size: int = 20):
        history = []
        current_labeled = labeled_df.copy()
        current_pool = pool_df.copy()

        for i in range(n_iterations):
            self.fit(current_labeled)
            metrics = self.evaluate(current_labeled, test_df)
            history.append({
                'iteration': i,
                'n_labeled': len(current_labeled),
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1']
            })
            if i == n_iterations - 1:
                break
            selected_indices = self.query(current_pool, strategy=strategy, batch_size=batch_size)
            new_samples = current_pool.iloc[selected_indices]
            current_labeled = pd.concat([current_labeled, new_samples], ignore_index=True)
            current_pool = current_pool.drop(current_pool.index[selected_indices]).reset_index(drop=True)

        return history

    def report(self, strategies: Dict[str, List[Dict]]):
        """
        Строит график качества vs количество размеченных данных.
        strategies: словарь {имя_стратегии: история}
        """
        plt.figure(figsize=(10, 6))
        for label, hist in strategies.items():
            n_labeled = [h['n_labeled'] for h in hist]
            acc = [h['accuracy'] for h in hist]
            plt.plot(n_labeled, acc, marker='o', label=label)

        plt.xlabel('Количество размеченных примеров')
        plt.ylabel('Accuracy')
        plt.title('Кривые обучения: сравнение стратегий')
        plt.legend()
        plt.grid(True)
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/learning_curve.png')
        plt.show()
