import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Optional, Union
from sklearn.metrics import cohen_kappa_score, accuracy_score
from transformers import pipeline
from datetime import datetime

class AnnotationAgent:
    """
    Агент для автоматической разметки текстовых данных.
    """
    def __init__(self, modality: str = 'text', model_name: str = 'typeform/distilbert-base-uncased-mnli'):
        """
        Инициализация агента.
        modality: 'text' (пока только текст)
        model_name: имя модели для zero-shot классификации
        """
        if modality != 'text':
            raise NotImplementedError("Пока поддерживается только текстовая модальность.")
        self.modality = modality
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=-1)  # cpu
        self.spec = None

    def auto_label(self, df: pd.DataFrame, candidate_labels: List[str], hypothesis_template: str = "This example is {}.") -> pd.DataFrame:
        """
        Автоматическая разметка текстов.
        df: должен содержать колонку 'content' с текстом.
        candidate_labels: список возможных меток (например, ['positive', 'negative']).
        hypothesis_template: шаблон для zero-shot (подставляется метка).
        Возвращает df с добавленными колонками:
          - 'auto_label': предсказанная метка
          - 'confidence': уверенность (вероятность для предсказанной метки)
        """
        if 'content' not in df.columns:
            raise ValueError("DataFrame должен содержать колонку 'content'")
        texts = df['content'].astype(str).tolist()
        predictions = self.classifier(texts, candidate_labels, hypothesis_template=hypothesis_template)
        auto_labels = [p['labels'][0] for p in predictions]
        confidences = [p['scores'][0] for p in predictions]
        df_result = df.copy()
        df_result['auto_label'] = auto_labels
        df_result['confidence'] = confidences
        return df_result

    def generate_spec(self, df: pd.DataFrame, task: str, classes: Dict[str, str], examples: Dict[str, List[str]]) -> str:
        """
        Генерирует спецификацию разметки в формате Markdown.
        task: описание задачи (например, "Классификация отзывов на фильмы")
        classes: словарь {метка: определение}
        examples: словарь {метка: [примеры текстов]}
        Возвращает путь к сохранённому файлу.
        """
        spec_lines = []
        spec_lines.append(f"# Спецификация разметки\n\n## Задача\n{task}\n")
        spec_lines.append("## Классы\n")
        for label, definition in classes.items():
            spec_lines.append(f"### {label}\n{definition}\n")
        spec_lines.append("## Примеры\n")
        for label, ex_list in examples.items():
            spec_lines.append(f"### {label}\n")
            for ex in ex_list:
                spec_lines.append(f"- {ex}\n")
            spec_lines.append("\n")
        spec_lines.append("## Граничные случаи\n")
        spec_lines.append("- Отзывы, содержащие сарказм или иронию.\n")
        spec_lines.append("- Смешанные эмоции (например, положительные по сюжету, но отрицательные по актёрам).\n")
        spec_lines.append("- Короткие отзывы (менее 10 слов) – классифицировать по тону.\n")
        spec_lines.append("- Отзывы на неанглийском языке – требовать перевода.\n")
        spec = "".join(spec_lines)

        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/annotation_spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(spec)
        self.spec = filename
        return filename

    def check_quality(self, df_labeled: pd.DataFrame, true_label_col: str = 'label', pred_label_col: str = 'auto_label', confidence_col: str = 'confidence') -> Dict:
        """
        Оценивает качество разметки.
        df_labeled: DataFrame с колонками true_label_col, pred_label_col, confidence_col.
        Возвращает словарь с метриками:
          - kappa: коэффициент Коэна (если метки бинарные/мультикласс)
          - accuracy: доля правильных ответов (если есть true метки)
          - label_distribution: распределение предсказанных меток
          - confidence_mean: средняя уверенность
        """
        label_dist = df_labeled[pred_label_col].value_counts().to_dict()
        conf_mean = df_labeled[confidence_col].mean() if confidence_col in df_labeled else None

        kappa = None
        accuracy = None
        if true_label_col in df_labeled.columns:
            y_true = df_labeled[true_label_col].astype(str)
            y_pred = df_labeled[pred_label_col].astype(str)
            accuracy = accuracy_score(y_true, y_pred)
            try:
                kappa = cohen_kappa_score(y_true, y_pred)
            except:
                kappa = None
        return {
            'kappa': kappa,
            'accuracy': accuracy,
            'label_distribution': label_dist,
            'confidence_mean': conf_mean
        }

    def export_to_labelstudio(self, df_labeled: pd.DataFrame, text_col: str = 'content', label_col: str = 'auto_label', confidence_col: str = 'confidence') -> str:
        """
        Экспортирует данные в формате LabelStudio JSON.
        Формат соответствует импорту в LabelStudio (tasks).
        Возвращает путь к сохранённому файлу.
        """
        tasks = []
        for idx, row in df_labeled.iterrows():
            task = {
                "data": {
                    "text": row[text_col]
                },
                "predictions": [{
                    "result": [{
                        "from_name": "sentiment",
                        "to_name": "text",
                        "type": "choices",
                        "value": {"choices": [row[label_col]]}
                    }],
                    "score": row[confidence_col] if confidence_col in row else 1.0,
                    "model_version": "zero-shot-bart"
                }]
            }
            tasks.append(task)

        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/labelstudio_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        return filename

    # Бонус: human-in-the-loop
    def export_low_confidence(self, df_labeled: pd.DataFrame, threshold: float = 0.7, text_col: str = 'content') -> str:
        """
        Сохраняет примеры с уверенностью ниже threshold в отдельный CSV для ручной разметки.
        """
        low_conf = df_labeled[df_labeled['confidence'] < threshold].copy()
        if low_conf.empty:
            print("Нет примеров с низкой уверенностью.")
            return None
        filename = f"outputs/low_confidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        low_conf[[text_col, 'auto_label', 'confidence']].to_csv(filename, index=False)
        print(f"Сохранено {len(low_conf)} примеров для ручной разметки в {filename}")
        return filename
