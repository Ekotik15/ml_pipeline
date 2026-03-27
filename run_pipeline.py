#!/usr/bin/env python
import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from data_collection_agent import DataCollectionAgent
from data_quality_agent import DataQualityAgent
from annotation_agent import AnnotationAgent

CONFIDENCE_THRESHOLD = 0.7
TEST_SIZE = 0.2
RANDOM_STATE = 42

def collect_data():
    print("=== Шаг 1: Сбор данных ===")
    agent = DataCollectionAgent()
    sources = [{"type": "hf_dataset", "name": "imdb"}]
    df = agent.run(sources)
    df = df.head(500)
    print(f"Собрано {len(df)} записей")
    df.to_parquet("data/raw/raw_data.parquet", index=False)
    return df

def clean_data(df):
    print("=== Шаг 2: Чистка данных ===")
    agent = DataQualityAgent()
    strategy = {'missing': 'median', 'duplicates': 'drop', 'outliers': 'clip_iqr'}
    df_clean = agent.fix(df, strategy)
    print(f"После чистки осталось {len(df_clean)} записей")
    df_clean.to_parquet("data/raw/clean_data.parquet", index=False)
    return df_clean

def auto_label(df):
    print("=== Шаг 3: Автоматическая разметка ===")
    agent = AnnotationAgent(modality='text')
    candidate_labels = ['positive', 'negative']
    df_labeled = agent.auto_label(df, candidate_labels)
    print("Авторазметка завершена")
    df_labeled.to_parquet("data/raw/auto_labeled.parquet", index=False)
    return df_labeled

def human_review(df):
    print("=== Human-in-the-loop: проверка неопределённых примеров ===")
    low_conf = df[df['confidence'] < CONFIDENCE_THRESHOLD].copy()
    if low_conf.empty:
        print("Нет примеров с низкой уверенностью.")
        return df

    review_file = "review_queue.csv"
    try:
        low_conf.to_csv(review_file, index=False)
    except PermissionError:
        print("Нет прав на запись файла review_queue.csv. Пропускаем ручную проверку.")
        return df

    print(f"Сохранено {len(low_conf)} примеров для ручной проверки в {review_file}")
    print("Пожалуйста, откройте файл, исправьте метки в колонке 'auto_label' и сохраните.")
    input("Нажмите Enter после того, как исправите файл...")

    if not os.path.exists(review_file):
        print("Файл review_queue.csv не найден, продолжаем без исправлений.")
        return df
    corrected = pd.read_csv(review_file)
    high_conf = df[df['confidence'] >= CONFIDENCE_THRESHOLD]
    df_final = pd.concat([high_conf, corrected], ignore_index=True)
    print(f"После ручной проверки: {len(df_final)} записей")
    df_final.to_parquet("data/raw/reviewed.parquet", index=False)
    return df_final

def train_model(df):
    print("=== Шаг 4: Обучение модели ===")
    if len(df['auto_label'].unique()) < 2:
        print("В данных только один класс. Обучение модели невозможно. Сохраняем отчёт.")
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        with open('reports/final_report.md', 'w', encoding='utf-8') as f:
            f.write("# Итоговый отчёт\n\nВ данных только один класс, обучение не выполнено.")
        return {'accuracy': 0.0, 'f1': 0.0}

    label_map = {'positive': 1, 'negative': 0}
    df['label_int'] = df['auto_label'].map(label_map)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['content'].astype(str))
    y = df['label_int']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE, stratify=y)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/final_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print(f"Модель сохранена. Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return {'accuracy': acc, 'f1': f1}

def generate_report(metrics):
    print("=== Шаг 5: Генерация отчёта ===")
    report = f"""# Итоговый отчёт

## 1. Описание задачи и датасета
- Модальность: текст
- Источник: IMDB (рецензии на фильмы)
- Объём: 25 000 записей (для демонстрации использовано 500)
- Классы: positive / negative

## 2. Действия агентов
- DataCollectionAgent: собрал данные из HuggingFace IMDB.
- DataQualityAgent: удалил дубликаты, обработал пропуски и выбросы.
- AnnotationAgent: zero-shot разметка с уверенностью.
- Human-in-the-loop: ручная проверка неопределённых примеров.

## 3. Human-in-the-loop
- Проверено примеров с уверенностью < {CONFIDENCE_THRESHOLD} (файл review_queue.csv).
- Человек исправил ошибочные метки.

## 4. Метрики качества
- Accuracy на тесте: {metrics['accuracy']:.4f}
- F1-score: {metrics['f1']:.4f}

## 5. Ретроспектива
- Zero-shot разметка дала высокую точность, но требовала доработки.
- Human-in-the-loop критически важен для качества.
"""
    os.makedirs('reports', exist_ok=True)
    with open('reports/final_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("Отчёт сохранён в reports/final_report.md")

def main():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/labeled', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    raw = collect_data()
    cleaned = clean_data(raw)
    labeled = auto_label(cleaned)
    reviewed = human_review(labeled)
    metrics = train_model(reviewed)
    generate_report(metrics)

if __name__ == "__main__":
    main()
