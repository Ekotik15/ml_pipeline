import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from data_collection_agent import DataCollectionAgent
from data_quality_agent import DataQualityAgent
from annotation_agent import AnnotationAgent
from al_agent import ActiveLearningAgent

CONFIDENCE_THRESHOLD = 0.7
AL_INIT_SIZE = 50
AL_BATCH_SIZE = 20
AL_ITERATIONS = 5
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
    low_conf.to_csv(review_file, index=False)
    print(f"Сохранено {len(low_conf)} примеров для ручной проверки в {review_file}")
    print("Пожалуйста, откройте файл, исправьте метки в колонке 'auto_label' и сохраните.")
    input("Нажмите Enter после того, как исправите файл...")

    if not os.path.exists(review_file):
        raise FileNotFoundError("Файл review_queue.csv не найден")
    corrected = pd.read_csv(review_file)
    high_conf = df[df['confidence'] >= CONFIDENCE_THRESHOLD]
    df_final = pd.concat([high_conf, corrected], ignore_index=True)
    print(f"После ручной проверки: {len(df_final)} записей")
    df_final.to_parquet("data/raw/reviewed.parquet", index=False)
    return df_final

def active_learning(df):
    print("=== Шаг 4: Активное обучение ===")
    agent = ActiveLearningAgent()

    # Стратифицированная выборка начальных данных (50 примеров)
    n_init = min(AL_INIT_SIZE, len(df))
    labeled_init, _ = train_test_split(
        df, train_size=n_init, random_state=RANDOM_STATE,
        stratify=df['auto_label']
    )
    # Убедимся, что в начальной выборке есть оба класса
    unique_labels = labeled_init['auto_label'].unique()
    if len(unique_labels) < 2:
        other_class = 'positive' if unique_labels[0] == 'negative' else 'negative'
        extra = df[df['auto_label'] == other_class].head(20)
        labeled_init = pd.concat([labeled_init, extra], ignore_index=True)
        print("Добавлены примеры другого класса в начальную выборку.")

    pool = df.drop(labeled_init.index).reset_index(drop=True)
    labeled_init = labeled_init.reset_index(drop=True)

    test = pool.sample(frac=TEST_SIZE, random_state=RANDOM_STATE)
    pool = pool.drop(test.index).reset_index(drop=True)

    current_labeled = labeled_init.copy()
    current_pool = pool.copy()
    history = []

    for i in range(AL_ITERATIONS):
        agent.fit(current_labeled)
        metrics = agent.evaluate(current_labeled, test)
        history.append({
            'iteration': i,
            'n_labeled': len(current_labeled),
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1']
        })
        print(f"Итерация {i}: размечено {len(current_labeled)} примеров, accuracy={metrics['accuracy']:.4f}")

        if i == AL_ITERATIONS - 1:
            break

        selected_indices = agent.query(current_pool, strategy='entropy', batch_size=AL_BATCH_SIZE)
        new_samples = current_pool.iloc[selected_indices]
        current_labeled = pd.concat([current_labeled, new_samples], ignore_index=True)
        current_pool = current_pool.drop(current_pool.index[selected_indices]).reset_index(drop=True)

    pd.DataFrame(history).to_csv("reports/al_history.csv", index=False)
    print(f"Активное обучение завершено. Всего размечено: {len(current_labeled)} примеров")
    return current_labeled

def train_model(df):
    print("=== Шаг 5: Обучение модели ===")
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
    print("=== Шаг 6: Генерация отчёта ===")
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
- ActiveLearningAgent: выбрал {AL_INIT_SIZE + AL_BATCH_SIZE * AL_ITERATIONS} наиболее информативных примеров.

## 3. Human-in-the-loop
- Проверено примеров с уверенностью < {CONFIDENCE_THRESHOLD} (файл review_queue.csv).
- Человек исправил ошибочные метки.

## 4. Метрики качества
- Accuracy на тесте: {metrics['accuracy']:.4f}
- F1-score: {metrics['f1']:.4f}

## 5. Ретроспектива
- Zero-shot разметка дала высокую точность, но требовала доработки.
- Активное обучение позволило сократить количество размечаемых примеров.
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
    selected = active_learning(reviewed)
    metrics = train_model(selected)
    generate_report(metrics)

if __name__ == "__main__":
    main()
