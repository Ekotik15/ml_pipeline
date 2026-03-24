import sys
sys.path.insert(0, '.')
import pandas as pd
import glob
from annotation_agent import AnnotationAgent

# Загрузка CSV из data/raw
files = sorted(glob.glob("data/raw/*.csv"))
if not files:
    raise FileNotFoundError("Нет CSV в data/raw/. Скопируйте файл из первого задания.")
df = pd.read_csv(files[-1])
df_sample = df.head(500).copy()
print(f"Загружено строк: {len(df_sample)}")

agent = AnnotationAgent(modality='text')
candidate_labels = ['positive', 'negative']
df_labeled = agent.auto_label(df_sample, candidate_labels)
print("Разметка выполнена.")
print(df_labeled[['content', 'label', 'auto_label', 'confidence']].head())

# Генерация спецификации
classes = {
    'positive': 'Отзыв, в котором пользователь выражает положительное мнение о фильме, хвалит игру актёров, сценарий, режиссуру и т.п.',
    'negative': 'Отзыв, в котором пользователь выражает недовольство фильмом, критикует недостатки, плохую игру, слабый сценарий и т.п.'
}
examples = {
    'positive': [
        "This movie was absolutely fantastic! Great acting and amazing story.",
        "I loved every minute of it. Highly recommend!"
    ],
    'negative': [
        "Waste of time. Terrible plot and boring characters.",
        "I regretted watching this. Poorly made."
    ]
}
spec_path = agent.generate_spec(df_sample,
                                task="Классификация отзывов на фильмы (положительный/отрицательный)",
                                classes=classes,
                                examples=examples)
print("Спецификация сохранена:", spec_path)

# Оценка качества
metrics = agent.check_quality(df_labeled, true_label_col='label', pred_label_col='auto_label', confidence_col='confidence')
print("Метрики качества:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Экспорт в LabelStudio
json_path = agent.export_to_labelstudio(df_labeled)
print("Экспорт LabelStudio сохранён:", json_path)

# Бонус: примеры с низкой уверенностью
low_conf_file = agent.export_low_confidence(df_labeled, threshold=0.7)
if low_conf_file:
    print("Создан файл для ручной разметки:", low_conf_file)
