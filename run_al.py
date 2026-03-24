import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from agents.al_agent import ActiveLearningAgent
import glob

# Загрузка данных
files = sorted(glob.glob("data/raw/*.csv"))
if not files:
    raise FileNotFoundError("Нет CSV в data/raw/")
df = pd.read_csv(files[-1])
df = df[['content', 'label']].dropna()
df['label'] = df['label'].astype(int)

print("Всего записей:", len(df))

# Разбиваем на train+pool и test (80/20)
train_pool, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Начальная размеченная выборка (50) и пул
initial_labeled, pool = train_test_split(train_pool, train_size=50, random_state=42, stratify=train_pool['label'])

print(f"Начальная разметка: {len(initial_labeled)} (positive={sum(initial_labeled['label']==1)})")
print(f"Пул: {len(pool)}")
print(f"Тест: {len(test)}")

# Стратегия entropy
agent_entropy = ActiveLearningAgent()
history_entropy = agent_entropy.run_cycle(
    labeled_df=initial_labeled.copy(),
    pool_df=pool.copy(),
    test_df=test,
    strategy='entropy',
    n_iterations=5,
    batch_size=20
)
print("\nEntropy история:")
for h in history_entropy:
    print(f"Iter {h['iteration']}: n_labeled={h['n_labeled']}, acc={h['accuracy']:.4f}")

# Стратегия random
agent_random = ActiveLearningAgent()
history_random = agent_random.run_cycle(
    labeled_df=initial_labeled.copy(),
    pool_df=pool.copy(),
    test_df=test,
    strategy='random',
    n_iterations=5,
    batch_size=20
)
print("\nRandom история:")
for h in history_random:
    print(f"Iter {h['iteration']}: n_labeled={h['n_labeled']}, acc={h['accuracy']:.4f}")

# Построение графика
agent_entropy.report(strategies={'entropy': history_entropy, 'random': history_random})

print("\nГрафик сохранён в outputs/learning_curve.png")
