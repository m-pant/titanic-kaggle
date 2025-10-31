#!/usr/bin/env python3
"""
Комплексное сравнение всех ML методов для датасета Titanic
Автор: Claude AI Assistant
Описание: Полный pipeline для загрузки данных, предобработки, обучения и сравнения моделей
"""

import numpy as np
import pandas as pd
import warnings
import time
from datetime import datetime

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)

# Модели
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Проверка и установка дополнительных библиотек
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost не установлен. Установите: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM не установлен. Установите: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost не установлен. Установите: pip install catboost")

warnings.filterwarnings('ignore')

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-darkgrid')

# ======================== ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ ========================

def load_titanic_data():
    """
    Загрузка данных Titanic с Kaggle или из локального файла
    """
    try:
        # Попытка загрузить локальные файлы
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        print("✅ Данные загружены из локальных файлов")
    except FileNotFoundError:
        print("📥 Загружаю данные с GitHub (публичный датасет)...")
        train_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        try:
            train_df = pd.read_csv(train_url)
            # Для GitHub версии нужно разделить на train/test вручную
            test_df = None
            print("✅ Данные загружены с GitHub")
        except:
            print("❌ Ошибка загрузки. Скачайте train.csv с Kaggle: https://www.kaggle.com/c/titanic")
            return None, None
    
    return train_df, test_df


def preprocess_titanic_data(train_df, test_df=None):
    """
    Полная предобработка данных Titanic
    """
    print("\n🔧 ПРЕДОБРАБОТКА ДАННЫХ")
    print("="*60)
    
    # Объединяем для единой обработки (если есть test)
    if test_df is not None:
        df_all = pd.concat([train_df, test_df], sort=False)
        len_train = len(train_df)
    else:
        df_all = train_df.copy()
        len_train = len(train_df)
    
    # ========== 1. FEATURE ENGINEERING ==========
    
    # Размер семьи
    df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1
    
    # Путешествует один?
    df_all['IsAlone'] = (df_all['FamilySize'] == 1).astype(int)
    
    # Титул из имени
    df_all['Title'] = df_all['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Группировка редких титулов
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Don': 'Rare', 'Dona': 'Rare', 'Lady': 'Rare', 'Countess': 'Rare',
        'Capt': 'Rare', 'Sir': 'Rare', 'Jonkheer': 'Rare'
    }
    df_all['Title'] = df_all['Title'].map(title_mapping).fillna('Rare')
    
    # Deck из каюты
    df_all['Deck'] = df_all['Cabin'].str[0]
    df_all['Deck'] = df_all['Deck'].fillna('Unknown')
    
    # Группы возрастов
    df_all['AgeGroup'] = pd.cut(df_all['Age'], 
                                bins=[0, 12, 18, 35, 60, 100],
                                labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Группы тарифов
    df_all['FareGroup'] = pd.qcut(df_all['Fare'].fillna(df_all['Fare'].median()), 
                                  q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # ========== 2. ЗАПОЛНЕНИЕ ПРОПУСКОВ ==========
    
    # Возраст - заполняем медианой по группам
    age_medians = df_all.groupby(['Title', 'Pclass'])['Age'].transform('median')
    df_all['Age'] = df_all['Age'].fillna(age_medians)
    df_all['Age'] = df_all['Age'].fillna(df_all['Age'].median())
    
    # Embarked - заполняем модой
    df_all['Embarked'] = df_all['Embarked'].fillna(df_all['Embarked'].mode()[0])
    
    # Fare - заполняем медианой по классу
    fare_medians = df_all.groupby('Pclass')['Fare'].transform('median')
    df_all['Fare'] = df_all['Fare'].fillna(fare_medians)
    
    # ========== 3. КОДИРОВАНИЕ КАТЕГОРИЙ ==========
    
    # Label Encoding для бинарных
    le = LabelEncoder()
    df_all['Sex'] = le.fit_transform(df_all['Sex'])
    
    # One-Hot Encoding для остальных категорий
    categorical_features = ['Embarked', 'Title', 'Deck', 'AgeGroup', 'FareGroup']
    df_all = pd.get_dummies(df_all, columns=categorical_features, drop_first=False)
    
    # ========== 4. ВЫБОР ПРИЗНАКОВ ==========
    
    # Удаляем ненужные колонки
    drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_all = df_all.drop([col for col in drop_columns if col in df_all.columns], axis=1)
    
    # ========== 5. РАЗДЕЛЕНИЕ ОБРАТНО ==========
    
    if test_df is not None:
        train = df_all[:len_train].copy()
        test = df_all[len_train:].copy()
        
        # Убираем Survived из test если есть
        if 'Survived' in test.columns:
            test = test.drop('Survived', axis=1)
        
        return train, test
    else:
        return df_all, None


def prepare_features(df):
    """
    Подготовка X и y из датафрейма
    """
    if 'Survived' in df.columns:
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        return X, y
    else:
        return df, None


# ======================== МОДЕЛИ И ИХ НАСТРОЙКИ ========================

class ModelComparison:
    """
    Класс для сравнения всех моделей
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self._initialize_models()
        self.results = []
        
    def _initialize_models(self):
        """
        Инициализация всех моделей с оптимальными параметрами для Titanic
        """
        models = {}
        
        # 1. Логистическая регрессия
        models['Logistic Regression'] = {
            'model': LogisticRegression(
                C=0.1, penalty='l2', max_iter=1000,
                random_state=self.random_state
            ),
            'needs_scaling': True
        }
        
        # 2. Random Forest
        models['Random Forest'] = {
            'model': RandomForestClassifier(
                n_estimators=100, max_depth=5,
                min_samples_split=10, min_samples_leaf=5,
                max_features='sqrt', random_state=self.random_state
            ),
            'needs_scaling': False
        }
        
        # 3. Gradient Boosting
        models['Gradient Boosting'] = {
            'model': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=3,
                min_samples_split=10, min_samples_leaf=5,
                subsample=0.8, random_state=self.random_state
            ),
            'needs_scaling': False
        }
        
        # 4. XGBoost
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = {
                'model': XGBClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=3,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    gamma=1, reg_alpha=1, reg_lambda=2,
                    random_state=self.random_state,
                    use_label_encoder=False, eval_metric='logloss'
                ),
                'needs_scaling': False
            }
        
        # 5. LightGBM
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': LGBMClassifier(
                    n_estimators=100, num_leaves=15, learning_rate=0.05,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1, reg_lambda=2,
                    random_state=self.random_state, verbose=-1
                ),
                'needs_scaling': False
            }
        
        # 6. CatBoost
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = {
                'model': CatBoostClassifier(
                    iterations=100, learning_rate=0.05, depth=4,
                    l2_leaf_reg=5, border_count=32,
                    random_state=self.random_state, verbose=False
                ),
                'needs_scaling': False
            }
        
        # 7. SVM
        models['SVM'] = {
            'model': SVC(
                C=1.0, kernel='rbf', gamma='scale',
                probability=True, random_state=self.random_state
            ),
            'needs_scaling': True
        }
        
        # 8. KNN
        models['KNN'] = {
            'model': KNeighborsClassifier(
                n_neighbors=10, weights='distance', metric='minkowski'
            ),
            'needs_scaling': True
        }
        
        # 9. Naive Bayes
        models['Naive Bayes'] = {
            'model': GaussianNB(),
            'needs_scaling': False
        }
        
        # 10. Extra Trees
        models['Extra Trees'] = {
            'model': ExtraTreesClassifier(
                n_estimators=100, max_depth=5,
                min_samples_split=10, min_samples_leaf=5,
                random_state=self.random_state
            ),
            'needs_scaling': False
        }
        
        # 11. AdaBoost
        models['AdaBoost'] = {
            'model': AdaBoostClassifier(
                n_estimators=100, learning_rate=0.5,
                random_state=self.random_state
            ),
            'needs_scaling': False
        }
        
        # 12. Neural Network
        models['Neural Network'] = {
            'model': MLPClassifier(
                hidden_layer_sizes=(50, 30), activation='relu',
                solver='adam', alpha=0.01, learning_rate='adaptive',
                max_iter=1000, early_stopping=True,
                validation_fraction=0.2, random_state=self.random_state
            ),
            'needs_scaling': True
        }
        
        return models
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test=None, cv_folds=5):
        """
        Обучение и оценка всех моделей
        """
        print("\n🔬 ТЕСТИРОВАНИЕ ВСЕХ МОДЕЛЕЙ")
        print("="*115)
        print(f"{'Модель':<25} {'CV AUC':>10} {'Val ACC':>10} {'Val AUC':>10} {'Kaggle ACC':>12} {'Время (с)':>12}")
        print("-" * 115)

        # Масштабирование данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)

        # Загрузка реальных данных для Kaggle
        real_data = None
        if X_test is not None:
            try:
                real_data = pd.read_csv('cheats/_correct_submission.csv')
            except FileNotFoundError:
                print("⚠️ Файл 'cheats/_correct_submission.csv' не найден. Реальная точность Kaggle не будет рассчитана.")

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for name, model_info in self.models.items():
            start_time = time.time()

            # Выбираем данные
            if model_info['needs_scaling']:
                X_tr, X_vl = X_train_scaled, X_val_scaled
            else:
                X_tr, X_vl = X_train, X_val

            try:
                # Кросс-валидация
                cv_scores = cross_val_score(
                    model_info['model'], X_tr, y_train,
                    cv=skf, scoring='roc_auc', n_jobs=-1
                )

                # Обучение
                model_info['model'].fit(X_tr, y_train)

                # Предсказания
                y_pred = model_info['model'].predict(X_vl)
                y_proba = model_info['model'].predict_proba(X_vl)[:, 1]

                # Метрики
                val_acc = accuracy_score(y_val, y_pred)
                val_auc = roc_auc_score(y_val, y_proba)

                # Real KAGGLE accuracy (проверка на реальных данных)
                real_accuracy = np.nan
                if X_test is not None and real_data is not None:
                    if model_info['needs_scaling']:
                        X_ts = X_test_scaled
                    else:
                        X_ts = X_test
                    y_test_pred = model_info['model'].predict(X_ts)
                    if len(real_data) == len(y_test_pred):
                        real_accuracy = (real_data["Survived"] == y_test_pred).mean()

                elapsed_time = time.time() - start_time

                # Сохранение результатов
                self.results.append({
                    'Model': name,
                    'CV_AUC_Mean': cv_scores.mean(),
                    'CV_AUC_Std': cv_scores.std(),
                    'Val_Accuracy': val_acc,
                    'Val_AUC': val_auc,
                    'Kaggle_Accuracy': real_accuracy,
                    'Training_Time': elapsed_time,
                    'model_object': model_info['model']
                })

                print(f"{name:<25} {cv_scores.mean():>10.4f} {val_acc:>10.4f} {val_auc:>10.4f} {real_accuracy:>12.4f} {elapsed_time:>12.2f}")

            except Exception as e:
                print(f"{name:<25} {'ОШИБКА':<10} - {str(e)[:50]}")

        print("-" * 115)

        # Создаем DataFrame с результатами
        self.results_df = pd.DataFrame(self.results)
        self.results_df = self.results_df.sort_values('Val_AUC', ascending=False)

        return self.results_df
    
    def plot_results(self):
        """
        Визуализация результатов
        """
        if not self.results:
            print("❌ Нет результатов для отображения")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        df = self.results_df
        
        # 1. Сравнение AUC
        ax1 = axes[0, 0]
        x_pos = np.arange(len(df))
        ax1.barh(x_pos, df['Val_AUC'], color='skyblue', label='Validation AUC')
        ax1.barh(x_pos, df['CV_AUC_Mean'], alpha=0.6, color='orange', label='CV AUC Mean')
        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(df['Model'])
        ax1.set_xlabel('AUC Score')
        ax1.set_title('Сравнение AUC моделей')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Accuracy vs AUC
        ax2 = axes[0, 1]
        ax2.scatter(df['Val_Accuracy'], df['Val_AUC'], s=100, alpha=0.6)
        for i, txt in enumerate(df['Model']):
            ax2.annotate(txt, (df['Val_Accuracy'].iloc[i], df['Val_AUC'].iloc[i]),
                        fontsize=8, ha='right')
        ax2.set_xlabel('Validation Accuracy')
        ax2.set_ylabel('Validation AUC')
        ax2.set_title('Accuracy vs AUC')
        ax2.grid(alpha=0.3)
        
        # 3. Время обучения
        ax3 = axes[1, 0]
        ax3.barh(x_pos, df['Training_Time'], color='green', alpha=0.6)
        ax3.set_yticks(x_pos)
        ax3.set_yticklabels(df['Model'])
        ax3.set_xlabel('Время обучения (секунды)')
        ax3.set_title('Скорость обучения моделей')
        ax3.grid(alpha=0.3)
        
        # 4. Топ-5 моделей
        ax4 = axes[1, 1]
        top5 = df.head(5)
        colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen']
        bars = ax4.bar(range(len(top5)), top5['Val_AUC'], color=colors)
        ax4.set_xticks(range(len(top5)))
        ax4.set_xticklabels(top5['Model'], rotation=45, ha='right')
        ax4.set_ylabel('Validation AUC')
        ax4.set_title('Топ-5 моделей по AUC')
        ax4.set_ylim([top5['Val_AUC'].min() - 0.01, top5['Val_AUC'].max() + 0.01])
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, top5['Val_AUC']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.suptitle('Результаты сравнения ML моделей на Titanic', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_ensemble(self, top_n=3):
        """
        Создание ансамбля из лучших моделей
        """
        if len(self.results_df) < top_n:
            print(f"⚠️ Недостаточно моделей для ансамбля. Доступно: {len(self.results_df)}")
            return None
        
        # Берем топ-N моделей
        top_models = self.results_df.head(top_n)
        
        # Создаем Voting Classifier
        estimators = [(row['Model'], row['model_object']) 
                     for _, row in top_models.iterrows()]
        
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        
        print(f"\n🎯 Создан ансамбль из топ-{top_n} моделей:")
        for model_name in top_models['Model']:
            print(f"  - {model_name}")
        
        return voting_clf
    
    def get_best_model(self):
        """
        Получение лучшей модели
        """
        if self.results_df.empty:
            print("❌ Нет обученных моделей")
            return None
        
        best = self.results_df.iloc[0]
        print(f"\n🏆 Лучшая модель: {best['Model']}")
        print(f"   Val AUC: {best['Val_AUC']:.4f}")
        print(f"   Val Accuracy: {best['Val_Accuracy']:.4f}")
        print(f"   CV AUC: {best['CV_AUC_Mean']:.4f} (±{best['CV_AUC_Std']:.4f})")
        
        return best['model_object']


# ======================== ПРОДВИНУТЫЕ МЕТОДЫ ========================

class AdvancedEnsembles:
    """
    Продвинутые ансамблевые методы
    """
    
    def __init__(self, base_models, random_state=42):
        self.base_models = base_models
        self.random_state = random_state
    
    def create_stacking_ensemble(self, X_train, y_train):
        """
        Stacking с мета-моделью
        """
        # Берем топ-4 базовые модели
        estimators = list(self.base_models.items())[:4]
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=0.1),
            cv=5,
            n_jobs=-1
        )
        
        print("\n📚 Обучаю Stacking ансамбль...")
        stacking_clf.fit(X_train, y_train)
        
        return stacking_clf
    
    def create_blending_ensemble(self, models, weights=None):
        """
        Blending - взвешенное голосование
        """
        class BlendingClassifier:
            def __init__(self, models, weights=None):
                self.models = models
                self.weights = weights if weights else [1/len(models)] * len(models)
            
            def fit(self, X, y):
                for name, model in self.models:
                    model.fit(X, y)
                return self
            
            def predict_proba(self, X):
                predictions = np.zeros((X.shape[0], 2))
                for (name, model), weight in zip(self.models, self.weights):
                    predictions += model.predict_proba(X) * weight
                return predictions
            
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        
        return BlendingClassifier(models, weights)


# ======================== HYPERPARAMETER TUNING ========================

def tune_best_model(model_name, X_train, y_train, X_val, y_val):
    """
    Тонкая настройка лучшей модели
    """
    print(f"\n🎯 Тонкая настройка {model_name}...")
    
    param_grids = {
        'LightGBM': {
            'n_estimators': [100, 150, 200],
            'num_leaves': [10, 15, 20, 25],
            'learning_rate': [0.03, 0.05, 0.08],
            'min_child_samples': [15, 20, 25],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        'XGBoost': {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.03, 0.05, 0.08],
            'min_child_weight': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        'CatBoost': {
            'iterations': [100, 150, 200],
            'depth': [3, 4, 5],
            'learning_rate': [0.03, 0.05, 0.08],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
    }
    
    if model_name not in param_grids:
        print(f"⚠️ Параметры для {model_name} не определены")
        return None
    
    # Используем RandomizedSearch для скорости
    if model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
        base_model = LGBMClassifier(random_state=42, verbose=-1)
    elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
        base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif model_name == 'CatBoost' and CATBOOST_AVAILABLE:
        base_model = CatBoostClassifier(random_state=42, verbose=False)
    else:
        return None
    
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grids[model_name],
        n_iter=20,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Лучшие параметры: {random_search.best_params_}")
    print(f"   Лучший CV Score: {random_search.best_score_:.4f}")
    
    # Проверка на валидации
    y_val_pred = random_search.predict(X_val)
    y_val_proba = random_search.predict_proba(X_val)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"   Val Accuracy: {val_acc:.4f}")
    print(f"   Val AUC: {val_auc:.4f}")
    
    return random_search.best_estimator_


# ======================== ГЛАВНАЯ ФУНКЦИЯ ========================

def main():
    """
    Основной pipeline
    """
    print("="*100)
    print(" "*30 + "🚢 TITANIC ML COMPARISON 🚢")
    print("="*100)
    print(f"Запущено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Загрузка данных
    print("\n📁 ЗАГРУЗКА ДАННЫХ")
    print("-"*60)
    train_df, test_df = load_titanic_data()
    
    if train_df is None:
        print("❌ Не удалось загрузить данные")
        return
    
    print(f"   Размер train: {train_df.shape}")
    if test_df is not None:
        print(f"   Размер test: {test_df.shape}")
    
    # 2. Предобработка
    train_processed, test_processed = preprocess_titanic_data(train_df, test_df)
    
    # 3. Подготовка данных
    X, y = prepare_features(train_processed)
    X_test = None
    if test_processed is not None:
        X_test, _ = prepare_features(test_processed)

    print(f"\n   Количество признаков после обработки: {X.shape[1]}")
    print(f"   Распределение классов: ")
    print(f"      - Выжившие: {y.sum()} ({y.mean():.1%})")
    print(f"      - Погибшие: {len(y) - y.sum()} ({1-y.mean():.1%})")
    
    # 4. Разделение на train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y
    )
    
    print(f"\n   Train set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    if X_test is not None:
        print(f"   Test set: {X_test.shape}")
    
    # 5. Сравнение моделей
    comparison = ModelComparison(random_state=42)
    results_df = comparison.train_and_evaluate(X_train, y_train, X_val, y_val, X_test=X_test)
    
    # 6. Визуализация результатов
    print("\n📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("-"*60)
    comparison.plot_results()
    
    # 7. Вывод топ-5 моделей
    print("\n🏆 ТОП-5 МОДЕЛЕЙ ПО VALIDATION AUC:")
    print("-"*60)
    top5 = results_df.head(5)
    for idx, (_, row) in enumerate(top5.iterrows(), 1):
        medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][idx-1]
        print(f"{medal} {row['Model']:<20} AUC: {row['Val_AUC']:.4f} | "
              f"Accuracy: {row['Val_Accuracy']:.4f} | "
              f"Время: {row['Training_Time']:.2f}с")
    
    # 8. Лучшая модель
    best_model = comparison.get_best_model()
    best_model_name = results_df.iloc[0]['Model']
    
    # 9. Создание ансамблей
    print("\n🔧 СОЗДАНИЕ АНСАМБЛЕЙ")
    print("-"*60)
    
    # Voting ensemble
    voting_ensemble = comparison.create_ensemble(top_n=3)
    if voting_ensemble:
        voting_ensemble.fit(X_train, y_train)
        y_voting_pred = voting_ensemble.predict(X_val)
        y_voting_proba = voting_ensemble.predict_proba(X_val)[:, 1]
        voting_acc = accuracy_score(y_val, y_voting_pred)
        voting_auc = roc_auc_score(y_val, y_voting_proba)
        print(f"\n   Voting Ensemble - AUC: {voting_auc:.4f} | Accuracy: {voting_acc:.4f}")
    
    # 10. Тонкая настройка лучшей модели
    if best_model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
        tuned_model = tune_best_model(best_model_name, X_train, y_train, X_val, y_val)
    
    # 11. Финальные рекомендации
    print("\n" + "="*100)
    print(" "*35 + "📋 ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print("="*100)
    
    print("\n1. ДЛЯ МАКСИМАЛЬНОЙ ТОЧНОСТИ:")
    print(f"   Используйте {results_df.iloc[0]['Model']} с AUC = {results_df.iloc[0]['Val_AUC']:.4f}")
    
    print("\n2. ДЛЯ БАЛАНСА СКОРОСТЬ/КАЧЕСТВО:")
    # Находим модель с лучшим соотношением
    results_df['Score_per_second'] = results_df['Val_AUC'] / results_df['Training_Time']
    best_efficiency = results_df.nlargest(1, 'Score_per_second').iloc[0]
    print(f"   Используйте {best_efficiency['Model']} (AUC = {best_efficiency['Val_AUC']:.4f}, "
          f"Время = {best_efficiency['Training_Time']:.2f}с)")
    
    print("\n3. ДЛЯ ПРОДАКШЕНА:")
    print("   Рекомендуется Voting ансамбль из топ-3 моделей или")
    print("   тонко настроенная версия лучшей модели")
    
    print("\n4. ДЛЯ ЭКСПЕРИМЕНТОВ:")
    if LIGHTGBM_AVAILABLE:
        print("   LightGBM - быстрый и эффективный для итераций")
    else:
        print("   Random Forest - стабильный и предсказуемый")
    
    # Сохранение результатов
    print("\n💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-"*60)
    
    # Сохраняем таблицу с результатами
    results_df.drop('model_object', axis=1).to_csv('titanic_model_comparison_results.csv', index=False)
    print("✅ Результаты сохранены в 'titanic_model_comparison_results.csv'")
    
    # Сохраняем лучшую модель
    try:
        import joblib
        joblib.dump(best_model, 'best_titanic_model.pkl')
        print(f"✅ Лучшая модель ({best_model_name}) сохранена в 'best_titanic_model.pkl'")
    except:
        print("⚠️ Для сохранения модели установите joblib: pip install joblib")
    
    print("\n" + "="*100)
    print(" "*40 + "✨ ГОТОВО! ✨")
    print("="*100)
    
    return results_df, best_model


if __name__ == "__main__":
    # Запуск основной программы
    results, model = main()
    
    # Дополнительный интерактивный режим
    print("\n💡 Совет: Для запуска отдельных экспериментов используйте функции из этого скрипта")
    print("   Например: comparison = ModelComparison() для создания нового сравнения")
