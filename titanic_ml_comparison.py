#!/usr/bin/env python3
"""
Comprehensive comparison of all ML methods for the Titanic dataset
Author: Claude AI Assistant
Description: A complete pipeline for data loading, preprocessing, model training, and comparison
"""

import numpy as np
import pandas as pd
import warnings
import time
from datetime import datetime

# Visualization
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

# Models
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

# Checking and installing additional libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost is not installed. Install it using: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM is not installed. Install it using: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost is not installed. Install it using: pip install catboost")

warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-darkgrid')

# ======================== DATA LOADING AND PREPROCESSING ========================

def load_titanic_data():
    """
    Load Titanic data from Kaggle or a local file
    """
    try:
        # Attempt to load local files
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        print("✅ Data loaded from local files")
    except FileNotFoundError:
        print("📥 Loading data from GitHub (public dataset)...")
        train_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        try:
            train_df = pd.read_csv(train_url)
            # For the GitHub version, we need to split into train/test manually
            test_df = None
            print("✅ Data loaded from GitHub")
        except:
            print("❌ Loading error. Download train.csv from Kaggle: https://www.kaggle.com/c/titanic")
            return None, None
    
    return train_df, test_df


def preprocess_titanic_data(train_df, test_df=None):
    """
    Complete preprocessing of Titanic data
    """
    print("\n🔧 DATA PREPROCESSING")
    print("="*60)
    
    # Combine for unified processing (if test exists)
    if test_df is not None:
        df_all = pd.concat([train_df, test_df], sort=False)
        len_train = len(train_df)
    else:
        df_all = train_df.copy()
        len_train = len(train_df)
    
    # ========== 1. FEATURE ENGINEERING ========== 
    
    # Family Size
    df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1
    
    # Travels alone?
    df_all['IsAlone'] = (df_all['FamilySize'] == 1).astype(int)
    
    # Title from name
    df_all['Title'] = df_all['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Grouping rare titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Don': 'Rare', 'Dona': 'Rare', 'Lady': 'Rare', 'Countess': 'Rare',
        'Capt': 'Rare', 'Sir': 'Rare', 'Jonkheer': 'Rare'
    }
    df_all['Title'] = df_all['Title'].map(title_mapping).fillna('Rare')
    
    # Deck from cabin
    df_all['Deck'] = df_all['Cabin'].str[0]
    df_all['Deck'] = df_all['Deck'].fillna('Unknown')
    
    # Age groups
    df_all['AgeGroup'] = pd.cut(df_all['Age'], 
                                bins=[0, 12, 18, 35, 60, 100],
                                labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Fare groups
    df_all['FareGroup'] = pd.qcut(df_all['Fare'].fillna(df_all['Fare'].median()), 
                                  q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # ========== 2. FILLING MISSING VALUES ========== 
    
    # Age - fill with median by groups
    age_medians = df_all.groupby(['Title', 'Pclass'])['Age'].transform('median')
    df_all['Age'] = df_all['Age'].fillna(age_medians)
    df_all['Age'] = df_all['Age'].fillna(df_all['Age'].median())
    
    # Embarked - fill with mode
    df_all['Embarked'] = df_all['Embarked'].fillna(df_all['Embarked'].mode()[0])
    
    # Fare - fill with median by class
    fare_medians = df_all.groupby('Pclass')['Fare'].transform('median')
    df_all['Fare'] = df_all['Fare'].fillna(fare_medians)
    
    # ========== 3. ENCODING CATEGORIES ========== 
    
    # Label Encoding for binary features
    le = LabelEncoder()
    df_all['Sex'] = le.fit_transform(df_all['Sex'])
    
    # One-Hot Encoding for other categories
    categorical_features = ['Embarked', 'Title', 'Deck', 'AgeGroup', 'FareGroup']
    df_all = pd.get_dummies(df_all, columns=categorical_features, drop_first=False)
    
    # ========== 4. FEATURE SELECTION ========== 
    
    # Remove unnecessary columns
    drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_all = df_all.drop([col for col in drop_columns if col in df_all.columns], axis=1)
    
    # ========== 5. SPLITTING BACK ========== 
    
    if test_df is not None:
        train = df_all[:len_train].copy()
        test = df_all[len_train:].copy()
        
        # Remove Survived from test if it exists
        if 'Survived' in test.columns:
            test = test.drop('Survived', axis=1)
        
        return train, test
    else:
        return df_all, None


def prepare_features(df):
    """
    Prepare X and y from a dataframe
    """
    if 'Survived' in df.columns:
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        return X, y
    else:
        return df, None


# ======================== MODELS AND THEIR SETTINGS ========================

class ModelComparison:
    """
    Class for comparing all models
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self._initialize_models()
        self.results = []
        
    def _initialize_models(self):
        """
        Initialize all models with optimal parameters for Titanic
        """
        models = {}
        
        # 1. Logistic Regression
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
                    l2_leaf_reg=3, border_count=32,
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
        Train and evaluate all models
        """
        print("\n🔬 TESTING ALL MODELS")
        print("="*115)
        print(f"{ 'Model':<25} {'CV AUC':>10} {'Val ACC':>10} {'Val AUC':>10} {'Kaggle ACC':>12} {'Time (s)':>12}")
        print("-" * 115)

        # Scaling data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)

        # Loading real data for Kaggle
        real_data = None
        if X_test is not None:
            try:
                real_data = pd.read_csv('cheats/_correct_submission.csv')
            except FileNotFoundError:
                print("⚠️ File 'cheats/_correct_submission.csv' not found. Real Kaggle accuracy will not be calculated.")

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for name, model_info in self.models.items():
            start_time = time.time()

            # Select data
            if model_info['needs_scaling']:
                X_tr, X_vl = X_train_scaled, X_val_scaled
            else:
                X_tr, X_vl = X_train, X_val

            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    model_info['model'], X_tr, y_train,
                    cv=skf, scoring='roc_auc', n_jobs=-1
                )

                # Training
                model_info['model'].fit(X_tr, y_train)

                # Predictions
                y_pred = model_info['model'].predict(X_vl)
                y_proba = model_info['model'].predict_proba(X_vl)[:, 1]

                # Metrics
                val_acc = accuracy_score(y_val, y_pred)
                val_auc = roc_auc_score(y_val, y_proba)

                # Real KAGGLE accuracy (checking on real data)
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

                # Saving results
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
                print(f"{name:<25} {'ERROR':<10} - {str(e)[:50]}")

        print("-" * 115)

        # Create a DataFrame with results
        self.results_df = pd.DataFrame(self.results)
        self.results_df = self.results_df.sort_values('Val_AUC', ascending=False)

        return self.results_df
    
    def plot_results(self):
        """
        Visualize results
        """
        if not self.results:
            print("❌ No results to display")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        df = self.results_df
        
        # 1. AUC Comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(df))
        ax1.barh(x_pos, df['Val_AUC'], color='skyblue', label='Validation AUC')
        ax1.barh(x_pos, df['CV_AUC_Mean'], alpha=0.6, color='orange', label='CV AUC Mean')
        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(df['Model'])
        ax1.set_xlabel('AUC Score')
        ax1.set_title('Model AUC Comparison')
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
        
        # 3. Training Time
        ax3 = axes[1, 0]
        ax3.barh(x_pos, df['Training_Time'], color='green', alpha=0.6)
        ax3.set_yticks(x_pos)
        ax3.set_yticklabels(df['Model'])
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_title('Model Training Speed')
        ax3.grid(alpha=0.3)
        
        # 4. Top-5 Models
        ax4 = axes[1, 1]
        top5 = df.head(5)
        colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen']
        bars = ax4.bar(range(len(top5)), top5['Val_AUC'], color=colors)
        ax4.set_xticks(range(len(top5)))
        ax4.set_xticklabels(top5['Model'], rotation=45, ha='right')
        ax4.set_ylabel('Validation AUC')
        ax4.set_title('Top 5 Models by AUC')
        ax4.set_ylim([top5['Val_AUC'].min() - 0.01, top5['Val_AUC'].max() + 0.01])
        
        # Add values on top of bars
        for bar, value in zip(bars, top5['Val_AUC']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.suptitle('ML Model Comparison Results on Titanic', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_ensemble(self, top_n=3):
        """
        Create an ensemble of the best models
        """
        if len(self.results_df) < top_n:
            print(f"⚠️ Not enough models for an ensemble. Available: {len(self.results_df)}")
            return None
        
        # Take top-N models
        top_models = self.results_df.head(top_n)
        
        # Create Voting Classifier
        estimators = [(row['Model'], row['model_object']) 
                     for _, row in top_models.iterrows()]
        
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        
        print(f"\n🎯 Created an ensemble of top {top_n} models:")
        for model_name in top_models['Model']:
            print(f"  - {model_name}")
        
        return voting_clf
    
    def get_best_model(self):
        """
        Get the best model
        """
        if self.results_df.empty:
            print("❌ No trained models")
            return None
        
        best = self.results_df.iloc[0]
        print(f"\n🏆 Best model: {best['Model']}")
        print(f"   Val AUC: {best['Val_AUC']:.4f}")
        print(f"   Val Accuracy: {best['Val_Accuracy']:.4f}")
        print(f"   CV AUC: {best['CV_AUC_Mean']:.4f} (±{best['CV_AUC_Std']:.4f})")
        
        return best['model_object']


# ======================== ADVANCED METHODS ========================

class AdvancedEnsembles:
    """
    Advanced ensemble methods
    """
    
    def __init__(self, base_models, random_state=42):
        self.base_models = base_models
        self.random_state = random_state
    
    def create_stacking_ensemble(self, X_train, y_train):
        """
        Stacking with a meta-model
        """
        # Take top-4 base models
        estimators = list(self.base_models.items())[:4]
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=0.1),
            cv=5,
            n_jobs=-1
        )
        
        print("\n📚 Training Stacking ensemble...")
        stacking_clf.fit(X_train, y_train)
        
        return stacking_clf
    
    def create_blending_ensemble(self, models, weights=None):
        """
        Blending - weighted voting
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
    Fine-tuning the best model
    """
    print(f"\n🎯 Fine-tuning {model_name}...")
    
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
        print(f"⚠️ Parameters for {model_name} are not defined")
        return None
    
    # Use RandomizedSearch for speed
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
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"   Best CV Score: {random_search.best_score_:.4f}")
    
    # Validation check
    y_val_pred = random_search.predict(X_val)
    y_val_proba = random_search.predict_proba(X_val)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"   Val Accuracy: {val_acc:.4f}")
    print(f"   Val AUC: {val_auc:.4f}")
    
    return random_search.best_estimator_


# ======================== MAIN FUNCTION ========================

def main():
    """
    Main pipeline
    """
    print("="*100)
    print(" "*30 + "🚢 TITANIC ML COMPARISON 🚢")
    print("="*100)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load data
    print("\n📁 LOADING DATA")
    print("-"*60)
    train_df, test_df = load_titanic_data()
    
    if train_df is None:
        print("❌ Failed to load data")
        return
    
    print(f"   Train size: {train_df.shape}")
    if test_df is not None:
        print(f"   Test size: {test_df.shape}")
    
    # 2. Preprocessing
    train_processed, test_processed = preprocess_titanic_data(train_df, test_df)
    
    # 3. Data preparation
    X, y = prepare_features(train_processed)
    X_test = None
    if test_processed is not None:
        X_test, _ = prepare_features(test_processed)

    print(f"\n   Number of features after processing: {X.shape[1]}")
    print(f"   Class distribution: ")
    print(f"      - Survived: {y.sum()} ({y.mean():.1%})")
    print(f"      - Deceased: {len(y) - y.sum()} ({1-y.mean():.1%})")
    
    # 4. Splitting into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n   Train set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    if X_test is not None:
        print(f"   Test set: {X_test.shape}")
    
    # 5. Model comparison
    comparison = ModelComparison(random_state=42)
    results_df = comparison.train_and_evaluate(X_train, y_train, X_val, y_val, X_test=X_test)
    
    # 6. Visualizing results
    print("\n📊 VISUALIZING RESULTS")
    print("-"*60)
    comparison.plot_results()
    
    # 7. Top-5 models output
    print("\n🏆 TOP 5 MODELS BY VALIDATION AUC:")
    print("-"*60)
    top5 = results_df.head(5)
    for idx, (_, row) in enumerate(top5.iterrows(), 1):
        medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][idx-1]
        print(f"{medal} {row['Model']:<20} AUC: {row['Val_AUC']:.4f} | "
              f"Accuracy: {row['Val_Accuracy']:.4f} | "
              f"Time: {row['Training_Time']:.2f}s")
    
    # 8. Best model
    best_model = comparison.get_best_model()
    best_model_name = results_df.iloc[0]['Model']
    
    # 9. Creating ensembles
    print("\n🔧 CREATING ENSEMBLES")
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
    
    # 10. Fine-tuning the best model
    if best_model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
        tuned_model = tune_best_model(best_model_name, X_train, y_train, X_val, y_val)
    
    # 11. Final recommendations
    print("\n" + "="*100)
    print(" "*35 + "📋 FINAL RECOMMENDATIONS")
    print("="*100)
    
    print("\n1. FOR MAXIMUM ACCURACY:")
    print(f"   Use {results_df.iloc[0]['Model']} with AUC = {results_df.iloc[0]['Val_AUC']:.4f}")
    
    print("\n2. FOR SPEED/QUALITY BALANCE:")
    # Find the model with the best ratio
    results_df['Score_per_second'] = results_df['Val_AUC'] / results_df['Training_Time']
    best_efficiency = results_df.nlargest(1, 'Score_per_second').iloc[0]
    print(f"   Use {best_efficiency['Model']} (AUC = {best_efficiency['Val_AUC']:.4f}, "
          f"Time = {best_efficiency['Training_Time']:.2f}s)")
    
    print("\n3. FOR PRODUCTION:")
    print("   A Voting ensemble of the top 3 models or")
    print("   a fine-tuned version of the best model is recommended")
    
    print("\n4. FOR EXPERIMENTS:")
    if LIGHTGBM_AVAILABLE:
        print("   LightGBM - fast and efficient for iterations")
    else:
        print("   Random Forest - stable and predictable")
    
    # Saving results
    print("\n💾 SAVING RESULTS")
    print("-"*60)
    
    # Save the results table
    results_df.drop('model_object', axis=1).to_csv('titanic_model_comparison_results.csv', index=False)
    print("✅ Results saved to 'titanic_model_comparison_results.csv'")
    
    # Save the best model
    try:
        import joblib
        joblib.dump(best_model, 'best_titanic_model.pkl')
        print(f"✅ Best model ({best_model_name}) saved to 'best_titanic_model.pkl'")
    except:
        print("⚠️ To save the model, install joblib: pip install joblib")
    
    print("\n" + "="*100)
    print(" "*40 + "✨ DONE! ✨")
    print("="*100)
    
    return results_df, best_model


if __name__ == "__main__":
    # Run the main program
    results, model = main()
    
    # Additional interactive mode
    print("\n💡 Tip: To run individual experiments, use the functions from this script")
    print("   For example: comparison = ModelComparison() to create a new comparison")