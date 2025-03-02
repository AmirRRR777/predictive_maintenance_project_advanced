import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier  # Используем более мощную модель
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

import optuna  # Для оптимизации гиперпараметров

# Импорт ClearML для отслеживания экспериментов
from clearml import Task, Logger


def create_multiclass_target(row):
    """
    Функция определяет тип отказа:
    0: Нет отказа
    1: TWF
    2: HDF
    3: PWF
    4: OSF
    5: RNF
    Если более одного отказа, выбираем первый по приоритету (TWF > HDF > PWF > OSF > RNF)
    """
    if row['TWF'] == 1:
        return 1
    elif row['HDF'] == 1:
        return 2
    elif row['PWF'] == 1:
        return 3
    elif row['OSF'] == 1:
        return 4
    elif row['RNF'] == 1:
        return 5
    else:
        return 0


def analysis_and_model_page():
    st.title("Анализ данных, мультиклассовая модель и ClearML")

    # 1) Инициализация ClearML Task для отслеживания эксперимента
    task = Task.init(project_name="Predictive Maintenance", task_name="Multiclass Model Training with Optuna")
    # Логируем некоторые параметры эксперимента
    task.connect({"model": "RandomForestClassifier", "n_trials": 30})
    logger = task.get_logger()

    # 2) Загрузка датасета через интерфейс
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # ----- Предобработка -----
        # Удаляем ненужные столбцы, оставляем столбцы отказов для формирования новой целевой переменной
        data.drop(columns=['UDI', 'Product ID'], inplace=True)

        # Создаем новую целевую переменную 'Failure_Type'
        data['Failure_Type'] = data.apply(create_multiclass_target, axis=1)
        data.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], inplace=True)

        # Преобразование категориального признака "Type"
        le = LabelEncoder()
        data['Type'] = le.fit_transform(data['Type'])

        # Разделение на признаки (X) и целевую переменную (y)
        X = data.drop(columns=['Machine failure', 'Failure_Type'])
        y = data['Failure_Type']

        # Определяем числовые признаки для масштабирования
        num_cols = ['Air temperature [K]',
                    'Process temperature [K]',
                    'Rotational speed [rpm]',
                    'Torque [Nm]',
                    'Tool wear [min]']
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Разделение данных на обучающую и тестовую выборки (с учетом стратификации по y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        st.subheader("Оптимизация гиперпараметров с использованием Optuna")

        # Определяем функцию-цель для Optuna
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                clf.fit(X_cv_train, y_cv_train)
                preds = clf.predict(X_cv_val)
                score = accuracy_score(y_cv_val, preds)
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)

        best_params = study.best_trial.params
        st.write("Лучшие гиперпараметры:", best_params)
        st.write("Лучшее значение CV accuracy:", study.best_trial.value)
        # Логируем лучшие параметры и CV accuracy в ClearML
        logger.report_scalar(title="CV Accuracy", series="Best Trial", value=study.best_trial.value, iteration=0)
        logger.report_text(f"Best parameters: {best_params}")

        # Обучаем финальную модель с найденными гиперпараметрами
        model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42
        )
        model.fit(X_train, y_train)

        # Оценка модели на тестовой выборке
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        st.subheader("Результаты финальной модели после оптимизации")
        st.write(f"**Accuracy**: {acc:.3f}")
        logger.report_scalar(title="Test Accuracy", series="Final Model", value=acc, iteration=0)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.text("Classification Report:")
        st.text(report)

        # ----- Предсказание на новых данных -----
        st.subheader("Предсказание на новых данных")
        with st.form("prediction_form"):
            input_type = st.selectbox("Тип продукта", ["L", "M", "H"])
            input_air_temp = st.number_input("Air temperature [K]", value=300.0)
            input_process_temp = st.number_input("Process temperature [K]", value=310.0)
            input_speed = st.number_input("Rotational speed [rpm]", value=1500)
            input_torque = st.number_input("Torque [Nm]", value=40.0)
            input_wear = st.number_input("Tool wear [min]", value=0)

            submit_button = st.form_submit_button("Предсказать")
            if submit_button:
                quality_mapping = {"L": 0, "M": 1, "H": 2}
                input_type_encoded = quality_mapping[input_type]

                input_data = pd.DataFrame({
                    'Type': [input_type_encoded],
                    'Air temperature [K]': [input_air_temp],
                    'Process temperature [K]': [input_process_temp],
                    'Rotational speed [rpm]': [input_speed],
                    'Torque [Nm]': [input_torque],
                    'Tool wear [min]': [input_wear]
                })
                input_data[num_cols] = scaler.transform(input_data[num_cols])

                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)

                st.write(f"**Предсказание (0=Нет отказа, 1=TWF, 2=HDF, 3=PWF, 4=OSF, 5=RNF)**: {prediction[0]}")
                st.write("**Вероятности по классам:**")
                st.write(dict(zip(np.arange(len(prediction_proba[0])), prediction_proba[0])))

        # Завершаем задачу ClearML после завершения эксперимента
        task.close()
