# Проект: Интеллектуальная система мультиклассовой классификации для предиктивного обслуживания оборудования с оптимизацией и MLOps
## Описание проекта
Цель проекта — разработать интеллектуальную систему машинного обучения для предиктивного обслуживания оборудования. В продвинутой версии реализованы следующие функции:
1) Мультиклассовая классификация отказов - модель предсказывает не только факт отказа, но и конкретный тип отказа (TWF, HDF, PWF, OSF, RNF);
2) Детальный анализ данных - реализованы интерактивные визуализации для глубокого изучения взаимосвязей между признаками и анализа распределения отказов;
3) Оптимизация модели - используется Optuna для автоматической настройки гиперпараметров модели с применением кросс-валидации;
4) Интеграция с MLOps - ClearML применяется для отслеживания экспериментов, логирования параметров и метрик, что обеспечивает удобное управление и версионирование моделей.
## Выполненные задачи
### **1. Изменение задачи на мультиклассовую классификацию**
1. Изменение задачи на мультиклассовую классификацию
- Описание: Реализована мультиклассовая классификация для предсказания типа отказа оборудования (TWF, HDF, PWF, OSF, RNF) на основе исходных столбцов отказов.
- Инструменты: RandomForestClassifier (модель RandomForest), преобразование целевой переменной через функцию create_multiclass_target.
- Результаты: Модель успешно различает типы отказов, что позволяет более детально анализировать неисправности оборудования.
### **2. Детальный анализ датасета**
- Описание: Создана страница в Streamlit с визуализацией распределений числовых признаков, корреляционной матрицей и scatter plot для анализа взаимосвязей между признаками.
- Инструменты: seaborn, matplotlib.
- Результаты: Интерактивные визуализации позволяют глубже понять структуру и взаимосвязи в данных.
### **3. Оптимизация обучения модели**
- Описание: Внедрена оптимизация гиперпараметров с использованием Optuna в сочетании с кросс-валидацией (StratifiedKFold) для повышения точности модели.
- Результаты:
   - Лучшие гиперпараметры:
    {
      "n_estimators": 93,
      "max_depth": 17,
      "min_samples_split": 2
    }
    - Лучшее значение CV accuracy: 0.9801
    - Итоговая точность модели на тестовой выборке улучшилась до 0.984.
### **4. Внедрение ClearML**
- Описание: Интеграция ClearML для отслеживания экспериментов, логирования параметров и метрик, а также управления и версионирования моделей.
- Результаты:
   - Эксперименты автоматически регистрируются и логируются на сервере ClearML.
   - Модель и метрики оптимизации доступны для анализа через ClearML Dashboard.
   - Возможна дальнейшая интеграция с ClearML Serving для развертывания модели через REST API.
## Датасет
Используется датасет **"AI4I 2020 Predictive Maintenance Dataset"**,
содержащий 10 000 записей с 14 признаками. Подробное описание датасе
можно найти в [документации]
(https://archive.ics.uci.edu/dataset/601/predictive+maintenance+data
## Установка и запуск
1. Клонируйте репозиторий:
git clone <https://github.com/AmirRRR777/predictive_maintenance_project_advanced.git>
2. Установите зависимости:
pip install -r requirements.txt
3. Запустите приложение:
streamlit run app.py
## Структура репозитория
- `app.py`: Основной файл приложения.
- `analysis_and_model.py`: Страница с анализом данных и моделью.
- `presentation.py`: Страница с презентацией проекта.
- `requirements.txt`: Файл с зависимостями.
- `data/`: Папка с данными.
- `README.md`: Описание проекта.
## Видео-демонстрация
[Ссылка на видео](video/video.mp4) или встроенное видео ниже:
<video src="video/video.mp4" controls width="100%"></video>

