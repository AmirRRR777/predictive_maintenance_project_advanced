import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

# Создаем объекты страниц с использованием st.Page
page_analysis = st.Page("analysis_and_model.py", title="Анализ и модель")
page_presentation = st.Page("presentation.py", title="Презентация")

# Переопределяем метод run, чтобы он вызывал нужную функцию, импортированную из модулей
page_analysis.run = analysis_and_model_page
page_presentation.run = presentation_page

# Формируем список страниц
pages = [page_analysis, page_presentation]

# Инициализируем навигацию, передавая список страниц
current_page = st.navigation(pages, position="sidebar", expanded=True)

# Запускаем выбранную страницу
current_page.run()
