import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def load_data(file_path):
    questions = []
    answers = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                last_comma_index = line.rfind(',')
                if last_comma_index != -1:
                    question = line[:last_comma_index].strip()
                    answer = line[last_comma_index + 1:].strip()
                    if answer in ['0', '1']:
                        questions.append(question)
                        answers.append(int(answer))

    return questions, answers


def create_and_train_model(questions, answers):
    global vectorizer  # Объявляем vectorizer как глобальную переменную
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(questions).toarray()  # Преобразуем в массив
    y = np.array(answers)

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создаем модель
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(1, activation='sigmoid')  # Для бинарной классификации
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Обучаем модель
    model.fit(X_train, y_train, epochs=10, batch_size=2)

    return model


# Загружаем данные
file_path = 'questions.txt'  # Путь к вашему файлу
questions, answers = load_data(file_path)

# Создаем и обучаем модель
model = create_and_train_model(questions, answers)


# Проверка модели
def predict(model, question):
    question_vec = vectorizer.transform([question]).toarray()  # Преобразуем вопрос в вектор
    prediction = model.predict(question_vec)
    return 'Да' if prediction[0] >= 0.5 else 'Нет'


# Цикл для ввода вопросов в реальном времени
while True:
    test_question = input("Введите ваш вопрос (или 'выход' для завершения): ")
    if test_question.lower() == 'выход':
        print("Завершение работы.")
        break
    print(f"Ответ на вопрос '{test_question}': {predict(model, test_question)}")
