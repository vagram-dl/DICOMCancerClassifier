import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox
from PyQt5.QtCore import Qt



class AttentionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(features, features // 2),
            nn.ReLU(),
            nn.Linear(features // 2, features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


class MultimodalCancerModel(nn.Module):
    def __init__(self, tabular_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.attention = AttentionBlock(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            self.attention,
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        img_features = self.cnn(x['image']).view(x['image'].size(0), -1)
        tab_features = self.tabular_net(x['tabular'])
        combined = torch.cat([img_features, tab_features], dim=1)
        return self.classifier(combined)



class CancerApp(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.img_size = 256

        # Сохраняем scaler из обучения
        # Здесь для примера используем стандартные значения
        self.mean = None
        self.std = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Cancer Classification Prototype")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()


        self.label = QLabel("Загрузите DICOM и введите данные")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 14px; padding: 10px;")
        self.layout.addWidget(self.label)


        self.age_input = QLineEdit(self)
        self.age_input.setPlaceholderText("Возраст (например: 45)")
        self.layout.addWidget(self.age_input)

        self.size_input = QLineEdit(self)
        self.size_input.setPlaceholderText("Размер опухоли (например: 12.5)")
        self.layout.addWidget(self.size_input)

        self.biopsy_input = QLineEdit(self)
        self.biopsy_input.setPlaceholderText("Biopsy score (например: 3)")
        self.layout.addWidget(self.biopsy_input)


        self.button = QPushButton("Загрузить DICOM и предсказать")
        self.button.clicked.connect(self.load_and_predict)
        self.button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.layout.addWidget(self.button)


        self.result_label = QLabel("Ожидание данных...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; padding: 20px; border: 1px solid #ccc; border-radius: 5px;")
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def preprocess_image(self, image_path):
        """Обработка DICOM изображения как в датасете"""
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array.astype(np.float32)

        # Нормализация по перцентилям
        image = (image - np.percentile(image, 1)) / (np.percentile(image, 99) - np.percentile(image, 1) + 1e-8)
        image = np.clip(image, 0, 1)

        # Преобразование в тензор и изменение размера
        image = torch.from_numpy(image).unsqueeze(0)
        image = F.interpolate(image.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False).squeeze(0)

        return image.unsqueeze(0)

    def validate_inputs(self):
        """Проверка введенных данных"""
        try:
            age = float(self.age_input.text()) if self.age_input.text() else None
            size = float(self.size_input.text()) if self.size_input.text() else None
            biopsy = float(self.biopsy_input.text()) if self.biopsy_input.text() else None

            if age is None or size is None or biopsy is None:
                QMessageBox.warning(self, "Ошибка", "Пожалуйста, заполните все поля")
                return None

            if age < 0 or age > 120:
                QMessageBox.warning(self, "Ошибка", "Некорректный возраст")
                return None

            if size < 0 or size > 100:
                QMessageBox.warning(self, "Ошибка", "Некорректный размер опухоли")
                return None

            if biopsy < 1 or biopsy > 10:
                QMessageBox.warning(self, "Ошибка", "Biopsy score должен быть от 1 до 10")
                return None

            return [[age, size, biopsy]]

        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите корректные числовые значения")
            return None

    def load_and_predict(self):
        try:
            # Проверка входных данных
            tabular_data = self.validate_inputs()
            if tabular_data is None:
                return

            # Выбор файла
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Выберите DICOM файл",
                "",
                "DICOM Files (*.dcm *.DCM);;All Files (*)"
            )

            if not file_path:
                return

            # Предобработка изображения
            self.result_label.setText("Обработка изображения...")
            self.result_label.setStyleSheet(
                "font-size: 14px; padding: 20px; border: 1px solid #FFA500; border-radius: 5px;")
            QApplication.processEvents()

            image = self.preprocess_image(file_path)
            tabular = torch.tensor(tabular_data, dtype=torch.float32)

            # Предсказание
            self.result_label.setText("Выполняется предсказание...")
            QApplication.processEvents()

            with torch.no_grad():
                output = self.model({'image': image, 'tabular': tabular})
                probability = output.item()
                prediction = int(probability > 0.5)

            # Отображение результата
            if prediction == 1:
                result_text = f" РАК ОБНАРУЖЕН\nВероятность: {probability:.2%}"
                style = "background-color: #FFE5E5; border: 2px solid #FF0000; border-radius: 5px;"
            else:
                result_text = f" РАК НЕ ОБНАРУЖЕН\nВероятность: {probability:.2%}"
                style = "background-color: #E5FFE5; border: 2px solid #00FF00; border-radius: 5px;"

            self.result_label.setText(result_text)
            self.result_label.setStyleSheet(f"font-size: 16px; padding: 20px; {style}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{str(e)}")
            self.result_label.setText("Ошибка при обработке")
            self.result_label.setStyleSheet(
                "font-size: 14px; padding: 20px; border: 1px solid #FF0000; border-radius: 5px;")



if __name__ == "__main__":
    app = QApplication(sys.argv)

    model = MultimodalCancerModel(tabular_size=3)

    try:
        model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
        print("Веса модели успешно загружены")
    except FileNotFoundError:
        print("Файл best_model.pt не найден, используется случайная инициализация")
    except Exception as e:
        print(f"Ошибка загрузки весов: {e}, используется случайная инициализация")

    window = CancerApp(model)
    window.show()
    sys.exit(app.exec_())