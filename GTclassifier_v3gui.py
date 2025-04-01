import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QScrollArea, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor
from PyQt5.QtCore import Qt, QPointF, QLineF, pyqtSlot
import tifffile as tiff
from sklearn.ensemble import RandomForestClassifier

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.drawing = False
        self.current_label = None
        self.labels = {'object': [], 'background': []}
        self.last_pan_point = None

    def set_image(self, image):
        self.image = image
        if image.ndim == 2:  # Grayscale image
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8)
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB image
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        else:
            raise ValueError("Unsupported image format")
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(0, 0, image.shape[1], image.shape[0])

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_label is not None:
            self.drawing = True
            self.start_point = self.mapToScene(event.pos())
            self.current_path = []
        elif event.button() == Qt.MiddleButton:
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drawing:
            end_point = self.mapToScene(event.pos())
            self.current_path.append(end_point)
            pen = QPen(QColor(0, 255, 0, 51) if self.current_label == 'object' else QColor(255, 0, 0, 51), 20)
            self.scene.addLine(QLineF(self.start_point, end_point), pen)
            self.start_point = end_point
        elif self.last_pan_point is not None:
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.labels[self.current_label].append(self.current_path)
            # self.parent().provide_feedback(self.current_label)
        elif event.button() == Qt.MiddleButton:
            self.last_pan_point = None
            self.setCursor(Qt.ArrowCursor)

    def clear_labels(self):
        if self.current_label and self.labels[self.current_label]:
            self.labels[self.current_label] = []
            self.redraw_labels()
            self.parent().provide_feedback(f"All {self.current_label} labels cleared")

    def redraw_labels(self):
        # Clear the scene
        self.scene.clear()
        # Redraw the image
        self.set_image(self.parent().images[self.parent().current_image_index])
        # Redraw all labels
        for label_type, paths in self.labels.items():
            for path in paths:
                for i in range(len(path) - 1):
                    start_point = path[i]
                    end_point = path[i + 1]
                    pen = QPen(QColor(0, 255, 0, 51) if label_type == 'object' else QColor(255, 0, 0, 51), 20)
                    self.scene.addLine(QLineF(start_point, end_point), pen)

class CFWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Segmentation App")
        self.setGeometry(100, 100, 1200, 800)

        self.image_folder = ""
        self.images = []
        self.current_image_index = 0
        self.label_counter = 0

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        button_layout = QHBoxLayout()

        self.input_button = QPushButton("Select Image Folder")
        self.input_button.clicked.connect(self.select_image_folder)
        button_layout.addWidget(self.input_button)

        self.object_label_button = QPushButton("Label Object")
        self.object_label_button.clicked.connect(self.label_object)
        button_layout.addWidget(self.object_label_button)

        self.background_label_button = QPushButton("Label Background")
        self.background_label_button.clicked.connect(self.label_background)
        button_layout.addWidget(self.background_label_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_labels)
        button_layout.addWidget(self.clear_button)

        self.save_label_button = QPushButton("Save")
        self.save_label_button.clicked.connect(self.save_label)
        button_layout.addWidget(self.save_label_button)

        self.segment_button = QPushButton("Segment Image")
        self.segment_button.clicked.connect(self.segment_image)
        button_layout.addWidget(self.segment_button)

        layout.addLayout(button_layout)

        self.image_viewer = ImageViewer(self)
        layout.addWidget(self.image_viewer)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder = folder
            self.load_images()

    def load_images(self):
        self.images = []
        for file_name in os.listdir(self.image_folder):
            if file_name.endswith(".tif"):
                image_path = os.path.join(self.image_folder, file_name)
                with tiff.TiffFile(image_path) as tif:
                    image = tif.asarray(out='memmap')  # use memory-mapped file to handle large images
                self.images.append(image)
        if self.images:
            self.current_image_index = 0
            self.display_image()

    def display_image(self):
        if self.images:
            image = self.images[self.current_image_index]
            self.image_viewer.set_image(image)

    def label_object(self):
        self.image_viewer.current_label = 'object'

    def label_background(self):
        self.image_viewer.current_label = 'background'

    def clear_labels(self):
        # self.image_viewer.clear_labels()
        mask_path = os.path.join(self.image_folder, f"mask{self.label_counter}.tif")
        if os.path.exists(mask_path):
            os.remove(mask_path)
            self.provide_feedback(f"Mask {self.label_counter} removed")

    def provide_feedback(self, label_type):
        QMessageBox.information(self, "Label Feedback", f"{label_type.capitalize()} label added")

    def save_labels_to_mask(self, image_shape, labels, mask_path):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for label_type, paths in labels.items():
            label_value = 1 if label_type == 'object' else 2
            for path in paths:
                for point in path:
                    px, py = int(point.x()), int(point.y())
                    mask[py, px] = label_value  # Set the pixel to 1 for 'object' and 2 for 'background'
        tiff.imwrite(mask_path, mask)

    def save_label(self):
        if not self.images:
            return

        self.label_counter += 1
        image = self.images[self.current_image_index]
        mask_path = os.path.join(self.image_folder, f"mask{self.label_counter}.tif")
        self.save_labels_to_mask(image.shape, self.image_viewer.labels, mask_path)
        QMessageBox.information(self, "Save Label", f"Labels have been saved to {mask_path}")

    def generate_feature_stack(image):
        # determine features
        # histogram = exposure.histogram(image)[0]
        blurred = filters.gaussian(image, sigma=2)
        edges = filters.sobel(blurred)
        
        intensity = image
        # collect features in a stack
        # The ravel() function turns a nD image into a 1-D image.
        # We need to use it because scikit-learn expects values in a 1-D format here. 
        feature_stack = [
            image.ravel(),  #intensity feature
            # histogram.ravel(),
            blurred.ravel(),#noise filter feature
            edges.ravel(),  #edge detection feature
            
            # intensity.ravel()
        ]
        
        # return stack as numpy-array
        return np.asarray(feature_stack)


    ## formating data
    def format_data(feature_stack, annotation):
        # reformat the data to match what scikit-learn expects
        # transpose the feature stack
        X = feature_stack.T
        # make the annotation 1-dimensional
        y = annotation.ravel()
        
        # remove all pixels from the feature and annotations which have not been annotated
        mask = y > 0
        X = X[mask]
        y = y[mask]

        return X, y
    def segment_image(self):
        raw_image_path = os.path.join(self.image_folder, "imageraw.tif")
        mask_path = os.path.join(self.image_folder, "finalmask.tif")

        if not os.path.exists(raw_image_path) or not os.path.exists(mask_path):
            QMessageBox.warning(self, "Error", "imageraw.tif or finalmask.tif not found in the image folder.")
            return

        raw_image = tiff.imread(raw_image_path)
        mask = tiff.imread(mask_path)

        # Overlay the current labels onto the final mask
        self.save_labels_to_mask(raw_image.shape, self.image_viewer.labels, mask_path)
        updated_mask = tiff.imread(mask_path)

        # Assuming generate_feature_stack and format_data are defined elsewhere
        feature_stack = generate_feature_stack(raw_image)
        X, y = format_data(feature_stack, updated_mask)

        classifier = RandomForestClassifier(max_depth=2, random_state=42)
        classifier.fit(X, y)
        res = classifier.predict(feature_stack.T) - 1  # we subtract 1 to make background = 0

        # Save the segmentation result
        segmentation_result_path = os.path.join(self.image_folder, "segmentation_result.tif")
        tiff.imwrite(segmentation_result_path, res.reshape(raw_image.shape).astype(np.uint8))
        QMessageBox.information(self, "Segmentation Result", f"Segmentation result saved to {segmentation_result_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CFWindow()
    window.show()
    sys.exit(app.exec_())