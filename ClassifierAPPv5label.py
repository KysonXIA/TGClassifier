import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog,QMessageBox
from PyQt5.QtCore import Qt
from sklearn.ensemble import RandomForestClassifier
import tifffile as tiff
from skimage.io import imread, imshow
import numpy as np
from numpy import ones
import napari
import matplotlib.pyplot as plt
from skimage import filters, exposure
import pylab 
from skimage.data import cells3d
from skimage.io import imsave, imread
import stackview
from pyclesperanto_prototype import imshow
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from skimage.segmentation import random_walker
from skimage.measure import label, regionprops
from skimage.filters import frangi

# Global variables
drawing = False
points = []
mask_created = False
current_class = 'object'
background_class = 'background'
object_count = 0
background_count = 0
background_value = 0

img = None
img_overlay = None
mask = None

# Zoom and pan
zoom_factor = 1.0
pan_offset_x = 0
pan_offset_y = 0

# Mouse states
right_dragging = False
middle_dragging = False
last_mouse_x = 0
last_mouse_y = 0

# Store transform info for reverse-mapping from display coords -> original image coords
transform_dict = {}

def display_to_image_coords(cx, cy):
    """
    Convert display/window coordinates (cx, cy) to the original image coordinates,
    taking into account zoom_factor and pan offsets.
    """
    if not transform_dict:
        return cx, cy
    scale = transform_dict["scale"]
    can_x1 = transform_dict["can_x1"]
    can_y1 = transform_dict["can_y1"]
    roi_x1 = transform_dict["roi_x1"]
    roi_y1 = transform_dict["roi_y1"]
    orig_w = transform_dict["orig_w"]
    orig_h = transform_dict["orig_h"]

    # Reverse the pan offset in the display
    rx = roi_x1 + (cx - can_x1)
    ry = roi_y1 + (cy - can_y1)
    # Reverse the zoom
    real_x = rx / scale
    real_y = ry / scale
    # Clamp
    real_x = max(0, min(real_x, orig_w - 1))
    real_y = max(0, min(real_y, orig_h - 1))
    return int(real_x), int(real_y)

def draw_roi(event, x, y, flags, param):
    """
    Left button for drawing ROI,
    Right button drag for zoom,
    Middle button drag for panning.
    """
    global drawing, points, img, img_overlay, mask_created
    global current_class, object_count, background_count
    global right_dragging, middle_dragging
    global last_mouse_x, last_mouse_y
    global zoom_factor, pan_offset_x, pan_offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        if not mask_created and img_overlay is not None:
            drawing = True
            real_x, real_y = display_to_image_coords(x, y)
            points = [(real_x, real_y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and img_overlay is not None:
            real_x, real_y = display_to_image_coords(x, y)
            points.append((real_x, real_y))
            color = (0, 255, 0) if current_class == 'object' else (0, 0, 255)
            cv2.line(img_overlay, points[-2], points[-1], color, 15)
            update_display()

        elif right_dragging:
            dy = y - last_mouse_y
            scale_speed = 0.005
            zoom_factor += dy * scale_speed
            zoom_factor = max(0.05, min(zoom_factor, 50.0))
            last_mouse_y = y
            update_display()

        elif middle_dragging:
            dx = x - last_mouse_x
            dy = y - last_mouse_y
            pan_offset_x += dx
            pan_offset_y += dy
            last_mouse_x = x
            last_mouse_y = y
            update_display()

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing and img_overlay is not None:
            drawing = False
            real_x, real_y = display_to_image_coords(x, y)
            points.append((real_x, real_y))
            color = (0, 255, 0) if current_class == 'object' else (0, 0, 255)
            cv2.line(img_overlay, points[-1], points[0], color, 15)
            create_mask()
            update_display()

            if current_class == 'object':
                object_count += 1
                if object_count == 6:
                    QMessageBox.information(self,"已标记 6 个 object ROI，请继续标记 5 个 background。")
            else:
                
                background_count += 1
                if background_count == 5:
                    mask_created = True
                    QMessageBox.information(self,"已标记 5 个 background ROI，可以保存或叠加。")

    elif event == cv2.EVENT_RBUTTONDOWN:
        right_dragging = True
        last_mouse_y = y

    elif event == cv2.EVENT_RBUTTONUP:
        right_dragging = False

    elif event == cv2.EVENT_MBUTTONDOWN:
        middle_dragging = True
        last_mouse_x = x
        last_mouse_y = y

    elif event == cv2.EVENT_MBUTTONUP:
        middle_dragging = False

def create_mask():
    """
    Create / update the binary mask (object=255, background=1, otherwise=0).
    """
    global points, img, img_overlay, mask, current_class
    if mask is None or img is None or len(points) < 3:
        return

    points_array = np.array(points, dtype=np.int32)
    mask_value = 255 
    if current_class == 'object' :
    elif current_class == 'background':
        pass  # Add appropriate logic here if needed
        mask_value = background_value if background_value > 0
    else:
        background_value
    
    cv2.fillPoly(mask, [points_array], mask_value)

def overlay_mask():
    """
    Overlay the mask onto the original image with certain alpha values.
    """
    global img, mask, img_overlay
    if mask is None or img is None:
        return
    colored_mask = np.zeros_like(img)
    colored_mask[mask == background_value] = [0, 0, 255]      # background blue
    colored_mask[mask == 255] = [0, 255, 0]    # object green
    img_overlay = cv2.addWeighted(img, 0.8, colored_mask, 0.7, 0)

def update_display():
    """
    Resizes and pans the overlay + mask to reflect zoom_factor + pan_offset,
    then shows them in "Image" and "Binary Mask" windows.
    """
    global img_overlay, mask, zoom_factor, pan_offset_x, pan_offset_y
    if img_overlay is None:
        return

    h, w = img_overlay.shape[:2]
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)

    resized_overlay = cv2.resize(img_overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = None
    if mask is not None and mask.size > 0:
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    display_h, display_w = h, w
    canvas_overlay = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((display_h, display_w), dtype=np.uint8)

    offset_x = int(pan_offset_x)
    offset_y = int(pan_offset_y)

    roi_x1 = max(0, -offset_x)
    roi_y1 = max(0, -offset_y)
    roi_x2 = min(display_w - offset_x, new_w)
    roi_y2 = min(display_h - offset_y, new_h)

    can_x1 = max(0, offset_x)
    can_y1 = max(0, offset_y)
    can_x2 = can_x1 + (roi_x2 - roi_x1)
    can_y2 = can_y1 + (roi_y2 - roi_y1)

    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
        canvas_overlay[can_y1:can_y2, can_x1:can_x2] = resized_overlay[roi_y1:roi_y2, roi_x1:roi_x2]
        if resized_mask is not None:
            canvas_mask[can_y1:can_y2, can_x1:can_x2] = resized_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    # Store transform for reverse mapping
    transform_dict["scale"] = zoom_factor
    transform_dict["can_x1"] = can_x1
    transform_dict["can_y1"] = can_y1
    transform_dict["roi_x1"] = roi_x1
    transform_dict["roi_y1"] = roi_y1
    transform_dict["orig_w"] = w
    transform_dict["orig_h"] = h

    cv2.imshow("Image", canvas_overlay)
    if resized_mask is not None:
        cv2.imshow("Binary Mask", canvas_mask)
    else:
        cv2.imshow("Binary Mask", np.zeros_like(canvas_mask))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Classifier")
        self.setGeometry(100, 100, 320, 600)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_tiff_image)
        layout.addWidget(self.load_button)

        self.object_button = QPushButton("Object")
        self.object_button.clicked.connect(self.select_object_class)
        layout.addWidget(self.object_button)

        self.background_button = QPushButton("Background")
        self.background_button.clicked.connect(self.select_background_class)
        layout.addWidget(self.background_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_drawing)
        layout.addWidget(self.clear_button)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_mask)
        layout.addWidget(self.save_button)

        self.overlay_button = QPushButton("Overlay")
        self.overlay_button.clicked.connect(self.overlay_mask_on_image)
        layout.addWidget(self.overlay_button)

        self.segment_button = QPushButton("Segment")
        self.segment_button.clicked.connect(self.segment_RF)
        layout.addWidget(self.segment_button)

        # Exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_app)
        layout.addWidget(self.exit_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # def format_data(feature_stack, annotation):
    #     # reformat the data to match what scikit-learn expects
    #     X = feature_stack.T # transpose the feature stack
    #     y = annotation.ravel() # make the annotation 1-dimensional
    #     mask = y > 0  # remove all pixels from the feature and annotations which have not been annotated
    #     X = X[mask]
    #     y = y[mask]
    #     return X, y

    def segment_RF(self):
        # Let the user select the input image
        QMessageBox.information(self, "File Selection", "Please select raw image file")
        input_image_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image", "", "TIFF Files (*.tif);;All Files (*)"
        )
        if not input_image_path:
            QMessageBox.warning(self, "File Selection", "No raw image file selected.")
            return

        # Let the user select the annotation file
        QMessageBox.information(self, "File Selection", "Please select annotation file")
        annotation_path, _ = QFileDialog.getOpenFileName(
            self, "Select Annotation File", "", "TIFF Files (*.tif);;All Files (*)"
        )
        if not annotation_path:
            QMessageBox.warning(self, "File Selection", "No annotation file selected.")
            return

        # Load the selected input image
        imageraw = imread(input_image_path)
        image = imageraw[:, :, 2]  # Assuming the third channel is used
        stackview.imshow(image, colormap='Greens')

        # Load the selected annotation file
        maskraw = imread(annotation_path)
        annotation = maskraw
        stackview.imshow(maskraw, vmin=0, vmax=2)

        # Let the user select the ground truth mask (optional)
        QMessageBox.information(self, "File Selection", "Please Ground Truth file")
        demo_mask_path, _ = QFileDialog.getOpenFileName(
            self, "Select Ground Truth Mask (Optional)", "", "TIFF Files (*.tif);;All Files (*)"
        )
        if demo_mask_path:
            GTdemo_mask = tiff.imread(demo_mask_path)
            QMessageBox.information(self, "File Selection", f"Ground truth mask selected: {demo_mask_path}")
        else:
            GTdemo_mask = None
            QMessageBox.information(self, "File Selection", "No ground truth mask selected. Skipping this step.")

        # Display processing messages
        QMessageBox.information(self, "Processing", f"Processing raw image: {input_image_path}")
        QMessageBox.information(self, "Processing", f"Processing annotation file: {annotation_path}")
        if demo_mask_path:
            QMessageBox.information(self, "Processing", f"Processing ground truth mask: {demo_mask_path}")
        else:
            QMessageBox.information(self, "Processing", "No ground truth mask provided.")

        # Generate the feature stack
        feature_stack = self.generate_feature_stack()
        fig, axes = plt.subplots(1, 3, figsize=(10, 10))

        # Visualize the feature stack
        axes[0].imshow(feature_stack[0].reshape(image.shape), cmap=plt.cm.Greens)
        axes[1].imshow(feature_stack[1].reshape(image.shape), cmap=plt.cm.Greens)
        axes[2].imshow(feature_stack[2].reshape(image.shape), cmap=plt.cm.Greens)

        # Prepare data for training
        X = feature_stack.T  # Transpose the feature stack
        y = annotation.ravel()  # Flatten the annotation
        mask = y > 0  # Remove unannotated pixels
        X = X[mask]
        y = y[mask]

        print("Input shape:", X.shape)  # Number of pixels
        print("Annotation shape:", y.shape)  # Number of features

        # Train and predict using Random Forest
        classifier = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=54)
        classifier.fit(X, y)
        resraw = classifier.predict(feature_stack.T)
        res = np.invert(resraw)
        res[res == 1] = 0

        # Ensure the ground truth mask has the same shape as the result
        if GTdemo_mask is not None:
            assert GTdemo_mask.shape == res.reshape(image.shape).shape, "Shape mismatch between result and ground truth mask"

        # Display the ground truth mask and the result
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        if GTdemo_mask is not None:
            axes[0].imshow(GTdemo_mask, cmap=plt.cm.gray)
            axes[0].set_title('Ground Truth Mask')
        axes[1].imshow(res.reshape(image.shape), cmap=plt.cm.gray)
        axes[1].set_title('Segmentation Result')

        plt.show()

        # Save the result
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Segmentation Result", "", "TIFF Files (*.tif);;All Files (*)"
        )
        if output_path:
            tiff.imwrite(output_path, res.reshape(image.shape).astype(np.uint8))
            QMessageBox.showinfo(self,f"Result saved to {output_path}")
    
    def generate_feature_stack(self):
    # determine features
        imageraw = imread('C:/ProjectCode/GTClassifier/demo5/APP1_Hippo005_overlay.tif')
        image = imageraw[:,:,2]
        intensity = image
        # median = filters.median(intensity)
        blurred = filters.gaussian(intensity, sigma=4)
        edges = filters.sobel(blurred)
        # threshold = filters.threshold_li(blurred)
        # hessian = filters.frangi(edges)
        # histogram = exposure.histogram(image)[0]
        # collect features in a stack
        feature_stack = [
        
            intensity.ravel(),  #intensity feature  # We need to use it because scikit-learn expects values in a 1-D format here. 
            edges.ravel(),  #edge detection feature # The ravel() function turns a nD image into a 1-D image.
            # median.ravel(), #noise filter feature
            blurred.ravel(),#noise filter feature
            # threshold.ravel()
            # threshold_otsu.ravel(),
            # histogram.ravel(),
            # hessian.ravel(),
            # intensity.ravel()
        ]
        # return stack as numpy-array
        return np.asarray(feature_stack)
    
    

    def load_tiff_image(self):
        global img, img_overlay, mask
        global zoom_factor, pan_offset_x, pan_offset_y
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load TIFF Image", "", "TIFF Files (*.tif);;All Files (*)", options=options
        )
        if file_path:
            try:
                loaded_image = tiff.imread(file_path)
            except Exception as e:
                QMessageBox.information(self,f"Error loading TIFF: {e}")
                return

            if len(loaded_image.shape) == 2:  # single channel -> convert
                loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_GRAY2BGR)
            QMessageBox.information(self, "Image Info", f"Loaded image shape: {loaded_image.shape}, dtype: {loaded_image.dtype}")

            img = loaded_image.copy()
            img_overlay = img.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

            zoom_factor = 1.0
            pan_offset_x = 0
            pan_offset_y = 0

            cv2.namedWindow("Image")
            cv2.namedWindow("Binary Mask")
            cv2.setMouseCallback("Image", draw_roi)
            update_display()

    def select_object_class(self):
        global current_class
        current_class = 'object'
        QMessageBox.information(self,"Selected class: object")

    def select_background_class(self):
        global current_class
        current_class = 'background'
        QMessageBox.information(self,"Selected class: background")

    def clear_drawing(self):
        global img_overlay, points, mask_created, mask
        global object_count, background_count
        if img is None:
            QMessageBox.information(self,"No image loaded yet.")
            return
        img_overlay[:] = img[:]
        points.clear()
        mask_created = False
        object_count = 0
        background_count = 0
        mask[:] = 0
        update_display()
        QMessageBox.information(self,"Cleared all ROIs and reset mask.")

    def save_mask(self):
        global mask, mask_created
        if not mask_created:
            QMessageBox.information(self,"尚未标记满 5 个 background ROI 和 3 个 object ROI，无法保存。")
            return
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        QMessageBox.information(self,"请选择保存二值掩码的文件路径和名称...")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", "", "TIFF Files (*.tif);;All Files (*)", options=options
        )
        if file_path:
            tiff.imwrite(file_path, mask)
            QMessageBox.information(self,f"Binary mask saved to {file_path}")

    def overlay_mask_on_image(self):
        global mask_created
        if not mask_created:
            QMessageBox.information(self,"尚未标记满 5 个 background ROI 和 3 个 object ROI，无法叠加。")
            return
        overlay_mask()
        update_display()

    def exit_app(self):
        QMessageBox.information(self, "Exiting application...")
        QApplication.quit()  # or sys.exit(0)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
