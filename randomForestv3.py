# from skimage.data import cells3d
# from skimage.io import imsave, imread
# import napari
# import numpy as np
# import matplotlib.pyplot as plt
# from pyclesperanto_prototype import imshow

# # image = imread('C:/Segment/TestData20241226/TestData20241226/2/ROI07(0.25)_[Mask]FL1_on_Overlay[2].tif')
# image = imread('C:/ProjectCode/GTClassifier/demo/APP1_Hippo001_overlay.tif')
# image_ch1 = image[:,:, 0]
# # imshow(image_ch1/255)
# image_ch2 = image[:,:, 1]
# # imshow(image_ch2)
# filename = 'C:/ProjectCode/GTClassifier/demo/APP1_Hippo001_Microglia_MASK.tif'
            
# annotation = imread(filename)
# fix, axs = plt.subplots(2,2, figsize=(10,10))

# # imshow(image_ch1, plot=axs[0,0], colormap="Greens_r")
# # imshow(image_ch2, plot=axs[0,1], colormap="Purples_r")
# # imshow(annotation, labels=True, plot=axs[1,0])
# # imshow(image_ch1, continue_drawing=True, plot=axs[1,1], colormap="Greens_r", alpha=0.5)
# # imshow(image_ch2, continue_drawing=True, plot=axs[1,1], colormap="Purples_r", alpha=0.5)
# # imshow(annotation, labels=True, plot=axs[1,1], alpha=0.5)

# from apoc import PixelClassifier

# # define features
# features = "sobel_of_gaussian_blur=2 laplace_box_of_gaussian_blur=2 gaussian_blur=2 sobel_of_gaussian_blur=4 laplace_box_of_gaussian_blur=4 gaussian_blur=4"

# # this is where the model will be saved
# cl_filename = 'test.cl'

# clf = PixelClassifier(opencl_filename=cl_filename)
# clf.train(features=features, ground_truth=annotation, image=[image_ch1, image_ch2])

# result = clf.predict(image=[image_ch1, image_ch2])
# imshow(result, labels=True)

# random forest model
from sklearn.ensemble import RandomForestClassifier
import tifffile as tiff
from skimage.io import imread, imshow
import numpy as np
from numpy import ones
import napari
import matplotlib.pyplot as plt
from skimage import filters, exposure
import pylab 
from tqdm import tqdm
from skimage.data import cells3d
from skimage.io import imsave, imread
import stackview
from pyclesperanto_prototype import imshow
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from skimage.segmentation import random_walker
from skimage.measure import label, regionprops
from skimage.filters import frangi
from skimage.feature import graycomatrix, graycoprops
def extract_feature_stack(image):
    feature_stack = []
    for c in range(image.shape[0]):
        channel = image[c]
        # 原始强度
        feature_stack.append(channel)
        # GLCM纹理（对比度、能量）
        glcm = graycomatrix(channel, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').reshape(channel.shape)
        energy = graycoprops(glcm, 'energy').reshape(channel.shape)
        feature_stack.append(contrast)
        feature_stack.append(energy)
        # feature_stack = [
        #     channel.ravel(),  # 原始强度
        #     contrast.ravel(),  # 对比度 
        #     energy.ravel(),  # 能量
            
        # ]
    return feature_stack  # 堆叠为特征张量（H×W×F）
def extract_glcm_features(image, window_size=5, levels=64, distance=1):
    """
    提取多通道图像的GLCM纹理特征（对比度、能量）堆栈
    
    参数：
    - image: 输入的多通道图像，形状为 (Channels, Height, Width)
    - window_size: 邻域窗口大小（奇数）
    - levels: 灰度量化级数
    - distance: GLCM计算的像素距离
    
    返回：
    - feature_stack: 特征堆栈，形状 (Height, Width, Features)
    """
    assert window_size % 2 == 1, "窗口大小必须为奇数"
    radius = window_size // 2
    channels, h, w = image.shape
    feature_stack = np.zeros((h, w, channels * 3))  # 每个通道3个特征（原始强度+对比度+能量）
    
    # 对每个通道单独处理
    for c in range(channels):
        channel_img = image[c]
        
        # 将图像归一化到 [0, levels-1]
        min_val = channel_img.min()
        max_val = channel_img.max()
        if max_val == min_val:
            normalized = np.zeros_like(channel_img)
        else:
            normalized = ((channel_img - min_val) / (max_val - min_val) * (levels - 1)).astype(np.uint8)
        
        # 预填充边界为0（避免边缘越界）
        padded = np.pad(normalized, pad_width=radius, mode='constant')
        
        # 遍历每个像素
        for i in tqdm(range(h), desc=f"Processing Channel {c+1}/{channels}"):
            for j in range(w):
                # 提取邻域窗口
                window = padded[i:i+window_size, j:j+window_size]
                
                # 计算GLCM
                glcm = graycomatrix(
                    window,
                    distances=[distance],
                    angles=[0],       # 可扩展多个角度并取平均
                    levels=levels,
                    symmetric=True,
                    normed=True
                )
                
                # 提取对比度和能量
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                
                # 存储特征（原始强度需归一化到[0,1]）
                feature_stack[i, j, c*3] = channel_img[i, j] / max_val if max_val != 0 else 0
                feature_stack[i, j, c*3+1] = contrast
                feature_stack[i, j, c*3+2] = energy
                
    return feature_stack

def generate_feature_stack(image):
    # determine features
    # histogram = exposure.histogram(image)[0]
    intensity = image
    # median = filters.median(intensity)
    blurred = filters.gaussian(intensity, sigma=2)
    edges = filters.sobel(blurred)
    # threshold = filters.threshold_otsu(edges)
    # threshold = filters.threshold_li(blurred)
    # hessian = filters.frangi(edges)
    # collect features in a stack
    # The ravel() function turns a nD image into a 1-D image.
    # We need to use it because scikit-learn expects values in a 1-D format here. 
    feature_stack = [
        intensity.ravel(),  #intensity feature
        edges.ravel(),  #edge detection feature
        # median.ravel(), #noise filter feature
        blurred.ravel(),#noise filter feature
        # threshold.ravel(),
        # threshold_otsu.ravel(),
        # histogram.ravel(),
        # hessian.ravel(),
        # intensity.ravel()
    ]
    return np.asarray(feature_stack)     # return stack as numpy-array


## formating data
def format_data(feature_stack, annotation):
    # reformat the data to match what scikit-learn expects
    # transpose the feature stack
    X = np.array(feature_stack).T
    # make the annotation 1-dimensional
    # annotation[annotation > 127] = 1
    y = annotation.ravel()
    
    # remove all pixels from the feature and annotations which have not been annotated
    # mask = y > 0
    # X = X[mask]
    # y = mask.ravel()

    x_pos = X[np.where(y == 1)]
    x_neg = X[np.where(y == 2)]

    y_pos = y[np.where(y == 1)]
    y_neg = y[np.where(y == 2)]

    X = np.concatenate((x_pos, x_neg))
    y = np.concatenate((y_pos, y_neg))
    return X, y
# imageraw = imread('C:/Segment/TestData20241226/TestData20241226/2/ROI07(0.25)_[Mask]FL1_on_Overlay[2].tif')
imageraw = imread('C:\ProjectCode\TGClassifier\demo\demopath.tif')
# imageraw = imread('C:/ProjectCode/GTClassifier/demo/demopath.tif')
# plt.imshow(image.astype('uint8'))
image = imageraw[:,:,1] #2 for blue dapi 1 for green 

# image = ones((4635,4641,3))
# plt.imshow(image)
stackview.imshow(image, colormap='Greens')
# plt.show()
# print(mpl.get_backend())
mask_posi = tiff.imread(r"C:\ProjectCode\TGClassifier\demo\Pos_mask_flag.tif")
mask_nega = tiff.imread(r"C:\ProjectCode\TGClassifier\demo\Neg_mask_flag.tif")
mask = np.zeros(mask_posi.shape)
mask[mask_posi>0] = 1
mask[mask_nega>0] = 2

# maskraw[maskraw < 127] = 0 
annotation = mask

class FrangiFilter():
    def __init__(self, image):
        self.image = image
    def generate_annotation(self):
        # Apply Frangi filter to select vessel-like regions
        vessel_mask = frangi(self.image) > 0.5  # Adjust the threshold as needed
        # Load the manual mask
        manual_mask_path = 'C:/ProjectCode/GTClassifier/demo/manumask.tif'
        manual_mask = tiff.imread(manual_mask_path)
        # Create the final label mask
        final_label = np.zeros(image.shape, dtype=np.uint8)
        # Assign background label to vessel-like regions
        final_label[vessel_mask] = 200
        # Assign object label to regions inside the manual mask
        final_label[manual_mask > 0] = 1
        # Display the final label mask
        # stackview.imshow(final_label, vmin=0, vmax=200)
        # Continue with the rest of the processing
        annotation = final_label
        return annotation


# Define intensity threshold and region volume threshold
class Annotation:
    def __init__(self, image, intensity_threshold=200, region_volume_threshold=1000):
        self.image = image
        self.intensity_threshold = intensity_threshold
        self.region_volume_threshold = region_volume_threshold

    def generate_annotation(self):
        # Generate binary mask for object label based on intensity threshold
        object_mask = self.image > self.intensity_threshold

        # Generate binary mask for background label based on region volume threshold
        labeled_image = label(object_mask)
        background_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        for region in regionprops(labeled_image):
            if region.area > self.region_volume_threshold:
                for coord in region.coords:
                    background_mask[coord[0], coord[1]] = 2  # Label background regions with 2

        # Combine object and background masks into a single annotation mask
        annotation = np.zeros(self.image.shape[:2], dtype=np.uint8)
        annotation[object_mask] = 1  # Label object regions with 1
        annotation[background_mask == 2] = 200  # Label background regions with 200

        return annotation

# Display the annotation
# annotation = Annotation(image).generate_annotation()
# stackview.imshow(maskraw, vmin=0, vmax=2)

# Load the demo mask
demo_mask_path = r"C:\ProjectCode\TGClassifier\demo\demomask.tif"
# demo_mask_path = "C:/ProjectCode/TGClassifier/demo/APP1_Hippo005_Microglia_MASK.tif"
GTdemo_mask = tiff.imread(demo_mask_path)

# # show feature images
feature_stack = generate_feature_stack(image)
# feature_stack = extract_glcm_features(imageraw)
# feature_stack = extract_feature_stack(image)
fig, axes = plt.subplots(1, 3, figsize=(10,10))

# reshape(image.shape) is the opposite of ravel() here. We just need it for visualization.
axes[0].imshow(feature_stack[0].reshape(image.shape), cmap=plt.cm.Greens)
axes[1].imshow(feature_stack[1].reshape(image.shape), cmap=plt.cm.Greens)
axes[2].imshow(feature_stack[2].reshape(image.shape), cmap=plt.cm.Greens)
# axes[2].imshow(GTdemo_mask, cmap=plt.cm.grey)

X, y = format_data(feature_stack, annotation)

# 提取特征并训练模型
# X = extract_features(image)
# y = annotation  # 需提供标注掩膜
# rf = RandomForestClassifier(n_estimators=100)
# rf.fit(X.reshape(-1, X.shape[-1]), y.ravel())
print("input shape", X.shape) # x is mumber of pixcels
print("annotation shape", y.shape) # y is number of features

import time
# train and predict pixels
classifier = RandomForestClassifier(n_estimators = 45,max_depth=40, random_state=42)

time_train_start = time.time()
classifier.fit(X, y)
# RandomForestClassifier(max_depth=2, random_state=0)
time_train_end = time.time()
print("train time : %.2f s"%(time_train_end - time_train_start))
res = classifier.predict(feature_stack.T) - 1 # we subtract 1 to make background = 0
time_predict_end = time.time()
print("predict time : %.2f s"%(time_predict_end - time_train_end))
# res = np.invert(resraw)
# res[res == 1] = 0
# stackview.imshow(res.reshape(image.shape))
# plt.imshow(res.reshape(image.shape))
# plt.title('Segmentation Result')
# plt.axis('off')
# plt.show()

# res = classifier.predict(X.reshape(-1, X.shape[-1]))
# Ensure the demo mask has the same shape as the result
assert GTdemo_mask.shape == res.reshape(image.shape).shape, "Shape mismatch between result and demo mask"
# GTdemo_mask = np.rot90(GTdemo_mask, 2)
# Display the GTdemo_mask and the result in the same plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(GTdemo_mask, cmap=plt.cm.gray)
axes[0].set_title('Ground Truth Mask')

res[res == 0] = 255
res[res != 255] = 0
axes[1].imshow(res.reshape(image.shape), cmap=plt.cm.gray)
axes[1].set_title('Segmentation Result')

plt.show()


y_true = GTdemo_mask.ravel()
y_pred = res.ravel()# reshape to 1-D nparray
# Calculate the Dice score
def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

# Calculate Dice score between the result and the demo mask
dice = dice_score(GTdemo_mask, res.reshape(image.shape))
print(f"Dice score: {dice}")


f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"Dice score: {dice}")
print(f"F1 score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save the result as a TIFF file
output_path = "C:/ProjectCode/TGClassifier/demo/segmentation_result.tif"
tiff.imwrite(output_path, res.reshape(image.shape).astype(np.uint8))
# tiff.imwrite(output_path, res)
print(f"Result saved to {output_path}")
# RandomForest 特征重要性
# 获取重要性
rfr = classifier
feat_important = rfr.feature_importances_
# 特征名
feat_name = ['intensity', 'edges', 'blurred']
plt.barh(range(len(feat_name)),feat_important,tick_label=feat_name)
