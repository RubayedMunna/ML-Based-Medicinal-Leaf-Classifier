#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import joblib
from skimage.io import imread
from skimage.transform import resize, rescale
from skimage.feature import hog
from skimage.color import rgb2gray
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.svm import SVC
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)

def resize_all(src, pklname, include, width=150, height=None):
    """
    Load images from path, resize them, and write them as arrays to a dictionary,
    together with labels and metadata. The dictionary is written to a pickle file.
    """
    height = height if height is not None else width
    data = dict()
    data['description'] = 'resized ({0}x{1})medicinal leaf images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    pklname = f"{pklname}_{width}x{height}px.pkl"

    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height))
                    data['label'].append(subdir[:-4])  # Remove 'Leaf' from label
                    data['filename'].append(file)
                    data['data'].append(im)
    joblib.dump(data, pklname)
    return data

def plot_bar(y, loc='left', relative=True):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]
    if relative:
        counts = 100 * counts[sorted_index] / len(y)
        ylabel_text = '% count'
    else:
        counts = counts[sorted_index]
        ylabel_text = 'count'
    xtemp = np.arange(len(unique))
    plt.bar(xtemp + n * width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, unique, rotation=45)
    plt.xlabel('leaf type')
    plt.ylabel(ylabel_text)

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return np.array([rgb2gray(img) for img in X])

class HogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, y=None, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        def local_hog(X):
            return hog(X, orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
        return np.array([local_hog(img) for img in X])

def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100 * cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()
    np.fill_diagonal(cmx_zero_diag, 0)
    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(12, 3)
    [a.set_xticks(range(len(cmx) + 1)) for a in ax]
    [a.set_yticks(range(len(cmx) + 1)) for a in ax]
    im1 = ax[0].imshow(cmx, vmax=vmax1)
    ax[0].set_title('as is')
    im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
    ax[1].set_title('%')
    im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
    ax[2].set_title('% and 0 diagonal')
    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) for divider in dividers]
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.tight_layout()

def main():
    # Data path
    data_path = r'D:\4-2\29(Rubayed)\Machine Learning Lab\Project\MedicinalLeafDataset'
    print("Directory contents:", os.listdir(data_path))

    # Resize images
    base_name = 'medicinal_leaf'  # Corrected typo from 'medician_leaf'
    width = 80
    include = {'AloeveraLeaf', 'BambooLeaf', 'CorianderLeaf', 'GingerLeaf', 'MangoLeaf', 'PepperLeaf', 'TurmericLeaf'}
    data = resize_all(src=data_path, pklname=base_name, width=width, include=include)

    # Load and inspect data
    data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    print('Number of samples:', len(data['data']))
    print('Keys:', list(data.keys()))
    print('Description:', data['description'])
    print('Image shape:', data['data'][0].shape)
    print('Labels:', np.unique(data['label']))
    print('Label counts:', Counter(data['label']))

    # Plot sample images
    labels = np.unique(data['label'])
    fig, axes = plt.subplots(1, len(labels))
    fig.set_size_inches(15, 4)
    fig.tight_layout()
    for ax, label in zip(axes, labels):
        idx = data['label'].index(label)
        ax.imshow(data['data'][idx])
        ax.axis('off')
        ax.set_title(label)
    plt.show()

    # Prepare data for training
    X = np.array(data['data'])
    y = np.array(data['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Plot train/test distribution
    plt.suptitle('Relative amount of photos per type')
    plot_bar(y_train, loc='left')
    plot_bar(y_test, loc='right')
    plt.legend(['train ({0} photos)'.format(len(y_train)), 'test ({0} photos)'.format(len(y_test))])
    plt.show()

    # HOG example
    if os.path.exists('Coriander-Leaf.jpg'):
        coriander = imread('Coriander-Leaf.jpg', as_gray=True)
        coriander = rescale(coriander, 1/3, mode='reflect')
        coriander_hog, coriander_hog_img = hog(coriander, pixels_per_cell=(14, 14), cells_per_block=(2, 2),
                                               orientations=9, visualize=True, block_norm='L2-Hys')
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(8, 6)
        [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False) for a in ax]
        ax[0].imshow(coriander, cmap='gray')
        ax[0].set_title('coriander')
        ax[1].imshow(coriander_hog_img, cmap='gray')
        ax[1].set_title('hog')
        plt.show()
        print('Number of pixels:', coriander.shape[0] * coriander.shape[1])
        print('Number of HOG features:', coriander_hog.shape[0])
    else:
        print("Image 'Coriander-Leaf.jpg' not found, skipping HOG visualization.")

    # Prepare training data with transformers
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(pixels_per_cell=(14, 14), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')
    scalify = StandardScaler()
    X_train_gray = grayify.fit_transform(X_train)
    X_train_hog = hogify.fit_transform(X_train_gray)
    X_train_prepared = scalify.fit_transform(X_train_hog)
    print("Prepared training data shape:", X_train_prepared.shape)

    # Train SGDClassifier
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train_prepared, y_train)
    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)
    y_pred = sgd_clf.predict(X_test_prepared)
    print("First 25 predictions vs actual:", np.array(y_pred == y_test)[:25])
    print('Percentage correct (SGD):', 100 * np.sum(y_pred == y_test) / len(y_test))

    # Example confusion matrix
    labels = ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no']
    predictions = ['yes', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no']
    df = pd.DataFrame(np.c_[labels, predictions], columns=['true_label', 'prediction'])
    print("Example confusion matrix data:\n", df)
    cmx_example = confusion_matrix(labels, predictions, labels=['yes', 'no'])
    df_cmx = pd.DataFrame(cmx_example, columns=['yes', 'no'], index=['yes', 'no'])
    df_cmx.columns.name = 'prediction'
    df_cmx.index.name = 'label'
    print("Example confusion matrix:\n", df_cmx)
    plt.imshow(cmx_example)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()

    # SGD confusion matrix
    cmx = confusion_matrix(y_test, y_pred)
    print("SGD Confusion Matrix:\n", cmx)
    plot_confusion_matrix(cmx)
    plt.show()
    print('\nSorted unique labels:', sorted(np.unique(y_test)))

    # Pipeline with SGDClassifier
    HOG_pipeline = Pipeline([
        ('grayify', RGB2GrayTransformer()),
        ('hogify', HogTransformer(pixels_per_cell=(14, 14), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')),
        ('scalify', StandardScaler()),
        ('classify', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
    ])
    clf = HOG_pipeline.fit(X_train, y_train)
    print('Percentage correct (Pipeline SGD):', 100 * np.sum(clf.predict(X_test) == y_test) / len(y_test))

    # GridSearchCV
    param_grid = [
        {
            'hogify__orientations': [8, 9],
            'hogify__cells_per_block': [(2, 2), (3, 3)],
            'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)]
        },
        {
            'hogify__orientations': [8],
            'hogify__cells_per_block': [(3, 3)],
            'hogify__pixels_per_cell': [(8, 8)],
            'classify': [SGDClassifier(random_state=42, max_iter=1000, tol=1e-3), SVC(kernel='linear')]
        }
    ]
    grid_search = GridSearchCV(HOG_pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1, return_train_score=True)
    grid_res = grid_search.fit(X_train, y_train)
    joblib.dump(grid_res, 'hog_sgd_model.pkl')
    print("Best estimator:", grid_res.best_estimator_)
    print("Best score:", grid_res.best_score_)
    pp.pprint(grid_res.best_params_)
    best_pred = grid_res.predict(X_test)
    print('Percentage correct (GridSearchCV):', 100 * np.sum(best_pred == y_test) / len(y_test))

    # GridSearchCV confusion matrix
    cmx_svm = confusion_matrix(y_test, best_pred)
    print("GridSearchCV Confusion Matrix:\n", cmx_svm)
    plot_confusion_matrix(cmx, vmax1=225, vmax2=100, vmax3=12)
    plt.show()
    plot_confusion_matrix(cmx_svm, vmax1=225, vmax2=100, vmax3=12)
    plt.show()

if __name__ == "__main__":
    main()
