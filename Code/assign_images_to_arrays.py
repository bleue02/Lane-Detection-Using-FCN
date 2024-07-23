'''images_set'의 이미지를 두 개의 개별 배열에 할당합니다.
도로 이미지를 'X'에 할당하고 실제 마스크를 'Y'에 할당합니다.
'''
# 해당 py는 RAM 사용

from train_data_loader import ImageLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import random


def calculate_batch(path):
    num_images = 7252
    num_batches = num_images // 64 + 1
    X = []
    Y = []

    # ImageLoader 인스턴스 생성
    image_loader = ImageLoader(path=path, seed=42, batch_size=64, num_workers=4)
    train_loader = image_loader.load_images()

    for i in tqdm(range(num_batches)):
        batch = next(iter(train_loader))
        batch_images = batch[0]  # this contains the images
        batch_labels = batch[1]  # this contains 0s and 1s
        for ind, lb in enumerate(batch_labels):
            '''
            라벨 0은 이미지가 실제 이미지에 속함
            라벨 1은 이미지가 실제 마스크에 속함
            '''
            if lb == 0:
                X.append(batch_images[ind].numpy())  # Convert tensor to numpy array
            else:
                Y.append(np.mean(batch_images[ind].numpy(), axis=0))  # Y shape is (m, 256, 320)
        if i % 10 == 0:
            pass
            print(f'Batch {i}')
    # convert the lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

def image_show():
    X, Y = calculate_batch(path)
    X, Y = shuffle(X, Y, random_state=100)
    X = np.array(X[:4000]) # get 4000 training samples
    Y = np.array(Y[:4000])
    display(X.shape)
    display(Y.shape)

    # 마스크 세트를 정규화하고 모양을 변경합니다(Y)
    Y = (Y >= 100).astype('int').reshape(-1, 256, 320, 1)
    Y.min(), Y.max()
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=.1, random_state=100)

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of Y_train:", Y_train.shape)
    print("Shape of Y_val:", Y_val.shape)

    plt.figure(figsize=(10, 40))
    s, e = 80, 84
    index = 1

    for i, j in zip(X_train[s:e], Y_train[s:e]):
        plt.subplot(10, 2, index)
        plt.imshow(i / 255.)
        plt.title('Ground truth image')

        plt.subplot(10, 2, index + 1)
        plt.imshow(j, cmap='gray')
        plt.title('Ground truth mask')
        index += 2

if __name__ == '__main__':
    path = r'C:\Users\jdah5454\PycharmProjects\Lane Detection Usin FCN\dataset\tusimple_preprocessed\training'
    calculate_batch(path)
    image_show()
