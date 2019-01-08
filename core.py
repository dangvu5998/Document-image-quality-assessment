'''
Document image quality assessment
'''
import numpy as np
import cv2
from .models import build_cnn_model
from .utils import get_document_corners, generate_patches, four_point_transform, image_normalize

class DIQA2:
    '''
    Document Image Quality Assessment
    Model version 2
    '''
    def __init__(self, model_path):
        self.model = build_cnn_model()
        self.model.load_weights(model_path)

    def get_quality_score(self, img, document_crop=True, img_type='RGB', blank_threshold=0.95):
        '''
        Get quality document image score
        '''
        if document_crop:
            corners = get_document_corners(img, img_type=img_type)
            img = four_point_transform(img, corners)
        patches = generate_patches(img, img_type=img_type, patch_size=(48, 48),
                                   blank_threshold=blank_threshold,
                                   col_step=48,
                                   row_step=48)
        patches_normalized = np.asarray(list(map(image_normalize, patches)))
        score = np.mean(self.model.predict(patches_normalized))
        return score

    def assessment(self, img, threshold=0.8, document_crop=True,
                   img_type='RGB', blank_threshold=0.9):
        '''
        Assess document image
        '''
        if document_crop:
            corners = get_document_corners(img, img_type=img_type)
            img = four_point_transform(img, corners)

        scale = 1820/img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img = img[200:-200, 200:-200]
        # break document into samples for assessment
        img_height, img_width = img.shape[:2]
        sample_width = int(img_width/5)
        sample_height = int(img_height/5)
        samples = []
        ret = np.ones((5, 5), dtype=np.float32)
        document_check_count = 0
        for i in range(5):
            for j in range(5):
                sample = img[i*sample_height:min((i+1)*sample_height, img_height), \
                    j*sample_width:min((j+1)*sample_width, img_width)]
                samples.append(sample)
                patches = generate_patches(sample, img_type=img_type,
                                           patch_size=(48, 48),
                                           blank_threshold=blank_threshold,
                                           col_step=48,
                                           row_step=48)
                patches_normalized = np.asarray(list(map(image_normalize, patches)))
                if patches_normalized.size > 10:
                    document_check_count += 1
                    quality = np.mean(self.model.predict(patches_normalized))
                    ret[i, j] = quality
        print(np.all(ret > threshold))
        return ret
        # return document_check_count >= 5 and np.all(ret > threshold), ret
