import numpy as np
import utils3d
import cv2

mask = np.random.rand(128, 128) > 0.99
print('original', mask.sum())

upsampled_mask = utils3d.numpy.masked_nearest_resize(
    mask=mask,
    size=(256, 256),
)
cv2_upsampled_mask = cv2.resize(mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
print('upsampled', upsampled_mask.sum(), cv2_upsampled_mask.sum())

downsampled_mask = utils3d.numpy.masked_nearest_resize(
    mask=mask,
    size=(64, 64),
)
cv2_downsampled_mask = cv2.resize(mask.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
print('downsampled', downsampled_mask.sum(), cv2_downsampled_mask.sum())


cv2.imwrite('original_mask.png', mask.astype(np.uint8) * 255)
cv2.imwrite('upsampled_mask.png', upsampled_mask.astype(np.uint8) * 255)
cv2.imwrite('cv2_upsampled_mask.png', cv2_upsampled_mask.astype(np.uint8) * 255)
cv2.imwrite('downsampled_mask.png', downsampled_mask.astype(np.uint8) * 255)
cv2.imwrite('cv2_downsampled_mask.png', cv2_downsampled_mask.astype(np.uint8) * 255)