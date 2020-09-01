"""
Align face and image sizes
"""
import cv2
import numpy as np

def PositiveCap(num):
  """ Cap a number to ensure positivity

  :param num: positive or negative number
  :returns: (overflow, capped_number)
  """
  if num < 0:
    return 0, abs(num)
  else:
    return num, 0

def RoiCoordinates(boundingRectangle, desiredSize, scalingFactor):
  """ Align the rectangle into the center and return the top-left coordinates
  within the new size. If rect is smaller, we add borders.

  :param boundingRectangle: (x, y, w, h) bounding rectangle of the face
  :param desiredSize: (width, height) are the desired dimensions
  :param scalingFactor: scaling factor of the rectangle to be resized
  :returns: 4 numbers. Top-left coordinates of the aligned ROI.
    (x, y, border_x, border_y). All values are > 0.
  """
  rectangle_x, rectangle_y, rectangle_w, rectangle_h = boundingRectangle
  desiredHeight, desiredWidth = desiredSize
  mid_x = int((rectangle_x + rectangle_w / 2) * scalingFactor)
  mid_y = int((rectangle_y + rectangle_h / 2) * scalingFactor)
  roi_x = mid_x - int(desiredWidth / 2)
  roi_y = mid_y - int(desiredHeight / 2)

  roi_x, border_x = PositiveCap(roi_x)
  roi_y, border_y = PositiveCap(roi_y)
  return roi_x, roi_y, border_x, border_y

def CalculateScalingFactor(boundingRectangle, desiredSize):
  """ Calculate the scaling factor for the current image to be
      resized to the new dimensions

  :param rect: (x, y, w, h) bounding rectangle of the face
  :param size: (width, height) are the desired dimensions
  :returns: floating point scaling factor
  """
  desiredHeight, desiredWidth = desiredSize
  rectangle_w, rectangle_h = boundingRectangle[2:]
  heightRatio = rectangle_h / desiredHeight
  widthRatio = rectangle_w / desiredWidth
  scalingFactor = 1
  if heightRatio > widthRatio:
    newRectangle_h = 0.95 * desiredHeight
    scalingFactor = newRectangle_h / rectangle_h
  else:
    newRectangle_w = 0.95 * desiredWidth
    scalingFactor = newRectangle_w / rectangle_w
  return scalingFactor

def ResizeImage(sourceImage, scalingFactor):
  """ Resize image with the provided scaling factor

  :param img: image to be resized
  :param scalingFactor: scaling factor for resizing the image
  """
  currentHeight, currnetWidth = sourceImage.shape[:2]
  newScaledHeight = int(scalingFactor * currentHeight)
  newScaledWidth = int(scalingFactor * currnetWidth)

  return cv2.resize(sourceImage, (newScaledWidth, newScaledHeight), interpolation = cv2.INTER_LANCZOS4)

def ResizeAlign(sourceImage, points, desiredSize):
  """ Resize image and associated points, align face to the center
    and crop to the desired size

  :param img: image to be resized
  :param points: *m* x 2 array of points
  :param size: (height, width) tuple of new desired size
  """
  desiredHeight, desiredWidth = desiredSize

  # Resize image based on bounding rectangle
  boundingRectangle = cv2.boundingRect(np.array([points], np.int32))
  scalingFactor = CalculateScalingFactor(boundingRectangle, desiredSize)
  resizedImage = ResizeImage(sourceImage, scalingFactor)

  # Align bounding rect to center
  currentHeight, currnetWidth = resizedImage.shape[:2]
  roi_x, roi_y, border_x, border_y = RoiCoordinates(boundingRectangle, desiredSize, scalingFactor)
  roi_h = np.min([desiredHeight, desiredHeight - border_y, currentHeight, currentHeight - roi_y])
  roi_w = np.min([desiredWidth, desiredWidth - border_x, currnetWidth, currnetWidth - roi_x])

  # Crop to supplied size
  crop = resizedImage[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
  crop = cv2.normalize(crop, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)

  paddingTop = border_y
  paddingBottom = desiredHeight - roi_h - border_y
  paddingLeft = border_x
  paddingRight = desiredWidth - roi_w - border_x
  crop = cv2.copyMakeBorder(crop, paddingTop, paddingBottom, paddingLeft, paddingRight, cv2.BORDER_REPLICATE)

  # Scale and align face points to the crop
  points[:, 0] = (points[:, 0] * scalingFactor) + (border_x - roi_x)
  points[:, 1] = (points[:, 1] * scalingFactor) + (border_y - roi_y)

  return (crop, points)
