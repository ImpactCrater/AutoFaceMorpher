"""
::

  Face averager

  Usage:
    averager.py --images=<images_folder> [--blur] [--plot]
              [--background=(black|transparent|average)]
              [--width=<width>] [--height=<height>]
              [--out=<filename>] [--destimg=<filename>]

  Options:
    -h, --help             Show this screen.
    --images=<folder>      Folder to images (.jpg, .jpeg, .png)
    --blur                 Flag to blur edges of image [default: False]
    --width=<width>        Custom width of the images/video [default: 500]
    --height=<height>      Custom height of the images/video [default: 600]
    --out=<filename>       Filename to save the average face [default: result.png]
    --destimg=<filename>   Destination face image to overlay average face
    --plot                 Flag to display the average face [default: False]
    --background=<bg>      Background of image to be one of (black|transparent|average) [default: average]
    --version              Show version.
"""

from docopt import docopt
import os
import cv2
import numpy as np

import locator
import aligner
import warper
import blender
import plotter

def ListImagePaths(imageFolder):
  for fname in os.listdir(imageFolder):
    if (fname.lower().endswith('.jpg') or
       fname.lower().endswith('.png') or
       fname.lower().endswith('.jpeg')):
      yield os.path.join(imageFolder, fname)

def UnsharpMasking(sourceImage, radius = (5, 5), sd = 2.5):
  """
  :param sourceImage: float32 BGR image.
  :param radius: taple(n, m): radius of Gaussian blur. it is must be odd number.
  :param sd: standard deviation.
  :returns: float32 BGR image.
  """
  blured = cv2.GaussianBlur(sourceImage, radius, sd)
  return cv2.addWeighted(sourceImage, 2.00, blured, -1.00, 0)

def AdjustTone(sourceImage):
  """
  :param sourceImage: float32 or float64 BGR image.
  :returns: float32 or float64 BGR image(range: 0.0 to 255.0).
  """
  adjustedImage = (sourceImage.max() - sourceImage.min()) * (-2.0 * cv2.pow(((sourceImage - sourceImage.min()) / (sourceImage.max() - sourceImage.min())), 3)
    + 3.0 * cv2.pow(((sourceImage - sourceImage.min()) / (sourceImage.max() - sourceImage.min())), 2))
  adjustedImage = cv2.normalize(adjustedImage, 0.0, 255.0, norm_type=cv2.NORM_MINMAX)
  adjustedImage = cv2.addWeighted(sourceImage, 0.9, adjustedImage, 0.1, 0)
  adjustedImage = adjustedImage.clip(0.0, 255.0)
  return adjustedImage

def loadImagePoints(path, desiredSize, background):
  sourceImage = cv2.imread(path)
  points = locator.face_points(sourceImage, background)

  if len(points) == 0:
    print('No face in %s' % path)
    return None, None
  else:
    return aligner.ResizeAlign(sourceImage, points, desiredSize)

def averager(imagePaths, destinationFilename = None, width = 500, height = 600, background = 'average',
             blurEdges = False, outputFilename = 'result.png', plot = False):

  size = (height, width)
  images = []
  pointSet = []
  counter = 0
  print("Now preparating images...")
  for path in imagePaths:
    sourceImage, points = loadImagePoints(path, size, background)
    if sourceImage is not None:
      images.append(sourceImage)
      pointSet.append(points)
    counter += 1
    print(counter)

  if len(images) == 0:
    raise FileNotFoundError('Could not find any valid images.' + ' Supported formats are .jpg, .png, .webp')

  if destinationFilename is not None:
    destinationImage, destinationPoints = loadImagePoints(destinationFilename, size, background)
    if destinationImage is None or destinationPoints is None:
      raise Exception('No face or detected face points in dest img: ' + destinationFilename)
  else:
    destinationImage = np.zeros(images[0].shape, np.uint8)
    destinationPoints = locator.average_points(pointSet)

  print("Now averaging and mixing images...")
  print("Maybe it takes a few minutes...")
  nImages = len(images)
  resultImage = np.zeros(images[0].shape, np.float32)
  counter = 0
  for i in range(0, nImages):
    resultImage += warper.warp_image(images[i], pointSet[i], destinationPoints, size, np.float64)
    counter += 1
    print(counter)

  resultImage = resultImage / nImages
  resultImage = (resultImage - np.mean(resultImage)) / np.std(resultImage) * 72 + 72
  minimumSize = min(size)
  unsharpRadius = int(minimumSize / 128)
  if unsharpRadius % 2 == 0:
    unsharpRadius += 1
  if unsharpRadius < 3:
    unsharpRadius = 3
  resultImage = UnsharpMasking(resultImage, radius = (unsharpRadius, unsharpRadius), sd = 2.5)
  resultImage = UnsharpMasking(resultImage, radius = (unsharpRadius, unsharpRadius), sd = 2.5)
  resultImage = AdjustTone(resultImage)

  facePixelIndexes = np.nonzero(resultImage)
  destinationImage[facePixelIndexes] = resultImage[facePixelIndexes]

  mask = blender.mask_from_points(size, destinationPoints)
  mask = mask.astype(np.float64)
  if blurEdges:
    blurKernelSize = 9
    mask = cv2.blur(mask, (blurKernelSize, blurKernelSize))

  if background in ('transparent', 'average'):
    destinationImage = np.dstack((destinationImage, mask))

    if background == 'average':
      averageBackground = locator.average_points(images)
      destinationImage = blender.overlay_image(destinationImage, mask, averageBackground)

  destinationImage = cv2.normalize(destinationImage, 0.0, 65535.0, norm_type=cv2.NORM_MINMAX)
  destinationImage = destinationImage.astype(np.uint16)
  print('Averaged {} images'.format(nImages))
  #cv2.imwrite("./mask.png", mask)
  cv2.imwrite(outputFilename, destinationImage)

def main():
  args = docopt(__doc__, version='Face Averager 1.0')
  try:
    averager(ListImagePaths(args['--images']), args['--destimg'],
             int(args['--width']), int(args['--height']),
             args['--background'], args['--blur'], args['--out'], args['--plot'])
  except Exception as e:
    print(e)


if __name__ == "__main__":
  main()
