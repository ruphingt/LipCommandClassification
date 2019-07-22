"""
SegmentMouth.py
    This file will extract out all mouth crop frames from videos in vidpath_train
    vidpath_test and save each video as a numpy array in npy file in data/train/filename.npy or
    data/test/filename.npy


"""
import os
import ntpath
import dlib
import cv2
import numpy as np
import imutils
import imageio
from scipy import ndimage
from scipy.misc import imresize, imsave
from skimage import io
from skimage.transform import resize


def get_mouth_frames(videoname):
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 80
    NUM_CHANNELS = 3
    FRAME_CAP = 70

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(facepredictorpath)

    cap = cv2.VideoCapture(videoname)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('framewidth', frameWidth, 'frameHeight', frameHeight, 'frameCount', frameCount)
    # Define the codec and create VideoWriter object
    outputname = ntpath.basename(videoname)

    normalize_ratio = None
    mouth_frames = []
    frames = []
    fc = 0
    ret = True

    # Store video frames in frames
    while (fc < FRAME_CAP and cap.isOpened()):
        mouth_crop_image = np.zeros((MOUTH_HEIGHT, MOUTH_WIDTH, NUM_CHANNELS))
        print('fc', fc)
        ret, frame = cap.read()
        if ret is True:

            print('ret', ret, 'frame.shape', frame.shape)
            frame = imutils.rotate(frame,90)

            print('frame shape ', frame.shape)
            dets = detector(frame, 1)
            print('length of dets ', len(dets))
            shape = None
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                i = -1
            if shape is None: # Detector doesn't detect face, just return as is
                print('No Face')
                continue
            mouth_points = []
            for part in shape.parts():
                i += 1
                if i < 48: # Only take mouth region
                    continue
                mouth_points.append((part.x,part.y))
            np_mouth_points = np.array(mouth_points)

            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1])
                mouth_right = np.max(np_mouth_points[:, :-1])
                normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

            new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
            resized_img = resize(frame, new_img_shape)
            print('Resized_img ', resized_img.shape)

            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
            mouth_r = mouth_l + MOUTH_WIDTH
            mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
            mouth_b = mouth_t + MOUTH_HEIGHT

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r, :]
            print('mouth_crop_image', mouth_crop_image.shape, 'dtype', mouth_crop_image.dtype)
        else:
            break

        mouth_frames.append(mouth_crop_image)
        fc +=1

    # Truncate if frames > 70, pad with zeros if < 70
    if fc < FRAME_CAP:
        zero_pad = [np.zeros((MOUTH_HEIGHT, MOUTH_WIDTH, NUM_CHANNELS))] * (FRAME_CAP - fc)
        mouth_frames.extend(zero_pad)

    #for i, mouth_crop_image in enumerate(mouth_frames):
    #    framename = os.path.join(tgtpath, os.path.splitext(filename)[0] + '_' + str(i) + '.jpeg')
    #    imageio.imwrite(framename, mouth_crop_image)


    cap.release()
    cv2.destroyAllWindows()

    return mouth_frames


def generate_mouth_array(train_or_test):
    """
    Generate mouth sequence array of shape T x H x W x 3 = 70 x 80 x 100 x 3
    Each mouth sequence example will be stored in data/train or data/test folder
    Arguments:
        train_or_test - a string.
                        If the string is 'train', then data from Videos/train is processed to data/train
                        If the string is 'test', then data from Videos/test is processed to data/test
    """

    if train_or_test == 'train':
        dir = vidpath_train
    elif train_or_test == 'test':
        dir = vidpath_test
    else:
        raise ValueError

    mouth_batches = []

    for root, subFolder, files in os.walk(dir):
        for filename in files:
            print('filename', filename, 'subfolder ',subFolder, 'root', root)

            if filename.endswith(".mp4") or filename.endswith(".avi"):
                mouth_frames = get_mouth_frames(os.path.join(root, filename))

                if mouth_frames:
                    # 4D array T * H * W * 3
                    mouth_array = np.stack((mouth_frame for mouth_frame in mouth_frames), axis=0)
                    # Train or Test folder
                    #folder = os.path.split(os.path.dirname(root))[1]

                    # Class folder
                    classfolder = None
                    for command in CommandSet:
                        if command in filename:
                            print('command', command, 'filename', filename)
                            classfolder = command
                            break

                    #print('classfolder', classfolder)
                    if classfolder == None:
                        raise ValueError

                    # Npy file containing Video name
                    npyname = os.path.splitext(filename)[0]
                    # data/train/filename.npy or data/test/filename,npy
                    #newvideo = os.path.join(tgtpath,folder,classfolder, npyname +'.npy')
                    newvideo = os.path.join(tgtpath,train_or_test,classfolder, npyname +'.npy')
                    print('tgtpath', tgtpath, 'newvideo', newvideo)
                    np.save(newvideo, mouth_array)
                    print('mouth_array.shape ', mouth_array.shape)
                    #mouth_batches.append(mouth_array)

            else:
                continue

def normalize_mouth_array(train_or_test):
    """
    Normalize all numpy array in data/test and data/train and store them in data_norm/test
    and data_norm/train
    """

    if train_or_test == 'train':
        dir = os.path.join(tgtpath,'train')
    elif train_or_test == 'test':
        dir = os.path.join(tgtpath,'test')
    else:
        raise ValueError

    for root, subFolder, files in os.walk(dir):
        for filename in files:
            print('filename', filename, 'subfolder ',subFolder, 'root', root)

            if filename.endswith(".npy"):
                vid = np.load(os.path.join(root, filename))
                print('vidarray', vid.shape)
                #Mean across time T for each RGB channel
                means_rgb = vid.mean(axis=(0,1,2))
                norm_vid = (vid - vid.mean(axis=(0, 1, 2), keepdims=True)) / \
                           vid.std(axis=(0, 1, 2), keepdims=True)

                # Class folder
                classfolder = None
                for command in CommandSet:
                    if command in filename:
                        print('command', command, 'filename', filename)
                        classfolder = command
                        break

                print('classfolder', classfolder)
                if classfolder == None:
                    raise ValueError

                # Npy file containing Video name
                npyname = os.path.splitext(filename)[0]
                # data_norm/train/filename.npy or data_norm/test/filename,npy
                normvideo = os.path.join(normpath,train_or_test,classfolder, npyname +'.npy')
                print('normpath', normpath, 'normvideo', normvideo)
                np.save(normvideo, norm_vid)

            else:
                continue

def augmentData(horizontal_flip, tgtdir, train_or_test):

    if train_or_test == 'train':
        dir = os.path.join(tgtdir,'train')
    elif train_or_test == 'test':
        dir = os.path.join(tgtdir,'test')
    else:
        raise ValueError

    for root, subFolder, files in os.walk(dir):
        for filename in files:
            print('filename', filename, 'subfolder ',subFolder, 'root', root)

            if filename.endswith(".npy"):


                # Class folder
                classfolder = None
                for command in CommandSet:
                    if command in filename:
                        print('command', command, 'filename', filename)
                        classfolder = command
                        break

                print('classfolder', classfolder)
                if classfolder == None:
                    raise ValueError

                if horizontal_flip is True:
                    vid = np.load(os.path.join(root, filename))
                    flip_vid = np.flip(vid,2)
                    print('vid', vid.shape, 'flip_vid', flip_vid.shape)
                    # Npy file containing Video name
                    npyname = os.path.splitext(filename)[0] + '_flip.npy'
                    # data_norm/train/filename_flip.npy or data_norm/test/filename_flip.npy
                    flipvideo = os.path.join(normpath,train_or_test,classfolder, npyname)
                    print('normpath', normpath, 'flipvideo', flipvideo)
                    np.save(flipvideo, flip_vid)

            else:
                continue

# Path names
dirpath = os.getcwd()
tgtpath = os.path.join(dirpath,'data')
normpath = os.path.join(dirpath,'data_norm')
#normpath = os.path.join(dirpath,'testing')
vidpath_train = os.path.join(dirpath,'Videos/train/')
#vidpath_test = os.path.join(dirpath,'Videos/test/')
vidpath_test = os.path.join(dirpath,'testing/test/')
facepredictorpath = os.path.join(dirpath,'common', 'predictors', 'shape_predictor_68_face_landmarks.dat')
CommandSet = {
    "fanhui" : 1,
    "zhuomian" : 2,
    "jieping" : 3,
    "wifi" : 4,
    "jingyin" : 5,
    "shoudiantong" : 6,
    "tongzhilan" : 7,
    "zuijin_yingyong" : 8,
    "lanya" : 9,
    "suoping" : 10,
}


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(facepredictorpath)

# Preprocess the training data into data/train
#generate_mouth_array('train')
# Preprocess the test data into data/test
#generate_mouth_array('test')

# Normalize the RGB channels of the numpy array over speaking interval T
#normalize_mouth_array('train')
#normalize_mouth_array('test')

# Horizontally augment the data
#augmentData(True, normpath, 'train')
augmentData(True, normpath, 'test')



