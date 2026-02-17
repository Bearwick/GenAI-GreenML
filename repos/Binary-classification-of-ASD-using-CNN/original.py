from tensorflow import keras
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from os import path
from tensorflow.keras import models
import cv2
from os import listdir
from os.path import isfile, join
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator




import os
import numpy as np
from nibabel.testing import data_path
import cv2
from os import walk
import matplotlib.pyplot as plt
#import AugmentImages as imgAugment
import nibabel as nib
import pandas as pd
import re


import os
from os import walk
from nilearn import image as nli
from nilearn import plotting
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import pylab as plt
import nilearn as ni

def performPreprocess(folder, outfile) :

    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1
    
    example1_filename = folder + "/rest_1/rest.nii.gz"
    bold=nli.load_img(example1_filename)
    
    bold = bold.slicer[..., 5:]
    img = nli.mean_img(bold)#mean of the image
    plotting.view_img(img, bg_img=img)
    
    mean = nli.mean_img(bold) # resample image to computed mean image
    print([mean.shape, img.shape])
    
    resampled_img = nli.resample_to_img(img, mean)
    resampled_img.shape
    
    from nilearn import plotting
    plotting.plot_anat(img, title='original img', display_mode='z', dim=-1,
                       cut_coords=[-20, -10, 0, 10, 20, 30])
    plotting.plot_anat(resampled_img, title='resampled img', display_mode='z', dim=-1,
                       cut_coords=[-20, -10, 0, 10, 20, 30])
    
    for fwhm in range(1, 12, 5):
        smoothed_img = nli.smooth_img(mean, fwhm)
        plotting.plot_epi(smoothed_img, title="Smoothing %imm" % fwhm,
                         display_mode='z', cmap='magma')
    
    TR = bold.header['pixdim'][4]#get TR value of functional image
    
    func_d = nli.clean_img(bold, detrend=True, standardize=False, t_r=TR)
    
    # Plot the original and detrended timecourse of a random voxel
    x, y, z = [31, 14, 7]
    plt.figure(figsize=(12, 4))
    plt.plot(np.transpose(bold.get_fdata()[x, y, z, :]))
    plt.plot(np.transpose(func_d.get_fdata()[x, y, z, :]))
    plt.legend(['Original', 'Detrend']);
    
    func_ds = nli.clean_img(bold, detrend=True, standardize=True, t_r=TR)
    
    plt.figure(figsize=(12, 4))
    plt.plot(np.transpose(func_d.get_fdata()[x, y, z, :]))
    plt.plot(np.transpose(func_ds.get_fdata()[x, y, z, :]))
    plt.legend(['Detrend', 'Detrend+standardize']);
    
    plt.figure(figsize=(12, 5))
    plt.plot(np.transpose(func_d.get_fdata()[x, y, z, :]))
    plt.plot(np.transpose(func_ds.get_fdata()[x, y, z, :]))
    plt.legend(['Det.+stand.', 'Det.+stand.-confounds']);
    
    mean = nli.mean_img(bold)
    
    thr = nli.threshold_img(mean, threshold='95%')
    plotting.view_img(thr, bg_img=img)
    voxel_size = np.prod(thr.header['pixdim'][1:4])  # Size of 1 voxel in mm^3
    
    from nilearn.regions import connected_regions
    cluster = connected_regions(thr, min_region_size=1000. / voxel_size, smoothing_fwhm=1)[0]
    mask = nli.math_img('np.mean(img,axis=3) > 0', img=cluster)
    from nilearn.plotting import plot_roi
    plotting.plot_roi(mask, bg_img=img, display_mode='z', dim=-.5, cmap='magma_r');
    # Apply mask to original functional image
    from nilearn.masking import apply_mask
    
    all_timecourses = apply_mask(bold, mask)
    all_timecourses.shape
    
    from nilearn.masking import unmask
    img_timecourse = unmask(all_timecourses, mask)
    
    mean_timecourse = all_timecourses.mean(axis=1)
    plt.plot(mean_timecourse)
    
    # Import CanICA module
    from nilearn.decomposition import CanICA
    
    # Specify relevant parameters
    n_components = 5
    fwhm = 6.
    
    # Specify CanICA object
    canica = CanICA(n_components=n_components, smoothing_fwhm=fwhm,
                    memory="nilearn_cache", memory_level=2,
                    threshold=3., verbose=10, random_state=0, n_jobs=-1,
                    standardize=True)
    # Run/fit CanICA on input data
    canica.fit(bold)
    # Retrieve the independent components in brain space
    components_img = canica.masker_.inverse_transform(canica.components_)
    
    from nilearn.image import iter_img
    from nilearn.plotting import plot_stat_map
    
    for i, cur_img in enumerate(iter_img(components_img)):
        plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                      cut_coords=[0, 10, 20, 30], colorbar=True, bg_img=img)
    
    from scipy.stats.stats import pearsonr
    
    # Get data of the functional image
    orig_data = bold.get_fdata()
    
    # Compute the pearson correlation between the components and the signal
    curves = np.array([[pearsonr(components_img.get_fdata()[..., j].ravel(),
          orig_data[..., i].ravel())[0] for i in range(orig_data.shape[-1])]
            for j in range(canica.n_components)])
    
    # Plot the components
    fig = plt.figure(figsize=(14, 4))
    centered_curves = curves - curves.mean(axis=1)[..., None]
    plt.plot(centered_curves.T)
    plt.legend(['IC %d' % i for i in range(canica.n_components)])
    
    # Import DictLearning module
    from nilearn.decomposition import DictLearning
    
    # Specify the dictionary learning object
    dict_learning = DictLearning(n_components=n_components, n_epochs=1,
                                 alpha=1., smoothing_fwhm=fwhm, standardize=True,
                                 memory="nilearn_cache", memory_level=2,
                                 verbose=1, random_state=0, n_jobs=-1)
    
    # As before, let's fit the model to the data
    dict_learning.fit(bold)
    
    # Retrieve the independent components in brain space
    components_img = dict_learning.masker_.inverse_transform(dict_learning.components_)
    
    # Now let's plot the components
    for i, cur_img in enumerate(iter_img(components_img)):
        plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                      cut_coords=[0, 10, 20, 30], colorbar=True, bg_img=img)
    
    nib.save(components_img, outfile)

def performPreprocessAnat(folder, outfile) :
    print('Anat preprocessing started...')
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1
    
    example_filename = folder + "/anat_1/mprage.nii.gz"
    
    img=nib.load(example_filename)
    t1_hdr = img.header
    t1_data = img.get_fdata()
    
    import numpy as np
    print(np.min(t1_data))
    print(np.max(t1_data))
    
    x_slice = t1_data[9, :, :]
    y_slice = t1_data[:, 19, :]
    z_slice = t1_data[:, :, 2]
    import matplotlib.pyplot as plt
    
    slices = [x_slice, y_slice, z_slice]
    
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    
    img.orthoview()
    affine=img.affine
    x, y, z, _ = np.linalg.pinv(affine).dot(np.array([0, 0, 0, 1])).astype(int)
    
    print("Affine:")
    print(affine)
    print("Center: ({:d}, {:d}, {:d})".format(x, y, z))
    nib.aff2axcodes(affine)
    nib.affines.voxel_sizes(affine)
    nib.aff2axcodes(affine)
    nib.affines.voxel_sizes(affine)
    img.orthoview()
    #to make values between 0-255 change datatype to unsigned 8 bit
    data=img.get_fdata()
    rescaled = ((data - data.min()) * 255. / (data.max() - data.min())).astype(np.uint8)
    new_img = nib.Nifti1Image(rescaled, affine=img.affine, header=img.header)
    orig_filename = img.get_filename()
    img.set_data_dtype(np.uint8)
    # Save image again
    new_img = nib.Nifti1Image(rescaled, affine=img.affine, header=img.header)
    nib.save(new_img, outfile)

dataFolder = 'kki'
y=[x[1] for x in os.walk(dataFolder)]
folders = y[0]

for count in range(1, len(folders)) :
    folderName = folders[count]
    fileName = "dataset/" + folderName + ".nii.gz"
    fileName2 = "dataset/" + folderName + "anat.nii.gz"
    
    print('Processing %s' % (folderName))
    performPreprocess(dataFolder + "/" + folderName + "/session_1", fileName)
    performPreprocessAnat(dataFolder + "/" + folderName + "/session_1", fileName2)
    
     

folder = 'dataset'
outFolder = 'nifti-results/'
filenames = next(walk(folder), (None, None, []))[2]

class_data = pd.read_csv('class_info.csv', header=None)
subjects = class_data[0]
classes = class_data[1]

rows = 256
cols = 256

def findNonZero(img) :
    img = np.asarray(img)
    return np.count_nonzero(img)

for count in range(0, len(filenames)) :
    try :
        filename = filenames[count]
        temp = re.findall(r'\d+', filename)
        res = list(map(int, temp))
        if(len(res) != 1) :
            continue
        res = res[0]
        x=subjects[subjects == res]
        idx = x.index[0]
        class_val = classes[idx]
        
        img = nib.load(folder + "/" + filename)
        print(img.shape)
        data = img.get_fdata()
        
        for count2 in range(0, len(data[0][0])) :
            img = data[:,:, count2]
            tot = len(img) * len(img[0])
            try :
                img = img[:,:,0:3]
                tot = len(img) * len(img[0]) * len(img[0][0])
            except :
                print('Dimensions are ok')
                
            nonZero = findNonZero(img)
            ratio = nonZero / tot
            if(ratio < 0.4) :
                continue
            
            fname = outFolder + str(class_val) + "/" + filename + "." + str(count2) + ".png"
            cv2.imwrite(fname, cv2.resize(img, (rows, cols)) )
                
            print('Writing %s' % (fname))
    except :
        print('Continue...')


model_name = "VGGNet.model"
rows = 128
cols = 128
channels = 3
data_directory = 'nifti-results'

if(path.exists(model_name)) :
    model = keras.models.load_model(model_name)
else :
    train_dir = data_directory
    test_dir = data_directory
    validation_dir = data_directory
    
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    train_batch_size = 16;
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    
    #Used to get the same results everytime
    np.random.seed(42)
    #tf.random.set_seed(42)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(rows,cols),
        batch_size=train_batch_size,
        class_mode='categorical')
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(rows,cols),
        batch_size=20,
        class_mode='categorical')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(rows,cols),
        batch_size=20,
        class_mode='categorical')
    
    ######################################################################
    #initialize the NN
    
    #Load the VGG16 model, use the ILSVRC competition's weights
    #include_top = False, means only include the Convolution Base (do not import the top layers or NN Layers)
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(rows,cols,channels))
    conv_base.trainable = False;
    model = models.Sequential()
    
    #Add the VGGNet model
    model.add(conv_base)
    
    #NN Layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation='relu'))
    model.add(keras.layers.Dense(2,activation='softmax'))
    
    print(model.summary())
    ######################################################################
    
    #Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    #Steps per epoch = Number of images in the training directory / batch_size (of the generator)
    #validation_steps = Number of images in the validation directory / batch_size (of the generator)
    checkpoint_callback = keras.callbacks.ModelCheckpoint("%s" % (model_name), save_best_only=True)
    model_history = model.fit_generator(
        train_generator,
        steps_per_epoch=5,
        epochs = 10,
        validation_data=validation_generator,
        validation_steps=30,
        callbacks = [checkpoint_callback])
    
    #Plot the model
    pd.DataFrame(model_history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    
    model.save(model_name)
    

#Check the testing set
#model.evaluate_generator(test_generator,steps=50)
folderName = "nifti-results/1/";
onlyfiles = [f for f in listdir(folderName) if isfile(join(folderName, f))]
out_csv = 'output.csv'
output = 'Frame, Obtained class'

for count in range(0,len(onlyfiles)):
    print(("Processing:%s\\%s\n" % (folderName, onlyfiles[count])))
    
    frame = cv2.imread(("%s\\%s" % (folderName, onlyfiles[count])),cv2.IMREAD_UNCHANGED)
    frame = cv2.resize(frame, (rows,cols), interpolation = cv2.INTER_AREA)
    frame_bkp = np.zeros(shape = (rows,cols,channels))
    try :
        frame_bkp[:,:,0] = frame
        frame_bkp[:,:,1] = frame
        frame_bkp[:,:,2] = frame
    except :
        frame_bkp = frame
        print('Dimensions done!')
    
    frame = np.asarray(frame_bkp).reshape((1,rows,cols,channels))
    #y_pred = model.predict_classes(frame)
    y_pred=np.argmax(model.predict(frame), axis=-1)
    y_pred = y_pred[0]
    label = str(y_pred)
    
    output = output + "\n" + onlyfiles[count] + "," + label
    
    print("Classified as type %s\n" % (label))
    cv2.putText(frame_bkp, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
    cv2.imshow("Result", frame_bkp)
    cv2.waitKey(1000)
cv2.destroyAllWindows()
file1 = open(out_csv,"w") 
file1.write(output)
file1.close()
#model.summary()

