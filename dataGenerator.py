import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator

def squeeze(batches): # change to enable x and y together
    while True:
        batch_x, batch_y = next(batches)
#         batch_y = batch_y[:,:,:,0]
        print(batch_x.shape, batch_y.shape)       
        yield (batch_x, batch_y) 
        

from scipy import signal

def hillshade(array,azimuth,angle_altitude):
    azimuth = 360.0 - azimuth 

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)
    shaded = (255*(shaded + 1)/2).reshape((128,128,1))
    

    minn, maxx = np.min(shaded), np.max(shaded)
    # print('aspect',minn,maxx)
    shaded = 0.1 + (shaded - minn) * 0.9 / (maxx - minn)
    return shaded
    
    
def terrain_analysis(array, size):
    """calculate terrain derivatives based on the Evans Young method

    Args:
      array (ndarray): elevation data array, must be 2D array
      size (float,float): size of sample in projected coordinates

    Returns:
      (ndarray): 3d array with original elevation data and derivatives (dim,dim,bands)
    """
    hi, low = 1.0, 0.1

    px, py = size[0]/array.shape[-1], size[1]/array.shape[-2]

    g = [[(-1/(6*px)), 0 , (1/(6*px))],
         [(-1/(6*px)), 0 , (1/(6*px))],
         [(-1/(6*px)), 0 , (1/(6*px))]]
    h = [[(1/(6*py)),(1/(6*py)),(1/(6*py))],
         [0,0,0],
         [(-1/(6*py)),(-1/(6*py)),(-1/(6*py))]]
    # d = [[(1/(3*(px**2))),(-2/(3*(px**2))),(1/(3*(px**2)))],
    #      [(1/(3*(px**2))),(-2/(3*(px**2))),(1/(3*(px**2)))],
    #      [(1/(3*(px**2))),(-2/(3*(px**2))),(1/(3*(px**2)))]]
    # e = [[(1/(3*(py**2))),(1/(3*(py**2))),(1/(3*(py**2)))],
    #      [(-2/(3*(py**2))),(-2/(3*(py**2))),(-2/(3*(py**2)))],
    #      [(1/(3*(py**2))),(1/(3*(py**2))),(1/(3*(py**2)))]]
    # f = [[(-1/(4*(px*py))),0, (1/(4*(px*py)))],
    #      [0,0,0],
    #      [(1/(4*(px*py))),0,(-1/(4*(px*py)))]]

    gi = signal.convolve2d(array, g, boundary='symm', mode='same')
    hi = signal.convolve2d(array, h, boundary='symm', mode='same')
    # di = signal.convolve2d(array, d, boundary='symm', mode='same')
    # ei = signal.convolve2d(array, e, boundary='symm', mode='same')
    # fi = signal.convolve2d(array, f, boundary='symm', mode='same')

    slope  = np.sqrt(np.power(hi,2)+np.power(gi,2))
    minn, maxx = np.min(slope), np.max(slope)
    # print(minn,maxx)
    slope = 0.1 + (slope - minn) * 0.9 / (maxx - minn)

    aspect = np.where(gi == 0, math.pi/2, np.arctan(hi/gi))
    minn, maxx = np.min(aspect), np.max(aspect)
    # # print('aspect',minn,maxx)
    aspect = 0.1 + (aspect - minn) * 0.9 / (maxx - minn)
    
    [dx,dy] = np.gradient(array)
    out = np.sqrt((dx*dx)+(dy*dy))
    # normalize 
    minn, maxx = np.min(out), np.max(out)
    gradient = 0.1 + (out - minn) * 0.9 / (maxx - minn)
#     planc  = -1*((np.power(hi, 2)*di)-(2*gi*hi*fi)+(np.power(gi,2)*ei)/(np.power((np.power(gi,2)+np.power(hi,2)),1.5)))
#     profc  = -1*(((np.power(gi,2)*di)+(2*gi*hi*fi) +(np.power(hi,2)*ei))/ ((np.power(gi,2)+np.power(hi,2))*(np.power( (1+np.power(gi,2)+np.power(hi,2)),1.5)) ))
#     meanc  = -1 *( ((1+np.power(hi,2))*di) -(2*gi*hi*fi) +((1+np.power(gi,2))*ei) / (2*np.power( (1+np.power(gi,2)+np.power(hi,2)),1.5)  ))
    
    return np.stack([array, slope, aspect, gradient], axis=-1)
    # return np.expand_dims(slope,axis=2)

    
def add_derivatives(batches, option):
    while True:
        batch_x, batch_y = next(batches)
        out_x = []
        for i in range(batch_x.shape[0]):
            if(option=='hillshade'):
                out_x.append(hillshade(batch_x[i,:,:,0], 30, 30))
            elif(option=='terrain'):
                out_x.append(terrain_analysis(batch_x[i,:,:,0],(1.5,1.5)))
        out_x = np.array(out_x)
        yield (out_x, batch_y) 

def use_gradient(batches):
    while True:
        batch_x, batch_y = next(batches)
        out_x = []
        for i in range(batch_x.shape[0]):
            [dx, dy] = np.gradient(batch_x[i,:,:,0])
            out = np.sqrt((dx*dx)+(dy*dy))
            # normalize 
            minn, maxx = np.min(out), np.max(out)
            out = 0.1 + (out - minn) * 0.9 / (maxx - minn)
            out_x.append(np.expand_dims(out, axis=2))
        out_x = np.array(out_x)
        yield (out_x, batch_y)



def custom_image_generator(data, target, batch_size=32, gradient=False, DEM=True):
    """Custom image generator that manipulates image/target pairs to prevent
    overfitting in the Convolutional Neural Network.
    Parameters
    ----------
    data : array
        Input images.
    target : array
        Target images.
    batch_size : int, optional
        Batch size for image manipulation.
    Yields
    ------
    Manipulated images and targets.
        
    """
    train_img, train_mask = data, target
    print(train_img.shape, train_mask.shape)
    
    data_gen_args = dict(
                         horizontal_flip = True,
                         vertical_flip = True,
                        #  rotation_range = 90,
                        #  width_shift_range=0.1,
                        #  height_shift_range=0.1,
                         zoom_range=0.05,
                         fill_mode='constant'
                        )
    img_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    img_datagen.fit(train_img)
    mask_datagen.fit(train_mask)
    
    seed = 2018
    img_gen = img_datagen.flow(train_img, seed = seed, batch_size=batch_size, shuffle=True)#shuffling
    mask_gen = mask_datagen.flow(train_mask, seed = seed, batch_size=batch_size, shuffle=True)
    train_gen = zip(img_gen, mask_gen)

    if(DEM):
        return train_gen
        
    # gradient or derivatives
    if(gradient):
        train_gen = use_gradient(train_gen)
    else:
        train_gen = add_derivatives(train_gen,'terrain')  # 8.3
        # train_gen = add_derivatives(train_gen, 'hillshade')
    return train_gen


def val_datagenerator(data, target, gradient=False, DEM=True):
    if(DEM):
        return (data, target)
    data_out = []
    for i in range(len(data)):
         if gradient:
             [dx, dy] = np.gradient(data[i,:,:,0])
             out = np.sqrt((dx*dx)+(dy*dy))
             # normalize
             minn, maxx = np.min(out), np.max(out)
             out = 0.1 + (out - minn) * 0.9 / (maxx - minn)
             data_out.append(np.expand_dims(out,axis=2))
         else:
             data_out.append(terrain_analysis(data[i,:,:,0],(1.5,1.5)))
            #  data_out.append(hillshade(data[i,:,:,0], 30, 30))
    data_out = np.array(data_out)
    return (data_out, target)
     

def no_aug_generator(data, target, batch_size=32, gradient=False):
    """Custom image generator that manipulates image/target pairs to prevent
    overfitting in the Convolutional Neural Network.
    Parameters
    ----------
    data : array
        Input images.
    target : array
        Target images.
    batch_size : int, optional
        Batch size for image manipulation.
    Yields
    ------
    Manipulated images and targets.
        
    """
    train_img, train_mask = data, target
    print(train_img.shape, train_mask.shape)
    
    data_gen_args = dict()
    img_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    img_datagen.fit(train_img)
    mask_datagen.fit(train_mask)
    
    seed = 2018
    img_gen = img_datagen.flow(train_img, seed = seed, batch_size=batch_size, shuffle=True)#shuffling
    mask_gen = mask_datagen.flow(train_mask, seed = seed, batch_size=batch_size, shuffle=True)
    train_gen = zip(img_gen, mask_gen)
    # if(gradient):
        # train_gen = use_gradient(train_gen)
    return train_gen

# def custom_image_generator2(data, target, batch_size=32):
#     """Custom image generator that manipulates image/target pairs to prevent
#     overfitting in the Convolutional Neural Network.
#     Parameters
#     ----------
#     data : array
#         Input images.
#     target : array
#         Target images.
#     batch_size : int, optional
#         Batch size for image manipulation.
#     Yields
#     ------
#     Manipulated images and targets.
        
#     """
#     L, W = data[0].shape[0], data[0].shape[1]
#     while True:
#         for i in range(0, (len(data)//batch_size)*batch_size, batch_size):
#             d, t = data[i:i + batch_size].copy(), target[i:i + batch_size].copy()

#             # Random color inversion
#             # for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
#             #     d[j][d[j] > 0.] = 1. - d[j][d[j] > 0.]

#             # Horizontal/vertical flips
#             for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
#                 d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])      # left/right
#             for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
#                 d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])      # up/down

#             # Random up/down & left/right pixel shifts, 90 degree rotations
#             npix = 15
#             h = np.random.randint(-npix, npix + 1, batch_size)    # Horizontal shift
#             v = np.random.randint(-npix, npix + 1, batch_size)    # Vertical shift
#             r = np.random.randint(0, 4, batch_size)               # 90 degree rotations
#             for j in range(batch_size):
#                 d[j] = np.pad(d[j], ((npix, npix), (npix, npix), (0, 0)),
#                               mode='constant')[npix + h[j]:L + h[j] + npix,
#                                               npix + v[j]:W + v[j] + npix, :]
#                 t[j] = np.pad(t[j], (npix,), mode='constant')[npix + h[j]:L + h[j] + npix, 
# 																npix + v[j]:W + v[j] + npix]
#                 d[j], t[j] = np.rot90(d[j], r[j]), np.rot90(t[j], r[j])
#             yield (d, t)
