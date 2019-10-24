import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def squeeze(batches): # change to enable x and y together
    while True:
        batch_x, batch_y = next(batches)
        batch_y = batch_y[:,:,:,0]
        
        yield (batch_x, batch_y) 
        

def custom_image_generator(data, target, batch_size=32):
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
    train_img, train_mask = data, np.expand_dims(target, axis=-1)
    
    data_gen_args = dict(
                         horizontal_flip = True,
                         vertical_flip = True,
                        )
    img_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    img_datagen.fit(train_img)
    mask_datagen.fit(train_mask)
    
    seed = 2018
    img_gen = img_datagen.flow(train_img, seed = seed, batch_size=batch_size, shuffle=True)#shuffling
    mask_gen = mask_datagen.flow(train_mask, seed = seed, batch_size=batch_size, shuffle=True)
    train_gen = zip(img_gen, mask_gen)
    train_gen = squeeze(train_gen)
    
    return train_gen


def custom_image_generator2(data, target, batch_size=32):
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
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, (len(data)//batch_size)*batch_size, batch_size):
            d, t = data[i:i + batch_size].copy(), target[i:i + batch_size].copy()

            # Random color inversion
            # for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
            #     d[j][d[j] > 0.] = 1. - d[j][d[j] > 0.]

            # Horizontal/vertical flips
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])      # left/right
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])      # up/down

            # Random up/down & left/right pixel shifts, 90 degree rotations
            npix = 15
            h = np.random.randint(-npix, npix + 1, batch_size)    # Horizontal shift
            v = np.random.randint(-npix, npix + 1, batch_size)    # Vertical shift
            r = np.random.randint(0, 4, batch_size)               # 90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix, npix), (npix, npix), (0, 0)),
                              mode='constant')[npix + h[j]:L + h[j] + npix,
                                               npix + v[j]:W + v[j] + npix, :]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix + h[j]:L + h[j] + npix, 
																npix + v[j]:W + v[j] + npix]
                d[j], t[j] = np.rot90(d[j], r[j]), np.rot90(t[j], r[j])
            yield (d, t)
