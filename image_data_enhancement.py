# -*- coding: utf-8 -*-

# import library
# eng : ImageDataGenerator --> Generate batches of tensor image data with real-time data augmentation.
# tr : ImageDataGenerator --> derin öğrenme için görüntü verilerinin ardışık düzenlenmesi için başvurulan sınıf.
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# eng : We describe the data augmentation techniques we use.
# tr : Kullanacağımız veri artırma tekniklerini tanımlıyoruz
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')

# eng : load image
# tr : Görüntü Yükleme İşlemi
img = load_img('cat.jpg')

#eng : We convert the image to array
#tr : görüntüyü diziye çeviriyoruz
x = img_to_array(img)
x = x.reshape((1,)+x.shape)

# eng : We make 20 different images from a single Image
# tr : Tek görüntüden 20 farklı görüntü yapıyoruz

# eng : reproduced_photos --> We save the duplicated images here
# tr : reproduced_photos --> Çoğaltılan görünütleri buraya kayıt ediyoruz
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='reproduced_photos',
                          save_format='jpeg'):
    i+=1
    if i > 20:
        break
