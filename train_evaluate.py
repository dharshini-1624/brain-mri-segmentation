import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import jaccard_score, confusion_matrix
from data_preprocessing import load_data, preprocess_images, get_data_generators
from unet_plus_plus import unet_plus_plus  
from attention_unet import attention_unet  



data_dir = 'C:/Users/saidh/Downloads/Data'  
mri_images, mask_images = load_data(data_dir) 
X_train, y_train = preprocess_images(mri_images, mask_images)  

train_generator = get_data_generators(X_train, y_train)


model_unet_plus_plus = unet_plus_plus()
model_unet_plus_plus.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_unet_plus_plus.fit(train_generator, 
                          steps_per_epoch=len(X_train) // 32, 
                          validation_data=(X_test, y_test), 
                          epochs=50, 
                          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])


y_pred_unet_plus_plus = model_unet_plus_plus.predict(X_test)
y_pred_unet_plus_plus = (y_pred_unet_plus_plus > 0.5).astype(np.uint8)  # Binarize predictions


dice_unet_plus_plus = 2 * np.sum(y_pred_unet_plus_plus * y_test) / (np.sum(y_pred_unet_plus_plus) + np.sum(y_test))
print(f'DICE Score for U-Net++: {dice_unet_plus_plus}')


model_attention_unet = attention_unet()
model_attention_unet.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_attention_unet.fit(train_generator, 
                          steps_per_epoch=len(X_train) // 32, 
                          validation_data=(X_test, y_test), 
                          epochs=50, 
                          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])


y_pred_attention_unet = model_attention_unet.predict(X_test)
y_pred_attention_unet = (y_pred_attention_unet > 0.5).astype(np.uint8)  # Binarize predictions


dice_attention_unet = 2 * np.sum(y_pred_attention_unet * y_test) / (np.sum(y_pred_attention_unet) + np.sum(y_test))
print(f'DICE Score for Attention U-Net: {dice_attention_unet}')


print(f'Comparison of DICE Scores:')
print(f'U-Net++ DICE Score: {dice_unet_plus_plus}')
print(f'Attention U-Net DICE Score: {dice_attention_unet}')


np.save('predictions_unet_plus_plus.npy', y_pred_unet_plus_plus)
np.save('predictions_attention_unet.npy', y_pred_attention_unet)
