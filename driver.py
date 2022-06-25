"""Driver."""
import os

from ResNet import ResNet

data_path = '/Users/justinsima/dir/implementations/datasets/ImageNet/dummy_data'
label_path = os.path.join(data_path, 'labels.json')
label_encoding = os.path.join(data_path, 'label_encodings.json')

model = ResNet(data_path, label_path, label_encoding)
model.fit(epochs=3)

test_image = '/Users/justinsima/dir/implementations/datasets/ImageNet/dummy_data/test/ILSVRC2012_test_00018560.JPEG'
pred = model.predict(test_image)
class_pred = model.predict_n(test_image, 1)
print(class_pred)