import numpy as np
import torch
from skimage.transform import resize
from PIL import Image
from models import PhenoProfiler

# List of image paths
img_paths = [
    './Sample_imgs/IXMtest_A01_s2_w1AA6B1894-F561-42EE-9D1D-E21E5C741B75.png',
    './Sample_imgs/IXMtest_A01_s2_w3A597237B-C3D7-43AE-8399-83E76DA1532D.png',
    './Sample_imgs/IXMtest_A01_s2_w50F1562CD-EBCF-408E-9D8D-F6F0FDD746C8.png',
    './Sample_imgs/IXMtest_A01_s2_w246FFAEE1-BEB6-4C81-913B-B979EC0C4BC3.png',
    './Sample_imgs/IXMtest_A01_s2_w46657239A-5AFE-4C29-BB9B-49978EFE4791.png',
]

# Load and preprocess images
images = np.stack([resize(np.array(Image.open(path)), (448, 448), anti_aliasing=True) for path in img_paths])
images_tensor = torch.tensor(images).float().cuda()

# Load model
model = PhenoProfiler().cuda()
model.load_state_dict(torch.load('./best.pt', weights_only=True))

# Generate embeddings
image_features = model.image_encoder(images_tensor.unsqueeze(0))
image_embeddings = model.image_projection(image_features)

# Print the shape of the embeddings
print(image_embeddings.shape)