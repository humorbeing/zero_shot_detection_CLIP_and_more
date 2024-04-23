from PIL import Image
from transformers import AutoProcessor, CLIPModel
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as matplotlibpatches


# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
model.to(device)

image_2cat = Image.open('cats.jpg')

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image_2cat,
    return_tensors="pt",
)
inputs = inputs.to(device)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)


inputs = processor(images=image_2cat, return_tensors="pt")
inputs = inputs.to(device)
image_features_2cat = model.get_image_features(**inputs)

inputs = processor(text="a photo of a cat", return_tensors="pt")
inputs = inputs.to(device)
text_features_cat = model.get_text_features(**inputs)


image_l2norm = image_features_2cat.norm(p=2, dim=-1).item()
image_features_2cat_N = image_features_2cat / image_l2norm

text_l2norm = text_features_cat.norm(p=2, dim=-1).item()
text_features_cat_N = text_features_cat / text_l2norm

cos_sim_wrong = torch.matmul(image_features_2cat, text_features_cat.t())
cos_sim = torch.matmul(image_features_2cat_N, text_features_cat_N.t())

cosineSimilarity = torch.nn.CosineSimilarity(dim=1)
cos_sim_torch = cosineSimilarity(image_features_2cat,text_features_cat)
cos_sim_torch_N = cosineSimilarity(image_features_2cat_N,text_features_cat_N)

logit_scale = model.logit_scale.exp()
clip_sim = cos_sim * logit_scale


def clip_sim_fn(feature1, feature2, logit_scale=100):
    feature1_l2norm = feature1.norm(p=2, dim=-1).item()
    feature1_N = feature1 / feature1_l2norm

    feature2_l2norm = feature2.norm(p=2, dim=-1).item()
    feature2_N = feature2 / feature2_l2norm
    
    cos_sim = torch.matmul(feature1_N, feature2_N.t())
    clip_sim = cos_sim * logit_scale
    return cos_sim, clip_sim


cos_s, clip_s = clip_sim_fn(image_features_2cat, text_features_cat)

image_2dog = Image.open('dogs.jpeg')
inputs = processor(images=image_2dog, return_tensors="pt")
inputs = inputs.to(device)
image_features_2dog = model.get_image_features(**inputs)

image_1cat = Image.open('cat1.webp')
inputs = processor(images=image_1cat, return_tensors="pt")
inputs = inputs.to(device)
image_features_1cat = model.get_image_features(**inputs)

_, cat_cat = clip_sim_fn(image_features_2cat, image_features_1cat)
_, cat_dog = clip_sim_fn(image_features_2cat, image_features_2dog)

score = torch.cat((cat_cat, cat_dog), 1)
probs = score.softmax(dim=1)


inputs = processor(text="a photo of a dog", return_tensors="pt")
inputs = inputs.to(device)
text_features_dog = model.get_text_features(**inputs)

inputs = processor(text="a photo of 2 cats", return_tensors="pt")
inputs = inputs.to(device)
text_features_2cat = model.get_text_features(**inputs)

inputs = processor(text="a photo of 3 cats", return_tensors="pt")
inputs = inputs.to(device)
text_features_3cat = model.get_text_features(**inputs)

_, t_cat_dog = clip_sim_fn(text_features_cat, text_features_dog)
_, t_cat_2cat = clip_sim_fn(text_features_cat, text_features_2cat)
_, t_cat_3cat = clip_sim_fn(text_features_cat, text_features_3cat)


score = torch.cat((t_cat_dog, t_cat_2cat, t_cat_3cat), 1)
probs = score.softmax(dim=1)

_, it_cat_1cat = clip_sim_fn(image_features_2cat, text_features_cat)
_, it_cat_2cat = clip_sim_fn(image_features_2cat, text_features_2cat)
_, it_cat_3cat = clip_sim_fn(image_features_2cat, text_features_3cat)
_, it_cat_dog = clip_sim_fn(image_features_2cat, text_features_dog)


score = torch.cat((it_cat_1cat, it_cat_2cat, it_cat_3cat, it_cat_dog), 1)
probs = score.softmax(dim=1)

# from datasets import load_dataset #pip install datasets

# data = load_dataset(
#     "jamescalam/image-text-demo",
#     split="train",
#     revision="180fdae"
# )
# image = data[2]["image"]


image_bc = Image.open('butterfly_landing_on_the_nose_of_a_cat.jpg')


img = transforms.functional.pil_to_tensor(image_bc)

transform = transforms.ToTensor()
# normalize the data (did not want).
# preprocess will normalize twice.
img_normalize = transform(image_bc)  


patches = img.data.unfold(0,3,3)
patch = 256
patches = patches.unfold(1, patch, patch)

X = patches.shape[1]

fig, ax = plt.subplots(X, 1, figsize=(40, 26))
# loop through each strip and display
for x in range(X):
    ax[x].imshow(patches[0, x].permute(2, 0, 1))
    ax[x].axis("off")
fig.tight_layout()
plt.savefig('butterfly_cat_01_preprocess_strips.jpg')


patches = patches.unfold(2, patch, patch)

X = patches.shape[1]
Y = patches.shape[2]

fig, ax = plt.subplots(X, Y, figsize=(Y*2, X*2))
for x in range(X):
    for y in range(Y):
        ax[x, y].imshow(patches[0, x, y].permute(1, 2, 0))
        ax[x, y].axis("off")
fig.tight_layout()
plt.savefig('butterfly_cat_02_preprocess_blocks.jpg')


# # set the 6x6 window
# window = 6

# big_patch = torch.zeros(patch*window, patch*window, 3)
# patch_batch = patches[0][:window][:window]

# # visualize patch
# for y in range(window):
#     for x in range(window):
#         big_patch[y*patch:(y+1)*patch, x*patch:(x+1)*patch, :] = patch_batch[y, x].permute(1, 2, 0)

# plt.imshow(big_patch/255)
# plt.savefig('new_here2.jpg')



window = 6
stride = 1

scores = torch.zeros(patches.shape[1], patches.shape[2])
# runs = torch.zeros(patches.shape[1], patches.shape[2])
runs = torch.ones(patches.shape[1], patches.shape[2])

for Y in range(0, patches.shape[1]-window+1, stride):
    for X in range(0, patches.shape[2]-window+1, stride):
        big_patch = torch.zeros(patch*window, patch*window, 3)
        patch_batch = patches[0, Y:Y+window, X:X+window]
        for y in range(window):
            for x in range(window):
                big_patch[
                    y*patch:(y+1)*patch, x*patch:(x+1)*patch, :
                ] = patch_batch[y, x].permute(1, 2, 0)
        # we preprocess the image and class label with the CLIP processor
        
        
        inputs = processor(
            images=big_patch,  # big patch image sent to CLIP
            return_tensors="pt",  # tell CLIP to return pytorch tensor
            text="a fluffy cat",  # class label sent to CLIP
            # text="a butterfly",
        ).to(device) # move to device if possible

        # calculate and retrieve similarity score
        score = model(**inputs).logits_per_image.item()
        # sum up similarity scores from current and previous big patches
        # that were calculated for patches within the current window
        scores[Y:Y+window, X:X+window] += score
        # calculate the number of runs on each patch within the current window
        runs[Y:Y+window, X:X+window] += 1

# average score for each patch
scores /= runs

for _ in range(1):
    scores = np.clip(scores-scores.mean(), 0, np.inf)
# transform the patches tensor 
adj_patches = patches.squeeze(0).permute(3, 4, 2, 0, 1)
adj_patches = adj_patches / 255
# normalize scores
scores = (
    scores - scores.min()) / (scores.max() - scores.min()
)
# multiply patches by scores

threshold = 0.6
# scores[scores>=threshold] = 1.0
# scores[scores<threshold] = 0.0
adj_patches = adj_patches * scores
# rotate patches to visualize
adj_patches = adj_patches.permute(3, 4, 2, 0, 1)

Y = adj_patches.shape[0]
X = adj_patches.shape[1]

fig, ax = plt.subplots(Y, X, figsize=(X*.5, Y*.5))
for y in range(Y):
    for x in range(X):
        ax[y, x].imshow(adj_patches[y, x].permute(1, 2, 0))
        ax[y, x].axis("off")
        ax[y, x].set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('butterfly_cat_03_attentions.jpg')

detection = scores > threshold

y_min, y_max = (
    np.nonzero(detection)[:,0].min().item(),
    np.nonzero(detection)[:,0].max().item()+1
)


x_min, x_max = (
    np.nonzero(detection)[:,1].min().item(),
    np.nonzero(detection)[:,1].max().item()+1
)

y_min *= patch
y_max *= patch
x_min *= patch
x_max *= patch


height = y_max - y_min
width = x_max - x_min


img.data.numpy().shape
image = np.moveaxis(img.data.numpy(), 0, -1)


fig, ax = plt.subplots(figsize=(Y*0.5, X*0.5))

ax.imshow(image)

# Create a Rectangle patch
rect = matplotlibpatches.Rectangle(
    (x_min, y_min), width, height,
    linewidth=3, edgecolor='#FAFF00', facecolor='none'
)

# Add the patch to the Axes
ax.add_patch(rect)

# plt.show()
plt.savefig('butterfly_cat_04_detection.jpg')
print('end')