import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent

class ShapeDepthMatchModel(nn.Module):
    def __init__(self):
        super(ShapeDepthMatchModel, self).__init__()

        self._shape_labels = ['circle frustum', 'cone', 'cube', 'cylinder', 'letter l', 'pyramid', 'square frustum', 'triangular prism']
        self.feature = nn.Sequential(nn.Conv2d(2, 6, 2),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(6, 6, 3),
                                     # nn.Conv2d(6, 12, 3),

                                     nn.LeakyReLU())
        self.shape_classifier = nn.Sequential(nn.Linear(22326, len(self._shape_labels)))
        # self.shape_classifier = nn.Sequential(nn.Linear(43200, len(self._shape_labels)))

    def shape_label(self, id):
        return self._shape_labels[id]

    def shape_labels(self):
        return self._shape_labels


    def forward(self, gray_image, depth):
        img =  torch.cat((gray_image, depth), dim=1)
        feature = self.feature(img)
        feature = feature.flatten(1)
        return self.shape_classifier(feature) #, self.color_classifier(color_feature)
        
        
class KindClassifier():
    def __init__(self, model_name):

        model_path = str(ROOT_DIR / model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ShapeDepthMatchModel()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.shape_labels = [
            'circle frustum',
            'cone',
            'cube',
            'cylinder',
            'letter l',
            'pyramid',
            'square frustum',
            'triangular prism'
        ]
    
    def run(self, rgb, depth):
        rgb_cropped = cv2.resize(rgb, (64,64)) # (64,64,3)
        depth_cropped = cv2.resize(depth, (64,64)) # (64,64)
        gray = cv2.cvtColor(rgb_cropped, cv2.COLOR_BGR2GRAY) # (64,64)
        mask = torch.FloatTensor(gray).unsqueeze(0).unsqueeze(0)  #(1,1,64,64)
        depth_cropped = torch.FloatTensor(depth_cropped).unsqueeze(0).unsqueeze(0)  #(1,1,64,64)

        predictions = self.test(mask, depth_cropped)
        top_class = predictions.argmax()
        
        return self.shape_labels[top_class], predictions[top_class]

    def test(self, mask, depth=False):
            with torch.no_grad():
                if  isinstance(depth, torch.Tensor):
                    obj_current_image_tensor = mask.to(self.device) 
                    obj_current_gray_image_tensor = depth.to(self.device)
                    object_shape_logit = self.model(obj_current_image_tensor, obj_current_gray_image_tensor)
                else:
                    obj_current_image_tensor = mask
                    object_shape_logit = self.model(obj_current_image_tensor)
    
                object_shape_logit = object_shape_logit
                object_shape_prob = F.softmax(object_shape_logit, dim=-1)

            return object_shape_prob.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    MODEL_NAME = "model.p"
    rgb = cv2.imread("rgb.png")
    depth = cv2.imread("depth.png", 0)

    rgb = rgb[0: 64, 0: 64, :]
    depth = depth[0: 64, 0: 64]

    print(KindClassifier(model_name=MODEL_NAME).run(rgb_cropped=rgb, depth_cropped=depth))