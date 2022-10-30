
"""
pose_scores 0.802927339778227

keypoint_scores [0.9966163  0.9873597  0.99463838 0.87494057 0.92514783 0.9907698
 0.95898128 0.84199125 0.65809762 0.56613177 0.81136292 0.81243712
 0.67141813 0.75180644 0.92078233 0.49843824 0.38884509]

keypoint_coords [
 [347.68612243 746.13035104] 1
 [343.26632583 754.44348238] 2
 [342.48052965 739.78269918] 3
 [349.06655362 762.03131309] 4
 [347.37552844 720.21655782] 5
 [395.90556764 782.08119253] 6
 [392.17379118 705.44334654] 7
 [457.93340248 796.01588973] 8
 [448.22022287 670.83262649] 9
 [455.23356923 780.42663039] 10
 [452.73005703 727.05245855] 11
 [517.44853237 767.12839183] 12
 [517.16083727 711.01250037] 13
 [619.07674287 761.58659585] 14
 [620.86821707 702.17011401] 15
 [725.26027278 748.96456297] 16
 [725.47197978 699.72492197] 17
 ]
"""

import torch
import cv2
from time import time

import numpy as np

import posenet


class MyPosenetPytorch:

    def __init__(self, model_dir):

        self.model = posenet.load_model(101, model_dir=model_dir)
        self.model = self.model.cuda()
        self.output_stride = self.model.output_stride
        self.scale_factor = 1

    def compute_image(self, img_in, threshold_pose, threshold_points):
        """A partir d'une image 1280x720 BGR, trouve maxi 4 squelettes.
        Retourne une liste des xys = all_xys
        Un xy = liste de 17 points (valide ou None) ou None si rien du tout.
        """

        # img_in = image 1281x721 et arrangée pour posenet
        img_in = my_process_input(img_in)

        with torch.no_grad():
            # Chargement du array sur le GPU, puis calcul sur le GPU
            img_in = torch.Tensor(img_in).cuda()

            heatmaps_result, offsets_result,\
            displacement_fwd_result, displacement_bwd_result = self.model(img_in)

            a = posenet.decode_multiple_poses(heatmaps_result.squeeze(0),
                                              offsets_result.squeeze(0),
                                              displacement_fwd_result.squeeze(0),
                                              displacement_bwd_result.squeeze(0),
                                              output_stride=self.output_stride,
                                              max_pose_detections=4,
                                              min_pose_score=threshold_pose)

            pose_scores, keypoint_scores, keypoint_coords = a


        all_xys = []
        for i in range(len(pose_scores)):
            xys_list = keypoint_coords_to_xys_list( keypoint_scores[i],
                                                    keypoint_coords[i],
                                                    threshold_points)
            if xys_list:
                all_xys.append(xys_list)

        return all_xys


class MyPosenetPytorchCamTest:

    def __init__(self, dev):

        self.model = posenet.load_model(101)
        self.model = self.model.cuda()
        self.output_stride = self.model.output_stride
        self.scale_factor = 1
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

    def run(self):

        start = time()
        frame_count = 0
        while True:

            res, img = self.cap.read()

            # img_in = image 1281x721 et arrangée pour posenet
            if res:
                img_in = my_process_input(img)
            else:
                break

            with torch.no_grad():
                # Chargement du array sur le GPU, puis calcul sur le GPU
                img_in = torch.Tensor(img_in).cuda()

                heatmaps_result, offsets_result,\
                displacement_fwd_result, displacement_bwd_result = self.model(img_in)

                a = posenet.decode_multiple_poses(heatmaps_result.squeeze(0),
                                                  offsets_result.squeeze(0),
                                                  displacement_fwd_result.squeeze(0),
                                                  displacement_bwd_result.squeeze(0),
                                                  output_stride=self.output_stride,
                                                  max_pose_detections=1,
                                                  min_pose_score=0.15)

                pose_scores, keypoint_scores, keypoint_coords = a


            # Je ne récupère que le 1er des sqelettes, de toute façon je n'en ai qu'un!
            xys_list = keypoint_coords_to_xys_list(keypoint_scores[0], keypoint_coords[0], 0.3)

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                img, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time() - start))



def my_process_input(source_img):
    """Conversion d'une image en array  (1, 3, 721, 1281)
    L'image d'entrée est 1280x720
    Le model demande une image 1281x721
    Je rajoute une bande noire de 1 pixels en bas et à droite

    source_img.shape = (720, 1280, 3)
    """

    row = np.zeros((1, 1280, 3), np.uint8)
    input_img = np.vstack((source_img, row))

    column = np.zeros((720, 1, 3), np.uint8)
    input_img = np.hstack((source_img, column))

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, 720, 1281)

    return input_img


def keypoint_coords_to_xys_list(keypoint_scores, keypoint_coords, score_mini):
    """
    pose_scores = 0.1808 > 0.15

    keypoint_scores et keypoint_coords sont numpy array

    keypoint_scores 0.997 keypoint_coords [149 895]
    keypoint_scores 0.858 keypoint_coords [111 903]
    ... etc ... 17 fois

    Les points avec keypoint_scores < score_mini sont mis à None

    Retourne liste de coordonnées (en array) ou None
    Si la liste ne contient que des None, retourne None
    """

    xys_list = [None]*17

    ks = list(keypoint_scores)
    for i in range(17):
        if keypoint_scores[i] > score_mini:
            xys_list[i] = [ int(keypoint_coords[i][1]),
                            int(keypoint_coords[i][0]) ]

    # Suppression des squelettes sans aucune articulation, possible lorsque
    # toutes les articulations sont en dessous du seuil de confiance ?
    if xys_list == [None]*17:
        xys_list = None

    return xys_list



if __name__ == "__main__":

    mpp = MyPosenetPytorchCamTest(0)
    mpp.run()
