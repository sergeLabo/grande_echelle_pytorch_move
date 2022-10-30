
# Posenet
# Copyright 2019 Google LLC
## License: Apache 2.0. See LICENSE file in root directory

## pyrealsense2
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.


"""
Echap pour finir proprement le script

Capture de 1 squelette, celui au plus près du centre,
avec
camera Intel RealSense D455, Google posenet et Google Coral.

Les distances en 3D sont en mm, comme dans grande echelle!
"""


import os
from time import time, sleep
from threading import Thread

import numpy as np
import cv2

import pyrealsense2 as rs
from my_realsense import MyRealSense
from edges import EDGES

from my_posenet_pytorch import MyPosenetPytorch
from gestures_detection import gestures_detection_run

from my_config import MyConfig
from filtre import moving_average



class PosenetRealsenseViewer:
    """Affichage dans une fenêtre OpenCV, et gestion des fenêtres"""

    def __init__(self, conn, config):
        """conn = Pipe avec GUI
        config = dict de config
        """
        self.conn = conn
        self.config = config
        self.t0 = time()

        self.img = None

        self.mode_expo = int(self.config['grandeechelle']['mode_expo'])
        self.full_screen = int(self.config['grandeechelle']['full_screen'])
        if self.mode_expo:
            self.info = 0
            self.full_screen = 0
        self.create_window()

    def create_window(self):
        if not self.mode_expo:
            cv2.namedWindow('posecolor', cv2.WND_PROP_FULLSCREEN)

    def set_window(self):
        """Bascule plein écran avec espace qui set et unset self.full_screen"""
        if not self.mode_expo:
            if self.full_screen:
                cv2.setWindowProperty(  'posecolor',
                                        cv2.WND_PROP_FULLSCREEN,
                                        cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(  'posecolor',
                                        cv2.WND_PROP_FULLSCREEN,
                                        cv2.WINDOW_NORMAL)

    def draw_line(self):
        """Ligne au centre"""
        h = self.img.shape[0]
        w = self.img.shape[1]
        cv2.line(self.img, (int(w/2), 0),
                                 (int(w/2), h),
                                 (255, 255, 255), 2)

    def viewer(self):

        # Affichage de l'image
        # La ligne au centre
        self.draw_line()

        if not self.mode_expo:
            cv2.imshow('posecolor', self.img)

        # Calcul du FPS, affichage toutes les 10 s
        if time() - self.t0 > 10:
            print("FPS Posenet =", self.nbr/10)
            self.t0, self.nbr = time(), 0

        k = cv2.waitKey(1)

        # Space pour full screen or not
        if k == 32:  # space
            if not self.mode_expo:
                if self.full_screen == 1:
                    self.full_screen = 0
                elif self.full_screen == 0:
                    self.full_screen = 1
                self.set_window()

        # Pour quitter
        if k == 27:  # Esc
            self.conn.send(['quit', 1])
            print("Quit envoyé de Posenet Realsense")
            cv2.destroyAllWindows()



class PosenetRealsense(MyRealSense, MyPosenetPytorch, PosenetRealsenseViewer):
    """ Capture avec  Camera RealSense D455
        Détection de la pose avec PyTorch sur CUDA
        Calcul des coordonnées 3D
            et envoi de la moyenne des profondeurs pour 1 personnage.
        La profondeur est le 3ème dans les coordonnées d'un point 3D,
        x = horizontale, y = verticale
    """

    def __init__(self, conn, current_dir, config):
        """
        conn = Pipe avec GUI
         current_dir: tout est dans le même dossier
        config = dict de config. Les paramètres sont à définir dans le fichier
        grande_echelle.ini
        """

        print("Lancement de PosenetRealsense")

        MyRealSense.__init__(self, config)
        PosenetRealsenseViewer.__init__(self, conn, config)

        self.conn = conn
        self.current_dir = current_dir

        self.pose_loop = 1
        self.pose_conn_loop = 1

        # Celui détecté
        self.who = None

        if self.conn:
            self.from_gui_receive_thread()

        self.config = config

        # Taille d'image possible 1280x720 seulement
        self.width = int(self.config['camera']['width_input'])
        self.height = int(self.config['camera']['height_input'])

        # model_dir est le dossier des model
        model_dir = os.path.join(current_dir, 'posenet')
        print("Dossier des model:", model_dir)
        if not os.path.exists(model_dir):
            print("Le dossier des model n'existe pas.")
        else:
            print("Le dossier des model existe.")
        MyPosenetPytorch.__init__(self, model_dir)

        self.img_without_skelets = None

        # Seuil de confiance de reconnaissance du squelette
        self.threshold_points = float(self.config['pose']['threshold_points'])
        self.threshold_pose = float(self.config['pose']['threshold_pose'])

        # Pour éliminer les poses trops loin, trop près, trop large en mmm
        self.profondeur_maxi = int(self.config['grandeechelle']['profondeur_maxi'])
        self.profondeur_mini = int(self.config['grandeechelle']['profondeur_mini'])
        self.largeur_maxi = int(self.config['grandeechelle']['largeur_maxi'])

        # Zoom
        self.decalage2000 = int(self.config['zoom']['decalage2000'])
        self.decalage8000 = int(self.config['zoom']['decalage8000'])
        self.size2000 = int(self.config['zoom']['size2000'])
        self.size8000 = int(self.config['zoom']['size8000'])

        self.full_screen = int(self.config['grandeechelle']['full_screen'])
        self.mode_expo = int(self.config['grandeechelle']['mode_expo'])

        self.depth = 1

    def from_gui_receive_thread(self):
        print("Lancement du thread from_gui_receive")
        t = Thread(target=self.from_gui_receive)
        t.start()

    def from_gui_receive(self):
        """Réception des datas du GUI, pour les réglages"""
        while self.pose_conn_loop:
            data = self.conn.recv()

            if data:
                if data[0] == 'quit':
                    print("Alerte: Quit reçu dans PosenetRealsense")
                    self.pose_conn_loop = 0
                    self.pose_loop = 0
                    self.pipeline.stop()

                elif data[0] == 'threshold_pose':
                    print('threshold_pose reçu dans posenet:', data[1])
                    self.threshold_pose = data[1]

                elif data[0] == 'threshold_points':
                    print('threshold_points reçu dans posenet:', data[1])
                    self.threshold_points = data[1]

                elif data[0] == 'profondeur_mini':
                    print('profondeur_mini reçu dans posenet::', data[1])
                    self.profondeur_mini = data[1]

                elif data[0] == 'profondeur_maxi':
                    print('profondeur_maxi reçu dans posenet::', data[1])
                    self.profondeur_maxi = data[1]

                elif data[0] == 'largeur_maxi':
                    print('largeur_maxi reçu dans posenet:', data[1])
                    self.largeur_maxi = data[1]

                elif data[0] == 'mode_expo':
                    print('mode_expo reçu dans posenet:', data[1])
                    self.mode_expo = data[1]

                elif data[0] == 'decalage2000':
                    print('decalage2000 reçu dans posenet:', data[1])
                    self.decalage2000 = data[1]

                elif data[0] == 'decalage8000':
                    print('decalage8000 reçu dans posenet:', data[1])
                    self.decalage8000 = data[1]

                elif data[0] == 'size2000':
                    print('size2000 reçu dans posenet:', data[1])
                    self.size2000 = data[1]

                elif data[0] == 'size8000':
                    print('size8000 reçu dans posenet:', data[1])
                    self.size8000 = data[1]

            sleep(0.05)

    def get_only_skelets_in_zone_and_valable(self):
        """Elimination des squelettes pas dans la zone
                profondeur mini à maxi et dans largeur_maxi
        et qui ne sont pas [None]*17
        """
        skelets_2D, skelets_3D, centers = [], [], []

        for i in range(len(self.skelets_2D)):
            # Si il n'y a pas de centre, il n'y a pas de personnage
            if self.centers[i]:
                # Entre mini maxi
                if self.profondeur_mini < self.centers[i][2] < self.profondeur_maxi:
                    # Pas trop écarté
                    if abs(self.centers[i][0]) < self.largeur_maxi:
                        # Squelette 3D valide, ne devrait pas arrivé si self.centers[i]
                        if self.skelets_3D[i] != [None]*17:
                            skelets_2D.append(self.skelets_2D[i])
                            skelets_3D.append(self.skelets_3D[i])
                            centers.append(self.centers[i])

        self.skelets_2D = skelets_2D
        self.skelets_2D = skelets_2D
        self.centers = centers

    def get_skelets_3D(self):
        """A partir des squelettes 2D détectés dans l'image,
        retourne les squelettes 3D
        """
        skelets_3D = []
        for xys in self.skelets_2D:
            if xys is not None:
                pts = self.get_points_3D(xys)
                # pts ne peut pas être [None]*17
                skelets_3D.append(pts)
        return skelets_3D

    def get_points_3D(self, xys):
        """Trouve les points 3D pour les squelettes xys"""
        # Les coordonnées des 17 points 3D avec qq None
        points_3D = [None]*17

        # Parcours des squelettes
        for i, xy in enumerate(xys):
            if xy:
                x = xy[0]
                y = xy[1]
                # Calcul de la profondeur du point
                profondeur = self.get_profondeur_du_point(x, y)
                if profondeur:
                    # Calcul les coordonnées 3D avec x et y coordonnées dans
                    # l'image et la profondeur du point
                    # Changement du nom de la fonction trop long
                    point_2D_to_3D = rs.rs2_deproject_pixel_to_point
                    point_with_deph = point_2D_to_3D(self.depth_intrinsic,
                                                     [x, y],
                                                     profondeur)
                    # Conversion des m en mm
                    points_3D[i] = [int(1000*x) for x in point_with_deph]

        return points_3D

    def get_profondeur_du_point(self, x, y):
        """Calcul la moyenne des profondeurs des pixels autour du point considéré
        Filtre les absurdes et les trop loins
        """
        profondeur = None
        distances = []
        # around = nombre de pixel autour du points
        x_min = max(x - 1, 0)
        x_max = min(x + 1, self.depth_frame.width)
        y_min = max(y - 1, 0)
        y_max = min(y + 1, self.depth_frame.height)

        for u in range(x_min, x_max):
            for v in range(y_min, y_max):
                # Profondeur du point de coordonnée (u, v) dans l'image
                distances.append(self.depth_frame.get_distance(u, v))

        # Si valeurs non trouvées, retourne [0.0, 0.0, 0.0, 0.0]
        # Remove the item 0.0 for all its occurrences
        dists = [i for i in distances if i != 0.0]
        dists_sort = sorted(dists)
        if len(dists_sort) > 2:
            # Suppression du plus petit et du plus grand
            goods = dists_sort[1:-1]
            # TODO: rajouter un filtre sur les absurdes ?

            # Calcul de la moyenne des profondeur
            profondeur = get_average_list_with_None(goods)

        return profondeur

    def get_who(self):
        """Détermination du squelette au centre
        self.centers = [[x, y , z], ...]
        all_x =    [-0.5, 0.8, 0.6]
        abs_all_x = [0.5, 0.8, 0.6]
        """
        if self.centers:
            # Recherche de celui au centre
            all_x = [c[0] for c in self.centers]
            abs_all_x = [abs(c[0]) for c in self.centers]

            if abs_all_x:  # si abs_all_x is not [], alors all_x est aussi
                mini = min(abs_all_x)
                # Index du mini des abs est le même que all_x
                self.who = abs_all_x.index(mini)
                self.depth = self.centers[self.who][2]
                # La position en x pour affichage
                self.x =  self.centers[self.who][0]
        else:
            self.who, self.depth, self.x = None, 0, self.x

    def from_all_xys_to_who(self, outputs):
        """ Appelé depuis la boucle infinie run()
        outputs = all_xys
        """
        self.skelets_2D, self.skelets_3D, self.centers = None, None, None

        # Récupération de tous les squelettes
        if outputs:
            # liste des squelettes 2D, un squelette = dict de 17 keypoints
            self.skelets_2D = outputs
            # Ajout de la profondeur pour 3D, et capture des couleurs
            # Liste des squelettes 3D, un squelette = list de 17 keypoints
            # un keypoint = liste de 3 coordonnées
            self.skelets_3D = self.get_skelets_3D()
            # Calcul de tous les centres
            self.centers = get_center_3D(self.skelets_3D)

            # Tri des squelettes dans la zone acceptable et pas tout None
            self.get_only_skelets_in_zone_and_valable()

            # self.skelets_2D est trié ici, ce n'est pas le même que ci-dessus
            if self.skelets_2D:
                # Détermination du squelette au centre
                self.get_who()

                # Envoi au GUI
                if self.conn and self.depth:
                    self.conn.send(['depth raw', int(self.depth)])

                # Dessin
                self.draw_all_poses()

    def frame_compute(self):
        # ############### Posenet
        outputs = self.compute_image(self.img,
                                     self.threshold_pose,
                                     self.threshold_points)

        # Recherche du bon squelette au centre
        # Ajoute les squelettes dans l'image self.img
        self.from_all_xys_to_who(outputs)

    def run(self):
        """Boucle infinie, quitter avec Echap dans la fenêtre OpenCV"""

        t0 = time()
        self.nbr = 0

        while self.pose_loop:
            self.nbr += 1

            # ############### RealSense
            frames = self.pipeline.wait_for_frames(timeout_ms=80)

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            color = aligned_frames.get_color_frame()
            self.depth_frame = aligned_frames.get_depth_frame()

            if not self.depth_frame and not color:
                continue

            color_data = color.as_frame().get_data()
            self.img = np.asanyarray(color_data)
            # L'image brute sans squelette pour y extraire le zoom
            self.img_without_skelets = self.img.copy()

            # Posenet
            self.frame_compute()

            # ############### Movenet
            # La partie d'image centrée sur who
            zoom = self.get_perso_zoom(self.img_without_skelets)
            # Envoi de zoom au GUI
            if zoom is not None:
                self.conn.send(['zoom', zoom])

            # Affichage
            self.viewer()

    def draw_all_poses(self):
        for i, skelet in enumerate(self.skelets_2D):
            if skelet:
                if i == self.who:
                    color = [0, 255, 0]
                else:
                    color = [0, 0, 255]
                self.draw_pose(skelet, color)

    def draw_pose(self, xys, color):
        """Affiche les points 2D, et les 'os' dans l'image pour un acteur
        xys = [[790, 331], [780, 313], None,  ... ]
        """
        # Dessin des points
        for point in xys:
            if point:
                x = point[0]
                y = point[1]
                cv2.circle(self.img, (x, y), 5, color=(100, 100, 100),
                                                                  thickness=-1)
                cv2.circle(self.img, (x, y), 6, color=color, thickness=1)

        # Dessin des os
        for a, b in EDGES:
            a = a.value  # 0 à 16
            b = b.value  # 0 à 16

            # Os seulement entre keypoints esxistants
            if not xys[a] or not xys[b] :
                continue

            # Les 2 keypoints existent
            ax, ay = xys[a]
            bx, by = xys[b]
            cv2.line(self.img, (ax, ay), (bx, by), color, 2)

    def get_perso_zoom(self, image):
        """Zoom sur la capture posenet,
        qui est au centre du perso dans la vue 2D
        Retourne une image de 256x256
        image = self.img_without_skelets est l'image realsense sans skelet
        self.img est l'image avec skelet
        """

        if self.skelets_2D:
            centre = get_center_2D(self.skelets_2D[self.who])

            # ############### Décalage vers le bas car jambes mal détectées
            # décalage à telle profondeur
            mini, d_min = self.decalage2000, 2000  # 20, 2000
            maxi, d_max = self.decalage8000, 8000  # 20, 8000

            a, b = get_a_b(d_min, mini, d_max, maxi)
            dec = int(a*self.depth + b)
            # Ajout du décalage vers le haut à y
            centre[1] += dec

            # ############### Calcul de la taille du zoom en fonction de depth
            # taille de la fenêtre à telle profondeur
            mini, d_min = self.size2000, 2000  # 600, 2000
            maxi, d_max = self.size8000, 8000  # 200, 8000

            a, b = get_a_b(d_min, mini, d_max, maxi)
            # Taille du zoom
            taille = int(a*self.depth + b)

            # Dessin du zoom r,s à u,v
            x, y = centre[0], centre[1]
            dt = int(taille/2)
            r = x - dt
            s = y - dt
            u = x + dt
            v = y + dt
            cv2.rectangle(self.img, (r, s), (u, v), [255, 0, 0], 2)

            # [rows, columns]
            zoom = image[s:v, r:u]
            if zoom.any():
                zoom = cv2.resize(zoom, (256, 256))
            else:
                zoom = None
        else:
            zoom = None

        return zoom



def get_a_b(x1, y1, x2, y2):
    """Calcule la pente et l'ordonnée à l'origine dans y = ax + b"""
    a = (y1 - y2)/(x1 - x2)
    b = y1 - a*x1
    return a, b


def get_center_2D(skelet):
    """Calcule le centre du skelet 2D dans l'image,
    retourne les positions x, y du centre dans l'image.
    """
    center = []
    for i in range(2):
        center.append(get_moyenne(skelet, i))
    return center


def get_center_3D(skelets_3D):
    """Retourne la liste des centres des squelettes 3D
    en coordonnées xyz dans l'espace en mm.
    x = latérale
    y = verticale
    z = profondeur
    """
    centers = []
    for skelet in skelets_3D:
        if skelet != [None]*17:
            center = []
            for i in range(3):
                center.append(get_moyenne(skelet, i))
            centers.append(center)
        else:
            centers.append(None)

    return centers


def get_moyenne(points, indice):
    """Calcul la moyenne d'une coordonnée des points,
    x = latérale
    y = verticale
    z = profondeur
    indice = 0 pour x, 1 pour y, 2 pour z
    """
    somme = 0
    n = 0
    for i in range(17):
        if points[i]:
            p = points[i][indice]
            if p:
                n += 1
                somme += p
    if n != 0:
        moyenne = int(somme/n)
    else:
        moyenne = None

    return moyenne


def get_average_list_with_None(liste):
    """Calcul de la moyenne des valeurs de la liste, sans tenir compte des None.
    liste = list de int ou float
    liste = [1, 2, None, ..., 10.0, None]

    Retourne un float
    Si la liste ne contient que des None, retourne None
    """
    # dtype permet d'accepter les None
    liste_array = np.array(liste, dtype=np.float64)

    return np.nanmean(liste_array)


def posenet_realsense_run(conn, current_dir, config):
    """Pour lancer ce script depuis le GUI en multiprocessing"""

    pnrs = PosenetRealsense(conn, current_dir, config)
    pnrs.run()
