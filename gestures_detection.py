
from time import time, sleep
from threading import Thread

import cv2
import numpy as np

from names import COMBINAISONS, ACTIONS, OS, NAMES, EDGES, COLOR
from my_movenet import MyMovenet



class GesturesDetection(MyMovenet):
    """Détection des gestes"""

    def __init__(self, conn, current_dir, config):

        MyMovenet.__init__(self, current_dir)

        self.move_conn = conn
        self.current_dir = current_dir

        self.config = config
        self.zoom = np.zeros((256, 256, 3), dtype = "uint8")
        self.move_loop = 1

        # Calcul du FPS
        self.t_gest = time()
        self.nbr_gest = 0

        self.threshold_m = float(self.config['move']['threshold'])

        self.move_conn_loop = 1
        if self.move_conn:
            self.gestures_detection_receive_thread()

        # Tous les angles
        self.angles = {}

        cv2.namedWindow('Movenet', cv2.WND_PROP_FULLSCREEN)

    def gestures_detection_receive_thread(self):
        print("Lancement du thread receive dans gestures_detection")
        t_move = Thread(target=self.gestures_detection_receive)
        t_move.start()

    def gestures_detection_receive(self):
        while self.move_conn_loop:
            if self.move_conn.poll():
                data = self.move_conn.recv()
                if data:
                    if data[0] == 'quit':
                        print("Alerte: Quit reçu dans gestures_detection")
                        self.move_loop = 0
                        self.move_conn_loop = 0

                    elif data[0] == 'zoom':
                        zoom = data[1]
                        if zoom.any():
                            self.get_gestures(zoom)

                    elif data[0] == 'threshold':
                        print('threshold reçu dans movenet:', data[1])
                        self.threshold_m = data[1]

            sleep(0.001)

    def get_gestures(self, zoom):
        """Traitement d'une image reçue
        Le squelette est défini dans self.movenet_keypoints
        Affichage du squelette,
        récupération des angles,
        détection des gestes
        L'image zoom est sans les squelettes posenet.
        """
        # # if self.zoom is not None:
        # Actualise les movenet_keypoints sans l'affichage des squelettes posenet
        self.skeleton_detection(zoom, self.threshold_m)

        self.zoom = zoom

        # Détection des gestes
        self.draw_keypoints_edges()
        self.angles = self.get_all_angles()
        self.draw_angles()
        self.gestures()

        # Calcul du FPS, affichage toutes les 10 s
        if time() - self.t_gest > 10:
            print("FPS Movenet =", self.nbr_gest/10)
            self.t_gest, self.nbr_gest = time(), 0

    def run(self):
        """Affichage du zoom"""

        while self.move_loop:
            self.nbr_gest += 1
            cv2.imshow('Movenet', self.zoom)

            k = cv2.waitKey(36)
            # Pour quitter
            if k == 27:  # Esc
                self.move_conn.send(['quit', 1])
                print("Quit envoyé de GesturesDetection")

        cv2.destroyAllWindows()

    def gestures(self):
        """Reconnaissance de gestes basée sur les angles"""

        toto = 1
        # TODO: toto peut-il être une liste ou un dict ?
        self.move_conn.send(['geste', toto])

    def draw_keypoints_edges(self):
        """
        keypoints = [None, [200, 300], None, [100, 700], ...] = 17 items
        L'index correspond aux valeurs dans NAMES
        """
        # Dessin des points détectés
        for point in self.movenet_keypoints:
            if point:
                x = int(point[0])
                y = int(point[1])
                cv2.circle(self.zoom, (x, y), 4, color=(0,0,255), thickness=-1)

        # Dessin des os
        for i, (a, b) in enumerate(EDGES):
            """EDGES = (   (0, 1)
            keypoints[0] = [513, 149]
            """
            if not self.movenet_keypoints[a] or not self.movenet_keypoints[b]:
                continue
            ax = int(self.movenet_keypoints[a][0])
            ay = int(self.movenet_keypoints[a][1])
            bx = int(self.movenet_keypoints[b][0])
            by = int(self.movenet_keypoints[b][1])
            cv2.line(self.zoom, (ax, ay), (bx, by), COLOR[i], 2)

    def get_angle(self, p1, p2):
        """Angle entre horizontal et l'os
        origin p1
        p1 = numéro d'os
        tg(alpha) = y2 - y1 / x2 - x1
        """
        alpha = None

        if self.movenet_keypoints[p1] and self.movenet_keypoints[p2]:
            x1, y1 = self.movenet_keypoints[p1][0], self.movenet_keypoints[p1][1]
            x2, y2 = self.movenet_keypoints[p2][0], self.movenet_keypoints[p2][1]
            if x2 - x1 != 0:
                tg_alpha = (y2 - y1) / (x2 - x1)
                alpha = np.arctan(tg_alpha)*180/np.pi
                # # if x2 > x1:
                    # # alpha = int((180/np.pi) * np.arctan(tg_alpha))
                # # else:
                    # # alpha = 180 - int((180/np.pi) * np.arctan(tg_alpha))

        return alpha

    def get_all_angles(self):
        """angles = {'tibia droit': 128}
        origine = 1er de OS (14, 16)
        angles idem cercle trigo
        """
        angles = {}
        for os, (p1, p2) in OS.items():
            angles[os] = self.get_angle(p1, p2)

        return angles

    def draw_angles(self):
        """dessin des valeurs d'angles
        angles = {'tibia droit': 128}
        """
        for os, (p1, p2) in OS.items():
            if os in ['bras gauche', 'avant bras gauche']:
                if self.movenet_keypoints[p1] and self.movenet_keypoints[p2]:
                    alpha = self.angles[os]
                    if os in ['bras gauche']:
                        v = 30
                    else:
                        v = 60
                    if alpha:
                        cv2.putText(self.zoom,                  # image
                                    str(int(alpha)),            # text
                                    (150, v),                  # position
                                    cv2.FONT_HERSHEY_SIMPLEX,   # police
                                    1,                        # taille police
                                    (0, 255, 255),              # couleur
                                    2)                          # épaisseur

            if os in ['bras droit', 'avant bras droit']:
                if self.movenet_keypoints[p1] and self.movenet_keypoints[p2]:
                    alpha = self.angles[os]
                    if os in ['bras droit']:
                        v = 30
                    else:
                        v = 60
                    if alpha:
                        cv2.putText(self.zoom,                  # image
                                    str(int(alpha)),            # text
                                    (5, v),                  # position
                                    cv2.FONT_HERSHEY_SIMPLEX,   # police
                                    1,                        # taille police
                                    (0, 255, 255),              # couleur
                                    2)                          # épaisseur




def gestures_detection_run(conn, current_dir, config):
    """Lancement depuis GUI"""
    gd = GesturesDetection(conn, current_dir, config)
    gd.run()
