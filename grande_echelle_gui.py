
"""
Interface graphique pour Grande Echelle
qui peut être lancé par run_grande_echelle.sh
et le raccourci sur le bureau
"""


from time import time, sleep
from pathlib import Path
from multiprocessing import Process, Pipe
from multiprocessing.sharedctypes import Value
from threading import Thread
import json
import gzip
from datetime import datetime

import numpy as np
from oscpy.client import OSCClient

import kivy
kivy.require('2.0.0')

from kivy.core.window import Window

k = 1
WS = (int(600*k), int(600*k))
Window.size = WS


from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty, NumericProperty

from posenet_realsense import posenet_realsense_run
from gestures_detection import gestures_detection_run
from filtre import moving_average


class MainScreen(Screen):
    """Ecran principal, l'appli s'ouvre sur cet écran
    root est le parent de cette classe dans la section <MainScreen> du kv
    """

    # Attribut de class, obligatoire pour appeler root.titre dans kv
    titre = StringProperty("Grande Echelle")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Trop fort
        self.app = App.get_running_app()

        # Lissage et Conversion des depth
        self.profondeur_maxi = int(self.app.config.get('grandeechelle', 'profondeur_maxi'))
        self.profondeur_mini = int(self.app.config.get('grandeechelle', 'profondeur_mini'))
        self.lissage = int(self.app.config.get('grandeechelle', 'lissage'))
        self.pile_size = self.lissage + 1
        self.histo = [self.profondeur_mini + 1000]*self.pile_size
        self.plage = int(self.app.config.get('grandeechelle', 'plage'))

        # Pour le Pipe
        self.p1_conn = None
        self.p3_conn = None
        self.kivy_receive_loop = 1

        # OSC
        self.ip = self.app.config.get('move', 'ip')
        self.port = int(self.app.config.get('move', 'port'))
        self.osc = OSCClient(self.ip, self.port)

        # Lancement de la capture pour Grande Echelle
        self.run_grande_echelle_capture()

        print("Initialisation du Screen MainScreen ok")

    def kivy_receive_thread(self):
        t_kivy = Thread(target=self.kivy_receive)
        t_kivy.start()

    def kivy_receive(self):
        """Réception des datas des autres processus"""
        while self.kivy_receive_loop:
            sleep(0.001)

            # De posenet realsense
            if self.p1_conn is not None:
                if self.p1_conn.poll():
                    try:
                        data1 = self.p1_conn.recv()
                    except:
                        data1 = None

                    if data1 is not None:
                        # depth
                        if data1[0] == 'depth raw':
                            depth = data1[1]
                            # # print("depth raw reçu", depth)
                            self.send_depth_smooth(depth)

                        # Relais des zoom
                        if data1[0] == 'zoom':
                            self.p3_conn.send(['zoom', data1[1]])

                        elif data1[0] == 'quit':
                            print("\nQuit reçu dans Kivy de Posenet Realsense ")
                            # Fait le quit dans PoseRealsense avec le quit
                            # envoyé par le Viewer
                            self.p1_conn.send(['quit', 1])
                            self.kivy_receive_loop = 0
                            self.app.do_quit()

            # De Movenet
            elif self.p3_conn is not None:
                if self.p3_conn.poll():
                    try:
                        data3 = self.p3_conn.recv()
                    except:
                        data3 = None

                    if data3 is not None:
                        if data3[0] == 'quit':
                            print("Quit reçu dans Kivy de Movenet")
                            self.app.do_quit()

                        # Détection de geste
                        elif data1[0] == 'geste':
                            g = data1[1]
                            print(f"Geste reçu dans Kivy {g}")
                            self.p2_conn.send(['geste', g])

    def run_grande_echelle_capture(self):
        """Chaqque processus communique avec ce script avec un Pipe"""

        print("Lancement de 3 processus:")

        current_dir = str(Path(__file__).parent.absolute())
        print("Dossier courrant:", current_dir)

        # Posenet et Realsense
        self.p1_conn, child_conn1 = Pipe()
        self.p1 = Process(target=posenet_realsense_run, args=(
                                                        child_conn1,
                                                        current_dir,
                                                        self.app.config, ))
        self.p1.start()
        print("Processus n°1 Posenet Realsense lancé ...")

        # Movenet
        self.p3_conn, child_conn3 = Pipe()
        self.p3 = Process(target=gestures_detection_run, args=(
                                                    child_conn3,
                                                    current_dir,
                                                    self.app.config, ))
        self.p3.start()
        print("Processus n°3 Gestures Detection lancé ...")

        self.kivy_receive_thread()
        print("Ca tourne ...")

    def send_depth_smooth(self, depth):
        """Calcule une depth lissé, converti sur une plage de self.plage
        Retourne depth_smooth
        """
        # Mise à jour de la pile
        self.histo.append(depth)
        del self.histo[0]

        try:
            depth_l = int(moving_average( np.array(self.histo),
                                        self.lissage,
                                        type_='simple')[0])
        except:
            depth_l = 0
            print("Erreur moving_average depth")

        # Pour bien comprendre
        mini = self.profondeur_mini + 1000  # frame 0 si mini
        maxi = self.profondeur_maxi - 500  # frame lenght si maxi

        # (x1, y1, x2, y2) = (mini, 0, maxi, self.plage)
        a, b = get_a_b(mini, self.plage, maxi, 0)
        depth_smooth = int(a*depth_l + b)
        # # print(depth_smooth)
        self.osc.send_message(b'/depth', [depth_smooth])



class ReglageProfondeur(Screen):

    threshold_pose = NumericProperty(0.8)
    threshold_points = NumericProperty(0.8)
    profondeur_mini = NumericProperty(1500)
    profondeur_maxi = NumericProperty(4000)
    largeur_maxi = NumericProperty(500)
    lissage = NumericProperty(11)
    d_mode = NumericProperty(0)
    info = NumericProperty(0)
    mode_expo = NumericProperty(0)
    threshold_move = NumericProperty(0.8)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Initialisation du Screen Settings")

        self.app = App.get_running_app()

        self.threshold_pose = float(self.app.config.get('pose',
                                                        'threshold_pose'))
        self.threshold_points = float(self.app.config.get('pose',
                                                        'threshold_points'))
        self.profondeur_mini = int(self.app.config.get('grandeechelle',
                                                        'profondeur_mini'))
        self.profondeur_maxi = int(self.app.config.get('grandeechelle',
                                                        'profondeur_maxi'))
        self.largeur_maxi = int(self.app.config.get('grandeechelle',
                                                    'largeur_maxi'))
        self.lissage = int(self.app.config.get('grandeechelle', 'lissage'))
        self.pile_size = self.lissage + 1
        self.mode_expo = int(self.app.config.get('grandeechelle', 'mode_expo'))
        self.threshold_move = float(self.app.config.get('move', 'threshold'))

    def do_slider(self, iD, instance, value):
        """Méthode appelée si action sur un slider.
        Il sont différenciée avec l'iD
        """
        scr = self.app.screen_manager.get_screen('Main')

        if iD == 'threshold_pose':
            # Maj de l'attribut
            self.threshold_pose = round(value, 2)
            # Maj de la config
            self.app.config.set('pose', 'threshold_pose', self.threshold_pose)
            # Sauvegarde dans le *.ini
            self.app.config.write()

            # Envoi de la valeur au process enfant
            if scr.p1_conn:
                scr.p1_conn.send(['threshold_pose', self.threshold_pose])

        if iD == 'threshold_points':
            # Maj de l'attribut
            self.threshold_points = round(value, 2)
            # Maj de la config
            self.app.config.set('pose', 'threshold_points', self.threshold_points)
            # Sauvegarde dans le *.ini
            self.app.config.write()

            # Envoi de la valeur au process enfant
            if scr.p1_conn:
                scr.p1_conn.send(['threshold_points', self.threshold_points])

        if iD == 'threshold_move':
            # Maj de l'attribut
            self.threshold_move = round(value, 2)
            # Maj de la config
            self.app.config.set('move', 'threshold', self.threshold_move)
            # Sauvegarde dans le *.ini
            self.app.config.write()

            # Envoi de la valeur au process enfant
            if scr.p3_conn:
                scr.p3_conn.send(['threshold', self.threshold_move])

        if iD == 'profondeur_mini':
            self.profondeur_mini = int(value)

            self.app.config.set('grandeechelle', 'profondeur_mini', self.profondeur_mini)
            self.app.config.write()

            if scr.p1_conn:
                scr.p1_conn.send(['profondeur_mini', self.profondeur_mini])

        if iD == 'profondeur_maxi':
            self.profondeur_maxi = int(value)

            self.app.config.set('grandeechelle', 'profondeur_maxi', self.profondeur_maxi)
            self.app.config.write()

            if scr.p1_conn:
                scr.p1_conn.send(['profondeur_maxi', self.profondeur_maxi])

        if iD == 'largeur_maxi':
            self.largeur_maxi = int(value)

            self.app.config.set('grandeechelle', 'largeur_maxi', self.largeur_maxi)
            self.app.config.write()
            if scr.p1_conn:
                scr.p1_conn.send(['largeur_maxi', self.largeur_maxi])

        if iD == 'lissage':
            self.lissage = int(value)
            self.pile_size = self.lissage + 1
            self.app.config.set('grandeechelle', 'lissage', self.lissage)
            self.app.config.write()

    def on_switch_mode_expo(self, instance, value):
        """Appelé par action sur switch mode expo"""
        scr = self.app.screen_manager.get_screen('Main')
        if value:
            value = 1
        else:
            value = 0
        self.mode_expo = value
        if scr.p1_conn:
            scr.p1_conn.send(['mode_expo', self.mode_expo])
        self.app.config.set('grandeechelle', 'mode_expo', self.mode_expo)
        self.app.config.write()
        print("mode_expo =", self.mode_expo)



class ReglageGestes(Screen):
    decalage2000 = NumericProperty(20)  # 0 à 50
    decalage8000 = NumericProperty(20)  # 0 à 50
    size2000 = NumericProperty(600)     # 400 à 720
    size8000 = NumericProperty(200)      # 100 à 300

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Initialisation du Screen Settings")

        self.app = App.get_running_app()

        self.decalage2000 = int(self.app.config.get('zoom', 'decalage2000'))
        self.decalage8000 = int(self.app.config.get('zoom', 'decalage8000'))
        self.size2000 = int(self.app.config.get('zoom', 'size2000'))
        self.size8000 = int(self.app.config.get('zoom', 'size8000'))

    def do_slider(self, iD, instance, value):
        """Méthode appelée si action sur un slider.
        Il sont différenciée avec l'iD
        """
        scr = self.app.screen_manager.get_screen('Main')

        if iD == 'decalage2000':
            # Maj de l'attribut
            self.decalage2000 = int(value)
            # Maj de la config
            self.app.config.set('zoom', 'decalage2000', self.decalage2000)
            # Sauvegarde dans le *.ini
            self.app.config.write()

            # Envoi de la valeur au process enfant Posenet et Realsense
            if scr.p1_conn:
                scr.p1_conn.send(['decalage2000', self.decalage2000])

        elif iD == 'decalage8000':
            # Maj de l'attribut
            self.decalage8000 = int(value)
            # Maj de la config
            self.app.config.set('zoom', 'decalage8000', self.decalage8000)
            # Sauvegarde dans le *.ini
            self.app.config.write()

            # Envoi de la valeur au process enfant Posenet et Realsense
            if scr.p1_conn:
                scr.p1_conn.send(['decalage8000', self.decalage8000])

        if iD == 'size2000':
            # Maj de l'attribut
            self.size2000 = int(value)
            # Maj de la config
            self.app.config.set('zoom', 'size2000', self.size2000)
            # Sauvegarde dans le *.ini
            self.app.config.write()

            # Envoi de la valeur au process enfant Posenet et Realsense
            if scr.p1_conn:
                scr.p1_conn.send(['size2000', self.size2000])

        elif iD == 'size8000':
            # Maj de l'attribut
            self.size8000 = int(value)
            # Maj de la config
            self.app.config.set('zoom', 'size8000', self.size8000)
            # Sauvegarde dans le *.ini
            self.app.config.write()

            # Envoi de la valeur au process enfant Posenet et Realsense
            if scr.p1_conn:
                scr.p1_conn.send(['size8000', self.size8000])



SCREENS = { 0: (MainScreen, 'Main'),
            1: (ReglageProfondeur, 'ReglageProfondeur'),
            2: (ReglageGestes, 'ReglageGestes')}



class Grande_EchelleApp(App):

    def build(self):
        """Exécuté après build_config, construit les écrans"""

        # Création des écrans
        self.screen_manager = ScreenManager()
        for i in range(len(SCREENS)):
            # Pour chaque écran, équivaut à
            # self.screen_manager.add_widget(MainScreen(name="Main"))
            self.screen_manager.add_widget(SCREENS[i][0](name=SCREENS[i][1]))

        return self.screen_manager

    def build_config(self, config):
        """Excécuté en premier (ou après __init__()).
        Si le fichier *.ini n'existe pas,
                il est créé avec ces valeurs par défaut.
        Il s'appelle comme le kv mais en ini
        Si il manque seulement des lignes, il ne fait rien !
        """

        print("Création du fichier *.ini si il n'existe pas")

        config.setdefaults( 'camera',
                                        {   'width_input': 1280,
                                            'height_input': 720})

        config.setdefaults( 'pose',
                                        {   'threshold_pose': 0.15,
                                            'threshold_points': 0.29})

        config.setdefaults( 'grandeechelle',
                                        {   'frame_rate_du_film': 25,
                                            'film': 'ICOS.mp4',
                                            'profondeur_mini': 1500,
                                            'profondeur_maxi': 4000,
                                            'largeur_maxi': 500,
                                            'lissage': 11,
                                            'full_screen': 0,
                                            'mode_expo': 0,
                                            'raz': 5})

        config.setdefaults( 'move',
                                        {   'threshold': 0.23,
                                            'ip': '127.0.0.1',
                                            'port': 8000})

        config.setdefaults( 'zoom',
                                        {'decalage2000': 20,
                                         'decalage8000': 20,
                                         'size2000': 600,
                                         'size8000': 200})

        print("self.config peut maintenant être appelé")

    def go_mainscreen(self):
        """Retour au menu principal depuis les autres écrans."""
        self.screen_manager.current = ("Main")

    def do_quit(self):
        print("Je quitte proprement, j'attends ....")
        scr = self.screen_manager.get_screen('Main')

        scr.p1_conn.send(['quit', 1])
        scr.p3_conn.send(['quit', 1])

        scr.kivy_receive_loop = 0
        sleep(0.2)

        scr.p3.terminate()
        sleep(0.2)
        scr.p1.terminate()
        sleep(0.2)

        # Kivy
        print("Quit final")
        Grande_EchelleApp.get_running_app().stop()



def get_a_b(x1, y1, x2, y2):
    """Calcule la pente et l'ordonnée à l'origine dans y = ax + b"""
    a = (y1 - y2)/(x1 - x2)
    b = y1 - a*x1
    return a, b



if __name__ == '__main__':
    """L'application s'appelle Grande_Echelle
    d'où
    la class
        Grande_EchelleApp()
    les fichiers:
        grande_echelle.kv
        grande_echelle.ini
    """

    Grande_EchelleApp().run()
