# Grande Echelle

Version avec PyTorch, sans Coral.<br>
Détection de gestes avec Movenet et Tensorflow

* GPU GTX1060
* Driver Version: 470.141.03
* CUDA Version: 11.4

Tensorflow 2.10 ne marche pas

### Merci à
* [Ross Wightman](https://github.com/rwightman)
* [rwightman posenet-pytorch](https://github.com/rwightman/posenet-pytorch)
pour la conversion du modèle PoseNet

### Le film n'est pas dans ce dépôt
Le bon film est ICOS-FINAL.mov de 3.7 Go

### Installation sur Ubuntu Mate 20.04
RealSense ne fonctionne pas avec python 3.10, il est impératif d'utiliser Ubuntu 20.04, et pas Ubuntu 22.04

### Installation de CUDA
``` bash
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt install nvidia-cuda-toolkit
sudo apt install libcurand10 libcusolver-11-4 libcusparse-11-4 libcufft10
```

#### RealSense D 455
``` bash
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo apt install software-properties-common
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo focal main" -u
sudo apt install librealsense2-dkms librealsense2-dev ?
```

#### Python
Installe tous les packages nécessaires dans un dossier /mon_env dans le dossier /grande_echelle
``` bash
# Mise à jour de pip
sudo apt install python3-pip python3-dev
python3 -m pip install --upgrade pip

# Installation de venv
sudo apt install python3-venv

# Installation de l'environnement
cd /le/dossier/de/grande_echelle/

# Création du dossier environnement si pas encore créé.
python3 -m venv mon_env

# Activation
source mon_env/bin/activate

# Installation de la version de pytorch nécessaire avec la carte GTX3050
python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Installation des packages, numpy, opencv-python, pyrealsense2, kivy, ...
python3 -m pip install -r requirements.txt
```

#### Bug dans Kivy
``` bash
sudo apt install xclip
```

### Délai pour éteindre le PC
[Modification du délai pour éteindre](https://ressources.labomedia.org/la_grande_echelle#modification_du_delai_pour_eteindre_une_debian)


### Excécution
Copier coller le lanceur grande-echelle.desktop sur le Bureau.<br>
Il faut le modifier avec Propriétés: adapter le chemin à votre cas.<br>
Ce lanceur lance un terminal et l'interface graphique, et il permet aussi de créer une application au démarrage en créant un lanceur dans ~./config/autostart

#### Utilisation
* Bascule full_screen en cours en activant la fenêtre à aggrandir puis:
    * espace
* Réglages à chaud permet de modifier tous les paramètres.
* En mode expo, démarrage directement en full screen sur le film, pas d'info, pas d'image de capture

#### Explications sur les  paramètres
* threshold: Seuil de confiance de la détection, plus c'est grand moins il y a d'erreur, mais plus la détection est difficile.
* [pose]
    * threshold_pose = 0.31 pour l'ensemble des points
    * threshold_points = 0.33 par point
* [move]
    * threshold = 0.23 pour le zoom
* pile_size = 80 lissage de la profondeur
* profondeur_mini = 1200, limite le mini
* profondeur_maxi = 4000, limite le maxi
* largeur_maxi = 1500, limite la plage des x

### LICENSE

#### Apache License, Version 2.0

* pose_engine.py
* posenet_histopocene.py
* pyrealsense2

#### Licence GPL v3

* tous les autres fichiers

#### Creative Commons

[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License](http://oer2go.org/mods/en-boundless/creativecommons.org/licenses/by-nc-nd/4.0/legalcode.html)
