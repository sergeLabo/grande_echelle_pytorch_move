

import tensorflow as tf



class MyMovenet:
    """La reconnaissance de squelette par MyMovenet Single Pose
    dans une image 256x256, pour la détection de gestes.
    """

    def __init__(self, current_dir):
        """mod = 1 ou 2 ou 3
        1 small
        2 medium
        3 big
        """

        self.mod = 3

        if self.mod == 3:
            model_path = current_dir + "/model_movenet_tflite/lite-model_movenet_singlepose_thunder_3.tflite"

        elif self.mod == 2:
            model_path = current_dir + "/model_movenet_tflite/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"

        elif self.mod == 1:
            model_path = current_dir + "/model_movenet_tflite/lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite"

        else:
            print("Ce model n'existe pas")

        print(f"movenet model path: {model_path}")

        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.movenet_keypoints = None

    def skeleton_detection(self, frame, threshold):
        if frame is not None:
            image = tf.expand_dims(frame, axis=0)
            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            image = tf.image.resize_with_pad(image, 256, 256)

            if self.mod == 3:
                input_image = tf.cast(image, dtype=tf.float32)
                ok = 1
            elif self.mod == 2:
                input_image = tf.cast(image, dtype=tf.uint8)
                ok = 1
            elif self.mod == 1:
                input_image = tf.cast(image, dtype=tf.uint8)
                ok = 1
            else:
                ok = 0
                print("Pas d'image en entrée")

            if ok:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
                self.interpreter.invoke()
                # Output is a [1, 1, 17, 3] numpy array.
                self.movenet_keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
                # Construction de ma liste de 17 keypoints = self.movenet_keypoints
                self.get_movenet_keypoints(threshold)
            else:
                print("Pas d'image pour movenet")
        else:
            print("Pas de frame pour movenet")

    def get_movenet_keypoints(self, threshold):
        """keypoints_with_scores = TODO à retrouver
        keypoints = [None, [200, 300], None, [100, 700], ...] = 17 items
        """
        keypoints = []
        for item in self.movenet_keypoints_with_scores[0][0]:
            if item[2] > threshold:
                x = int(item[1]*256)
                y = int((item[0]*256))
                keypoints.append([x, y])

            else:
                keypoints.append(None)
        self.movenet_keypoints = keypoints

    def movenet_close(self):
        """Marche pas"""
        print("Fermeture de MyMovenet")
        del self.interpreter
