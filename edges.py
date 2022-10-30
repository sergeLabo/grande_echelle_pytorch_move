
import enum


class KeypointType(enum.IntEnum):
    """Pose kepoints."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


EDGES = (   (KeypointType.NOSE, KeypointType.LEFT_EYE),
            (KeypointType.NOSE, KeypointType.RIGHT_EYE),
            (KeypointType.NOSE, KeypointType.LEFT_EAR),
            (KeypointType.NOSE, KeypointType.RIGHT_EAR),
            (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
            (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
            (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
            (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
            (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
            (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
            (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
            (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
            (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
            (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
            (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
            (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
            (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
            (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
            (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE))
