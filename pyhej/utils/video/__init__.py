import os
import cv2 as cv
from pydub import AudioSegment


def mp4_to_mp3(mp4, mp3):
    """
    假设你有mp4和flv视频,并且你想把它们转换成mp3

    # Arguments
        mp4: Path to input video file
        mp3: Path to output audio file
    """
    if not os.path.isdir(os.path.dirname(mp3)):
        os.system("mkdir -m 777 -p {}".format(os.path.dirname(mp3)))
    AudioSegment.from_file(mp4, format="mp4").export(mp3, format="mp3")
    return mp3


def audio_slice_mp3(mp3, seconds=30, output=None):
    """
    # Arguments
        mp3: Path to mp3 file or AudioSegment object
        seconds: seconds of the audio
    """
    if isinstance(mp3, str):
        mp3 = AudioSegment.from_mp3(mp3)
    if output is None:
        return mp3[:seconds*1000]
    else:
        if not os.path.isdir(os.path.dirname(output)):
            os.system("mkdir -m 777 -p {}".format(os.path.dirname(output)))
        mp3[:seconds*1000].export(output, format="mp3")
        return output


def video_capture(dev, size=3):
    """
    # Arguments
        dev: str, such as "/dev/video0"
        size: int, num of show images
    """
    vcapture = cv.VideoCapture(dev)

    assert vcapture.isOpened(), "not find dev {}".format(dev)
    print("capture fps: {}".format(vcapture.get(cv.CAP_PROP_FPS)))

    for i in range(size):
        success, image = vcapture.read()
        if success:
            plt.imshow(image[..., ::-1])
            plt.show()
        else:
            print("warning: {}".format(success))

    vcapture.release()