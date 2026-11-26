# Third-party model notices

Models are downloaded unchanged from the official OpenCV Zoo repository. The
top-level repository license and individual model licenses are distinct.

## YuNet

- Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- File: `face_detection_yunet_2023mar.onnx`
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- License: MIT; see `licenses/YuNet-MIT.txt`.

## SFace

- Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- File: `face_recognition_sface_2021dec.onnx`
- SHA-256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
- License: Apache License 2.0; see `licenses/SFace-Apache-2.0.txt`.
- Upstream license: https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE

Python dependency licenses are included with their installed distributions.

## MediaPipe Face Landmarker

- Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- Versioned task: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- The bundle is downloaded unchanged; Face Landmarker model documentation lists
  Apache-2.0 licensing. MediaPipe source: https://github.com/google-ai-edge/mediapipe
- Mesh coordinates/pose are predictions from RGB, not measured physical depth.
