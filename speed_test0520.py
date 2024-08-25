import argparse

import cv2
from ultralytics import YOLO

import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse. ArgumentParser(
    description="Vehicle Speed Estimation using Inference and Supervision"
    )
    parser.add_argument(
    '--source_video_path',
    required=True,
    help="Path to the source video file",
    type=str,
    )
    return parser.parse_args()


#C:\Users\user\Desktop\無人機影片\100MEDIA\學長姐/DJI_0001.MOV


if __name__ == "__main__":  
    args = parse_arguments()

    model= YOLO("yolov8x.pt")

    bounding_box_annotator = sv.BoundingBoxAnnotator()

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    for frame in frame_generator:
        result = model(frame) [0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy ()
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)

        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cv2. destroyAllWindows()

