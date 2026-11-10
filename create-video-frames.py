# import cv2
# import numpy as np

# out_path = 'data/sample_videos/synthetic_test.mp4'
# fps = 15
# duration_s = 10
# w, h = 320, 240
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
# for i in range(fps * duration_s):
#     frame = np.full((h, w, 3), 40, dtype=np.uint8)
#     x = int((i / (fps * duration_s)) * (w - 40))
#     cv2.rectangle(frame, (x, 100), (x + 40, 180), (0, 200, 255), -1)
#     writer.write(frame)
# writer.release()
# print('done')



from pathlib import Path
from datetime import datetime
from ingestion.multimodal_detector import MultimodalDetector

detector = MultimodalDetector()
result = detector.detect(
    frame_path=Path('.frame_cache/CAM_01_frame000000.jpg'),
    camera_id='CAM_01',
    timestamp=datetime.now()
)
print(result.model_dump_json(indent=2))
