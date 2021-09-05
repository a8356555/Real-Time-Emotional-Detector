# Real-Time-Emotional-Detector

## Target
1. familiar with pytorch pipelines
    * Custom Pytorch Dataset
    * Custom Pytorch Transform Class
    * Fine-tuning Pytorch Model (Resnet)

2. Deploy on Flask
    * try to use base64 to encode & decode image bytes
    * try to use multi-thread to speed up 
 
## TODO
    * speed up frame transfer using multi-threading / multi-processing (need more try) on resnet18
        1. raw method (single thread): FPS 5.7
        2. multi-thread: FPS 4.4
        3. multi-process (using process=2): FPS 1.5
