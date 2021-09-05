# Real-Time-Emotional-Detector
https://colab.research.google.com/drive/1B-gQMwyIadw3_SvZWKRIV8EYNdaq_esW#scrollTo=yWajLJ9RtVv2&uniqifier=1

## Target
1. familiar with pytorch pipelines
    * Custom Pytorch Dataset
    * Custom Pytorch Transform Class
    * Fine-tuning Pytorch Model (Resnet)

2. Deploy on Flask
    * try to use base64 to encode & decode image bytes
    * try to use multi-thread to speed up 
 
## TEST
1. Evaluate training transform
   * torchvision.transforms (for pil) (slow)
   * Albumentation (for np) (resize the fastest)
   * custom implementation (for np) (fastest)
         ToTensor
         - transform (for pil)
         10 loops, best of 5: 2.87 ms per loop
         - transform (for np)
         10 loops, best of 5: 1.9 ms per loop
         - A.ToTensorV2 (for np)
         10 loops, best of 5: 7.67 µs per loop
         - custom (for np)
         10 loops, best of 5: 794 ns per loop

         Resize
         - transform (for pil)
         10 loops, best of 5: 2.8 ms per loop
         - A.Resize (for np)
         10 loops, best of 5: 1.19 ms per loop
         - custom (for np)
         10 loops, best of 5: 1.63 ms per loop

         RandomCrop
         - transform (for pil)
         The slowest run took 6.38 times longer than the fastest. This could mean that an intermediate result is being cached.
         10 loops, best of 5: 34.3 µs per loop
         - A.randomcrop (for np)
         10 loops, best of 5: 8.74 µs per loop
         - custom (for np)
         10 loops, best of 5: 7.23 µs per loop

3. Evaluate transform function for prediction
    * numpy to torch (raw) (t3) (fastest)
    * numpy + Augmentation (t2) (2nd)
    * pil + transforms (3rd)
    * numpy to pil + transforms (t1) (slowest)

3. Evaluate image transfer approach through flask
    * open image through bytes (io) (fastest)
    * use io.Bytes to wrap (wio) (2th)
    * encode through base64 (b64) (slowest)

2+3. Combination results: approach, total time in 100 loops (second)
         [['approach_io_np_t3', 3.908006429672241],
          ['approach_io_np_t2', 3.9259140491485596],
          ['approach_io_pil', 4.82680869102478],
          ['approach_io_np_t1', 5.604491949081421],
          ['approach_wio_np_t3', 11.306720972061157],
          ['approach_wio_np_t2', 11.346932411193848],
          ['approach_b64_np_t2', 11.60984992980957],
          ['approach_b64_np_t3', 11.618371725082397],
          ['approach_wio_pil', 12.65789532661438],
          ['approach_b64_pil', 13.032054662704468],
          ['approach_wio_np_t1', 13.037485122680664],
          ['approach_b64_np_t1', 13.37218427658081]]

       
4. Speed up frame transfer using multi-threading / multi-processing (need more try) on resnet18
    * raw method (single thread): FPS 5.7
    * multi-thread: FPS 4.4
    * multi-process (using process=2): FPS 1.5
