# [Reproduce]-Correction-Filter-for-Single-Image-Super-Resolution
This project reproduced the paper ['Correction Filter for Single Image Super-Resolution: Robustifying Off-the-Shelf Deep Super-Resolvers'](https://arxiv.org/abs/1912.00157) by Shady et al..

Through my experiments, it can be concluded that the main contribution of the paper is reproducible.

## Requirements
- matplotlib
- numpy
- scipy
- scikit-image
- torch
- torchvision


## Usage
First of all, install the prerequisites by running:
```
$pip install requirements.txt
```
To get the downsampled test image (generated using the Gaussian kernel), run the command line:
```
$ python downsample_imgs.py --in_dir 'YOUR INPUT DIRECTORY' --out_dir 'YOUR OUTPUT DIRECTORY' --standard_deviation 'YOUR DESIRED STD OF GAUSSIAN KERNEL' --scale_factor 'YOUR DESIRED SCALING FACTOR'
```

The downsampled test image is stored in the path:
```
'YOUR OUTPUT DIRECTORY DEFINED ABOVE' + '/Gaussian_mat/*.mat' 
```

### The Non-blind Setting
To correct the test image, run the command line:
```
$ python downsample_imgs.py --in_dir 'YOUR INPUT DIRECTORY' --out_dir 'YOUR OUTPUT DIRECTORY' --standard_deviation 'YOUR DESIRED STD OF GAUSSIAN KERNEL' --scale_factor 'YOUR DESIRED SCALING FACTOR'
```

Your corrected image is stored in the path:
```
'YOUR OUTPUT DIRECTORY DEFINED ABOVE' + '/LR_Images_Corrected/*.png' 
```

### The blind Setting
To use CNN to estimate the correct image, run the command line:
```
$ python estimate_correction.py --in_dir 'YOUR INPUT DIRECTORY' --out_dir 'YOUR OUTPUT DIRECTORY' --standard_deviation 'YOUR DESIRED STD OF GAUSSIAN KERNEL' --iterations 'YOUR DESIRED NUMBER OF ITERATIONS FOR CNN'
```
Your estimated image is stored in the path:
```
'YOUR OUTPUT DIRECTORY DEFINED ABOVE' + '/Estimate_Output/*.png' 
```

### To use super-resolvers
This project uses DBPN super-resolver. This repository does not include DBPN due to its large file size. However, you can find it [here](https://github.com/alterzero/DBPN-Pytorch).

#### Node: In my experiment, the code of DBPN can't be run directly. If you are like me, got the error like follows:
<img width="949" alt="Screen Shot 2021-04-21 at 12 38 55 AM" src="https://user-images.githubusercontent.com/29801160/115497688-25bedb80-a23a-11eb-93ce-c297a79dc064.png">

You can use the code below to avoid this error:
```
dictionary_modified = torch.load(opt.model, map_location=lambda storage, loc: storage)
dictionary_copy = dictionary_modified.copy()
for key in dictionary_copy:
    res = list(key)
    res = res[7:]
    res = ''.join(res)
    dictionary_modified[res] = dictionary_modified.pop(key)
model.load_state_dict(dictionary_modified)
```
### To evaluate the performance 
This project uses the same metrics as in the paper, namely, PSNR AND SSIM. To evaluate the result, run the command line:
```
$ python metric.py
```
#### Node: you might need to change the default input and output directory.



### Results:
The left images are DBPN without correction filter, the right images are DBPN with correction filter.
<div class="row">
  <div class="column", float="left">
    <img src="https://user-images.githubusercontent.com/29801160/115499623-ba770880-a23d-11eb-81b9-762f2cd72e32.png", width=300>
  </div>
    <div class="column">
  </div>
  <div class="column", float="right">
    <img src="https://user-images.githubusercontent.com/29801160/115499605-b21ecd80-a23d-11eb-83c2-bcc707d8079c.png", width=300>
  </div>
</div>
<br>
<br>


<div class="row">
  <div class="column", float="left">
    <img src="https://user-images.githubusercontent.com/29801160/115500678-c794f700-a23f-11eb-99e5-365bab179aab.png", width=300>
  </div>
    <div class="column">
  </div>
  <div class="column", float="right">
    <img src="https://user-images.githubusercontent.com/29801160/115500662-c2d04300-a23f-11eb-9bd6-8c61e99119c1.png", width=300>
  </div>
</div>
<br>
<br>

<div class="row">
  <div class="column", float="left">
    <img src="https://user-images.githubusercontent.com/29801160/115501158-b13b6b00-a240-11eb-98ef-541a3bb6a323.png", width=300>
  </div>
    <div class="column">
  </div>
  <div class="column", float="right">
    <img src="https://user-images.githubusercontent.com/29801160/115501203-c3b5a480-a240-11eb-9681-a8975d36fca5.png", width=300>
  </div>
</div>



### Metric:
<img width="800" alt="Screen Shot 2021-04-21 at 1 30 22 AM" src="https://user-images.githubusercontent.com/29801160/115501459-3de62900-a241-11eb-8feb-c5f005fb7ec1.png">
<img width="800" alt="Screen Shot 2021-04-21 at 1 32 35 AM" src="https://user-images.githubusercontent.com/29801160/115501593-7dad1080-a241-11eb-8d02-2aa7fa60b338.png">
<img width="800" alt="Screen Shot 2021-04-21 at 1 33 50 AM" src="https://user-images.githubusercontent.com/29801160/115501684-a59c7400-a241-11eb-9978-1c0b013379e0.png">
<img width="800" alt="Screen Shot 2021-04-21 at 1 34 28 AM" src="https://user-images.githubusercontent.com/29801160/115501734-bcdb6180-a241-11eb-917f-9f55f82b2f95.png">



### References:
- [Correction Filter for Single Image Super-Resolution: Robustifying Off-the-Shelf Deep Super-Resolvers](https://arxiv.org/abs/1912.00157)
- [DBPN](https://github.com/alterzero/DBPN-Pytorch)


