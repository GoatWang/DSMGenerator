# Introduction
This repo aims to generate the 3D point cloud from two 2D images. All processes includes **Tie Point Finding**, **Rectification**, **Dense Matching** and **Point Cloud Generating**.

# Result
## CASE1: Aerial Image
1. Source
    | Left | Right |
    | --- | --- | 
    |![Aerial_Source_l](static/Aerial_Source_l.jpg) | ![Aerial_Source_r](static/Aerial_Source_r.jpg) |

2. Result
    - ![SGBM_CloudPoint](static/SGBM_CloudPoint.png)

## CASE2: Road Quality (Smoothness) Measurement: with hand-crafted processes
1. Source 
    - Video: 
        - ![RoadQuality_Source](static/RoadQuality_Source4.gif)


    - Images
        | Left | Right |
        | --- | --- | 
        |![RoadQuality_Source0](static/RoadQuality_Source0.jpg) | ![RoadQuality_Source1](static/RoadQuality_Source1.jpg) |

2. Result
    - ![RoadQuality_Result](static/RoadQuality_Result.gif)

# Theory
Please see [Pinhole Camera Model PPT](https://docs.google.com/presentation/d/1d6rK1UMtkgI-SxBe5717WRXsQCkIL9Wc/edit?usp=sharing&ouid=110278970958064225979&rtpof=true&sd=true)

# Scripts
- TiePoints.py: Use SIFT, ORB and SURF to find the sparse matching point (same point in real world) from 2 images.
    ```
    python TiePoints.py
    ```
    - ![TP_00](static/TP_00.png)
    - ![TP_01](static/TP_01.png)
    - ![TP_02](static/TP_02.png)
    - ![TP_10](static/TP_10.png)
    - ![TP_11](static/TP_11.png)
    - ![TP_12](static/TP_12.png)
    - ![TP_20](static/TP_20.png)
    - ![TP_21](static/TP_21.png)
    - ![TP_22](static/TP_22.png)
    - ![TP_unFiltered](static/TP_unFiltered.png)
    - ![TP_Filtered](static/TP_Filtered.png)
    
- Rectification.py: Rectify the image for application of SGBM reduce the searching dimension from two dimensions to one dimension.)
    ```
    python Rectification.py
    ```
    - ![Rectification_EpipolarLine](static/Rectification_EpipolarLine.png)
    - ![Rectification_Rectification](static/Rectification_Rectification.png)
    - ![Rectification_Add](static/Rectification_Add.png)

- SGBM_opencv.py: OpenCV version SGBM.
    ```
    python SGBM_opencv.py
    ```
    - ![SGBM_TP](static/SGBM_TP.png)
    - ![SGBM_Rectification](static/SGBM_Rectification.png)
    - ![SGBM_TP_Rectification](static/SGBM_TP_Rectification.png)
    - ![SGBM_Disparity](static/SGBM_Disparity.png)
    - ![SGBM_CloudPoint](static/SGBM_CloudPoint.png)

- AeroTriangulation.py: Conversion between 2D image pixels into 3D object points.
    ```
    python AeroTriangulation.py
    ```

- AeroTriangulation_tf.py: This is used for calibrate the relational extrinsic parameters of stereo camera. But its recommended to use the chessboard to calibrate the extrinsic parameters of stereo camera.
    ```
    python AeroTriangulation_tf.py
    ```
    - ![AereoTriangulation](static/AereoTriangulation.png)

- io_aereo_params.py: This is used for get the saved aereo parameters (OPK, L_XYZ, DMC_ROWS_LABEL, DMC_COLS_LABEL, DMC_FOCAL_LENGTH, DMC_PIXEL_SIZE, XOFFSET, YOFFSET)

- temp_refine_resolution.py: This is used to lower the resolution of original images in order to push the images on to the gitlab.

- SGBM.py: This is jit implementation of SGBM which can be used to learn SGBM, but its efficiency is not terrible. **currently not working**.
