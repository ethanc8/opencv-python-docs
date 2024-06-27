# `cv2.aruco`
```{py:module} cv2.aruco
None
```
## Attributes
```{py:attribute} CORNER_REFINE_APRILTAG
:type: int
```


```{py:attribute} CORNER_REFINE_CONTOUR
:type: int
```


```{py:attribute} CORNER_REFINE_NONE
:type: int
```


```{py:attribute} CORNER_REFINE_SUBPIX
:type: int
```


```{py:attribute} DICT_4X4_100
:type: int
```


```{py:attribute} DICT_4X4_1000
:type: int
```


```{py:attribute} DICT_4X4_250
:type: int
```


```{py:attribute} DICT_4X4_50
:type: int
```


```{py:attribute} DICT_5X5_100
:type: int
```


```{py:attribute} DICT_5X5_1000
:type: int
```


```{py:attribute} DICT_5X5_250
:type: int
```


```{py:attribute} DICT_5X5_50
:type: int
```


```{py:attribute} DICT_6X6_100
:type: int
```


```{py:attribute} DICT_6X6_1000
:type: int
```


```{py:attribute} DICT_6X6_250
:type: int
```


```{py:attribute} DICT_6X6_50
:type: int
```


```{py:attribute} DICT_7X7_100
:type: int
```


```{py:attribute} DICT_7X7_1000
:type: int
```


```{py:attribute} DICT_7X7_250
:type: int
```


```{py:attribute} DICT_7X7_50
:type: int
```


```{py:attribute} DICT_APRILTAG_16H5
:type: int
```


```{py:attribute} DICT_APRILTAG_16h5
:type: int
```


```{py:attribute} DICT_APRILTAG_25H9
:type: int
```


```{py:attribute} DICT_APRILTAG_25h9
:type: int
```


```{py:attribute} DICT_APRILTAG_36H10
:type: int
```


```{py:attribute} DICT_APRILTAG_36H11
:type: int
```


```{py:attribute} DICT_APRILTAG_36h10
:type: int
```


```{py:attribute} DICT_APRILTAG_36h11
:type: int
```


```{py:attribute} DICT_ARUCO_MIP_36H12
:type: int
```


```{py:attribute} DICT_ARUCO_MIP_36h12
:type: int
```


```{py:attribute} DICT_ARUCO_ORIGINAL
:type: int
```


```{py:attribute} 
```


```{py:attribute} 
```


```{py:attribute} 
```


```{py:attribute} 
```


```{py:attribute} 
```


```{py:attribute} 
```


```{py:attribute} 
```


```{py:attribute} 
```



## Classes
`````{py:class} ArucoDetector




````{py:method} detectMarkers(image[, corners[, ids[, rejectedImgPoints]]]) -> corners, ids, rejectedImgPoints

Basic marker detection

Performs marker detection in the input image. Only markers included in the specific dictionary are searched. For each detected marker, it returns the 2D position of its corner in the image and its corresponding identifier. Note that this function does not perform pose estimation. 
```{note}
The function does not correct lens distortion or takes it into account. It's recommended to undistortinput image with corresponding camera model, if camera parameters are known 
```
**See also:** undistort, estimatePoseSingleMarkers,  estimatePoseBoard


:param self: 
:type self: 
:param image: input image
:type image: cv2.typing.MatLike
:param corners: vector of detected marker corners. For each marker, its four cornersare provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array is Nx4. The order of the corners is clockwise. 
:type corners: _typing.Sequence[cv2.typing.MatLike] | None
:param ids: vector of identifiers of the detected markers. The identifier is of type int(e.g. std::vector<int>). For N detected markers, the size of ids is also N. The identifiers have the same order than the markers in the imgPoints array. 
:type ids: cv2.typing.MatLike | None
:param rejectedImgPoints: contains the imgPoints of those squares whose inner code has not acorrect codification. Useful for debugging purposes. 
:type rejectedImgPoints: _typing.Sequence[cv2.typing.MatLike] | None
:rettype: tuple[_typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike]]
````

````{py:method} detectMarkers(image[, corners[, ids[, rejectedImgPoints]]]) -> corners, ids, rejectedImgPoints

Basic marker detection

Performs marker detection in the input image. Only markers included in the specific dictionary are searched. For each detected marker, it returns the 2D position of its corner in the image and its corresponding identifier. Note that this function does not perform pose estimation. 
```{note}
The function does not correct lens distortion or takes it into account. It's recommended to undistortinput image with corresponding camera model, if camera parameters are known 
```
**See also:** undistort, estimatePoseSingleMarkers,  estimatePoseBoard


:param self: 
:type self: 
:param image: input image
:type image: cv2.UMat
:param corners: vector of detected marker corners. For each marker, its four cornersare provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array is Nx4. The order of the corners is clockwise. 
:type corners: _typing.Sequence[cv2.UMat] | None
:param ids: vector of identifiers of the detected markers. The identifier is of type int(e.g. std::vector<int>). For N detected markers, the size of ids is also N. The identifiers have the same order than the markers in the imgPoints array. 
:type ids: cv2.UMat | None
:param rejectedImgPoints: contains the imgPoints of those squares whose inner code has not acorrect codification. Useful for debugging purposes. 
:type rejectedImgPoints: _typing.Sequence[cv2.UMat] | None
:rettype: tuple[_typing.Sequence[cv2.UMat], cv2.UMat, _typing.Sequence[cv2.UMat]]
````

````{py:method} refineDetectedMarkers(image, board, detectedCorners, detectedIds, rejectedCorners[, cameraMatrix[, distCoeffs[, recoveredIdxs]]]) -> detectedCorners, detectedIds, rejectedCorners, recoveredIdxs

Refine not detected markers based on the already detected and the board layout

This function tries to find markers that were not detected in the basic detecMarkers function. First, based on the current detected marker and the board layout, the function interpolates the position of the missing markers. Then it tries to find correspondence between the reprojected markers and the rejected candidates based on the minRepDistance and errorCorrectionRate parameters. If camera parameters and distortion coefficients are provided, missing markers are reprojected using projectPoint function. If not, missing marker projections are interpolated using global homography, and all the marker corners in the board must have the same Z coordinate. 


:param self: 
:type self: 
:param image: input image
:type image: cv2.typing.MatLike
:param board: layout of markers in the board.
:type board: Board
:param detectedCorners: vector of already detected marker corners.
:type detectedCorners: _typing.Sequence[cv2.typing.MatLike]
:param detectedIds: vector of already detected marker identifiers.
:type detectedIds: cv2.typing.MatLike
:param rejectedCorners: vector of rejected candidates during the marker detection process.
:type rejectedCorners: _typing.Sequence[cv2.typing.MatLike]
:param cameraMatrix: optional input 3x3 floating-point camera matrix$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ 
:type cameraMatrix: cv2.typing.MatLike | None
:param distCoeffs: optional vector of distortion coefficients$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])$ of 4, 5, 8 or 12 elements 
:type distCoeffs: cv2.typing.MatLike | None
:param recoveredIdxs: Optional array to returns the indexes of the recovered candidates in theoriginal rejectedCorners array. 
:type recoveredIdxs: cv2.typing.MatLike | None
:rettype: tuple[_typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike]
````

````{py:method} refineDetectedMarkers(image, board, detectedCorners, detectedIds, rejectedCorners[, cameraMatrix[, distCoeffs[, recoveredIdxs]]]) -> detectedCorners, detectedIds, rejectedCorners, recoveredIdxs

Refine not detected markers based on the already detected and the board layout

This function tries to find markers that were not detected in the basic detecMarkers function. First, based on the current detected marker and the board layout, the function interpolates the position of the missing markers. Then it tries to find correspondence between the reprojected markers and the rejected candidates based on the minRepDistance and errorCorrectionRate parameters. If camera parameters and distortion coefficients are provided, missing markers are reprojected using projectPoint function. If not, missing marker projections are interpolated using global homography, and all the marker corners in the board must have the same Z coordinate. 


:param self: 
:type self: 
:param image: input image
:type image: cv2.UMat
:param board: layout of markers in the board.
:type board: Board
:param detectedCorners: vector of already detected marker corners.
:type detectedCorners: _typing.Sequence[cv2.UMat]
:param detectedIds: vector of already detected marker identifiers.
:type detectedIds: cv2.UMat
:param rejectedCorners: vector of rejected candidates during the marker detection process.
:type rejectedCorners: _typing.Sequence[cv2.UMat]
:param cameraMatrix: optional input 3x3 floating-point camera matrix$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ 
:type cameraMatrix: cv2.UMat | None
:param distCoeffs: optional vector of distortion coefficients$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])$ of 4, 5, 8 or 12 elements 
:type distCoeffs: cv2.UMat | None
:param recoveredIdxs: Optional array to returns the indexes of the recovered candidates in theoriginal rejectedCorners array. 
:type recoveredIdxs: cv2.UMat | None
:rettype: tuple[_typing.Sequence[cv2.UMat], cv2.UMat, _typing.Sequence[cv2.UMat], cv2.UMat]
````

````{py:method} __init__(self, dictionary: Dictionary=..., detectorParams: DetectorParameters=..., refineParams: RefineParameters=...)


Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param dictionary: 
:type dictionary: Dictionary
:param detectorParams: 
:type detectorParams: DetectorParameters
:param refineParams: 
:type refineParams: RefineParameters
:rettype: None
````

````{py:method} getDictionary() -> retval





:param self: 
:type self: 
:rettype: Dictionary
````

````{py:method} setDictionary(dictionary) -> None





:param self: 
:type self: 
:param dictionary: 
:type dictionary: Dictionary
:rettype: None
````

````{py:method} getDetectorParameters() -> retval





:param self: 
:type self: 
:rettype: DetectorParameters
````

````{py:method} setDetectorParameters(detectorParameters) -> None





:param self: 
:type self: 
:param detectorParameters: 
:type detectorParameters: DetectorParameters
:rettype: None
````

````{py:method} getRefineParameters() -> retval





:param self: 
:type self: 
:rettype: RefineParameters
````

````{py:method} setRefineParameters(refineParameters) -> None





:param self: 
:type self: 
:param refineParameters: 
:type refineParameters: RefineParameters
:rettype: None
````

````{py:method} write(fs, name) -> None
simplified API for language bindings




:param self: 
:type self: 
:param fs: 
:type fs: cv2.FileStorage
:param name: 
:type name: str
:rettype: None
````

````{py:method} read(fn) -> None
Reads algorithm parameters from a file storage




:param self: 
:type self: 
:param fn: 
:type fn: cv2.FileNode
:rettype: None
````


`````


`````{py:class} Board




````{py:method} __init__(self, objPoints: _typing.Sequence[cv2.typing.MatLike], dictionary: Dictionary, ids: cv2.typing.MatLike)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param objPoints: 
:type objPoints: _typing.Sequence[cv2.typing.MatLike]
:param dictionary: 
:type dictionary: Dictionary
:param ids: 
:type ids: cv2.typing.MatLike
:rettype: None
````

````{py:method} __init__(self, objPoints: _typing.Sequence[cv2.UMat], dictionary: Dictionary, ids: cv2.UMat)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param objPoints: 
:type objPoints: _typing.Sequence[cv2.UMat]
:param dictionary: 
:type dictionary: Dictionary
:param ids: 
:type ids: cv2.UMat
:rettype: None
````

````{py:method} matchImagePoints(detectedCorners, detectedIds[, objPoints[, imgPoints]]) -> objPoints, imgPoints

Given a board configuration and a set of detected markers, returns the correspondingimage points and object points, can be used in solvePnP() 


**See also:** solvePnP


:param self: 
:type self: 
:param detectedCorners: List of detected marker corners of the board.For cv::Board and cv::GridBoard the method expects std::vector<std::vector<Point2f>> or std::vector<Mat> with Aruco marker corners. For cv::CharucoBoard the method expects std::vector<Point2f> or Mat with ChAruco corners (chess board corners matched with Aruco markers). 
:type detectedCorners: _typing.Sequence[cv2.typing.MatLike]
:param detectedIds: List of identifiers for each marker or charuco corner.For any Board class the method expects std::vector<int> or Mat. 
:type detectedIds: cv2.typing.MatLike
:param objPoints: Vector of marker points in the board coordinate space.For any Board class the method expects std::vector<cv::Point3f> objectPoints or cv::Mat 
:type objPoints: cv2.typing.MatLike | None
:param imgPoints: Vector of marker points in the image coordinate space.For any Board class the method expects std::vector<cv::Point2f> objectPoints or cv::Mat 
:type imgPoints: cv2.typing.MatLike | None
:rettype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} matchImagePoints(detectedCorners, detectedIds[, objPoints[, imgPoints]]) -> objPoints, imgPoints

Given a board configuration and a set of detected markers, returns the correspondingimage points and object points, can be used in solvePnP() 


**See also:** solvePnP


:param self: 
:type self: 
:param detectedCorners: List of detected marker corners of the board.For cv::Board and cv::GridBoard the method expects std::vector<std::vector<Point2f>> or std::vector<Mat> with Aruco marker corners. For cv::CharucoBoard the method expects std::vector<Point2f> or Mat with ChAruco corners (chess board corners matched with Aruco markers). 
:type detectedCorners: _typing.Sequence[cv2.UMat]
:param detectedIds: List of identifiers for each marker or charuco corner.For any Board class the method expects std::vector<int> or Mat. 
:type detectedIds: cv2.UMat
:param objPoints: Vector of marker points in the board coordinate space.For any Board class the method expects std::vector<cv::Point3f> objectPoints or cv::Mat 
:type objPoints: cv2.UMat | None
:param imgPoints: Vector of marker points in the image coordinate space.For any Board class the method expects std::vector<cv::Point2f> objectPoints or cv::Mat 
:type imgPoints: cv2.UMat | None
:rettype: tuple[cv2.UMat, cv2.UMat]
````

````{py:method} generateImage(outSize[, img[, marginSize[, borderBits]]]) -> img

Draw a planar board

This function return the image of the board, ready to be printed. 


:param self: 
:type self: 
:param outSize: size of the output image in pixels.
:type outSize: cv2.typing.Size
:param img: output image with the board. The size of this image will be outSizeand the board will be on the center, keeping the board proportions. 
:type img: cv2.typing.MatLike | None
:param marginSize: minimum margins (in pixels) of the board in the output image
:type marginSize: int
:param borderBits: width of the marker borders.
:type borderBits: int
:rettype: cv2.typing.MatLike
````

````{py:method} generateImage(outSize[, img[, marginSize[, borderBits]]]) -> img

Draw a planar board

This function return the image of the board, ready to be printed. 


:param self: 
:type self: 
:param outSize: size of the output image in pixels.
:type outSize: cv2.typing.Size
:param img: output image with the board. The size of this image will be outSizeand the board will be on the center, keeping the board proportions. 
:type img: cv2.UMat | None
:param marginSize: minimum margins (in pixels) of the board in the output image
:type marginSize: int
:param borderBits: width of the marker borders.
:type borderBits: int
:rettype: cv2.UMat
````

````{py:method} getDictionary() -> retval
return the Dictionary of markers employed for this board




:param self: 
:type self: 
:rettype: Dictionary
````

````{py:method} getObjPoints() -> retval
return array of object points of all the marker corners in the board.

Each marker include its 4 corners in this order: -   objPoints[i][0] - left-top point of i-th marker -   objPoints[i][1] - right-top point of i-th marker -   objPoints[i][2] - right-bottom point of i-th marker -   objPoints[i][3] - left-bottom point of i-th marker Markers are placed in a certain order - row by row, left to right in every row. For M markers, the size is Mx4. 


:param self: 
:type self: 
:rettype: _typing.Sequence[_typing.Sequence[cv2.typing.Point3f]]
````

````{py:method} getIds() -> retval
vector of the identifiers of the markers in the board (should be the same size as objPoints)




:param self: 
:type self: 
:return: vector of the identifiers of the markers
:rettype: _typing.Sequence[int]
````

````{py:method} getRightBottomCorner() -> retval
get coordinate of the bottom right corner of the board, is set when calling the function create()




:param self: 
:type self: 
:rettype: cv2.typing.Point3f
````


`````


`````{py:class} CharucoBoard




````{py:method} __init__(self, size: cv2.typing.Size, squareLength: float, markerLength: float, dictionary: Dictionary, ids: cv2.typing.MatLike | None=...)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param squareLength: 
:type squareLength: float
:param markerLength: 
:type markerLength: float
:param dictionary: 
:type dictionary: Dictionary
:param ids: 
:type ids: cv2.typing.MatLike | None
:rettype: None
````

````{py:method} __init__(self, size: cv2.typing.Size, squareLength: float, markerLength: float, dictionary: Dictionary, ids: cv2.UMat | None=...)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param squareLength: 
:type squareLength: float
:param markerLength: 
:type markerLength: float
:param dictionary: 
:type dictionary: Dictionary
:param ids: 
:type ids: cv2.UMat | None
:rettype: None
````

````{py:method} checkCharucoCornersCollinear(charucoIds) -> retval

check whether the ChArUco markers are collinear

The number of ids in charucoIDs should be <= the number of chessboard corners in the board. This functions checks whether the charuco corners are on a straight line (returns true, if so), or not (false). Axis parallel, as well as diagonal and other straight lines detected.  Degenerate cases: for number of charucoIDs <= 2,the function returns true. 


:param self: 
:type self: 
:param charucoIds: list of identifiers for each corner in charucoCorners per frame.
:type charucoIds: cv2.typing.MatLike
:return: bool value, 1 (true) if detected corners form a line, 0 (false) if they do not.solvePnP, calibration functions will fail if the corners are collinear (true). 
:rettype: bool
````

````{py:method} checkCharucoCornersCollinear(charucoIds) -> retval

check whether the ChArUco markers are collinear

The number of ids in charucoIDs should be <= the number of chessboard corners in the board. This functions checks whether the charuco corners are on a straight line (returns true, if so), or not (false). Axis parallel, as well as diagonal and other straight lines detected.  Degenerate cases: for number of charucoIDs <= 2,the function returns true. 


:param self: 
:type self: 
:param charucoIds: list of identifiers for each corner in charucoCorners per frame.
:type charucoIds: cv2.UMat
:return: bool value, 1 (true) if detected corners form a line, 0 (false) if they do not.solvePnP, calibration functions will fail if the corners are collinear (true). 
:rettype: bool
````

````{py:method} setLegacyPattern(legacyPattern) -> None
set legacy chessboard pattern.

Legacy setting creates chessboard patterns starting with a white box in the upper left corner if there is an even row count of chessboard boxes, otherwise it starts with a black box. This setting ensures compatibility to patterns created with OpenCV versions prior OpenCV 4.6.0. See https://github.com/opencv/opencv/issues/23152. Default value: false. 


:param self: 
:type self: 
:param legacyPattern: 
:type legacyPattern: bool
:rettype: None
````

````{py:method} getLegacyPattern() -> retval





:param self: 
:type self: 
:rettype: bool
````

````{py:method} getChessboardSize() -> retval





:param self: 
:type self: 
:rettype: cv2.typing.Size
````

````{py:method} getSquareLength() -> retval





:param self: 
:type self: 
:rettype: float
````

````{py:method} getMarkerLength() -> retval





:param self: 
:type self: 
:rettype: float
````

````{py:method} getChessboardCorners() -> retval
get CharucoBoard::chessboardCorners




:param self: 
:type self: 
:rettype: _typing.Sequence[cv2.typing.Point3f]
````


`````


`````{py:class} CharucoDetector




````{py:method} detectBoard(image[, charucoCorners[, charucoIds[, markerCorners[, markerIds]]]]) -> charucoCorners, charucoIds, markerCorners, markerIds

detect aruco markers and interpolate position of ChArUco board corners

This function receives the detected markers and returns the 2D position of the chessboard corners from a ChArUco board using the detected Aruco markers. If markerCorners and markerCorners are empty, the detectMarkers() will run and detect aruco markers and ids. If camera parameters are provided, the process is based in an approximated pose estimation, else it is based on local homography. Only visible corners are returned. For each corner, its corresponding identifier is also returned in charucoIds. 
**See also:** findChessboardCorners


:param self: 
:type self: 
:param image: input image necesary for corner refinement. Note that markers are not detected andshould be sent in corners and ids parameters. 
:type image: cv2.typing.MatLike
:param charucoCorners: interpolated chessboard corners.
:type charucoCorners: cv2.typing.MatLike | None
:param charucoIds: interpolated chessboard corners identifiers.
:type charucoIds: cv2.typing.MatLike | None
:param markerCorners: vector of already detected markers corners. For each marker, its fourcorners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array should be Nx4. The order of the corners should be clockwise. If markerCorners and markerCorners are empty, the function detect aruco markers and ids. 
:type markerCorners: _typing.Sequence[cv2.typing.MatLike] | None
:param markerIds: list of identifiers for each marker in corners.If markerCorners and markerCorners are empty, the function detect aruco markers and ids. 
:type markerIds: cv2.typing.MatLike | None
:rettype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike]
````

````{py:method} detectBoard(image[, charucoCorners[, charucoIds[, markerCorners[, markerIds]]]]) -> charucoCorners, charucoIds, markerCorners, markerIds

detect aruco markers and interpolate position of ChArUco board corners

This function receives the detected markers and returns the 2D position of the chessboard corners from a ChArUco board using the detected Aruco markers. If markerCorners and markerCorners are empty, the detectMarkers() will run and detect aruco markers and ids. If camera parameters are provided, the process is based in an approximated pose estimation, else it is based on local homography. Only visible corners are returned. For each corner, its corresponding identifier is also returned in charucoIds. 
**See also:** findChessboardCorners


:param self: 
:type self: 
:param image: input image necesary for corner refinement. Note that markers are not detected andshould be sent in corners and ids parameters. 
:type image: cv2.UMat
:param charucoCorners: interpolated chessboard corners.
:type charucoCorners: cv2.UMat | None
:param charucoIds: interpolated chessboard corners identifiers.
:type charucoIds: cv2.UMat | None
:param markerCorners: vector of already detected markers corners. For each marker, its fourcorners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array should be Nx4. The order of the corners should be clockwise. If markerCorners and markerCorners are empty, the function detect aruco markers and ids. 
:type markerCorners: _typing.Sequence[cv2.UMat] | None
:param markerIds: list of identifiers for each marker in corners.If markerCorners and markerCorners are empty, the function detect aruco markers and ids. 
:type markerIds: cv2.UMat | None
:rettype: tuple[cv2.UMat, cv2.UMat, _typing.Sequence[cv2.UMat], cv2.UMat]
````

````{py:method} detectDiamonds(image[, diamondCorners[, diamondIds[, markerCorners[, markerIds]]]]) -> diamondCorners, diamondIds, markerCorners, markerIds

Detect ChArUco Diamond markers

This function detects Diamond markers from the previous detected ArUco markers. The diamonds are returned in the diamondCorners and diamondIds parameters. If camera calibration parameters are provided, the diamond search is based on reprojection. If not, diamond search is based on homography. Homography is faster than reprojection, but less accurate. 


:param self: 
:type self: 
:param image: input image necessary for corner subpixel.
:type image: cv2.typing.MatLike
:param diamondCorners: output list of detected diamond corners (4 corners per diamond). The orderis the same than in marker corners: top left, top right, bottom right and bottom left. Similar format than the corners returned by detectMarkers (e.g std::vector<std::vector<cv::Point2f> > ). 
:type diamondCorners: _typing.Sequence[cv2.typing.MatLike] | None
:param diamondIds: ids of the diamonds in diamondCorners. The id of each diamond is in fact oftype Vec4i, so each diamond has 4 ids, which are the ids of the aruco markers composing the diamond. 
:type diamondIds: cv2.typing.MatLike | None
:param markerCorners: list of detected marker corners from detectMarkers function.If markerCorners and markerCorners are empty, the function detect aruco markers and ids. 
:type markerCorners: _typing.Sequence[cv2.typing.MatLike] | None
:param markerIds: list of marker ids in markerCorners.If markerCorners and markerCorners are empty, the function detect aruco markers and ids. 
:type markerIds: cv2.typing.MatLike | None
:rettype: tuple[_typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike]
````

````{py:method} detectDiamonds(image[, diamondCorners[, diamondIds[, markerCorners[, markerIds]]]]) -> diamondCorners, diamondIds, markerCorners, markerIds

Detect ChArUco Diamond markers

This function detects Diamond markers from the previous detected ArUco markers. The diamonds are returned in the diamondCorners and diamondIds parameters. If camera calibration parameters are provided, the diamond search is based on reprojection. If not, diamond search is based on homography. Homography is faster than reprojection, but less accurate. 


:param self: 
:type self: 
:param image: input image necessary for corner subpixel.
:type image: cv2.UMat
:param diamondCorners: output list of detected diamond corners (4 corners per diamond). The orderis the same than in marker corners: top left, top right, bottom right and bottom left. Similar format than the corners returned by detectMarkers (e.g std::vector<std::vector<cv::Point2f> > ). 
:type diamondCorners: _typing.Sequence[cv2.UMat] | None
:param diamondIds: ids of the diamonds in diamondCorners. The id of each diamond is in fact oftype Vec4i, so each diamond has 4 ids, which are the ids of the aruco markers composing the diamond. 
:type diamondIds: cv2.UMat | None
:param markerCorners: list of detected marker corners from detectMarkers function.If markerCorners and markerCorners are empty, the function detect aruco markers and ids. 
:type markerCorners: _typing.Sequence[cv2.UMat] | None
:param markerIds: list of marker ids in markerCorners.If markerCorners and markerCorners are empty, the function detect aruco markers and ids. 
:type markerIds: cv2.UMat | None
:rettype: tuple[_typing.Sequence[cv2.UMat], cv2.UMat, _typing.Sequence[cv2.UMat], cv2.UMat]
````

````{py:method} __init__(self, board: CharucoBoard, charucoParams: CharucoParameters=..., detectorParams: DetectorParameters=..., refineParams: RefineParameters=...)


Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param board: 
:type board: CharucoBoard
:param charucoParams: 
:type charucoParams: CharucoParameters
:param detectorParams: 
:type detectorParams: DetectorParameters
:param refineParams: 
:type refineParams: RefineParameters
:rettype: None
````

````{py:method} getBoard() -> retval





:param self: 
:type self: 
:rettype: CharucoBoard
````

````{py:method} setBoard(board) -> None





:param self: 
:type self: 
:param board: 
:type board: CharucoBoard
:rettype: None
````

````{py:method} getCharucoParameters() -> retval





:param self: 
:type self: 
:rettype: CharucoParameters
````

````{py:method} setCharucoParameters(charucoParameters) -> None





:param self: 
:type self: 
:param charucoParameters: 
:type charucoParameters: CharucoParameters
:rettype: None
````

````{py:method} getDetectorParameters() -> retval





:param self: 
:type self: 
:rettype: DetectorParameters
````

````{py:method} setDetectorParameters(detectorParameters) -> None





:param self: 
:type self: 
:param detectorParameters: 
:type detectorParameters: DetectorParameters
:rettype: None
````

````{py:method} getRefineParameters() -> retval





:param self: 
:type self: 
:rettype: RefineParameters
````

````{py:method} setRefineParameters(refineParameters) -> None





:param self: 
:type self: 
:param refineParameters: 
:type refineParameters: RefineParameters
:rettype: None
````


`````


`````{py:class} CharucoParameters




````{py:method} __init__(self)


Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rettype: None
````

```{py:attribute} cameraMatrix
:type: cv2.typing.MatLike
```

```{py:attribute} distCoeffs
:type: cv2.typing.MatLike
```

```{py:attribute} minMarkers
:type: int
```

```{py:attribute} tryRefineMarkers
:type: bool
```


`````


`````{py:class} DetectorParameters




````{py:method} __init__(self)


Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rettype: None
````

````{py:method} readDetectorParameters(fn) -> retval
Read a new set of DetectorParameters from FileNode (use FileStorage.root()).




:param self: 
:type self: 
:param fn: 
:type fn: cv2.FileNode
:rettype: bool
````

````{py:method} writeDetectorParameters(fs[, name]) -> retval
Write a set of DetectorParameters to FileStorage




:param self: 
:type self: 
:param fs: 
:type fs: cv2.FileStorage
:param name: 
:type name: str
:rettype: bool
````

```{py:attribute} adaptiveThreshWinSizeMin
:type: int
```

```{py:attribute} adaptiveThreshWinSizeMax
:type: int
```

```{py:attribute} adaptiveThreshWinSizeStep
:type: int
```

```{py:attribute} adaptiveThreshConstant
:type: float
```

```{py:attribute} minMarkerPerimeterRate
:type: float
```

```{py:attribute} maxMarkerPerimeterRate
:type: float
```

```{py:attribute} polygonalApproxAccuracyRate
:type: float
```

```{py:attribute} minCornerDistanceRate
:type: float
```

```{py:attribute} minDistanceToBorder
:type: int
```

```{py:attribute} minMarkerDistanceRate
:type: float
```

```{py:attribute} minGroupDistance
:type: float
```

```{py:attribute} cornerRefinementMethod
:type: int
```

```{py:attribute} cornerRefinementWinSize
:type: int
```

```{py:attribute} relativeCornerRefinmentWinSize
:type: float
```

```{py:attribute} cornerRefinementMaxIterations
:type: int
```

```{py:attribute} cornerRefinementMinAccuracy
:type: float
```

```{py:attribute} markerBorderBits
:type: int
```

```{py:attribute} perspectiveRemovePixelPerCell
:type: int
```

```{py:attribute} perspectiveRemoveIgnoredMarginPerCell
:type: float
```

```{py:attribute} maxErroneousBitsInBorderRate
:type: float
```

```{py:attribute} minOtsuStdDev
:type: float
```

```{py:attribute} errorCorrectionRate
:type: float
```

```{py:attribute} aprilTagQuadDecimate
:type: float
```

```{py:attribute} aprilTagQuadSigma
:type: float
```

```{py:attribute} aprilTagMinClusterPixels
:type: int
```

```{py:attribute} aprilTagMaxNmaxima
:type: int
```

```{py:attribute} aprilTagCriticalRad
:type: float
```

```{py:attribute} aprilTagMaxLineFitMse
:type: float
```

```{py:attribute} aprilTagMinWhiteBlackDiff
:type: int
```

```{py:attribute} aprilTagDeglitch
:type: int
```

```{py:attribute} detectInvertedMarker
:type: bool
```

```{py:attribute} useAruco3Detection
:type: bool
```

```{py:attribute} minSideLengthCanonicalImg
:type: int
```

```{py:attribute} minMarkerLengthRatioOriginalImg
:type: float
```


`````


`````{py:class} Dictionary




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rettype: None
````

````{py:method} __init__(self, bytesList: cv2.typing.MatLike, _markerSize: int, maxcorr: int=...)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param bytesList: 
:type bytesList: cv2.typing.MatLike
:param _markerSize: 
:type _markerSize: int
:param maxcorr: 
:type maxcorr: int
:rettype: None
````

````{py:method} getDistanceToId(bits, id[, allRotations]) -> retval

Returns Hamming distance of the input bits to the specific id.

If `allRotations` flag is set, the four posible marker rotations are considered 


:param self: 
:type self: 
:param bits: 
:type bits: cv2.typing.MatLike
:param id: 
:type id: int
:param allRotations: 
:type allRotations: bool
:rettype: int
````

````{py:method} getDistanceToId(bits, id[, allRotations]) -> retval

Returns Hamming distance of the input bits to the specific id.

If `allRotations` flag is set, the four posible marker rotations are considered 


:param self: 
:type self: 
:param bits: 
:type bits: cv2.UMat
:param id: 
:type id: int
:param allRotations: 
:type allRotations: bool
:rettype: int
````

````{py:method} generateImageMarker(id, sidePixels[, _img[, borderBits]]) -> _img

Generate a canonical marker image




:param self: 
:type self: 
:param id: 
:type id: int
:param sidePixels: 
:type sidePixels: int
:param _img: 
:type _img: cv2.typing.MatLike | None
:param borderBits: 
:type borderBits: int
:rettype: cv2.typing.MatLike
````

````{py:method} generateImageMarker(id, sidePixels[, _img[, borderBits]]) -> _img

Generate a canonical marker image




:param self: 
:type self: 
:param id: 
:type id: int
:param sidePixels: 
:type sidePixels: int
:param _img: 
:type _img: cv2.UMat | None
:param borderBits: 
:type borderBits: int
:rettype: cv2.UMat
````

````{py:method} readDictionary(fn) -> retval
Read a new dictionary from FileNode.

Dictionary example in YAML format:\n nmarkers: 35\n markersize: 6\n maxCorrectionBits: 5\n marker_0: "101011111011111001001001101100000000"\n \n marker_34: "011111010000111011111110110101100101" 


:param self: 
:type self: 
:param fn: 
:type fn: cv2.FileNode
:rettype: bool
````

````{py:method} writeDictionary(fs[, name]) -> None
Write a dictionary to FileStorage, format is the same as in readDictionary().




:param self: 
:type self: 
:param fs: 
:type fs: cv2.FileStorage
:param name: 
:type name: str
:rettype: None
````

````{py:method} identify(onlyBits, maxCorrectionRate) -> retval, idx, rotation
Given a matrix of bits. Returns whether if marker is identified or not.

Returns reference to the marker id in the dictionary (if any) and its rotation. 


:param self: 
:type self: 
:param onlyBits: 
:type onlyBits: cv2.typing.MatLike
:param maxCorrectionRate: 
:type maxCorrectionRate: float
:rettype: tuple[bool, int, int]
````

````{py:method} getByteListFromBits(bits) -> retval
:staticmethod:
Transform matrix of bits to list of bytes with 4 marker rotations




:param bits: 
:type bits: cv2.typing.MatLike
:rettype: cv2.typing.MatLike
````

````{py:method} getBitsFromByteList(byteList, markerSize) -> retval
:staticmethod:
Transform list of bytes to matrix of bits




:param byteList: 
:type byteList: cv2.typing.MatLike
:param markerSize: 
:type markerSize: int
:rettype: cv2.typing.MatLike
````

```{py:attribute} bytesList
:type: cv2.typing.MatLike
```

```{py:attribute} markerSize
:type: int
```

```{py:attribute} maxCorrectionBits
:type: int
```


`````


`````{py:class} GridBoard




````{py:method} __init__(self, size: cv2.typing.Size, markerLength: float, markerSeparation: float, dictionary: Dictionary, ids: cv2.typing.MatLike | None=...)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param markerLength: 
:type markerLength: float
:param markerSeparation: 
:type markerSeparation: float
:param dictionary: 
:type dictionary: Dictionary
:param ids: 
:type ids: cv2.typing.MatLike | None
:rettype: None
````

````{py:method} __init__(self, size: cv2.typing.Size, markerLength: float, markerSeparation: float, dictionary: Dictionary, ids: cv2.UMat | None=...)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param markerLength: 
:type markerLength: float
:param markerSeparation: 
:type markerSeparation: float
:param dictionary: 
:type dictionary: Dictionary
:param ids: 
:type ids: cv2.UMat | None
:rettype: None
````

````{py:method} getGridSize() -> retval





:param self: 
:type self: 
:rettype: cv2.typing.Size
````

````{py:method} getMarkerLength() -> retval





:param self: 
:type self: 
:rettype: float
````

````{py:method} getMarkerSeparation() -> retval





:param self: 
:type self: 
:rettype: float
````


`````


`````{py:class} RefineParameters




````{py:method} __init__(self, minRepDistance: float=..., errorCorrectionRate: float=..., checkAllOrders: bool=...)


Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param minRepDistance: 
:type minRepDistance: float
:param errorCorrectionRate: 
:type errorCorrectionRate: float
:param checkAllOrders: 
:type checkAllOrders: bool
:rettype: None
````

````{py:method} readRefineParameters(fn) -> retval
Read a new set of RefineParameters from FileNode (use FileStorage.root()).




:param self: 
:type self: 
:param fn: 
:type fn: cv2.FileNode
:rettype: bool
````

````{py:method} writeRefineParameters(fs[, name]) -> retval
Write a set of RefineParameters to FileStorage




:param self: 
:type self: 
:param fs: 
:type fs: cv2.FileStorage
:param name: 
:type name: str
:rettype: bool
````

```{py:attribute} minRepDistance
:type: float
```

```{py:attribute} errorCorrectionRate
:type: float
```

```{py:attribute} checkAllOrders
:type: bool
```


`````



## Functions
````{py:function} Dictionary_getBitsFromByteList(byteList, markerSize) -> retval

Transform list of bytes to matrix of bits




:rettype: object
````


````{py:function} Dictionary_getByteListFromBits(bits) -> retval

Transform matrix of bits to list of bytes with 4 marker rotations




:rettype: object
````


````{py:function} drawDetectedCornersCharuco(image, charucoCorners[, charucoIds[, cornerColor]]) -> image

Draws a set of Charuco corners

This function draws a set of detected Charuco corners. If identifiers vector is provided, it also draws the id of each corner. 


:param image: input/output image. It must have 1 or 3 channels. The number of channels is notaltered. 
:type image: cv2.typing.MatLike
:param charucoCorners: vector of detected charuco corners
:type charucoCorners: cv2.typing.MatLike
:param charucoIds: list of identifiers for each corner in charucoCorners
:type charucoIds: cv2.typing.MatLike | None
:param cornerColor: color of the square surrounding each corner
:type cornerColor: cv2.typing.Scalar
:rettype: cv2.typing.MatLike
````


````{py:function} drawDetectedDiamonds(image, diamondCorners[, diamondIds[, borderColor]]) -> image

Draw a set of detected ChArUco Diamond markers

Given an array of detected diamonds, this functions draws them in the image. The marker borders are painted and the markers identifiers if provided. Useful for debugging purposes. 


:param image: input/output image. It must have 1 or 3 channels. The number of channels is notaltered. 
:type image: cv2.typing.MatLike
:param diamondCorners: positions of diamond corners in the same format returned bydetectCharucoDiamond(). (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array should be Nx4. The order of the corners should be clockwise. 
:type diamondCorners: _typing.Sequence[cv2.typing.MatLike]
:param diamondIds: vector of identifiers for diamonds in diamondCorners, in the same formatreturned by detectCharucoDiamond() (e.g. std::vector<Vec4i>). Optional, if not provided, ids are not painted. 
:type diamondIds: cv2.typing.MatLike | None
:param borderColor: color of marker borders. Rest of colors (text color and first corner color)are calculated based on this one. 
:type borderColor: cv2.typing.Scalar
:rettype: cv2.typing.MatLike
````


````{py:function} drawDetectedMarkers(image, corners[, ids[, borderColor]]) -> image

Draw detected markers in image

Given an array of detected marker corners and its corresponding ids, this functions draws the markers in the image. The marker borders are painted and the markers identifiers if provided. Useful for debugging purposes. 


:param image: input/output image. It must have 1 or 3 channels. The number of channels is not altered.
:type image: cv2.typing.MatLike
:param corners: positions of marker corners on input image.(e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array should be Nx4. The order of the corners should be clockwise. 
:type corners: _typing.Sequence[cv2.typing.MatLike]
:param ids: vector of identifiers for markers in markersCorners .Optional, if not provided, ids are not painted. 
:type ids: cv2.typing.MatLike | None
:param borderColor: color of marker borders. Rest of colors (text color and first corner color)are calculated based on this one to improve visualization. 
:type borderColor: cv2.typing.Scalar
:rettype: cv2.typing.MatLike
````


````{py:function} extendDictionary(nMarkers, markerSize[, baseDictionary[, randomSeed]]) -> retval

Extend base dictionary by new nMarkers

This function creates a new dictionary composed by nMarkers markers and each markers composed by markerSize x markerSize bits. If baseDictionary is provided, its markers are directly included and the rest are generated based on them. If the size of baseDictionary is higher than nMarkers, only the first nMarkers in baseDictionary are taken and no new marker is added. 


:param nMarkers: number of markers in the dictionary
:type nMarkers: int
:param markerSize: number of bits per dimension of each markers
:type markerSize: int
:param baseDictionary: Include the markers in this dictionary at the beginning (optional)
:type baseDictionary: Dictionary
:param randomSeed: a user supplied seed for theRNG()
:type randomSeed: int
:rettype: Dictionary
````


````{py:function} generateImageMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img

Generate a canonical marker image

This function returns a marker image in its canonical form (i.e. ready to be printed) 


:param dictionary: dictionary of markers indicating the type of markers
:type dictionary: Dictionary
:param id: identifier of the marker that will be returned. It has to be a valid id in the specified dictionary.
:type id: int
:param sidePixels: size of the image in pixels
:type sidePixels: int
:param img: output image with the marker
:type img: cv2.typing.MatLike | None
:param borderBits: width of the marker border.
:type borderBits: int
:rettype: cv2.typing.MatLike
````


````{py:function} getPredefinedDictionary(dict) -> retval

Returns one of the predefined dictionaries referenced by DICT_*.




:param dict: 
:type dict: int
:rettype: Dictionary
````



