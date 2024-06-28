# `cv2`
```{py:module} cv2

OpenCV Python binary extension loader

```
## Classes
`````{py:class} AKAZE




````{py:method} create([, descriptor_type[, descriptor_size[, descriptor_channels[, threshold[, nOctaves[, nOctaveLayers[, diffusivity[, max_points]]]]]]]]) -> retval
:classmethod:
The AKAZE constructor




:param cls: 
:type cls: 
:param descriptor_type: Type of the extracted descriptor: DESCRIPTOR_KAZE,DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT. 
:type descriptor_type: AKAZE_DescriptorType
:param descriptor_size: Size of the descriptor in bits. 0 -\> Full size
:type descriptor_size: int
:param descriptor_channels: Number of channels in the descriptor (1, 2, 3)
:type descriptor_channels: int
:param threshold: Detector response threshold to accept point
:type threshold: float
:param nOctaves: Maximum octave evolution of the image
:type nOctaves: int
:param nOctaveLayers: Default number of sublevels per scale level
:type nOctaveLayers: int
:param diffusivity: Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT orDIFF_CHARBONNIER 
:type diffusivity: KAZE_DiffusivityType
:param max_points: Maximum amount of returned points. In case if image containsmore features, then the features with highest response are returned. Negative value means no limitation. 
:type max_points: int
:rtype: AKAZE
````

````{py:method} setDescriptorType(dtype) -> None





:param self: 
:type self: 
:param dtype: 
:type dtype: AKAZE_DescriptorType
:rtype: None
````

````{py:method} getDescriptorType() -> retval





:param self: 
:type self: 
:rtype: AKAZE_DescriptorType
````

````{py:method} setDescriptorSize(dsize) -> None





:param self: 
:type self: 
:param dsize: 
:type dsize: int
:rtype: None
````

````{py:method} getDescriptorSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setDescriptorChannels(dch) -> None





:param self: 
:type self: 
:param dch: 
:type dch: int
:rtype: None
````

````{py:method} getDescriptorChannels() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setThreshold(threshold) -> None





:param self: 
:type self: 
:param threshold: 
:type threshold: float
:rtype: None
````

````{py:method} getThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setNOctaves(octaves) -> None





:param self: 
:type self: 
:param octaves: 
:type octaves: int
:rtype: None
````

````{py:method} getNOctaves() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setNOctaveLayers(octaveLayers) -> None





:param self: 
:type self: 
:param octaveLayers: 
:type octaveLayers: int
:rtype: None
````

````{py:method} getNOctaveLayers() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setDiffusivity(diff) -> None





:param self: 
:type self: 
:param diff: 
:type diff: KAZE_DiffusivityType
:rtype: None
````

````{py:method} getDiffusivity() -> retval





:param self: 
:type self: 
:rtype: KAZE_DiffusivityType
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````

````{py:method} setMaxPoints(max_points) -> None





:param self: 
:type self: 
:param max_points: 
:type max_points: int
:rtype: None
````

````{py:method} getMaxPoints() -> retval





:param self: 
:type self: 
:rtype: int
````


`````


`````{py:class} AffineFeature




````{py:method} create(backend[, maxTilt[, minTilt[, tiltStep[, rotateStepBase]]]]) -> retval
:classmethod:





:param cls: 
:type cls: 
:param backend: The detector/extractor you want to use as backend.
:type backend: Feature2D
:param maxTilt: The highest power index of tilt factor. 5 is used in the paper as tilt sampling range n.
:type maxTilt: int
:param minTilt: The lowest power index of tilt factor. 0 is used in the paper.
:type minTilt: int
:param tiltStep: Tilt sampling step $\delta_t$ in Algorithm 1 in the paper.
:type tiltStep: float
:param rotateStepBase: Rotation sampling step factor b in Algorithm 1 in the paper.
:type rotateStepBase: float
:rtype: AffineFeature
````

````{py:method} setViewParams(tilts, rolls) -> None





:param self: 
:type self: 
:param tilts: 
:type tilts: _typing.Sequence[float]
:param rolls: 
:type rolls: _typing.Sequence[float]
:rtype: None
````

````{py:method} getViewParams(tilts, rolls) -> None





:param self: 
:type self: 
:param tilts: 
:type tilts: _typing.Sequence[float]
:param rolls: 
:type rolls: _typing.Sequence[float]
:rtype: None
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} AgastFeatureDetector




````{py:method} create([, threshold[, nonmaxSuppression[, type]]]) -> retval
:classmethod:





:param cls: 
:type cls: 
:param threshold: 
:type threshold: int
:param nonmaxSuppression: 
:type nonmaxSuppression: bool
:param type: 
:type type: AgastFeatureDetector_DetectorType
:rtype: AgastFeatureDetector
````

````{py:method} setThreshold(threshold) -> None





:param self: 
:type self: 
:param threshold: 
:type threshold: int
:rtype: None
````

````{py:method} getThreshold() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setNonmaxSuppression(f) -> None





:param self: 
:type self: 
:param f: 
:type f: bool
:rtype: None
````

````{py:method} getNonmaxSuppression() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} setType(type) -> None





:param self: 
:type self: 
:param type: 
:type type: AgastFeatureDetector_DetectorType
:rtype: None
````

````{py:method} getType() -> retval





:param self: 
:type self: 
:rtype: AgastFeatureDetector_DetectorType
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} Algorithm




````{py:method} write(fs) -> None

Stores algorithm parameters in a file storage


write(fs, name) -> None @overload 


:param self: 
:type self: 
:param fs: 
:type fs: FileStorage
:rtype: None
````

````{py:method} write(fs) -> None

Stores algorithm parameters in a file storage


write(fs, name) -> None @overload 


:param self: 
:type self: 
:param fs: 
:type fs: FileStorage
:param name: 
:type name: str
:rtype: None
````

````{py:method} clear() -> None
Clears the algorithm state




:param self: 
:type self: 
:rtype: None
````

````{py:method} read(fn) -> None
Reads algorithm parameters from a file storage




:param self: 
:type self: 
:param fn: 
:type fn: FileNode
:rtype: None
````

````{py:method} empty() -> retval
Returns true if the Algorithm is empty (e.g. in the very beginning or after unsuccessful read




:param self: 
:type self: 
:rtype: bool
````

````{py:method} save(filename) -> None



Saves the algorithm to a file. In order to make this method work, the derived class must implement Algorithm::write(FileStorage& fs). 


:param self: 
:type self: 
:param filename: 
:type filename: str
:rtype: None
````

````{py:method} getDefaultName() -> retval



Returns the algorithm string identifier. This string is used as top level xml/yml node tag when the object is saved to a file or string. 


:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} AlignExposures




````{py:method} process(src, dst, times, response) -> None

Aligns images




:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[cv2.typing.MatLike]
:param dst: vector of aligned images
:type dst: _typing.Sequence[cv2.typing.MatLike]
:param times: vector of exposure time values for each image
:type times: cv2.typing.MatLike
:param response: 256x1 matrix with inverse camera response function for each pixel value, it shouldhave the same number of channels as images. 
:type response: cv2.typing.MatLike
:rtype: None
````

````{py:method} process(src, dst, times, response) -> None

Aligns images




:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[UMat]
:param dst: vector of aligned images
:type dst: _typing.Sequence[cv2.typing.MatLike]
:param times: vector of exposure time values for each image
:type times: UMat
:param response: 256x1 matrix with inverse camera response function for each pixel value, it shouldhave the same number of channels as images. 
:type response: UMat
:rtype: None
````


`````


`````{py:class} AlignMTB




````{py:method} process(src, dst, times, response) -> None

Short version of process, that doesn't take extra arguments.


process(src, dst) -> None 


:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[cv2.typing.MatLike]
:param dst: vector of aligned images
:type dst: _typing.Sequence[cv2.typing.MatLike]
:param times: 
:type times: cv2.typing.MatLike
:param response: 
:type response: cv2.typing.MatLike
:rtype: None
````

````{py:method} process(src, dst, times, response) -> None

Short version of process, that doesn't take extra arguments.


process(src, dst) -> None 


:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[UMat]
:param dst: vector of aligned images
:type dst: _typing.Sequence[cv2.typing.MatLike]
:param times: 
:type times: UMat
:param response: 
:type response: UMat
:rtype: None
````

````{py:method} process(src, dst, times, response) -> None

Short version of process, that doesn't take extra arguments.


process(src, dst) -> None 


:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[cv2.typing.MatLike]
:param dst: vector of aligned images
:type dst: _typing.Sequence[cv2.typing.MatLike]
:rtype: None
````

````{py:method} process(src, dst, times, response) -> None

Short version of process, that doesn't take extra arguments.


process(src, dst) -> None 


:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[UMat]
:param dst: vector of aligned images
:type dst: _typing.Sequence[cv2.typing.MatLike]
:rtype: None
````

````{py:method} calculateShift(img0, img1) -> retval

Calculates shift between two images, i. e. how to shift the second image to correspond it with thefirst. 




:param self: 
:type self: 
:param img0: first image
:type img0: cv2.typing.MatLike
:param img1: second image
:type img1: cv2.typing.MatLike
:rtype: cv2.typing.Point
````

````{py:method} calculateShift(img0, img1) -> retval

Calculates shift between two images, i. e. how to shift the second image to correspond it with thefirst. 




:param self: 
:type self: 
:param img0: first image
:type img0: UMat
:param img1: second image
:type img1: UMat
:rtype: cv2.typing.Point
````

````{py:method} shiftMat(src, shift[, dst]) -> dst

Helper function, that shift Mat filling new regions with zeros.




:param self: 
:type self: 
:param src: input image
:type src: cv2.typing.MatLike
:param shift: shift value
:type shift: cv2.typing.Point
:param dst: result image
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} shiftMat(src, shift[, dst]) -> dst

Helper function, that shift Mat filling new regions with zeros.




:param self: 
:type self: 
:param src: input image
:type src: UMat
:param shift: shift value
:type shift: cv2.typing.Point
:param dst: result image
:type dst: UMat | None
:rtype: UMat
````

````{py:method} computeBitmaps(img[, tb[, eb]]) -> tb, eb

Computes median threshold and exclude bitmaps of given image.




:param self: 
:type self: 
:param img: input image
:type img: cv2.typing.MatLike
:param tb: median threshold bitmap
:type tb: cv2.typing.MatLike | None
:param eb: exclude bitmap
:type eb: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} computeBitmaps(img[, tb[, eb]]) -> tb, eb

Computes median threshold and exclude bitmaps of given image.




:param self: 
:type self: 
:param img: input image
:type img: UMat
:param tb: median threshold bitmap
:type tb: UMat | None
:param eb: exclude bitmap
:type eb: UMat | None
:rtype: tuple[UMat, UMat]
````

````{py:method} getMaxBits() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMaxBits(max_bits) -> None





:param self: 
:type self: 
:param max_bits: 
:type max_bits: int
:rtype: None
````

````{py:method} getExcludeRange() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setExcludeRange(exclude_range) -> None





:param self: 
:type self: 
:param exclude_range: 
:type exclude_range: int
:rtype: None
````

````{py:method} getCut() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} setCut(value) -> None





:param self: 
:type self: 
:param value: 
:type value: bool
:rtype: None
````


`````


`````{py:class} AsyncArray




````{py:method} get([, dst]) -> dst




Fetch the result. 
Waits for result until container has valid result. Throws exception if exception was stored as a result. 
Throws exception on invalid container state. 
get(timeoutNs[, dst]) -> retval, dst Retrieving the result with timeout 
```{note}
Result or stored exception can be fetched only once.
```
```{note}
Result or stored exception can be fetched only once.
```


:param self: 
:type self: 
:param dst: [out] destination array
:type dst: cv2.typing.MatLike | None
:param timeoutNs: [in] timeout in nanoseconds, -1 for infinite wait
:type timeoutNs: 
:return: true if result is ready, false if the timeout has expired
:rtype: cv2.typing.MatLike
````

````{py:method} get([, dst]) -> dst




Fetch the result. 
Waits for result until container has valid result. Throws exception if exception was stored as a result. 
Throws exception on invalid container state. 
get(timeoutNs[, dst]) -> retval, dst Retrieving the result with timeout 
```{note}
Result or stored exception can be fetched only once.
```
```{note}
Result or stored exception can be fetched only once.
```


:param self: 
:type self: 
:param dst: [out] destination array
:type dst: UMat | None
:param timeoutNs: [in] timeout in nanoseconds, -1 for infinite wait
:type timeoutNs: 
:return: true if result is ready, false if the timeout has expired
:rtype: UMat
````

````{py:method} get([, dst]) -> dst




Fetch the result. 
Waits for result until container has valid result. Throws exception if exception was stored as a result. 
Throws exception on invalid container state. 
get(timeoutNs[, dst]) -> retval, dst Retrieving the result with timeout 
```{note}
Result or stored exception can be fetched only once.
```
```{note}
Result or stored exception can be fetched only once.
```


:param self: 
:type self: 
:param timeoutNs: [in] timeout in nanoseconds, -1 for infinite wait
:type timeoutNs: float
:param dst: [out] destination array
:type dst: cv2.typing.MatLike | None
:return: true if result is ready, false if the timeout has expired
:rtype: tuple[bool, cv2.typing.MatLike]
````

````{py:method} get([, dst]) -> dst




Fetch the result. 
Waits for result until container has valid result. Throws exception if exception was stored as a result. 
Throws exception on invalid container state. 
get(timeoutNs[, dst]) -> retval, dst Retrieving the result with timeout 
```{note}
Result or stored exception can be fetched only once.
```
```{note}
Result or stored exception can be fetched only once.
```


:param self: 
:type self: 
:param timeoutNs: [in] timeout in nanoseconds, -1 for infinite wait
:type timeoutNs: float
:param dst: [out] destination array
:type dst: UMat | None
:return: true if result is ready, false if the timeout has expired
:rtype: tuple[bool, UMat]
````

````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} release() -> None





:param self: 
:type self: 
:rtype: None
````

````{py:method} wait_for(timeoutNs) -> retval





:param self: 
:type self: 
:param timeoutNs: 
:type timeoutNs: float
:rtype: bool
````

````{py:method} valid() -> retval





:param self: 
:type self: 
:rtype: bool
````


`````


`````{py:class} BFMatcher




````{py:method} create([, normType[, crossCheck]]) -> retval
:classmethod:
Brute-force matcher create method.




:param cls: 
:type cls: 
:param normType: One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms arepreferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description). 
:type normType: int
:param crossCheck: If it is false, this is will be default BFMatcher behaviour when it finds the knearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper. 
:type crossCheck: bool
:rtype: BFMatcher
````

````{py:method} __init__(self, normType: int=..., crossCheck: bool=...)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param normType: 
:type normType: int
:param crossCheck: 
:type crossCheck: bool
:rtype: None
````


`````


`````{py:class} BOWImgDescriptorExtractor




````{py:method} __init__(self, dextractor: Feature2D, dmatcher: DescriptorMatcher)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param dextractor: 
:type dextractor: Feature2D
:param dmatcher: 
:type dmatcher: DescriptorMatcher
:rtype: None
````

````{py:method} setVocabulary(vocabulary) -> None
Sets a visual vocabulary.




:param self: 
:type self: 
:param vocabulary: Vocabulary (can be trained using the inheritor of BOWTrainer ). Each row of thevocabulary is a visual word (cluster center). 
:type vocabulary: cv2.typing.MatLike
:rtype: None
````

````{py:method} getVocabulary() -> retval
Returns the set vocabulary.




:param self: 
:type self: 
:rtype: cv2.typing.MatLike
````

````{py:method} compute(image, keypoints[, imgDescriptor]) -> imgDescriptor



@overload 


:param self: 
:type self: 
:param image: 
:type image: cv2.typing.MatLike
:param keypoints: 
:type keypoints: _typing.Sequence[KeyPoint]
:param imgDescriptor: Computed output image descriptor.
:type imgDescriptor: cv2.typing.MatLike | None
:param keypointDescriptors: Computed descriptors to match with vocabulary.
:type keypointDescriptors: 
:param pointIdxsOfClusters: Indices of keypoints that belong to the cluster. This means thatpointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary) returned if it is non-zero. 
:type pointIdxsOfClusters: 
:rtype: cv2.typing.MatLike
````

````{py:method} descriptorSize() -> retval
Returns an image descriptor size if the vocabulary is set. Otherwise, it returns 0.




:param self: 
:type self: 
:rtype: int
````

````{py:method} descriptorType() -> retval
Returns an image descriptor type.




:param self: 
:type self: 
:rtype: int
````


`````


`````{py:class} BOWKMeansTrainer




````{py:method} cluster() -> retval




cluster(descriptors) -> retval 


:param self: 
:type self: 
:rtype: cv2.typing.MatLike
````

````{py:method} cluster() -> retval




cluster(descriptors) -> retval 


:param self: 
:type self: 
:param descriptors: 
:type descriptors: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````

````{py:method} __init__(self, clusterCount: int, termcrit: cv2.typing.TermCriteria=..., attempts: int=..., flags: int=...)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param clusterCount: 
:type clusterCount: int
:param termcrit: 
:type termcrit: cv2.typing.TermCriteria
:param attempts: 
:type attempts: int
:param flags: 
:type flags: int
:rtype: None
````


`````


`````{py:class} BOWTrainer




````{py:method} cluster() -> retval

Clusters train descriptors.


@overload 
cluster(descriptors) -> retval 
The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first variant of the method, train descriptors stored in the object are clustered. In the second variant, input descriptors are clustered. 


:param self: 
:type self: 
:param descriptors: Descriptors to cluster. Each row of the descriptors matrix is a descriptor.Descriptors are not added to the inner train descriptor set. 
:type descriptors: 
:rtype: cv2.typing.MatLike
````

````{py:method} cluster() -> retval

Clusters train descriptors.


@overload 
cluster(descriptors) -> retval 
The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first variant of the method, train descriptors stored in the object are clustered. In the second variant, input descriptors are clustered. 


:param self: 
:type self: 
:param descriptors: Descriptors to cluster. Each row of the descriptors matrix is a descriptor.Descriptors are not added to the inner train descriptor set. 
:type descriptors: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````

````{py:method} add(descriptors) -> None
Adds descriptors to a training set.


The training set is clustered using clustermethod to construct the vocabulary. 


:param self: 
:type self: 
:param descriptors: Descriptors to add to a training set. Each row of the descriptors matrix is adescriptor. 
:type descriptors: cv2.typing.MatLike
:rtype: None
````

````{py:method} getDescriptors() -> retval
Returns a training set of descriptors.




:param self: 
:type self: 
:rtype: _typing.Sequence[cv2.typing.MatLike]
````

````{py:method} descriptorsCount() -> retval
Returns the count of all descriptors stored in the training set.




:param self: 
:type self: 
:rtype: int
````

````{py:method} clear() -> None





:param self: 
:type self: 
:rtype: None
````


`````


`````{py:class} BRISK




````{py:method} create([, thresh[, octaves[, patternScale]]]) -> retval
:classmethod:
The BRISK constructor for a custom pattern, detection threshold and octaves


create(radiusList, numberList[, dMax[, dMin[, indexChange]]]) -> retval 
create(thresh, octaves, radiusList, numberList[, dMax[, dMin[, indexChange]]]) -> retval 


:param cls: 
:type cls: 
:param thresh: AGAST detection threshold score.
:type thresh: int
:param octaves: detection octaves. Use 0 to do single scale.
:type octaves: int
:param patternScale: apply this scale to the pattern used for sampling the neighbourhood of akeypoint. 
:type patternScale: float
:param radiusList: defines the radii (in pixels) where the samples around a keypoint are taken (forkeypoint scale 1). 
:type radiusList: 
:param numberList: defines the number of sampling points on the sampling circle. Must be the samesize as radiusList.. 
:type numberList: 
:param dMax: threshold for the short pairings used for descriptor formation (in pixels for keypointscale 1). 
:type dMax: 
:param dMin: threshold for the long pairings used for orientation determination (in pixels forkeypoint scale 1). 
:type dMin: 
:param indexChange: index remapping of the bits.
:type indexChange: 
:rtype: BRISK
````

````{py:method} create([, thresh[, octaves[, patternScale]]]) -> retval
:classmethod:
The BRISK constructor for a custom pattern, detection threshold and octaves


create(radiusList, numberList[, dMax[, dMin[, indexChange]]]) -> retval 
create(thresh, octaves, radiusList, numberList[, dMax[, dMin[, indexChange]]]) -> retval 


:param cls: 
:type cls: 
:param radiusList: defines the radii (in pixels) where the samples around a keypoint are taken (forkeypoint scale 1). 
:type radiusList: _typing.Sequence[float]
:param numberList: defines the number of sampling points on the sampling circle. Must be the samesize as radiusList.. 
:type numberList: _typing.Sequence[int]
:param dMax: threshold for the short pairings used for descriptor formation (in pixels for keypointscale 1). 
:type dMax: float
:param dMin: threshold for the long pairings used for orientation determination (in pixels forkeypoint scale 1). 
:type dMin: float
:param indexChange: index remapping of the bits.
:type indexChange: _typing.Sequence[int]
:param thresh: AGAST detection threshold score.
:type thresh: 
:param octaves: detection octaves. Use 0 to do single scale.
:type octaves: 
:param patternScale: apply this scale to the pattern used for sampling the neighbourhood of akeypoint. 
:type patternScale: 
:rtype: BRISK
````

````{py:method} create([, thresh[, octaves[, patternScale]]]) -> retval
:classmethod:
The BRISK constructor for a custom pattern, detection threshold and octaves


create(radiusList, numberList[, dMax[, dMin[, indexChange]]]) -> retval 
create(thresh, octaves, radiusList, numberList[, dMax[, dMin[, indexChange]]]) -> retval 


:param cls: 
:type cls: 
:param thresh: AGAST detection threshold score.
:type thresh: int
:param octaves: detection octaves. Use 0 to do single scale.
:type octaves: int
:param radiusList: defines the radii (in pixels) where the samples around a keypoint are taken (forkeypoint scale 1). 
:type radiusList: _typing.Sequence[float]
:param numberList: defines the number of sampling points on the sampling circle. Must be the samesize as radiusList.. 
:type numberList: _typing.Sequence[int]
:param dMax: threshold for the short pairings used for descriptor formation (in pixels for keypointscale 1). 
:type dMax: float
:param dMin: threshold for the long pairings used for orientation determination (in pixels forkeypoint scale 1). 
:type dMin: float
:param indexChange: index remapping of the bits.
:type indexChange: _typing.Sequence[int]
:param patternScale: apply this scale to the pattern used for sampling the neighbourhood of akeypoint. 
:type patternScale: 
:rtype: BRISK
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````

````{py:method} setThreshold(threshold) -> None
Set detection threshold.




:param self: 
:type self: 
:param threshold: AGAST detection threshold score.
:type threshold: int
:rtype: None
````

````{py:method} getThreshold() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setOctaves(octaves) -> None
Set detection octaves.




:param self: 
:type self: 
:param octaves: detection octaves. Use 0 to do single scale.
:type octaves: int
:rtype: None
````

````{py:method} getOctaves() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setPatternScale(patternScale) -> None
Set detection patternScale.




:param self: 
:type self: 
:param patternScale: apply this scale to the pattern used for sampling the neighbourhood of akeypoint. 
:type patternScale: float
:rtype: None
````

````{py:method} getPatternScale() -> retval





:param self: 
:type self: 
:rtype: float
````


`````


`````{py:class} BackgroundSubtractor




````{py:method} apply(image[, fgmask[, learningRate]]) -> fgmask

Computes a foreground mask.




:param self: 
:type self: 
:param image: Next video frame.
:type image: cv2.typing.MatLike
:param fgmask: The output foreground mask as an 8-bit binary image.
:type fgmask: cv2.typing.MatLike | None
:param learningRate: The value between 0 and 1 that indicates how fast the background model islearnt. Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame. 
:type learningRate: float
:rtype: cv2.typing.MatLike
````

````{py:method} apply(image[, fgmask[, learningRate]]) -> fgmask

Computes a foreground mask.




:param self: 
:type self: 
:param image: Next video frame.
:type image: UMat
:param fgmask: The output foreground mask as an 8-bit binary image.
:type fgmask: UMat | None
:param learningRate: The value between 0 and 1 that indicates how fast the background model islearnt. Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame. 
:type learningRate: float
:rtype: UMat
````

````{py:method} getBackgroundImage([, backgroundImage]) -> backgroundImage

Computes a background image.


```{note}
Sometimes the background image can be very blurry, as it contain the average backgroundstatistics. 
```


:param self: 
:type self: 
:param backgroundImage: The output background image.
:type backgroundImage: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} getBackgroundImage([, backgroundImage]) -> backgroundImage

Computes a background image.


```{note}
Sometimes the background image can be very blurry, as it contain the average backgroundstatistics. 
```


:param self: 
:type self: 
:param backgroundImage: The output background image.
:type backgroundImage: UMat | None
:rtype: UMat
````


`````


`````{py:class} BackgroundSubtractorKNN




````{py:method} getHistory() -> retval
Returns the number of last frames that affect the background model




:param self: 
:type self: 
:rtype: int
````

````{py:method} setHistory(history) -> None
Sets the number of last frames that affect the background model




:param self: 
:type self: 
:param history: 
:type history: int
:rtype: None
````

````{py:method} getNSamples() -> retval
Returns the number of data samples in the background model




:param self: 
:type self: 
:rtype: int
````

````{py:method} setNSamples(_nN) -> None
Sets the number of data samples in the background model.


The model needs to be reinitalized to reserve memory. 


:param self: 
:type self: 
:param _nN: 
:type _nN: int
:rtype: None
````

````{py:method} getDist2Threshold() -> retval
Returns the threshold on the squared distance between the pixel and the sample


The threshold on the squared distance between the pixel and the sample to decide whether a pixel is close to a data sample. 


:param self: 
:type self: 
:rtype: float
````

````{py:method} setDist2Threshold(_dist2Threshold) -> None
Sets the threshold on the squared distance




:param self: 
:type self: 
:param _dist2Threshold: 
:type _dist2Threshold: float
:rtype: None
````

````{py:method} getkNNSamples() -> retval
Returns the number of neighbours, the k in the kNN.


K is the number of samples that need to be within dist2Threshold in order to decide that that pixel is matching the kNN background model. 


:param self: 
:type self: 
:rtype: int
````

````{py:method} setkNNSamples(_nkNN) -> None
Sets the k in the kNN. How many nearest neighbours need to match.




:param self: 
:type self: 
:param _nkNN: 
:type _nkNN: int
:rtype: None
````

````{py:method} getDetectShadows() -> retval
Returns the shadow detection flag


If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorKNN for details. 


:param self: 
:type self: 
:rtype: bool
````

````{py:method} setDetectShadows(detectShadows) -> None
Enables or disables shadow detection




:param self: 
:type self: 
:param detectShadows: 
:type detectShadows: bool
:rtype: None
````

````{py:method} getShadowValue() -> retval
Returns the shadow value


Shadow value is the value used to mark shadows in the foreground mask. Default value is 127. Value 0 in the mask always means background, 255 means foreground. 


:param self: 
:type self: 
:rtype: int
````

````{py:method} setShadowValue(value) -> None
Sets the shadow value




:param self: 
:type self: 
:param value: 
:type value: int
:rtype: None
````

````{py:method} getShadowThreshold() -> retval
Returns the shadow threshold


A shadow is detected if pixel is a darker version of the background. The shadow threshold (Tau in the paper) is a threshold defining how much darker the shadow can be. Tau= 0.5 means that if a pixel is more than twice darker then it is not shadow. See Prati, Mikic, Trivedi and Cucchiara, Detecting Moving Shadows...*, IEEE PAMI,2003. 


:param self: 
:type self: 
:rtype: float
````

````{py:method} setShadowThreshold(threshold) -> None
Sets the shadow threshold




:param self: 
:type self: 
:param threshold: 
:type threshold: float
:rtype: None
````


`````


`````{py:class} BackgroundSubtractorMOG2




````{py:method} apply(image[, fgmask[, learningRate]]) -> fgmask

Computes a foreground mask.




:param self: 
:type self: 
:param image: Next video frame. Floating point frame will be used without scaling and should be in range $[0,255]$.
:type image: cv2.typing.MatLike
:param fgmask: The output foreground mask as an 8-bit binary image.
:type fgmask: cv2.typing.MatLike | None
:param learningRate: The value between 0 and 1 that indicates how fast the background model islearnt. Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame. 
:type learningRate: float
:rtype: cv2.typing.MatLike
````

````{py:method} apply(image[, fgmask[, learningRate]]) -> fgmask

Computes a foreground mask.




:param self: 
:type self: 
:param image: Next video frame. Floating point frame will be used without scaling and should be in range $[0,255]$.
:type image: UMat
:param fgmask: The output foreground mask as an 8-bit binary image.
:type fgmask: UMat | None
:param learningRate: The value between 0 and 1 that indicates how fast the background model islearnt. Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame. 
:type learningRate: float
:rtype: UMat
````

````{py:method} getHistory() -> retval
Returns the number of last frames that affect the background model




:param self: 
:type self: 
:rtype: int
````

````{py:method} setHistory(history) -> None
Sets the number of last frames that affect the background model




:param self: 
:type self: 
:param history: 
:type history: int
:rtype: None
````

````{py:method} getNMixtures() -> retval
Returns the number of gaussian components in the background model




:param self: 
:type self: 
:rtype: int
````

````{py:method} setNMixtures(nmixtures) -> None
Sets the number of gaussian components in the background model.


The model needs to be reinitalized to reserve memory. 


:param self: 
:type self: 
:param nmixtures: 
:type nmixtures: int
:rtype: None
````

````{py:method} getBackgroundRatio() -> retval
Returns the "background ratio" parameter of the algorithm


If a foreground pixel keeps semi-constant value for about backgroundRatio\*history frames, it's considered background and added to the model as a center of a new component. It corresponds to TB parameter in the paper. 


:param self: 
:type self: 
:rtype: float
````

````{py:method} setBackgroundRatio(ratio) -> None
Sets the "background ratio" parameter of the algorithm




:param self: 
:type self: 
:param ratio: 
:type ratio: float
:rtype: None
````

````{py:method} getVarThreshold() -> retval
Returns the variance threshold for the pixel-model match


The main threshold on the squared Mahalanobis distance to decide if the sample is well described by the background model or not. Related to Cthr from the paper. 


:param self: 
:type self: 
:rtype: float
````

````{py:method} setVarThreshold(varThreshold) -> None
Sets the variance threshold for the pixel-model match




:param self: 
:type self: 
:param varThreshold: 
:type varThreshold: float
:rtype: None
````

````{py:method} getVarThresholdGen() -> retval
Returns the variance threshold for the pixel-model match used for new mixture component generation


Threshold for the squared Mahalanobis distance that helps decide when a sample is close to the existing components (corresponds to Tg in the paper). If a pixel is not close to any component, it is considered foreground or added as a new component. 3 sigma =\> Tg=3\*3=9 is default. A smaller Tg value generates more components. A higher Tg value may result in a small number of components but they can grow too large. 


:param self: 
:type self: 
:rtype: float
````

````{py:method} setVarThresholdGen(varThresholdGen) -> None
Sets the variance threshold for the pixel-model match used for new mixture component generation




:param self: 
:type self: 
:param varThresholdGen: 
:type varThresholdGen: float
:rtype: None
````

````{py:method} getVarInit() -> retval
Returns the initial variance of each gaussian component




:param self: 
:type self: 
:rtype: float
````

````{py:method} setVarInit(varInit) -> None
Sets the initial variance of each gaussian component




:param self: 
:type self: 
:param varInit: 
:type varInit: float
:rtype: None
````

````{py:method} getVarMin() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setVarMin(varMin) -> None





:param self: 
:type self: 
:param varMin: 
:type varMin: float
:rtype: None
````

````{py:method} getVarMax() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setVarMax(varMax) -> None





:param self: 
:type self: 
:param varMax: 
:type varMax: float
:rtype: None
````

````{py:method} getComplexityReductionThreshold() -> retval
Returns the complexity reduction threshold


This parameter defines the number of samples needed to accept to prove the component exists. CT=0.05 is a default value for all the samples. By setting CT=0 you get an algorithm very similar to the standard Stauffer&Grimson algorithm. 


:param self: 
:type self: 
:rtype: float
````

````{py:method} setComplexityReductionThreshold(ct) -> None
Sets the complexity reduction threshold




:param self: 
:type self: 
:param ct: 
:type ct: float
:rtype: None
````

````{py:method} getDetectShadows() -> retval
Returns the shadow detection flag


If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorMOG2 for details. 


:param self: 
:type self: 
:rtype: bool
````

````{py:method} setDetectShadows(detectShadows) -> None
Enables or disables shadow detection




:param self: 
:type self: 
:param detectShadows: 
:type detectShadows: bool
:rtype: None
````

````{py:method} getShadowValue() -> retval
Returns the shadow value


Shadow value is the value used to mark shadows in the foreground mask. Default value is 127. Value 0 in the mask always means background, 255 means foreground. 


:param self: 
:type self: 
:rtype: int
````

````{py:method} setShadowValue(value) -> None
Sets the shadow value




:param self: 
:type self: 
:param value: 
:type value: int
:rtype: None
````

````{py:method} getShadowThreshold() -> retval
Returns the shadow threshold


A shadow is detected if pixel is a darker version of the background. The shadow threshold (Tau in the paper) is a threshold defining how much darker the shadow can be. Tau= 0.5 means that if a pixel is more than twice darker then it is not shadow. See Prati, Mikic, Trivedi and Cucchiara, Detecting Moving Shadows...*, IEEE PAMI,2003. 


:param self: 
:type self: 
:rtype: float
````

````{py:method} setShadowThreshold(threshold) -> None
Sets the shadow threshold




:param self: 
:type self: 
:param threshold: 
:type threshold: float
:rtype: None
````


`````


`````{py:class} BaseCascadeClassifier





`````


`````{py:class} CLAHE




````{py:method} apply(src[, dst]) -> dst

Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.




:param self: 
:type self: 
:param src: Source image of type CV_8UC1 or CV_16UC1.
:type src: cv2.typing.MatLike
:param dst: Destination image.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} apply(src[, dst]) -> dst

Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.




:param self: 
:type self: 
:param src: Source image of type CV_8UC1 or CV_16UC1.
:type src: UMat
:param dst: Destination image.
:type dst: UMat | None
:rtype: UMat
````

````{py:method} setClipLimit(clipLimit) -> None
Sets threshold for contrast limiting.




:param self: 
:type self: 
:param clipLimit: threshold value.
:type clipLimit: float
:rtype: None
````

````{py:method} getClipLimit() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setTilesGridSize(tileGridSize) -> None
Sets size of grid for histogram equalization. Input image will be divided intoequally sized rectangular tiles. 




:param self: 
:type self: 
:param tileGridSize: defines the number of tiles in row and column.
:type tileGridSize: cv2.typing.Size
:rtype: None
````

````{py:method} getTilesGridSize() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} collectGarbage() -> None





:param self: 
:type self: 
:rtype: None
````


`````


`````{py:class} CalibrateCRF




````{py:method} process(src, times[, dst]) -> dst

Recovers inverse camera response.




:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[cv2.typing.MatLike]
:param times: vector of exposure time values for each image
:type times: cv2.typing.MatLike
:param dst: 256x1 matrix with inverse camera response function
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} process(src, times[, dst]) -> dst

Recovers inverse camera response.




:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[UMat]
:param times: vector of exposure time values for each image
:type times: UMat
:param dst: 256x1 matrix with inverse camera response function
:type dst: UMat | None
:rtype: UMat
````


`````


`````{py:class} CalibrateDebevec




````{py:method} getLambda() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setLambda(lambda_) -> None





:param self: 
:type self: 
:param lambda_: 
:type lambda_: float
:rtype: None
````

````{py:method} getSamples() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setSamples(samples) -> None





:param self: 
:type self: 
:param samples: 
:type samples: int
:rtype: None
````

````{py:method} getRandom() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} setRandom(random) -> None





:param self: 
:type self: 
:param random: 
:type random: bool
:rtype: None
````


`````


`````{py:class} CalibrateRobertson




````{py:method} getMaxIter() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMaxIter(max_iter) -> None





:param self: 
:type self: 
:param max_iter: 
:type max_iter: int
:rtype: None
````

````{py:method} getThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setThreshold(threshold) -> None





:param self: 
:type self: 
:param threshold: 
:type threshold: float
:rtype: None
````

````{py:method} getRadiance() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.MatLike
````


`````


`````{py:class} CascadeClassifier




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, filename: str)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:rtype: None
````

````{py:method} detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects

Detects objects of different sizes in the input image. The detected objects are returned as a listof rectangles. 




:param self: 
:type self: 
:param image: Matrix of the type CV_8U containing an image where objects are detected.
:type image: cv2.typing.MatLike
:param scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
:type scaleFactor: float
:param minNeighbors: Parameter specifying how many neighbors each candidate rectangle should haveto retain it. 
:type minNeighbors: int
:param flags: Parameter with the same meaning for an old cascade as in the functioncvHaarDetectObjects. It is not used for a new cascade. 
:type flags: int
:param minSize: Minimum possible object size. Objects smaller than that are ignored.
:type minSize: cv2.typing.Size
:param maxSize: Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
:type maxSize: cv2.typing.Size
:param objects: Vector of rectangles where each rectangle contains the detected object, therectangles may be partially outside the original image. 
:type objects: 
:rtype: _typing.Sequence[cv2.typing.Rect]
````

````{py:method} detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects

Detects objects of different sizes in the input image. The detected objects are returned as a listof rectangles. 




:param self: 
:type self: 
:param image: Matrix of the type CV_8U containing an image where objects are detected.
:type image: UMat
:param scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
:type scaleFactor: float
:param minNeighbors: Parameter specifying how many neighbors each candidate rectangle should haveto retain it. 
:type minNeighbors: int
:param flags: Parameter with the same meaning for an old cascade as in the functioncvHaarDetectObjects. It is not used for a new cascade. 
:type flags: int
:param minSize: Minimum possible object size. Objects smaller than that are ignored.
:type minSize: cv2.typing.Size
:param maxSize: Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
:type maxSize: cv2.typing.Size
:param objects: Vector of rectangles where each rectangle contains the detected object, therectangles may be partially outside the original image. 
:type objects: 
:rtype: _typing.Sequence[cv2.typing.Rect]
````

````{py:method} detectMultiScale2(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects, numDetections




@overload 


:param self: 
:type self: 
:param image: Matrix of the type CV_8U containing an image where objects are detected.
:type image: cv2.typing.MatLike
:param scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
:type scaleFactor: float
:param minNeighbors: Parameter specifying how many neighbors each candidate rectangle should haveto retain it. 
:type minNeighbors: int
:param flags: Parameter with the same meaning for an old cascade as in the functioncvHaarDetectObjects. It is not used for a new cascade. 
:type flags: int
:param minSize: Minimum possible object size. Objects smaller than that are ignored.
:type minSize: cv2.typing.Size
:param maxSize: Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
:type maxSize: cv2.typing.Size
:param objects: Vector of rectangles where each rectangle contains the detected object, therectangles may be partially outside the original image. 
:type objects: 
:param numDetections: Vector of detection numbers for the corresponding objects. An object's numberof detections is the number of neighboring positively classified rectangles that were joined together to form the object. 
:type numDetections: 
:rtype: tuple[_typing.Sequence[cv2.typing.Rect], _typing.Sequence[int]]
````

````{py:method} detectMultiScale2(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects, numDetections




@overload 


:param self: 
:type self: 
:param image: Matrix of the type CV_8U containing an image where objects are detected.
:type image: UMat
:param scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
:type scaleFactor: float
:param minNeighbors: Parameter specifying how many neighbors each candidate rectangle should haveto retain it. 
:type minNeighbors: int
:param flags: Parameter with the same meaning for an old cascade as in the functioncvHaarDetectObjects. It is not used for a new cascade. 
:type flags: int
:param minSize: Minimum possible object size. Objects smaller than that are ignored.
:type minSize: cv2.typing.Size
:param maxSize: Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
:type maxSize: cv2.typing.Size
:param objects: Vector of rectangles where each rectangle contains the detected object, therectangles may be partially outside the original image. 
:type objects: 
:param numDetections: Vector of detection numbers for the corresponding objects. An object's numberof detections is the number of neighboring positively classified rectangles that were joined together to form the object. 
:type numDetections: 
:rtype: tuple[_typing.Sequence[cv2.typing.Rect], _typing.Sequence[int]]
````

````{py:method} detectMultiScale3(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) -> objects, rejectLevels, levelWeights




@overload This function allows you to retrieve the final stage decision certainty of classification. For this, one needs to set `outputRejectLevels` on true and provide the `rejectLevels` and `levelWeights` parameter. For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage. This value can then be used to separate strong from weaker classifications. 
A code sample on how to use it efficiently can be found below: 
```c++
Mat img;
vector<double> weights;
vector<int> levels;
vector<Rect> detections;
CascadeClassifier model("/path/to/your/model.xml");
model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
```



:param self: 
:type self: 
:param image: 
:type image: cv2.typing.MatLike
:param scaleFactor: 
:type scaleFactor: float
:param minNeighbors: 
:type minNeighbors: int
:param flags: 
:type flags: int
:param minSize: 
:type minSize: cv2.typing.Size
:param maxSize: 
:type maxSize: cv2.typing.Size
:param outputRejectLevels: 
:type outputRejectLevels: bool
:rtype: tuple[_typing.Sequence[cv2.typing.Rect], _typing.Sequence[int], _typing.Sequence[float]]
````

````{py:method} detectMultiScale3(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) -> objects, rejectLevels, levelWeights




@overload This function allows you to retrieve the final stage decision certainty of classification. For this, one needs to set `outputRejectLevels` on true and provide the `rejectLevels` and `levelWeights` parameter. For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage. This value can then be used to separate strong from weaker classifications. 
A code sample on how to use it efficiently can be found below: 
```c++
Mat img;
vector<double> weights;
vector<int> levels;
vector<Rect> detections;
CascadeClassifier model("/path/to/your/model.xml");
model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
```



:param self: 
:type self: 
:param image: 
:type image: UMat
:param scaleFactor: 
:type scaleFactor: float
:param minNeighbors: 
:type minNeighbors: int
:param flags: 
:type flags: int
:param minSize: 
:type minSize: cv2.typing.Size
:param maxSize: 
:type maxSize: cv2.typing.Size
:param outputRejectLevels: 
:type outputRejectLevels: bool
:rtype: tuple[_typing.Sequence[cv2.typing.Rect], _typing.Sequence[int], _typing.Sequence[float]]
````

````{py:method} empty() -> retval
Checks whether the classifier has been loaded.




:param self: 
:type self: 
:rtype: bool
````

````{py:method} load(filename) -> retval
Loads a classifier from a file.




:param self: 
:type self: 
:param filename: Name of the file from which the classifier is loaded. The file may contain an oldHAAR classifier trained by the haartraining application or a new cascade classifier trained by the traincascade application. 
:type filename: str
:rtype: bool
````

````{py:method} read(node) -> retval
Reads a classifier from a FileStorage node.


```{note}
The file may contain a new cascade classifier (trained by the traincascade application) only.
```


:param self: 
:type self: 
:param node: 
:type node: FileNode
:rtype: bool
````

````{py:method} isOldFormatCascade() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} getOriginalWindowSize() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} getFeatureType() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} convert(oldcascade, newcascade) -> retval
:staticmethod:





:param oldcascade: 
:type oldcascade: str
:param newcascade: 
:type newcascade: str
:rtype: bool
````


`````


`````{py:class} CirclesGridFinderParameters




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

```{py:attribute} densityNeighborhoodSize
:type: cv2.typing.Size2f
```

```{py:attribute} minDensity
:type: float
```

```{py:attribute} kmeansAttempts
:type: int
```

```{py:attribute} minDistanceToAddKeypoint
:type: int
```

```{py:attribute} keypointScale
:type: int
```

```{py:attribute} minGraphConfidence
:type: float
```

```{py:attribute} vertexGain
:type: float
```

```{py:attribute} vertexPenalty
:type: float
```

```{py:attribute} existingVertexGain
:type: float
```

```{py:attribute} edgeGain
:type: float
```

```{py:attribute} edgePenalty
:type: float
```

```{py:attribute} convexHullFactor
:type: float
```

```{py:attribute} minRNGEdgeSwitchDist
:type: float
```

```{py:attribute} squareSize
:type: float
```

```{py:attribute} maxRectifiedDistance
:type: float
```


`````


`````{py:class} DISOpticalFlow




````{py:method} create([, preset]) -> retval
:classmethod:
Creates an instance of DISOpticalFlow




:param cls: 
:type cls: 
:param preset: one of PRESET_ULTRAFAST, PRESET_FAST and PRESET_MEDIUM
:type preset: int
:rtype: DISOpticalFlow
````

````{py:method} getFinestScale() -> retval
Finest level of the Gaussian pyramid on which the flow is computed (zero levelcorresponds to the original image resolution). The final flow is obtained by bilinear upscaling. 


**See also:** setFinestScale


:param self: 
:type self: 
:rtype: int
````

````{py:method} setFinestScale(val) -> None



@copybrief getFinestScale @see getFinestScale 


:param self: 
:type self: 
:param val: 
:type val: int
:rtype: None
````

````{py:method} getPatchSize() -> retval
Size of an image patch for matching (in pixels). Normally, default 8x8 patches work wellenough in most cases. 


**See also:** setPatchSize


:param self: 
:type self: 
:rtype: int
````

````{py:method} setPatchSize(val) -> None



@copybrief getPatchSize @see getPatchSize 


:param self: 
:type self: 
:param val: 
:type val: int
:rtype: None
````

````{py:method} getPatchStride() -> retval
Stride between neighbor patches. Must be less than patch size. Lower values correspondto higher flow quality. 


**See also:** setPatchStride


:param self: 
:type self: 
:rtype: int
````

````{py:method} setPatchStride(val) -> None



@copybrief getPatchStride @see getPatchStride 


:param self: 
:type self: 
:param val: 
:type val: int
:rtype: None
````

````{py:method} getGradientDescentIterations() -> retval
Maximum number of gradient descent iterations in the patch inverse search stage. Higher valuesmay improve quality in some cases. 


**See also:** setGradientDescentIterations


:param self: 
:type self: 
:rtype: int
````

````{py:method} setGradientDescentIterations(val) -> None



@copybrief getGradientDescentIterations @see getGradientDescentIterations 


:param self: 
:type self: 
:param val: 
:type val: int
:rtype: None
````

````{py:method} getVariationalRefinementIterations() -> retval
Number of fixed point iterations of variational refinement per scale. Set to zero todisable variational refinement completely. Higher values will typically result in more smooth and high-quality flow. 


**See also:** setGradientDescentIterations


:param self: 
:type self: 
:rtype: int
````

````{py:method} setVariationalRefinementIterations(val) -> None



@copybrief getGradientDescentIterations @see getGradientDescentIterations 


:param self: 
:type self: 
:param val: 
:type val: int
:rtype: None
````

````{py:method} getVariationalRefinementAlpha() -> retval
Weight of the smoothness term


**See also:** setVariationalRefinementAlpha


:param self: 
:type self: 
:rtype: float
````

````{py:method} setVariationalRefinementAlpha(val) -> None



@copybrief getVariationalRefinementAlpha @see getVariationalRefinementAlpha 


:param self: 
:type self: 
:param val: 
:type val: float
:rtype: None
````

````{py:method} getVariationalRefinementDelta() -> retval
Weight of the color constancy term


**See also:** setVariationalRefinementDelta


:param self: 
:type self: 
:rtype: float
````

````{py:method} setVariationalRefinementDelta(val) -> None



@copybrief getVariationalRefinementDelta @see getVariationalRefinementDelta 


:param self: 
:type self: 
:param val: 
:type val: float
:rtype: None
````

````{py:method} getVariationalRefinementGamma() -> retval
Weight of the gradient constancy term


**See also:** setVariationalRefinementGamma


:param self: 
:type self: 
:rtype: float
````

````{py:method} setVariationalRefinementGamma(val) -> None



@copybrief getVariationalRefinementGamma @see getVariationalRefinementGamma 


:param self: 
:type self: 
:param val: 
:type val: float
:rtype: None
````

````{py:method} getUseMeanNormalization() -> retval
Whether to use mean-normalization of patches when computing patch distance. It is turned onby default as it typically provides a noticeable quality boost because of increased robustness to illumination variations. Turn it off if you are certain that your sequence doesn't contain any changes in illumination. 


**See also:** setUseMeanNormalization


:param self: 
:type self: 
:rtype: bool
````

````{py:method} setUseMeanNormalization(val) -> None



@copybrief getUseMeanNormalization @see getUseMeanNormalization 


:param self: 
:type self: 
:param val: 
:type val: bool
:rtype: None
````

````{py:method} getUseSpatialPropagation() -> retval
Whether to use spatial propagation of good optical flow vectors. This option is turned on bydefault, as it tends to work better on average and can sometimes help recover from major errors introduced by the coarse-to-fine scheme employed by the DIS optical flow algorithm. Turning this option off can make the output flow field a bit smoother, however. 


**See also:** setUseSpatialPropagation


:param self: 
:type self: 
:rtype: bool
````

````{py:method} setUseSpatialPropagation(val) -> None



@copybrief getUseSpatialPropagation @see getUseSpatialPropagation 


:param self: 
:type self: 
:param val: 
:type val: bool
:rtype: None
````


`````


`````{py:class} DMatch




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, _queryIdx: int, _trainIdx: int, _distance: float)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param _queryIdx: 
:type _queryIdx: int
:param _trainIdx: 
:type _trainIdx: int
:param _distance: 
:type _distance: float
:rtype: None
````

````{py:method} __init__(self, _queryIdx: int, _trainIdx: int, _imgIdx: int, _distance: float)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param _queryIdx: 
:type _queryIdx: int
:param _trainIdx: 
:type _trainIdx: int
:param _imgIdx: 
:type _imgIdx: int
:param _distance: 
:type _distance: float
:rtype: None
````

```{py:attribute} queryIdx
:type: int
```

```{py:attribute} trainIdx
:type: int
```

```{py:attribute} imgIdx
:type: int
```

```{py:attribute} distance
:type: float
```


`````


`````{py:class} DenseOpticalFlow




````{py:method} calc(I0, I1, flow) -> flow

Calculates an optical flow.




:param self: 
:type self: 
:param I0: first 8-bit single-channel input image.
:type I0: cv2.typing.MatLike
:param I1: second input image of the same size and the same type as prev.
:type I1: cv2.typing.MatLike
:param flow: computed flow image that has the same size as prev and type CV_32FC2.
:type flow: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````

````{py:method} calc(I0, I1, flow) -> flow

Calculates an optical flow.




:param self: 
:type self: 
:param I0: first 8-bit single-channel input image.
:type I0: UMat
:param I1: second input image of the same size and the same type as prev.
:type I1: UMat
:param flow: computed flow image that has the same size as prev and type CV_32FC2.
:type flow: UMat
:rtype: UMat
````

````{py:method} collectGarbage() -> None
Releases all inner buffers.




:param self: 
:type self: 
:rtype: None
````


`````


`````{py:class} DescriptorMatcher




````{py:method} create(descriptorMatcherType) -> retval
:classmethod:
Creates a descriptor matcher of a given type with the default parameters (using defaultconstructor). 


create(matcherType) -> retval 


:param cls: 
:type cls: 
:param descriptorMatcherType: Descriptor matcher type. Now the following matcher types aresupported: -   `BruteForce` (it uses L2 ) -   `BruteForce-L1` -   `BruteForce-Hamming` -   `BruteForce-Hamming(2)` -   `FlannBased` 
:type descriptorMatcherType: str
:rtype: DescriptorMatcher
````

````{py:method} create(descriptorMatcherType) -> retval
:classmethod:
Creates a descriptor matcher of a given type with the default parameters (using defaultconstructor). 


create(matcherType) -> retval 


:param cls: 
:type cls: 
:param matcherType: 
:type matcherType: DescriptorMatcher_MatcherType
:param descriptorMatcherType: Descriptor matcher type. Now the following matcher types aresupported: -   `BruteForce` (it uses L2 ) -   `BruteForce-L1` -   `BruteForce-Hamming` -   `BruteForce-Hamming(2)` -   `FlannBased` 
:type descriptorMatcherType: 
:rtype: DescriptorMatcher
````

````{py:method} add(descriptors) -> None

Adds descriptors to train a CPU(trainDescCollectionis) or GPU(utrainDescCollectionis) descriptorcollection. 


If the collection is not empty, the new descriptors are added to existing train descriptors. 


:param self: 
:type self: 
:param descriptors: Descriptors to add. Each descriptors[i] is a set of descriptors from the sametrain image. 
:type descriptors: _typing.Sequence[cv2.typing.MatLike]
:rtype: None
````

````{py:method} add(descriptors) -> None

Adds descriptors to train a CPU(trainDescCollectionis) or GPU(utrainDescCollectionis) descriptorcollection. 


If the collection is not empty, the new descriptors are added to existing train descriptors. 


:param self: 
:type self: 
:param descriptors: Descriptors to add. Each descriptors[i] is a set of descriptors from the sametrain image. 
:type descriptors: _typing.Sequence[UMat]
:rtype: None
````

````{py:method} match(queryDescriptors, trainDescriptors[, mask]) -> matches

Finds the best match for each descriptor from a query set.


In the first variant of this method, the train descriptors are passed as an input argument. In the second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is used. Optional mask (or masks) can be passed to specify which query and training descriptors can be matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if mask.at\<uchar\>(i,j) is non-zero. 
match(queryDescriptors[, masks]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: cv2.typing.MatLike
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: cv2.typing.MatLike
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: cv2.typing.MatLike | None
:param matches: Matches. If a query descriptor is masked out in mask , no match is added for thisdescriptor. So, matches size may be smaller than the query descriptors count. 
:type matches: 
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: 
:rtype: _typing.Sequence[DMatch]
````

````{py:method} match(queryDescriptors, trainDescriptors[, mask]) -> matches

Finds the best match for each descriptor from a query set.


In the first variant of this method, the train descriptors are passed as an input argument. In the second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is used. Optional mask (or masks) can be passed to specify which query and training descriptors can be matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if mask.at\<uchar\>(i,j) is non-zero. 
match(queryDescriptors[, masks]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: UMat
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: UMat
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: UMat | None
:param matches: Matches. If a query descriptor is masked out in mask , no match is added for thisdescriptor. So, matches size may be smaller than the query descriptors count. 
:type matches: 
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: 
:rtype: _typing.Sequence[DMatch]
````

````{py:method} match(queryDescriptors, trainDescriptors[, mask]) -> matches

Finds the best match for each descriptor from a query set.


In the first variant of this method, the train descriptors are passed as an input argument. In the second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is used. Optional mask (or masks) can be passed to specify which query and training descriptors can be matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if mask.at\<uchar\>(i,j) is non-zero. 
match(queryDescriptors[, masks]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: cv2.typing.MatLike
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: _typing.Sequence[cv2.typing.MatLike] | None
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: 
:param matches: Matches. If a query descriptor is masked out in mask , no match is added for thisdescriptor. So, matches size may be smaller than the query descriptors count. 
:type matches: 
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: 
:rtype: _typing.Sequence[DMatch]
````

````{py:method} match(queryDescriptors, trainDescriptors[, mask]) -> matches

Finds the best match for each descriptor from a query set.


In the first variant of this method, the train descriptors are passed as an input argument. In the second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is used. Optional mask (or masks) can be passed to specify which query and training descriptors can be matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if mask.at\<uchar\>(i,j) is non-zero. 
match(queryDescriptors[, masks]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: UMat
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: _typing.Sequence[UMat] | None
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: 
:param matches: Matches. If a query descriptor is masked out in mask , no match is added for thisdescriptor. So, matches size may be smaller than the query descriptors count. 
:type matches: 
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: 
:rtype: _typing.Sequence[DMatch]
````

````{py:method} knnMatch(queryDescriptors, trainDescriptors, k[, mask[, compactResult]]) -> matches

Finds the k best matches for each descriptor from a query set.


These extended variants of DescriptorMatcher::match methods find several best matches for each query descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match for the details about query and train descriptors. 
knnMatch(queryDescriptors, k[, masks[, compactResult]]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: cv2.typing.MatLike
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: cv2.typing.MatLike
:param k: Count of best matches found per each query descriptor or less if a query descriptor hasless than k possible matches in total. 
:type k: int
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: cv2.typing.MatLike | None
:param compactResult: Parameter used when the mask (or masks) is not empty. If compactResult isfalse, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors. 
:type compactResult: bool
:param matches: Matches. Each matches[i] is k or less matches for the same query descriptor.
:type matches: 
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: 
:rtype: _typing.Sequence[_typing.Sequence[DMatch]]
````

````{py:method} knnMatch(queryDescriptors, trainDescriptors, k[, mask[, compactResult]]) -> matches

Finds the k best matches for each descriptor from a query set.


These extended variants of DescriptorMatcher::match methods find several best matches for each query descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match for the details about query and train descriptors. 
knnMatch(queryDescriptors, k[, masks[, compactResult]]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: UMat
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: UMat
:param k: Count of best matches found per each query descriptor or less if a query descriptor hasless than k possible matches in total. 
:type k: int
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: UMat | None
:param compactResult: Parameter used when the mask (or masks) is not empty. If compactResult isfalse, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors. 
:type compactResult: bool
:param matches: Matches. Each matches[i] is k or less matches for the same query descriptor.
:type matches: 
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: 
:rtype: _typing.Sequence[_typing.Sequence[DMatch]]
````

````{py:method} knnMatch(queryDescriptors, trainDescriptors, k[, mask[, compactResult]]) -> matches

Finds the k best matches for each descriptor from a query set.


These extended variants of DescriptorMatcher::match methods find several best matches for each query descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match for the details about query and train descriptors. 
knnMatch(queryDescriptors, k[, masks[, compactResult]]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: cv2.typing.MatLike
:param k: Count of best matches found per each query descriptor or less if a query descriptor hasless than k possible matches in total. 
:type k: int
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: _typing.Sequence[cv2.typing.MatLike] | None
:param compactResult: Parameter used when the mask (or masks) is not empty. If compactResult isfalse, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors. 
:type compactResult: bool
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: 
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: 
:param matches: Matches. Each matches[i] is k or less matches for the same query descriptor.
:type matches: 
:rtype: _typing.Sequence[_typing.Sequence[DMatch]]
````

````{py:method} knnMatch(queryDescriptors, trainDescriptors, k[, mask[, compactResult]]) -> matches

Finds the k best matches for each descriptor from a query set.


These extended variants of DescriptorMatcher::match methods find several best matches for each query descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match for the details about query and train descriptors. 
knnMatch(queryDescriptors, k[, masks[, compactResult]]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: UMat
:param k: Count of best matches found per each query descriptor or less if a query descriptor hasless than k possible matches in total. 
:type k: int
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: _typing.Sequence[UMat] | None
:param compactResult: Parameter used when the mask (or masks) is not empty. If compactResult isfalse, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors. 
:type compactResult: bool
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: 
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: 
:param matches: Matches. Each matches[i] is k or less matches for the same query descriptor.
:type matches: 
:rtype: _typing.Sequence[_typing.Sequence[DMatch]]
````

````{py:method} radiusMatch(queryDescriptors, trainDescriptors, maxDistance[, mask[, compactResult]]) -> matches

For each query descriptor, finds the training descriptors not farther than the specified distance.


For each query descriptor, the methods find such training descriptors that the distance between the query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are returned in the distance increasing order. 
radiusMatch(queryDescriptors, maxDistance[, masks[, compactResult]]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: cv2.typing.MatLike
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: cv2.typing.MatLike
:param maxDistance: Threshold for the distance between matched descriptors. Distance means heremetric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)! 
:type maxDistance: float
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: cv2.typing.MatLike | None
:param compactResult: Parameter used when the mask (or masks) is not empty. If compactResult isfalse, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors. 
:type compactResult: bool
:param matches: Found matches.
:type matches: 
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: 
:rtype: _typing.Sequence[_typing.Sequence[DMatch]]
````

````{py:method} radiusMatch(queryDescriptors, trainDescriptors, maxDistance[, mask[, compactResult]]) -> matches

For each query descriptor, finds the training descriptors not farther than the specified distance.


For each query descriptor, the methods find such training descriptors that the distance between the query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are returned in the distance increasing order. 
radiusMatch(queryDescriptors, maxDistance[, masks[, compactResult]]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: UMat
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: UMat
:param maxDistance: Threshold for the distance between matched descriptors. Distance means heremetric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)! 
:type maxDistance: float
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: UMat | None
:param compactResult: Parameter used when the mask (or masks) is not empty. If compactResult isfalse, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors. 
:type compactResult: bool
:param matches: Found matches.
:type matches: 
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: 
:rtype: _typing.Sequence[_typing.Sequence[DMatch]]
````

````{py:method} radiusMatch(queryDescriptors, trainDescriptors, maxDistance[, mask[, compactResult]]) -> matches

For each query descriptor, finds the training descriptors not farther than the specified distance.


For each query descriptor, the methods find such training descriptors that the distance between the query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are returned in the distance increasing order. 
radiusMatch(queryDescriptors, maxDistance[, masks[, compactResult]]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: cv2.typing.MatLike
:param maxDistance: Threshold for the distance between matched descriptors. Distance means heremetric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)! 
:type maxDistance: float
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: _typing.Sequence[cv2.typing.MatLike] | None
:param compactResult: Parameter used when the mask (or masks) is not empty. If compactResult isfalse, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors. 
:type compactResult: bool
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: 
:param matches: Found matches.
:type matches: 
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: 
:rtype: _typing.Sequence[_typing.Sequence[DMatch]]
````

````{py:method} radiusMatch(queryDescriptors, trainDescriptors, maxDistance[, mask[, compactResult]]) -> matches

For each query descriptor, finds the training descriptors not farther than the specified distance.


For each query descriptor, the methods find such training descriptors that the distance between the query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are returned in the distance increasing order. 
radiusMatch(queryDescriptors, maxDistance[, masks[, compactResult]]) -> matches @overload 


:param self: 
:type self: 
:param queryDescriptors: Query set of descriptors.
:type queryDescriptors: UMat
:param maxDistance: Threshold for the distance between matched descriptors. Distance means heremetric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)! 
:type maxDistance: float
:param masks: Set of masks. Each masks[i] specifies permissible matches between the input querydescriptors and stored train descriptors from the i-th image trainDescCollection[i]. 
:type masks: _typing.Sequence[UMat] | None
:param compactResult: Parameter used when the mask (or masks) is not empty. If compactResult isfalse, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors. 
:type compactResult: bool
:param trainDescriptors: Train set of descriptors. This set is not added to the train descriptorscollection stored in the class object. 
:type trainDescriptors: 
:param matches: Found matches.
:type matches: 
:param mask: Mask specifying permissible matches between an input query and train matrices ofdescriptors. 
:type mask: 
:rtype: _typing.Sequence[_typing.Sequence[DMatch]]
````

````{py:method} write(fileName) -> None




write(fs, name) -> None 


:param self: 
:type self: 
:param fileName: 
:type fileName: str
:rtype: None
````

````{py:method} write(fileName) -> None




write(fs, name) -> None 


:param self: 
:type self: 
:param fs: 
:type fs: FileStorage
:param name: 
:type name: str
:rtype: None
````

````{py:method} read(fileName) -> None




read(arg1) -> None 


:param self: 
:type self: 
:param fileName: 
:type fileName: str
:rtype: None
````

````{py:method} read(fileName) -> None




read(arg1) -> None 


:param self: 
:type self: 
:param arg1: 
:type arg1: FileNode
:rtype: None
````

````{py:method} getTrainDescriptors() -> retval
Returns a constant link to the train descriptor collection trainDescCollection .




:param self: 
:type self: 
:rtype: _typing.Sequence[cv2.typing.MatLike]
````

````{py:method} clear() -> None
Clears the train descriptor collections.




:param self: 
:type self: 
:rtype: None
````

````{py:method} empty() -> retval
Returns true if there are no train descriptors in the both collections.




:param self: 
:type self: 
:rtype: bool
````

````{py:method} isMaskSupported() -> retval
Returns true if the descriptor matcher supports masking permissible matches.




:param self: 
:type self: 
:rtype: bool
````

````{py:method} train() -> None
Trains a descriptor matcher


Trains a descriptor matcher (for example, the flann index). In all methods to match, the method train() is run every time before matching. Some descriptor matchers (for example, BruteForceMatcher) have an empty implementation of this method. Other matchers really train their inner structures (for example, FlannBasedMatcher trains flann::Index ). 


:param self: 
:type self: 
:rtype: None
````

````{py:method} clone([, emptyTrainData]) -> retval
Clones the matcher.




:param self: 
:type self: 
:param emptyTrainData: If emptyTrainData is false, the method creates a deep copy of the object,that is, copies both parameters and train data. If emptyTrainData is true, the method creates an object copy with the current parameters but with empty train data. 
:type emptyTrainData: bool
:rtype: DescriptorMatcher
````


`````


`````{py:class} FaceDetectorYN




````{py:method} create(model, config, input_size[, score_threshold[, nms_threshold[, top_k[, backend_id[, target_id]]]]]) -> retval
:classmethod:
Creates an instance of face detector class with given parameters


create(framework, bufferModel, bufferConfig, input_size[, score_threshold[, nms_threshold[, top_k[, backend_id[, target_id]]]]]) -> retval @overload 


:param cls: 
:type cls: 
:param model: the path to the requested model
:type model: str
:param config: the path to the config file for compability, which is not requested for ONNX models
:type config: str
:param input_size: the size of the input image
:type input_size: cv2.typing.Size
:param score_threshold: the threshold to filter out bounding boxes of score smaller than the given value
:type score_threshold: float
:param nms_threshold: the threshold to suppress bounding boxes of IoU bigger than the given value
:type nms_threshold: float
:param top_k: keep top K bboxes before NMS
:type top_k: int
:param backend_id: the id of backend
:type backend_id: int
:param target_id: the id of target device
:type target_id: int
:param framework: Name of origin framework
:type framework: 
:param bufferModel: A buffer with a content of binary file with weights
:type bufferModel: 
:param bufferConfig: A buffer with a content of text file contains network configuration
:type bufferConfig: 
:rtype: FaceDetectorYN
````

````{py:method} create(model, config, input_size[, score_threshold[, nms_threshold[, top_k[, backend_id[, target_id]]]]]) -> retval
:classmethod:
Creates an instance of face detector class with given parameters


create(framework, bufferModel, bufferConfig, input_size[, score_threshold[, nms_threshold[, top_k[, backend_id[, target_id]]]]]) -> retval @overload 


:param cls: 
:type cls: 
:param framework: Name of origin framework
:type framework: str
:param bufferModel: A buffer with a content of binary file with weights
:type bufferModel: numpy.ndarray[_typing.Any, numpy.dtype[numpy.uint8]]
:param bufferConfig: A buffer with a content of text file contains network configuration
:type bufferConfig: numpy.ndarray[_typing.Any, numpy.dtype[numpy.uint8]]
:param input_size: the size of the input image
:type input_size: cv2.typing.Size
:param score_threshold: the threshold to filter out bounding boxes of score smaller than the given value
:type score_threshold: float
:param nms_threshold: the threshold to suppress bounding boxes of IoU bigger than the given value
:type nms_threshold: float
:param top_k: keep top K bboxes before NMS
:type top_k: int
:param backend_id: the id of backend
:type backend_id: int
:param target_id: the id of target device
:type target_id: int
:param model: the path to the requested model
:type model: 
:param config: the path to the config file for compability, which is not requested for ONNX models
:type config: 
:rtype: FaceDetectorYN
````

````{py:method} detect(image[, faces]) -> retval, faces

Detects faces in the input image. Following is an example output.


![image](pics/lena-face-detection.jpg) 


:param self: 
:type self: 
:param image: an image to detect
:type image: cv2.typing.MatLike
:param faces: detection results stored in a 2D cv::Mat of shape [num_faces, 15]- 0-1: x, y of bbox top left corner - 2-3: width, height of bbox - 4-5: x, y of right eye (blue point in the example image) - 6-7: x, y of left eye (red point in the example image) - 8-9: x, y of nose tip (green point in the example image) - 10-11: x, y of right corner of mouth (pink point in the example image) - 12-13: x, y of left corner of mouth (yellow point in the example image) - 14: face score 
:type faces: cv2.typing.MatLike | None
:rtype: tuple[int, cv2.typing.MatLike]
````

````{py:method} detect(image[, faces]) -> retval, faces

Detects faces in the input image. Following is an example output.


![image](pics/lena-face-detection.jpg) 


:param self: 
:type self: 
:param image: an image to detect
:type image: UMat
:param faces: detection results stored in a 2D cv::Mat of shape [num_faces, 15]- 0-1: x, y of bbox top left corner - 2-3: width, height of bbox - 4-5: x, y of right eye (blue point in the example image) - 6-7: x, y of left eye (red point in the example image) - 8-9: x, y of nose tip (green point in the example image) - 10-11: x, y of right corner of mouth (pink point in the example image) - 12-13: x, y of left corner of mouth (yellow point in the example image) - 14: face score 
:type faces: UMat | None
:rtype: tuple[int, UMat]
````

````{py:method} setInputSize(input_size) -> None
Set the size for the network input, which overwrites the input size of creating model. Call this method when the size of input image does not match the input size when creating model




:param self: 
:type self: 
:param input_size: the size of the input image
:type input_size: cv2.typing.Size
:rtype: None
````

````{py:method} getInputSize() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} setScoreThreshold(score_threshold) -> None
Set the score threshold to filter out bounding boxes of score less than the given value




:param self: 
:type self: 
:param score_threshold: threshold for filtering out bounding boxes
:type score_threshold: float
:rtype: None
````

````{py:method} getScoreThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setNMSThreshold(nms_threshold) -> None
Set the Non-maximum-suppression threshold to suppress bounding boxes that have IoU greater than the given value




:param self: 
:type self: 
:param nms_threshold: threshold for NMS operation
:type nms_threshold: float
:rtype: None
````

````{py:method} getNMSThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setTopK(top_k) -> None
Set the number of bounding boxes preserved before NMS




:param self: 
:type self: 
:param top_k: the number of bounding boxes to preserve from top rank based on score
:type top_k: int
:rtype: None
````

````{py:method} getTopK() -> retval





:param self: 
:type self: 
:rtype: int
````


`````


`````{py:class} FaceRecognizerSF




````{py:method} create(model, config[, backend_id[, target_id]]) -> retval
:classmethod:
Creates an instance of this class with given parameters




:param cls: 
:type cls: 
:param model: the path of the onnx model used for face recognition
:type model: str
:param config: the path to the config file for compability, which is not requested for ONNX models
:type config: str
:param backend_id: the id of backend
:type backend_id: int
:param target_id: the id of target device
:type target_id: int
:rtype: FaceRecognizerSF
````

````{py:method} alignCrop(src_img, face_box[, aligned_img]) -> aligned_img

Aligning image to put face on the standard position




:param self: 
:type self: 
:param src_img: input image
:type src_img: cv2.typing.MatLike
:param face_box: the detection result used for indicate face in input image
:type face_box: cv2.typing.MatLike
:param aligned_img: output aligned image
:type aligned_img: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} alignCrop(src_img, face_box[, aligned_img]) -> aligned_img

Aligning image to put face on the standard position




:param self: 
:type self: 
:param src_img: input image
:type src_img: UMat
:param face_box: the detection result used for indicate face in input image
:type face_box: UMat
:param aligned_img: output aligned image
:type aligned_img: UMat | None
:rtype: UMat
````

````{py:method} feature(aligned_img[, face_feature]) -> face_feature

Extracting face feature from aligned image




:param self: 
:type self: 
:param aligned_img: input aligned image
:type aligned_img: cv2.typing.MatLike
:param face_feature: output face feature
:type face_feature: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} feature(aligned_img[, face_feature]) -> face_feature

Extracting face feature from aligned image




:param self: 
:type self: 
:param aligned_img: input aligned image
:type aligned_img: UMat
:param face_feature: output face feature
:type face_feature: UMat | None
:rtype: UMat
````

````{py:method} match(face_feature1, face_feature2[, dis_type]) -> retval

Calculating the distance between two face features




:param self: 
:type self: 
:param face_feature1: the first input feature
:type face_feature1: cv2.typing.MatLike
:param face_feature2: the second input feature of the same size and the same type as face_feature1
:type face_feature2: cv2.typing.MatLike
:param dis_type: defining the similarity with optional values "FR_OSINE" or "FR_NORM_L2"
:type dis_type: int
:rtype: float
````

````{py:method} match(face_feature1, face_feature2[, dis_type]) -> retval

Calculating the distance between two face features




:param self: 
:type self: 
:param face_feature1: the first input feature
:type face_feature1: UMat
:param face_feature2: the second input feature of the same size and the same type as face_feature1
:type face_feature2: UMat
:param dis_type: defining the similarity with optional values "FR_OSINE" or "FR_NORM_L2"
:type dis_type: int
:rtype: float
````


`````


`````{py:class} FarnebackOpticalFlow




````{py:method} create([, numLevels[, pyrScale[, fastPyramids[, winSize[, numIters[, polyN[, polySigma[, flags]]]]]]]]) -> retval
:classmethod:





:param cls: 
:type cls: 
:param numLevels: 
:type numLevels: int
:param pyrScale: 
:type pyrScale: float
:param fastPyramids: 
:type fastPyramids: bool
:param winSize: 
:type winSize: int
:param numIters: 
:type numIters: int
:param polyN: 
:type polyN: int
:param polySigma: 
:type polySigma: float
:param flags: 
:type flags: int
:rtype: FarnebackOpticalFlow
````

````{py:method} getNumLevels() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setNumLevels(numLevels) -> None





:param self: 
:type self: 
:param numLevels: 
:type numLevels: int
:rtype: None
````

````{py:method} getPyrScale() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setPyrScale(pyrScale) -> None





:param self: 
:type self: 
:param pyrScale: 
:type pyrScale: float
:rtype: None
````

````{py:method} getFastPyramids() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} setFastPyramids(fastPyramids) -> None





:param self: 
:type self: 
:param fastPyramids: 
:type fastPyramids: bool
:rtype: None
````

````{py:method} getWinSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setWinSize(winSize) -> None





:param self: 
:type self: 
:param winSize: 
:type winSize: int
:rtype: None
````

````{py:method} getNumIters() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setNumIters(numIters) -> None





:param self: 
:type self: 
:param numIters: 
:type numIters: int
:rtype: None
````

````{py:method} getPolyN() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setPolyN(polyN) -> None





:param self: 
:type self: 
:param polyN: 
:type polyN: int
:rtype: None
````

````{py:method} getPolySigma() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setPolySigma(polySigma) -> None





:param self: 
:type self: 
:param polySigma: 
:type polySigma: float
:rtype: None
````

````{py:method} getFlags() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setFlags(flags) -> None





:param self: 
:type self: 
:param flags: 
:type flags: int
:rtype: None
````


`````


`````{py:class} FastFeatureDetector




````{py:method} create([, threshold[, nonmaxSuppression[, type]]]) -> retval
:classmethod:





:param cls: 
:type cls: 
:param threshold: 
:type threshold: int
:param nonmaxSuppression: 
:type nonmaxSuppression: bool
:param type: 
:type type: FastFeatureDetector_DetectorType
:rtype: FastFeatureDetector
````

````{py:method} setThreshold(threshold) -> None





:param self: 
:type self: 
:param threshold: 
:type threshold: int
:rtype: None
````

````{py:method} getThreshold() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setNonmaxSuppression(f) -> None





:param self: 
:type self: 
:param f: 
:type f: bool
:rtype: None
````

````{py:method} getNonmaxSuppression() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} setType(type) -> None





:param self: 
:type self: 
:param type: 
:type type: FastFeatureDetector_DetectorType
:rtype: None
````

````{py:method} getType() -> retval





:param self: 
:type self: 
:rtype: FastFeatureDetector_DetectorType
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} Feature2D




````{py:method} detect(image[, mask]) -> keypoints

Detects keypoints in an image (first variant) or image set (second variant).


detect(images[, masks]) -> keypoints @overload 


:param self: 
:type self: 
:param image: Image.
:type image: cv2.typing.MatLike
:param mask: Mask specifying where to look for keypoints (optional). It must be a 8-bit integermatrix with non-zero values in the region of interest. 
:type mask: cv2.typing.MatLike | None
:param keypoints: The detected keypoints. In the second variant of the method keypoints[i] is a setof keypoints detected in images[i] . 
:type keypoints: 
:param images: Image set.
:type images: 
:param masks: Masks for each input image specifying where to look for keypoints (optional).masks[i] is a mask for images[i]. 
:type masks: 
:rtype: _typing.Sequence[KeyPoint]
````

````{py:method} detect(image[, mask]) -> keypoints

Detects keypoints in an image (first variant) or image set (second variant).


detect(images[, masks]) -> keypoints @overload 


:param self: 
:type self: 
:param image: Image.
:type image: UMat
:param mask: Mask specifying where to look for keypoints (optional). It must be a 8-bit integermatrix with non-zero values in the region of interest. 
:type mask: UMat | None
:param keypoints: The detected keypoints. In the second variant of the method keypoints[i] is a setof keypoints detected in images[i] . 
:type keypoints: 
:param images: Image set.
:type images: 
:param masks: Masks for each input image specifying where to look for keypoints (optional).masks[i] is a mask for images[i]. 
:type masks: 
:rtype: _typing.Sequence[KeyPoint]
````

````{py:method} detect(image[, mask]) -> keypoints

Detects keypoints in an image (first variant) or image set (second variant).


detect(images[, masks]) -> keypoints @overload 


:param self: 
:type self: 
:param images: Image set.
:type images: _typing.Sequence[cv2.typing.MatLike]
:param masks: Masks for each input image specifying where to look for keypoints (optional).masks[i] is a mask for images[i]. 
:type masks: _typing.Sequence[cv2.typing.MatLike] | None
:param image: Image.
:type image: 
:param keypoints: The detected keypoints. In the second variant of the method keypoints[i] is a setof keypoints detected in images[i] . 
:type keypoints: 
:param mask: Mask specifying where to look for keypoints (optional). It must be a 8-bit integermatrix with non-zero values in the region of interest. 
:type mask: 
:rtype: _typing.Sequence[_typing.Sequence[KeyPoint]]
````

````{py:method} detect(image[, mask]) -> keypoints

Detects keypoints in an image (first variant) or image set (second variant).


detect(images[, masks]) -> keypoints @overload 


:param self: 
:type self: 
:param images: Image set.
:type images: _typing.Sequence[UMat]
:param masks: Masks for each input image specifying where to look for keypoints (optional).masks[i] is a mask for images[i]. 
:type masks: _typing.Sequence[UMat] | None
:param image: Image.
:type image: 
:param keypoints: The detected keypoints. In the second variant of the method keypoints[i] is a setof keypoints detected in images[i] . 
:type keypoints: 
:param mask: Mask specifying where to look for keypoints (optional). It must be a 8-bit integermatrix with non-zero values in the region of interest. 
:type mask: 
:rtype: _typing.Sequence[_typing.Sequence[KeyPoint]]
````

````{py:method} compute(image, keypoints[, descriptors]) -> keypoints, descriptors

Computes the descriptors for a set of keypoints detected in an image (first variant) or image set(second variant). 


compute(images, keypoints[, descriptors]) -> keypoints, descriptors @overload 


:param self: 
:type self: 
:param image: Image.
:type image: cv2.typing.MatLike
:param keypoints: Input collection of keypoints. Keypoints for which a descriptor cannot becomputed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation). 
:type keypoints: _typing.Sequence[KeyPoint]
:param descriptors: Computed descriptors. In the second variant of the method descriptors[i] aredescriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint. 
:type descriptors: cv2.typing.MatLike | None
:param images: Image set.
:type images: 
:rtype: tuple[_typing.Sequence[KeyPoint], cv2.typing.MatLike]
````

````{py:method} compute(image, keypoints[, descriptors]) -> keypoints, descriptors

Computes the descriptors for a set of keypoints detected in an image (first variant) or image set(second variant). 


compute(images, keypoints[, descriptors]) -> keypoints, descriptors @overload 


:param self: 
:type self: 
:param image: Image.
:type image: UMat
:param keypoints: Input collection of keypoints. Keypoints for which a descriptor cannot becomputed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation). 
:type keypoints: _typing.Sequence[KeyPoint]
:param descriptors: Computed descriptors. In the second variant of the method descriptors[i] aredescriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint. 
:type descriptors: UMat | None
:param images: Image set.
:type images: 
:rtype: tuple[_typing.Sequence[KeyPoint], UMat]
````

````{py:method} compute(image, keypoints[, descriptors]) -> keypoints, descriptors

Computes the descriptors for a set of keypoints detected in an image (first variant) or image set(second variant). 


compute(images, keypoints[, descriptors]) -> keypoints, descriptors @overload 


:param self: 
:type self: 
:param images: Image set.
:type images: _typing.Sequence[cv2.typing.MatLike]
:param keypoints: Input collection of keypoints. Keypoints for which a descriptor cannot becomputed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation). 
:type keypoints: _typing.Sequence[_typing.Sequence[KeyPoint]]
:param descriptors: Computed descriptors. In the second variant of the method descriptors[i] aredescriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint. 
:type descriptors: _typing.Sequence[cv2.typing.MatLike] | None
:param image: Image.
:type image: 
:rtype: tuple[_typing.Sequence[_typing.Sequence[KeyPoint]], _typing.Sequence[cv2.typing.MatLike]]
````

````{py:method} compute(image, keypoints[, descriptors]) -> keypoints, descriptors

Computes the descriptors for a set of keypoints detected in an image (first variant) or image set(second variant). 


compute(images, keypoints[, descriptors]) -> keypoints, descriptors @overload 


:param self: 
:type self: 
:param images: Image set.
:type images: _typing.Sequence[UMat]
:param keypoints: Input collection of keypoints. Keypoints for which a descriptor cannot becomputed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation). 
:type keypoints: _typing.Sequence[_typing.Sequence[KeyPoint]]
:param descriptors: Computed descriptors. In the second variant of the method descriptors[i] aredescriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint. 
:type descriptors: _typing.Sequence[UMat] | None
:param image: Image.
:type image: 
:rtype: tuple[_typing.Sequence[_typing.Sequence[KeyPoint]], _typing.Sequence[UMat]]
````

````{py:method} detectAndCompute(image, mask[, descriptors[, useProvidedKeypoints]]) -> keypoints, descriptors




Detects keypoints and computes the descriptors 


:param self: 
:type self: 
:param image: 
:type image: cv2.typing.MatLike
:param mask: 
:type mask: cv2.typing.MatLike
:param descriptors: 
:type descriptors: cv2.typing.MatLike | None
:param useProvidedKeypoints: 
:type useProvidedKeypoints: bool
:rtype: tuple[_typing.Sequence[KeyPoint], cv2.typing.MatLike]
````

````{py:method} detectAndCompute(image, mask[, descriptors[, useProvidedKeypoints]]) -> keypoints, descriptors




Detects keypoints and computes the descriptors 


:param self: 
:type self: 
:param image: 
:type image: UMat
:param mask: 
:type mask: UMat
:param descriptors: 
:type descriptors: UMat | None
:param useProvidedKeypoints: 
:type useProvidedKeypoints: bool
:rtype: tuple[_typing.Sequence[KeyPoint], UMat]
````

````{py:method} write(fileName) -> None




write(fs, name) -> None 


:param self: 
:type self: 
:param fileName: 
:type fileName: str
:rtype: None
````

````{py:method} write(fileName) -> None




write(fs, name) -> None 


:param self: 
:type self: 
:param fs: 
:type fs: FileStorage
:param name: 
:type name: str
:rtype: None
````

````{py:method} read(fileName) -> None




read(arg1) -> None 


:param self: 
:type self: 
:param fileName: 
:type fileName: str
:rtype: None
````

````{py:method} read(fileName) -> None




read(arg1) -> None 


:param self: 
:type self: 
:param arg1: 
:type arg1: FileNode
:rtype: None
````

````{py:method} descriptorSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} descriptorType() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} defaultNorm() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} empty() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} FileNode




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} getNode(nodename) -> retval



@overload 


:param self: 
:type self: 
:param nodename: Name of an element in the mapping node.
:type nodename: str
:rtype: FileNode
````

````{py:method} at(i) -> retval



@overload 


:param self: 
:type self: 
:param i: Index of an element in the sequence node.
:type i: int
:rtype: FileNode
````

````{py:method} keys() -> retval
Returns keys of a mapping node.




:param self: 
:type self: 
:return: Keys of a mapping node.
:rtype: _typing.Sequence[str]
````

````{py:method} type() -> retval
Returns type of the node.




:param self: 
:type self: 
:return: Type of the node. See FileNode::Type
:rtype: int
````

````{py:method} empty() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isNone() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isSeq() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isMap() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isInt() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isReal() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isString() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isNamed() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} name() -> retval





:param self: 
:type self: 
:rtype: str
````

````{py:method} size() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} rawSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} real() -> retval



Internal method used when reading FileStorage. Sets the type (int, real or string) and value of the previously created node. 


:param self: 
:type self: 
:rtype: float
````

````{py:method} string() -> retval





:param self: 
:type self: 
:rtype: str
````

````{py:method} mat() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.MatLike
````


`````


`````{py:class} FileStorage




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, filename: str, flags: int, encoding: str=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param flags: 
:type flags: int
:param encoding: 
:type encoding: str
:rtype: None
````

````{py:method} write(name, val) -> None

Simplified writing API to use with bindings.




:param self: 
:type self: 
:param name: Name of the written object. When writing to sequences (a.k.a. "arrays"), pass an empty string.
:type name: str
:param val: Value of the written object.
:type val: int
:rtype: None
````

````{py:method} write(name, val) -> None

Simplified writing API to use with bindings.




:param self: 
:type self: 
:param name: Name of the written object. When writing to sequences (a.k.a. "arrays"), pass an empty string.
:type name: str
:param val: Value of the written object.
:type val: float
:rtype: None
````

````{py:method} write(name, val) -> None

Simplified writing API to use with bindings.




:param self: 
:type self: 
:param name: Name of the written object. When writing to sequences (a.k.a. "arrays"), pass an empty string.
:type name: str
:param val: Value of the written object.
:type val: str
:rtype: None
````

````{py:method} write(name, val) -> None

Simplified writing API to use with bindings.




:param self: 
:type self: 
:param name: Name of the written object. When writing to sequences (a.k.a. "arrays"), pass an empty string.
:type name: str
:param val: Value of the written object.
:type val: cv2.typing.MatLike
:rtype: None
````

````{py:method} write(name, val) -> None

Simplified writing API to use with bindings.




:param self: 
:type self: 
:param name: Name of the written object. When writing to sequences (a.k.a. "arrays"), pass an empty string.
:type name: str
:param val: Value of the written object.
:type val: _typing.Sequence[str]
:rtype: None
````

````{py:method} open(filename, flags[, encoding]) -> retval
Opens a file.


See description of parameters in FileStorage::FileStorage. The method calls FileStorage::release before opening the file. 


:param self: 
:type self: 
:param filename: Name of the file to open or the text string to read the data from.Extension of the file (.xml, .yml/.yaml or .json) determines its format (XML, YAML or JSON respectively). Also you can append .gz to work with compressed files, for example myHugeMatrix.xml.gz. If both FileStorage::WRITE and FileStorage::MEMORY flags are specified, source is used just to specify the output file format (e.g. mydata.xml, .yml etc.). A file name can also contain parameters. You can use this format, "*?base64" (e.g. "file.json?base64" (case sensitive)), as an alternative to FileStorage::BASE64 flag. 
:type filename: str
:param flags: Mode of operation. One of FileStorage::Mode
:type flags: int
:param encoding: Encoding of the file. Note that UTF-16 XML encoding is not supported currently andyou should use 8-bit encoding instead of it. 
:type encoding: str
:rtype: bool
````

````{py:method} isOpened() -> retval
Checks whether the file is opened.




:param self: 
:type self: 
:return: true if the object is associated with the current file and false otherwise. It is agood practice to call this method after you tried to open a file. 
:rtype: bool
````

````{py:method} release() -> None
Closes the file and releases all the memory buffers.


Call this method after all I/O operations with the storage are finished. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} releaseAndGetString() -> retval
Closes the file and releases all the memory buffers.


Call this method after all I/O operations with the storage are finished. If the storage was opened for writing data and FileStorage::WRITE was specified 


:param self: 
:type self: 
:rtype: str
````

````{py:method} getFirstTopLevelNode() -> retval
Returns the first element of the top-level mapping.




:param self: 
:type self: 
:return: The first element of the top-level mapping.
:rtype: FileNode
````

````{py:method} root([, streamidx]) -> retval
Returns the top-level mapping




:param self: 
:type self: 
:param streamidx: Zero-based index of the stream. In most cases there is only one stream in the file.However, YAML supports multiple streams and so there can be several. 
:type streamidx: int
:return: The top-level mapping.
:rtype: FileNode
````

````{py:method} getNode(nodename) -> retval



@overload 


:param self: 
:type self: 
:param nodename: 
:type nodename: str
:rtype: FileNode
````

````{py:method} writeComment(comment[, append]) -> None
Writes a comment.


The function writes a comment into file storage. The comments are skipped when the storage is read. 


:param self: 
:type self: 
:param comment: The written comment, single-line or multi-line
:type comment: str
:param append: If true, the function tries to put the comment at the end of current line.Else if the comment is multi-line, or if it does not fit at the end of the current line, the comment starts a new line. 
:type append: bool
:rtype: None
````

````{py:method} startWriteStruct(name, flags[, typeName]) -> None
Starts to write a nested structure (sequence or a mapping).




:param self: 
:type self: 
:param name: name of the structure. When writing to sequences (a.k.a. "arrays"), pass an empty string.
:type name: str
:param flags: type of the structure (FileNode::MAP or FileNode::SEQ (both with optional FileNode::FLOW)).
:type flags: int
:param typeName: optional name of the type you store. The effect of setting this depends on the storage format.I.e. if the format has a specification for storing type information, this parameter is used. 
:type typeName: str
:rtype: None
````

````{py:method} endWriteStruct() -> None
Finishes writing nested structure (should pair startWriteStruct())




:param self: 
:type self: 
:rtype: None
````

````{py:method} getFormat() -> retval
Returns the current format.




:param self: 
:type self: 
:return: The current format, see FileStorage::Mode
:rtype: int
````


`````


`````{py:class} FlannBasedMatcher




````{py:method} create() -> retval
:classmethod:





:param cls: 
:type cls: 
:rtype: FlannBasedMatcher
````

````{py:method} __init__(self, indexParams: cv2.typing.IndexParams=..., searchParams: cv2.typing.SearchParams=...)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param indexParams: 
:type indexParams: cv2.typing.IndexParams
:param searchParams: 
:type searchParams: cv2.typing.SearchParams
:rtype: None
````


`````


`````{py:class} 





`````


`````{py:class} GArrayDesc





`````


`````{py:class} GArrayT




````{py:method} __init__(self, type: cv2.gapi.ArgType)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param type: 
:type type: cv2.gapi.ArgType
:rtype: None
````

````{py:method} type() -> retval





:param self: 
:type self: 
:rtype: cv2.gapi.ArgType
````


`````


`````{py:class} GCompileArg




````{py:method} __init__(self, arg: GKernelPackage)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arg: 
:type arg: GKernelPackage
:rtype: None
````

````{py:method} __init__(self, arg: cv2.gapi.GNetPackage)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arg: 
:type arg: cv2.gapi.GNetPackage
:rtype: None
````

````{py:method} __init__(self, arg: cv2.gapi.streaming.queue_capacity)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arg: 
:type arg: cv2.gapi.streaming.queue_capacity
:rtype: None
````

````{py:method} __init__(self, arg: cv2.gapi.ot.ObjectTrackerParams)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arg: 
:type arg: cv2.gapi.ot.ObjectTrackerParams
:rtype: None
````


`````


`````{py:class} GComputation




````{py:method} __init__(self, ins: cv2.typing.GProtoInputArgs, outs: cv2.typing.GProtoOutputArgs)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param ins: 
:type ins: cv2.typing.GProtoInputArgs
:param outs: 
:type outs: cv2.typing.GProtoOutputArgs
:rtype: None
````

````{py:method} __init__(self, in_: GMat, out: GMat)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param in_: 
:type in_: GMat
:param out: 
:type out: GMat
:rtype: None
````

````{py:method} __init__(self, in_: GMat, out: GScalar)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param in_: 
:type in_: GMat
:param out: 
:type out: GScalar
:rtype: None
````

````{py:method} __init__(self, in1: GMat, in2: GMat, out: GMat)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param in1: 
:type in1: GMat
:param in2: 
:type in2: GMat
:param out: 
:type out: GMat
:rtype: None
````

````{py:method} compileStreaming(in_metas[, args]) -> retval

Compile the computation for streaming mode.


This method triggers compilation process and produces a new GStreamingCompiled object which then can process video stream data of the given format. Passing a stream in a different format to the compiled computation will generate a run-time exception. 
compileStreaming([, args]) -> retval 
This method triggers compilation process and produces a new GStreamingCompiled object which then can process video stream data in any format. Underlying mechanisms will be adjusted to every new input video stream automatically, but please note that _not all_ existing backends support this (see reshape()). 
compileStreaming(callback[, args]) -> retval 
**See also:** @ref gapi_compile_args
**See also:** @ref gapi_compile_args


:param self: 
:type self: 
:param in_metas: vector of input metadata configuration. Grabmetadata from real data objects (like cv::Mat or cv::Scalar) using cv::descr_of(), or create it on your own. 
:type in_metas: _typing.Sequence[cv2.typing.GMetaArg]
:param args: compilation arguments for this compilationprocess. Compilation arguments directly affect what kind of executable object would be produced, e.g. which kernels (and thus, devices) would be used to execute computation. 
:type args: _typing.Sequence[GCompileArg]
:return: GStreamingCompiled, a streaming-oriented executablecomputation compiled for any input image format. 
:rtype: GStreamingCompiled
````

````{py:method} compileStreaming(in_metas[, args]) -> retval

Compile the computation for streaming mode.


This method triggers compilation process and produces a new GStreamingCompiled object which then can process video stream data of the given format. Passing a stream in a different format to the compiled computation will generate a run-time exception. 
compileStreaming([, args]) -> retval 
This method triggers compilation process and produces a new GStreamingCompiled object which then can process video stream data in any format. Underlying mechanisms will be adjusted to every new input video stream automatically, but please note that _not all_ existing backends support this (see reshape()). 
compileStreaming(callback[, args]) -> retval 
**See also:** @ref gapi_compile_args
**See also:** @ref gapi_compile_args


:param self: 
:type self: 
:param args: compilation arguments for this compilationprocess. Compilation arguments directly affect what kind of executable object would be produced, e.g. which kernels (and thus, devices) would be used to execute computation. 
:type args: _typing.Sequence[GCompileArg]
:param in_metas: vector of input metadata configuration. Grabmetadata from real data objects (like cv::Mat or cv::Scalar) using cv::descr_of(), or create it on your own. 
:type in_metas: 
:return: GStreamingCompiled, a streaming-oriented executablecomputation compiled for any input image format. 
:rtype: GStreamingCompiled
````

````{py:method} compileStreaming(in_metas[, args]) -> retval

Compile the computation for streaming mode.


This method triggers compilation process and produces a new GStreamingCompiled object which then can process video stream data of the given format. Passing a stream in a different format to the compiled computation will generate a run-time exception. 
compileStreaming([, args]) -> retval 
This method triggers compilation process and produces a new GStreamingCompiled object which then can process video stream data in any format. Underlying mechanisms will be adjusted to every new input video stream automatically, but please note that _not all_ existing backends support this (see reshape()). 
compileStreaming(callback[, args]) -> retval 
**See also:** @ref gapi_compile_args
**See also:** @ref gapi_compile_args


:param self: 
:type self: 
:param callback: 
:type callback: cv2.typing.ExtractMetaCallback
:param args: compilation arguments for this compilationprocess. Compilation arguments directly affect what kind of executable object would be produced, e.g. which kernels (and thus, devices) would be used to execute computation. 
:type args: _typing.Sequence[GCompileArg]
:param in_metas: vector of input metadata configuration. Grabmetadata from real data objects (like cv::Mat or cv::Scalar) using cv::descr_of(), or create it on your own. 
:type in_metas: 
:return: GStreamingCompiled, a streaming-oriented executablecomputation compiled for any input image format. 
:rtype: GStreamingCompiled
````

````{py:method} apply(callback[, args]) -> retval
Compile graph on-the-fly and immediately execute it onthe inputs data vectors. 


Number of input/output data objects must match GComputation's protocol, also types of host data objects (cv::Mat, cv::Scalar) must match the shapes of data objects from protocol (cv::GMat, cv::GScalar). If there's a mismatch, a run-time exception will be generated. 
Internally, a cv::GCompiled object is created for the given input format configuration, which then is executed on the input data immediately. cv::GComputation caches compiled objects produced within apply() -- if this method would be called next time with the same input parameters (image formats, image resolution, etc), the underlying compiled graph will be reused without recompilation. If new metadata doesn't match the cached one, the underlying compiled graph is regenerated. 
```{note}
compile() always triggers a compilation process andproduces a new GCompiled object regardless if a similar one has been cached via apply() or not. 
```
**See also:** @ref gapi_data_objects, @ref gapi_compile_args


:param self: 
:type self: 
:param callback: 
:type callback: cv2.typing.ExtractArgsCallback
:param args: a list of compilation arguments to pass to theunderlying compilation process. Don't create GCompileArgs object manually, use cv::compile_args() wrapper instead. 
:type args: _typing.Sequence[GCompileArg]
:param ins: vector of input data to process. Don't createGRunArgs object manually, use cv::gin() wrapper instead. 
:type ins: 
:param outs: vector of output data to fill results in. cv::Matobjects may be empty in this vector, G-API will automatically initialize it with the required format & dimensions. Don't create GRunArgsP object manually, use cv::gout() wrapper instead. 
:type outs: 
:rtype: _typing.Sequence[cv2.typing.GRunArg]
````


`````


`````{py:class} GFTTDetector




````{py:method} create([, maxCorners[, qualityLevel[, minDistance[, blockSize[, useHarrisDetector[, k]]]]]]) -> retval
:classmethod:



create(maxCorners, qualityLevel, minDistance, blockSize, gradiantSize[, useHarrisDetector[, k]]) -> retval 


:param cls: 
:type cls: 
:param maxCorners: 
:type maxCorners: int
:param qualityLevel: 
:type qualityLevel: float
:param minDistance: 
:type minDistance: float
:param blockSize: 
:type blockSize: int
:param useHarrisDetector: 
:type useHarrisDetector: bool
:param k: 
:type k: float
:rtype: GFTTDetector
````

````{py:method} create([, maxCorners[, qualityLevel[, minDistance[, blockSize[, useHarrisDetector[, k]]]]]]) -> retval
:classmethod:



create(maxCorners, qualityLevel, minDistance, blockSize, gradiantSize[, useHarrisDetector[, k]]) -> retval 


:param cls: 
:type cls: 
:param maxCorners: 
:type maxCorners: int
:param qualityLevel: 
:type qualityLevel: float
:param minDistance: 
:type minDistance: float
:param blockSize: 
:type blockSize: int
:param gradiantSize: 
:type gradiantSize: int
:param useHarrisDetector: 
:type useHarrisDetector: bool
:param k: 
:type k: float
:rtype: GFTTDetector
````

````{py:method} setMaxFeatures(maxFeatures) -> None





:param self: 
:type self: 
:param maxFeatures: 
:type maxFeatures: int
:rtype: None
````

````{py:method} getMaxFeatures() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setQualityLevel(qlevel) -> None





:param self: 
:type self: 
:param qlevel: 
:type qlevel: float
:rtype: None
````

````{py:method} getQualityLevel() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMinDistance(minDistance) -> None





:param self: 
:type self: 
:param minDistance: 
:type minDistance: float
:rtype: None
````

````{py:method} getMinDistance() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setBlockSize(blockSize) -> None





:param self: 
:type self: 
:param blockSize: 
:type blockSize: int
:rtype: None
````

````{py:method} getBlockSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setGradientSize(gradientSize_) -> None





:param self: 
:type self: 
:param gradientSize_: 
:type gradientSize_: int
:rtype: None
````

````{py:method} getGradientSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setHarrisDetector(val) -> None





:param self: 
:type self: 
:param val: 
:type val: bool
:rtype: None
````

````{py:method} getHarrisDetector() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} setK(k) -> None





:param self: 
:type self: 
:param k: 
:type k: float
:rtype: None
````

````{py:method} getK() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} GFrame




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````


`````


`````{py:class} GInferInputs




````{py:method} setInput(name, value) -> retval






:param self: 
:type self: 
:param name: 
:type name: str
:param value: 
:type value: GMat
:rtype: GInferInputs
````

````{py:method} setInput(name, value) -> retval






:param self: 
:type self: 
:param name: 
:type name: str
:param value: 
:type value: GFrame
:rtype: GInferInputs
````

````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````


`````


`````{py:class} GInferListInputs




````{py:method} setInput(name, value) -> retval






:param self: 
:type self: 
:param name: 
:type name: str
:param value: 
:type value: GArrayT
:rtype: GInferListInputs
````

````{py:method} setInput(name, value) -> retval






:param self: 
:type self: 
:param name: 
:type name: str
:param value: 
:type value: GArrayT
:rtype: GInferListInputs
````

````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````


`````


`````{py:class} GInferListOutputs




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} at(name) -> retval





:param self: 
:type self: 
:param name: 
:type name: str
:rtype: GArrayT
````


`````


`````{py:class} GInferOutputs




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} at(name) -> retval





:param self: 
:type self: 
:param name: 
:type name: str
:rtype: GMat
````


`````


`````{py:class} GKernelPackage




````{py:method} size() -> retval
Returns total number of kernelsin the package (across all backends included) 




:param self: 
:type self: 
:return: a number of kernels in the package
:rtype: int
````


`````


`````{py:class} GMat




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````


`````


`````{py:class} GMatDesc




````{py:method} depth






:param self: 
:type self: 
:rtype: int
````

````{py:method} chan






:param self: 
:type self: 
:rtype: int
````

````{py:method} size






:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} planar






:param self: 
:type self: 
:rtype: bool
````

````{py:method} dims






:param self: 
:type self: 
:rtype: _typing.Sequence[int]
````

````{py:method} __init__(self, d: int, c: int, s: cv2.typing.Size, p: bool=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param d: 
:type d: int
:param c: 
:type c: int
:param s: 
:type s: cv2.typing.Size
:param p: 
:type p: bool
:rtype: None
````

````{py:method} __init__(self, d: int, dd: _typing.Sequence[int])




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param d: 
:type d: int
:param dd: 
:type dd: _typing.Sequence[int]
:rtype: None
````

````{py:method} __init__(self, d: int, dd: _typing.Sequence[int])




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param d: 
:type d: int
:param dd: 
:type dd: _typing.Sequence[int]
:rtype: None
````

````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} withSizeDelta(delta) -> retval




withSizeDelta(dx, dy) -> retval 


:param self: 
:type self: 
:param delta: 
:type delta: cv2.typing.Size
:rtype: GMatDesc
````

````{py:method} withSizeDelta(delta) -> retval




withSizeDelta(dx, dy) -> retval 


:param self: 
:type self: 
:param dx: 
:type dx: int
:param dy: 
:type dy: int
:rtype: GMatDesc
````

````{py:method} asPlanar() -> retval




asPlanar(planes) -> retval 


:param self: 
:type self: 
:rtype: GMatDesc
````

````{py:method} asPlanar() -> retval




asPlanar(planes) -> retval 


:param self: 
:type self: 
:param planes: 
:type planes: int
:rtype: GMatDesc
````

````{py:method} withSize(sz) -> retval





:param self: 
:type self: 
:param sz: 
:type sz: cv2.typing.Size
:rtype: GMatDesc
````

````{py:method} withDepth(ddepth) -> retval





:param self: 
:type self: 
:param ddepth: 
:type ddepth: int
:rtype: GMatDesc
````

````{py:method} withType(ddepth, dchan) -> retval





:param self: 
:type self: 
:param ddepth: 
:type ddepth: int
:param dchan: 
:type dchan: int
:rtype: GMatDesc
````

````{py:method} asInterleaved() -> retval





:param self: 
:type self: 
:rtype: GMatDesc
````


`````


`````{py:class} 





`````


`````{py:class} GOpaqueDesc





`````


`````{py:class} GOpaqueT




````{py:method} __init__(self, type: cv2.gapi.ArgType)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param type: 
:type type: cv2.gapi.ArgType
:rtype: None
````

````{py:method} type() -> retval





:param self: 
:type self: 
:rtype: cv2.gapi.ArgType
````


`````


`````{py:class} GScalar




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, s: cv2.typing.Scalar)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param s: 
:type s: cv2.typing.Scalar
:rtype: None
````


`````


`````{py:class} GScalarDesc





`````


`````{py:class} GStreamingCompiled




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} setSource(callback) -> None
Specify the input data to GStreamingCompiled forprocessing, a generic version. 


Use gin() to create an input parameter vector. 
Input vectors must have the same number of elements as defined in the cv::GComputation protocol (at the moment of its construction). Shapes of elements also must conform to protocol (e.g. cv::Mat needs to be passed where cv::GMat has been declared as input, and so on). Run-time exception is generated on type mismatch. 
In contrast with regular GCompiled, user can also pass an object of type GVideoCapture for a GMat parameter of the parent GComputation.  The compiled pipeline will start fetching data from that GVideoCapture and feeding it into the pipeline. Pipeline stops when a GVideoCapture marks end of the stream (or when stop() is called). 
Passing a regular Mat for a GMat parameter makes it "infinite" source -- pipeline may run forever feeding with this Mat until stopped explicitly. 
Currently only a single GVideoCapture is supported as input. If the parent GComputation is declared with multiple input GMat's, one of those can be specified as GVideoCapture but all others must be regular Mat objects. 
Throws if pipeline is already running. Use stop() and then setSource() to run the graph on a new video stream. 
```{note}
This method is not thread-safe (with respect to the userside) at the moment. Protect the access if start()/stop()/setSource() may be called on the same object in multiple threads in your application. 
```
**See also:** gin


:param self: 
:type self: 
:param callback: 
:type callback: cv2.typing.ExtractArgsCallback
:param ins: vector of inputs to process.
:type ins: 
:rtype: None
````

````{py:method} start() -> None
Start the pipeline execution.


Use pull()/try_pull() to obtain data. Throws an exception if a video source was not specified. 
setSource() must be called first, even if the pipeline has been working already and then stopped (explicitly via stop() or due stream completion) 
```{note}
This method is not thread-safe (with respect to the userside) at the moment. Protect the access if start()/stop()/setSource() may be called on the same object in multiple threads in your application. 
```


:param self: 
:type self: 
:rtype: None
````

````{py:method} pull() -> retval
Get the next processed frame from the pipeline.


Use gout() to create an output parameter vector. 
Output vectors must have the same number of elements as defined in the cv::GComputation protocol (at the moment of its construction). Shapes of elements also must conform to protocol (e.g. cv::Mat needs to be passed where cv::GMat has been declared as output, and so on). Run-time exception is generated on type mismatch. 
This method writes new data into objects passed via output vector.  If there is no data ready yet, this method blocks. Use try_pull() if you need a non-blocking version. 


:param self: 
:type self: 
:param outs: vector of output parameters to obtain.
:type outs: 
:return: true if next result has been obtained,false marks end of the stream. 
:rtype: tuple[bool, _typing.Sequence[cv2.typing.GRunArg] | _typing.Sequence[cv2.typing.GOptRunArg]]
````

````{py:method} stop() -> None
Stop (abort) processing the pipeline.


Note - it is not pause but a complete stop. Calling start() will cause G-API to start processing the stream from the early beginning. 
Throws if the pipeline is not running. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} running() -> retval
Test if the pipeline is running.


```{note}
This method is not thread-safe (with respect to the userside) at the moment. Protect the access if start()/stop()/setSource() may be called on the same object in multiple threads in your application. 
```


:param self: 
:type self: 
:return: true if the current stream is not over yet.
:rtype: bool
````


`````


`````{py:class} GeneralizedHough




````{py:method} setTemplate(templ[, templCenter]) -> None




setTemplate(edges, dx, dy[, templCenter]) -> None 


:param self: 
:type self: 
:param templ: 
:type templ: cv2.typing.MatLike
:param templCenter: 
:type templCenter: cv2.typing.Point
:rtype: None
````

````{py:method} setTemplate(templ[, templCenter]) -> None




setTemplate(edges, dx, dy[, templCenter]) -> None 


:param self: 
:type self: 
:param templ: 
:type templ: UMat
:param templCenter: 
:type templCenter: cv2.typing.Point
:rtype: None
````

````{py:method} setTemplate(templ[, templCenter]) -> None




setTemplate(edges, dx, dy[, templCenter]) -> None 


:param self: 
:type self: 
:param edges: 
:type edges: cv2.typing.MatLike
:param dx: 
:type dx: cv2.typing.MatLike
:param dy: 
:type dy: cv2.typing.MatLike
:param templCenter: 
:type templCenter: cv2.typing.Point
:rtype: None
````

````{py:method} setTemplate(templ[, templCenter]) -> None




setTemplate(edges, dx, dy[, templCenter]) -> None 


:param self: 
:type self: 
:param edges: 
:type edges: UMat
:param dx: 
:type dx: UMat
:param dy: 
:type dy: UMat
:param templCenter: 
:type templCenter: cv2.typing.Point
:rtype: None
````

````{py:method} detect(image[, positions[, votes]]) -> positions, votes




detect(edges, dx, dy[, positions[, votes]]) -> positions, votes 


:param self: 
:type self: 
:param image: 
:type image: cv2.typing.MatLike
:param positions: 
:type positions: cv2.typing.MatLike | None
:param votes: 
:type votes: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} detect(image[, positions[, votes]]) -> positions, votes




detect(edges, dx, dy[, positions[, votes]]) -> positions, votes 


:param self: 
:type self: 
:param image: 
:type image: UMat
:param positions: 
:type positions: UMat | None
:param votes: 
:type votes: UMat | None
:rtype: tuple[UMat, UMat]
````

````{py:method} detect(image[, positions[, votes]]) -> positions, votes




detect(edges, dx, dy[, positions[, votes]]) -> positions, votes 


:param self: 
:type self: 
:param edges: 
:type edges: cv2.typing.MatLike
:param dx: 
:type dx: cv2.typing.MatLike
:param dy: 
:type dy: cv2.typing.MatLike
:param positions: 
:type positions: cv2.typing.MatLike | None
:param votes: 
:type votes: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} detect(image[, positions[, votes]]) -> positions, votes




detect(edges, dx, dy[, positions[, votes]]) -> positions, votes 


:param self: 
:type self: 
:param edges: 
:type edges: UMat
:param dx: 
:type dx: UMat
:param dy: 
:type dy: UMat
:param positions: 
:type positions: UMat | None
:param votes: 
:type votes: UMat | None
:rtype: tuple[UMat, UMat]
````

````{py:method} setCannyLowThresh(cannyLowThresh) -> None





:param self: 
:type self: 
:param cannyLowThresh: 
:type cannyLowThresh: int
:rtype: None
````

````{py:method} getCannyLowThresh() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setCannyHighThresh(cannyHighThresh) -> None





:param self: 
:type self: 
:param cannyHighThresh: 
:type cannyHighThresh: int
:rtype: None
````

````{py:method} getCannyHighThresh() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMinDist(minDist) -> None





:param self: 
:type self: 
:param minDist: 
:type minDist: float
:rtype: None
````

````{py:method} getMinDist() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setDp(dp) -> None





:param self: 
:type self: 
:param dp: 
:type dp: float
:rtype: None
````

````{py:method} getDp() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMaxBufferSize(maxBufferSize) -> None





:param self: 
:type self: 
:param maxBufferSize: 
:type maxBufferSize: int
:rtype: None
````

````{py:method} getMaxBufferSize() -> retval





:param self: 
:type self: 
:rtype: int
````


`````


`````{py:class} GeneralizedHoughBallard




````{py:method} setLevels(levels) -> None





:param self: 
:type self: 
:param levels: 
:type levels: int
:rtype: None
````

````{py:method} getLevels() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setVotesThreshold(votesThreshold) -> None





:param self: 
:type self: 
:param votesThreshold: 
:type votesThreshold: int
:rtype: None
````

````{py:method} getVotesThreshold() -> retval





:param self: 
:type self: 
:rtype: int
````


`````


`````{py:class} GeneralizedHoughGuil




````{py:method} setXi(xi) -> None





:param self: 
:type self: 
:param xi: 
:type xi: float
:rtype: None
````

````{py:method} getXi() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setLevels(levels) -> None





:param self: 
:type self: 
:param levels: 
:type levels: int
:rtype: None
````

````{py:method} getLevels() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setAngleEpsilon(angleEpsilon) -> None





:param self: 
:type self: 
:param angleEpsilon: 
:type angleEpsilon: float
:rtype: None
````

````{py:method} getAngleEpsilon() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMinAngle(minAngle) -> None





:param self: 
:type self: 
:param minAngle: 
:type minAngle: float
:rtype: None
````

````{py:method} getMinAngle() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMaxAngle(maxAngle) -> None





:param self: 
:type self: 
:param maxAngle: 
:type maxAngle: float
:rtype: None
````

````{py:method} getMaxAngle() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setAngleStep(angleStep) -> None





:param self: 
:type self: 
:param angleStep: 
:type angleStep: float
:rtype: None
````

````{py:method} getAngleStep() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setAngleThresh(angleThresh) -> None





:param self: 
:type self: 
:param angleThresh: 
:type angleThresh: int
:rtype: None
````

````{py:method} getAngleThresh() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMinScale(minScale) -> None





:param self: 
:type self: 
:param minScale: 
:type minScale: float
:rtype: None
````

````{py:method} getMinScale() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMaxScale(maxScale) -> None





:param self: 
:type self: 
:param maxScale: 
:type maxScale: float
:rtype: None
````

````{py:method} getMaxScale() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setScaleStep(scaleStep) -> None





:param self: 
:type self: 
:param scaleStep: 
:type scaleStep: float
:rtype: None
````

````{py:method} getScaleStep() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setScaleThresh(scaleThresh) -> None





:param self: 
:type self: 
:param scaleThresh: 
:type scaleThresh: int
:rtype: None
````

````{py:method} getScaleThresh() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setPosThresh(posThresh) -> None





:param self: 
:type self: 
:param posThresh: 
:type posThresh: int
:rtype: None
````

````{py:method} getPosThresh() -> retval





:param self: 
:type self: 
:rtype: int
````


`````


`````{py:class} GraphicalCodeDetector




````{py:method} detect(img[, points]) -> retval, points

Detects graphical code in image and returns the quadrangle containing the code.




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing (or not) graphical code.
:type img: cv2.typing.MatLike
:param points: Output vector of vertices of the minimum-area quadrangle containing the code.
:type points: cv2.typing.MatLike | None
:rtype: tuple[bool, cv2.typing.MatLike]
````

````{py:method} detect(img[, points]) -> retval, points

Detects graphical code in image and returns the quadrangle containing the code.




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing (or not) graphical code.
:type img: UMat
:param points: Output vector of vertices of the minimum-area quadrangle containing the code.
:type points: UMat | None
:rtype: tuple[bool, UMat]
````

````{py:method} decode(img, points[, straight_code]) -> retval, straight_code

Decodes graphical code in image once it's found by the detect() method.


Returns UTF8-encoded output string or empty string if the code cannot be decoded. 


:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing graphical code.
:type img: cv2.typing.MatLike
:param points: Quadrangle vertices found by detect() method (or some other algorithm).
:type points: cv2.typing.MatLike
:param straight_code: The optional output image containing binarized code, will be empty if not found.
:type straight_code: cv2.typing.MatLike | None
:rtype: tuple[str, cv2.typing.MatLike]
````

````{py:method} decode(img, points[, straight_code]) -> retval, straight_code

Decodes graphical code in image once it's found by the detect() method.


Returns UTF8-encoded output string or empty string if the code cannot be decoded. 


:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing graphical code.
:type img: UMat
:param points: Quadrangle vertices found by detect() method (or some other algorithm).
:type points: UMat
:param straight_code: The optional output image containing binarized code, will be empty if not found.
:type straight_code: UMat | None
:rtype: tuple[str, UMat]
````

````{py:method} detectAndDecode(img[, points[, straight_code]]) -> retval, points, straight_code

Both detects and decodes graphical code




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing graphical code.
:type img: cv2.typing.MatLike
:param points: optional output array of vertices of the found graphical code quadrangle, will be empty if not found.
:type points: cv2.typing.MatLike | None
:param straight_code: The optional output image containing binarized code
:type straight_code: cv2.typing.MatLike | None
:rtype: tuple[str, cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} detectAndDecode(img[, points[, straight_code]]) -> retval, points, straight_code

Both detects and decodes graphical code




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing graphical code.
:type img: UMat
:param points: optional output array of vertices of the found graphical code quadrangle, will be empty if not found.
:type points: UMat | None
:param straight_code: The optional output image containing binarized code
:type straight_code: UMat | None
:rtype: tuple[str, UMat, UMat]
````

````{py:method} detectMulti(img[, points]) -> retval, points

Detects graphical codes in image and returns the vector of the quadrangles containing the codes.




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing (or not) graphical codes.
:type img: cv2.typing.MatLike
:param points: Output vector of vector of vertices of the minimum-area quadrangle containing the codes.
:type points: cv2.typing.MatLike | None
:rtype: tuple[bool, cv2.typing.MatLike]
````

````{py:method} detectMulti(img[, points]) -> retval, points

Detects graphical codes in image and returns the vector of the quadrangles containing the codes.




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing (or not) graphical codes.
:type img: UMat
:param points: Output vector of vector of vertices of the minimum-area quadrangle containing the codes.
:type points: UMat | None
:rtype: tuple[bool, UMat]
````

````{py:method} decodeMulti(img, points[, straight_code]) -> retval, decoded_info, straight_code

Decodes graphical codes in image once it's found by the detect() method.




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing graphical codes.
:type img: cv2.typing.MatLike
:param points: vector of Quadrangle vertices found by detect() method (or some other algorithm).
:type points: cv2.typing.MatLike
:param straight_code: The optional output vector of images containing binarized codes
:type straight_code: _typing.Sequence[cv2.typing.MatLike] | None
:param decoded_info: UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
:type decoded_info: 
:rtype: tuple[bool, _typing.Sequence[str], _typing.Sequence[cv2.typing.MatLike]]
````

````{py:method} decodeMulti(img, points[, straight_code]) -> retval, decoded_info, straight_code

Decodes graphical codes in image once it's found by the detect() method.




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing graphical codes.
:type img: UMat
:param points: vector of Quadrangle vertices found by detect() method (or some other algorithm).
:type points: UMat
:param straight_code: The optional output vector of images containing binarized codes
:type straight_code: _typing.Sequence[UMat] | None
:param decoded_info: UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
:type decoded_info: 
:rtype: tuple[bool, _typing.Sequence[str], _typing.Sequence[UMat]]
````

````{py:method} detectAndDecodeMulti(img[, points[, straight_code]]) -> retval, decoded_info, points, straight_code

Both detects and decodes graphical codes




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing graphical codes.
:type img: cv2.typing.MatLike
:param points: optional output vector of vertices of the found graphical code quadrangles. Will be empty if not found.
:type points: cv2.typing.MatLike | None
:param straight_code: The optional vector of images containing binarized codes
:type straight_code: _typing.Sequence[cv2.typing.MatLike] | None
:param decoded_info: UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
:type decoded_info: 
:rtype: tuple[bool, _typing.Sequence[str], cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike]]
````

````{py:method} detectAndDecodeMulti(img[, points[, straight_code]]) -> retval, decoded_info, points, straight_code

Both detects and decodes graphical codes




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing graphical codes.
:type img: UMat
:param points: optional output vector of vertices of the found graphical code quadrangles. Will be empty if not found.
:type points: UMat | None
:param straight_code: The optional vector of images containing binarized codes
:type straight_code: _typing.Sequence[UMat] | None
:param decoded_info: UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
:type decoded_info: 
:rtype: tuple[bool, _typing.Sequence[str], UMat, _typing.Sequence[UMat]]
````


`````


`````{py:class} HOGDescriptor




````{py:method} winSize






:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} blockSize






:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} blockStride






:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} cellSize






:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} nbins






:param self: 
:type self: 
:rtype: int
````

````{py:method} derivAperture






:param self: 
:type self: 
:rtype: int
````

````{py:method} winSigma






:param self: 
:type self: 
:rtype: float
````

````{py:method} histogramNormType






:param self: 
:type self: 
:rtype: HOGDescriptor_HistogramNormType
````

````{py:method} L2HysThreshold






:param self: 
:type self: 
:rtype: float
````

````{py:method} gammaCorrection






:param self: 
:type self: 
:rtype: bool
````

````{py:method} svmDetector






:param self: 
:type self: 
:rtype: _typing.Sequence[float]
````

````{py:method} nlevels






:param self: 
:type self: 
:rtype: int
````

````{py:method} signedGradient






:param self: 
:type self: 
:rtype: bool
````

````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, _winSize: cv2.typing.Size, _blockSize: cv2.typing.Size, _blockStride: cv2.typing.Size, _cellSize: cv2.typing.Size, _nbins: int, _derivAperture: int=..., _winSigma: float=..., _histogramNormType: HOGDescriptor_HistogramNormType=..., _L2HysThreshold: float=..., _gammaCorrection: bool=..., _nlevels: int=..., _signedGradient: bool=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param _winSize: 
:type _winSize: cv2.typing.Size
:param _blockSize: 
:type _blockSize: cv2.typing.Size
:param _blockStride: 
:type _blockStride: cv2.typing.Size
:param _cellSize: 
:type _cellSize: cv2.typing.Size
:param _nbins: 
:type _nbins: int
:param _derivAperture: 
:type _derivAperture: int
:param _winSigma: 
:type _winSigma: float
:param _histogramNormType: 
:type _histogramNormType: HOGDescriptor_HistogramNormType
:param _L2HysThreshold: 
:type _L2HysThreshold: float
:param _gammaCorrection: 
:type _gammaCorrection: bool
:param _nlevels: 
:type _nlevels: int
:param _signedGradient: 
:type _signedGradient: bool
:rtype: None
````

````{py:method} __init__(self, filename: str)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:rtype: None
````

````{py:method} setSVMDetector(svmdetector) -> None

Sets coefficients for the linear SVM classifier.




:param self: 
:type self: 
:param svmdetector: coefficients for the linear SVM classifier.
:type svmdetector: cv2.typing.MatLike
:rtype: None
````

````{py:method} setSVMDetector(svmdetector) -> None

Sets coefficients for the linear SVM classifier.




:param self: 
:type self: 
:param svmdetector: coefficients for the linear SVM classifier.
:type svmdetector: UMat
:rtype: None
````

````{py:method} compute(img[, winStride[, padding[, locations]]]) -> descriptors

Computes HOG descriptors of given image.




:param self: 
:type self: 
:param img: Matrix of the type CV_8U containing an image where HOG features will be calculated.
:type img: cv2.typing.MatLike
:param winStride: Window stride. It must be a multiple of block stride.
:type winStride: cv2.typing.Size
:param padding: Padding
:type padding: cv2.typing.Size
:param locations: Vector of Point
:type locations: _typing.Sequence[cv2.typing.Point]
:param descriptors: Matrix of the type CV_32F
:type descriptors: 
:rtype: _typing.Sequence[float]
````

````{py:method} compute(img[, winStride[, padding[, locations]]]) -> descriptors

Computes HOG descriptors of given image.




:param self: 
:type self: 
:param img: Matrix of the type CV_8U containing an image where HOG features will be calculated.
:type img: UMat
:param winStride: Window stride. It must be a multiple of block stride.
:type winStride: cv2.typing.Size
:param padding: Padding
:type padding: cv2.typing.Size
:param locations: Vector of Point
:type locations: _typing.Sequence[cv2.typing.Point]
:param descriptors: Matrix of the type CV_32F
:type descriptors: 
:rtype: _typing.Sequence[float]
````

````{py:method} detect(img[, hitThreshold[, winStride[, padding[, searchLocations]]]]) -> foundLocations, weights

Performs object detection without a multi-scale window.




:param self: 
:type self: 
:param img: Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
:type img: cv2.typing.MatLike
:param hitThreshold: Threshold for the distance between features and SVM classifying plane.Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here. 
:type hitThreshold: float
:param winStride: Window stride. It must be a multiple of block stride.
:type winStride: cv2.typing.Size
:param padding: Padding
:type padding: cv2.typing.Size
:param searchLocations: Vector of Point includes set of requested locations to be evaluated.
:type searchLocations: _typing.Sequence[cv2.typing.Point]
:param foundLocations: Vector of point where each point contains left-top corner point of detected object boundaries.
:type foundLocations: 
:param weights: Vector that will contain confidence values for each detected object.
:type weights: 
:rtype: tuple[_typing.Sequence[cv2.typing.Point], _typing.Sequence[float]]
````

````{py:method} detect(img[, hitThreshold[, winStride[, padding[, searchLocations]]]]) -> foundLocations, weights

Performs object detection without a multi-scale window.




:param self: 
:type self: 
:param img: Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
:type img: UMat
:param hitThreshold: Threshold for the distance between features and SVM classifying plane.Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here. 
:type hitThreshold: float
:param winStride: Window stride. It must be a multiple of block stride.
:type winStride: cv2.typing.Size
:param padding: Padding
:type padding: cv2.typing.Size
:param searchLocations: Vector of Point includes set of requested locations to be evaluated.
:type searchLocations: _typing.Sequence[cv2.typing.Point]
:param foundLocations: Vector of point where each point contains left-top corner point of detected object boundaries.
:type foundLocations: 
:param weights: Vector that will contain confidence values for each detected object.
:type weights: 
:rtype: tuple[_typing.Sequence[cv2.typing.Point], _typing.Sequence[float]]
````

````{py:method} detectMultiScale(img[, hitThreshold[, winStride[, padding[, scale[, groupThreshold[, useMeanshiftGrouping]]]]]]) -> foundLocations, foundWeights

Detects objects of different sizes in the input image. The detected objects are returned as a listof rectangles. 




:param self: 
:type self: 
:param img: Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
:type img: cv2.typing.MatLike
:param hitThreshold: Threshold for the distance between features and SVM classifying plane.Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here. 
:type hitThreshold: float
:param winStride: Window stride. It must be a multiple of block stride.
:type winStride: cv2.typing.Size
:param padding: Padding
:type padding: cv2.typing.Size
:param scale: Coefficient of the detection window increase.
:type scale: float
:param groupThreshold: Coefficient to regulate the similarity threshold. When detected, some objects can be coveredby many rectangles. 0 means not to perform grouping. 
:type groupThreshold: float
:param useMeanshiftGrouping: indicates grouping algorithm
:type useMeanshiftGrouping: bool
:param foundLocations: Vector of rectangles where each rectangle contains the detected object.
:type foundLocations: 
:param foundWeights: Vector that will contain confidence values for each detected object.
:type foundWeights: 
:rtype: tuple[_typing.Sequence[cv2.typing.Rect], _typing.Sequence[float]]
````

````{py:method} detectMultiScale(img[, hitThreshold[, winStride[, padding[, scale[, groupThreshold[, useMeanshiftGrouping]]]]]]) -> foundLocations, foundWeights

Detects objects of different sizes in the input image. The detected objects are returned as a listof rectangles. 




:param self: 
:type self: 
:param img: Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
:type img: UMat
:param hitThreshold: Threshold for the distance between features and SVM classifying plane.Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here. 
:type hitThreshold: float
:param winStride: Window stride. It must be a multiple of block stride.
:type winStride: cv2.typing.Size
:param padding: Padding
:type padding: cv2.typing.Size
:param scale: Coefficient of the detection window increase.
:type scale: float
:param groupThreshold: Coefficient to regulate the similarity threshold. When detected, some objects can be coveredby many rectangles. 0 means not to perform grouping. 
:type groupThreshold: float
:param useMeanshiftGrouping: indicates grouping algorithm
:type useMeanshiftGrouping: bool
:param foundLocations: Vector of rectangles where each rectangle contains the detected object.
:type foundLocations: 
:param foundWeights: Vector that will contain confidence values for each detected object.
:type foundWeights: 
:rtype: tuple[_typing.Sequence[cv2.typing.Rect], _typing.Sequence[float]]
````

````{py:method} computeGradient(img, grad, angleOfs[, paddingTL[, paddingBR]]) -> grad, angleOfs

 Computes gradients and quantized gradient orientations.




:param self: 
:type self: 
:param img: Matrix contains the image to be computed
:type img: cv2.typing.MatLike
:param grad: Matrix of type CV_32FC2 contains computed gradients
:type grad: cv2.typing.MatLike
:param angleOfs: Matrix of type CV_8UC2 contains quantized gradient orientations
:type angleOfs: cv2.typing.MatLike
:param paddingTL: Padding from top-left
:type paddingTL: cv2.typing.Size
:param paddingBR: Padding from bottom-right
:type paddingBR: cv2.typing.Size
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} computeGradient(img, grad, angleOfs[, paddingTL[, paddingBR]]) -> grad, angleOfs

 Computes gradients and quantized gradient orientations.




:param self: 
:type self: 
:param img: Matrix contains the image to be computed
:type img: UMat
:param grad: Matrix of type CV_32FC2 contains computed gradients
:type grad: UMat
:param angleOfs: Matrix of type CV_8UC2 contains quantized gradient orientations
:type angleOfs: UMat
:param paddingTL: Padding from top-left
:type paddingTL: cv2.typing.Size
:param paddingBR: Padding from bottom-right
:type paddingBR: cv2.typing.Size
:rtype: tuple[UMat, UMat]
````

````{py:method} getDescriptorSize() -> retval
Returns the number of coefficients required for the classification.




:param self: 
:type self: 
:rtype: int
````

````{py:method} checkDetectorSize() -> retval
Checks if detector size equal to descriptor size.




:param self: 
:type self: 
:rtype: bool
````

````{py:method} getWinSigma() -> retval
Returns winSigma value




:param self: 
:type self: 
:rtype: float
````

````{py:method} load(filename[, objname]) -> retval
loads HOGDescriptor parameters and coefficients for the linear SVM classifier from a file




:param self: 
:type self: 
:param filename: Name of the file to read.
:type filename: str
:param objname: The optional name of the node to read (if empty, the first top-level node will be used).
:type objname: str
:rtype: bool
````

````{py:method} save(filename[, objname]) -> None
saves HOGDescriptor parameters and coefficients for the linear SVM classifier to a file




:param self: 
:type self: 
:param filename: File name
:type filename: str
:param objname: Object name
:type objname: str
:rtype: None
````

````{py:method} getDefaultPeopleDetector() -> retval
:staticmethod:
Returns coefficients of the classifier trained for people detection (for 64x128 windows).




:rtype: _typing.Sequence[float]
````

````{py:method} getDaimlerPeopleDetector() -> retval
:staticmethod:
Returns coefficients of the classifier trained for people detection (for 48x96 windows).




:rtype: _typing.Sequence[float]
````


`````


`````{py:class} KAZE




````{py:method} create([, extended[, upright[, threshold[, nOctaves[, nOctaveLayers[, diffusivity]]]]]]) -> retval
:classmethod:
The KAZE constructor




:param cls: 
:type cls: 
:param extended: Set to enable extraction of extended (128-byte) descriptor.
:type extended: bool
:param upright: Set to enable use of upright descriptors (non rotation-invariant).
:type upright: bool
:param threshold: Detector response threshold to accept point
:type threshold: float
:param nOctaves: Maximum octave evolution of the image
:type nOctaves: int
:param nOctaveLayers: Default number of sublevels per scale level
:type nOctaveLayers: int
:param diffusivity: Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT orDIFF_CHARBONNIER 
:type diffusivity: KAZE_DiffusivityType
:rtype: KAZE
````

````{py:method} setExtended(extended) -> None





:param self: 
:type self: 
:param extended: 
:type extended: bool
:rtype: None
````

````{py:method} getExtended() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} setUpright(upright) -> None





:param self: 
:type self: 
:param upright: 
:type upright: bool
:rtype: None
````

````{py:method} getUpright() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} setThreshold(threshold) -> None





:param self: 
:type self: 
:param threshold: 
:type threshold: float
:rtype: None
````

````{py:method} getThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setNOctaves(octaves) -> None





:param self: 
:type self: 
:param octaves: 
:type octaves: int
:rtype: None
````

````{py:method} getNOctaves() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setNOctaveLayers(octaveLayers) -> None





:param self: 
:type self: 
:param octaveLayers: 
:type octaveLayers: int
:rtype: None
````

````{py:method} getNOctaveLayers() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setDiffusivity(diff) -> None





:param self: 
:type self: 
:param diff: 
:type diff: KAZE_DiffusivityType
:rtype: None
````

````{py:method} getDiffusivity() -> retval





:param self: 
:type self: 
:rtype: KAZE_DiffusivityType
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} KalmanFilter




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, dynamParams: int, measureParams: int, controlParams: int=..., type: int=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param dynamParams: 
:type dynamParams: int
:param measureParams: 
:type measureParams: int
:param controlParams: 
:type controlParams: int
:param type: 
:type type: int
:rtype: None
````

````{py:method} predict([, control]) -> retval
Computes a predicted state.




:param self: 
:type self: 
:param control: The optional input control
:type control: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} correct(measurement) -> retval
Updates the predicted state from the measurement.




:param self: 
:type self: 
:param measurement: The measured system parameters
:type measurement: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````

```{py:attribute} statePre
:type: cv2.typing.MatLike
```

```{py:attribute} statePost
:type: cv2.typing.MatLike
```

```{py:attribute} transitionMatrix
:type: cv2.typing.MatLike
```

```{py:attribute} controlMatrix
:type: cv2.typing.MatLike
```

```{py:attribute} measurementMatrix
:type: cv2.typing.MatLike
```

```{py:attribute} processNoiseCov
:type: cv2.typing.MatLike
```

```{py:attribute} measurementNoiseCov
:type: cv2.typing.MatLike
```

```{py:attribute} errorCovPre
:type: cv2.typing.MatLike
```

```{py:attribute} gain
:type: cv2.typing.MatLike
```

```{py:attribute} errorCovPost
:type: cv2.typing.MatLike
```


`````


`````{py:class} KeyPoint




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, x: float, y: float, size: float, angle: float=..., response: float=..., octave: int=..., class_id: int=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param x: 
:type x: float
:param y: 
:type y: float
:param size: 
:type size: float
:param angle: 
:type angle: float
:param response: 
:type response: float
:param octave: 
:type octave: int
:param class_id: 
:type class_id: int
:rtype: None
````

````{py:method} convert(keypoints[, keypointIndexes]) -> points2f
:staticmethod:



This method converts vector of keypoints to vector of points or the reverse, where each keypoint is assigned the same size and the same orientation. 
convert(points2f[, size[, response[, octave[, class_id]]]]) -> keypoints @overload 


:param keypoints: Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB
:type keypoints: _typing.Sequence[KeyPoint]
:param keypointIndexes: Array of indexes of keypoints to be converted to points. (Acts like a mask toconvert only specified keypoints) 
:type keypointIndexes: _typing.Sequence[int]
:param points2f: Array of (x,y) coordinates of each keypoint
:type points2f: 
:param size: keypoint diameter
:type size: 
:param response: keypoint detector response on the keypoint (that is, strength of the keypoint)
:type response: 
:param octave: pyramid octave in which the keypoint has been detected
:type octave: 
:param class_id: object id
:type class_id: 
:rtype: _typing.Sequence[cv2.typing.Point2f]
````

````{py:method} convert(keypoints[, keypointIndexes]) -> points2f
:staticmethod:



This method converts vector of keypoints to vector of points or the reverse, where each keypoint is assigned the same size and the same orientation. 
convert(points2f[, size[, response[, octave[, class_id]]]]) -> keypoints @overload 


:param points2f: Array of (x,y) coordinates of each keypoint
:type points2f: _typing.Sequence[cv2.typing.Point2f]
:param size: keypoint diameter
:type size: float
:param response: keypoint detector response on the keypoint (that is, strength of the keypoint)
:type response: float
:param octave: pyramid octave in which the keypoint has been detected
:type octave: int
:param class_id: object id
:type class_id: int
:param keypoints: Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB
:type keypoints: 
:param keypointIndexes: Array of indexes of keypoints to be converted to points. (Acts like a mask toconvert only specified keypoints) 
:type keypointIndexes: 
:rtype: _typing.Sequence[KeyPoint]
````

````{py:method} overlap(kp1, kp2) -> retval
:staticmethod:



This method computes overlap for pair of keypoints. Overlap is the ratio between area of keypoint regions' intersection and area of keypoint regions' union (considering keypoint region as circle). If they don't overlap, we get zero. If they coincide at same location with same size, we get 1. 


:param kp1: First keypoint
:type kp1: KeyPoint
:param kp2: Second keypoint
:type kp2: KeyPoint
:rtype: float
````

```{py:attribute} pt
:type: cv2.typing.Point2f
```

```{py:attribute} size
:type: float
```

```{py:attribute} angle
:type: float
```

```{py:attribute} response
:type: float
```

```{py:attribute} octave
:type: int
```

```{py:attribute} class_id
:type: int
```


`````


`````{py:class} LineSegmentDetector




````{py:method} detect(image[, lines[, width[, prec[, nfa]]]]) -> lines, width, prec, nfa

Finds lines in the input image.


This is the output of the default parameters of the algorithm on the above shown image. 
![image](pics/building_lsd.png) 


:param self: 
:type self: 
:param image: A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use:`lsd_ptr-\>detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);` 
:type image: cv2.typing.MatLike
:param lines: A vector of Vec4f elements specifying the beginning and ending point of a line. WhereVec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly oriented depending on the gradient. 
:type lines: cv2.typing.MatLike | None
:param width: Vector of widths of the regions, where the lines are found. E.g. Width of line.
:type width: cv2.typing.MatLike | None
:param prec: Vector of precisions with which the lines are found.
:type prec: cv2.typing.MatLike | None
:param nfa: Vector containing number of false alarms in the line region, with precision of 10%. Thebigger the value, logarithmically better the detection. - -1 corresponds to 10 mean false alarms - 0 corresponds to 1 mean false alarm - 1 corresponds to 0.1 mean false alarms This vector will be calculated only when the objects type is #LSD_REFINE_ADV. 
:type nfa: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} detect(image[, lines[, width[, prec[, nfa]]]]) -> lines, width, prec, nfa

Finds lines in the input image.


This is the output of the default parameters of the algorithm on the above shown image. 
![image](pics/building_lsd.png) 


:param self: 
:type self: 
:param image: A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use:`lsd_ptr-\>detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);` 
:type image: UMat
:param lines: A vector of Vec4f elements specifying the beginning and ending point of a line. WhereVec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly oriented depending on the gradient. 
:type lines: UMat | None
:param width: Vector of widths of the regions, where the lines are found. E.g. Width of line.
:type width: UMat | None
:param prec: Vector of precisions with which the lines are found.
:type prec: UMat | None
:param nfa: Vector containing number of false alarms in the line region, with precision of 10%. Thebigger the value, logarithmically better the detection. - -1 corresponds to 10 mean false alarms - 0 corresponds to 1 mean false alarm - 1 corresponds to 0.1 mean false alarms This vector will be calculated only when the objects type is #LSD_REFINE_ADV. 
:type nfa: UMat | None
:rtype: tuple[UMat, UMat, UMat, UMat]
````

````{py:method} drawSegments(image, lines) -> image

Draws the line segments on a given image.




:param self: 
:type self: 
:param image: The image, where the lines will be drawn. Should be bigger or equal to the image,where the lines were found. 
:type image: cv2.typing.MatLike
:param lines: A vector of the lines that needed to be drawn.
:type lines: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````

````{py:method} drawSegments(image, lines) -> image

Draws the line segments on a given image.




:param self: 
:type self: 
:param image: The image, where the lines will be drawn. Should be bigger or equal to the image,where the lines were found. 
:type image: UMat
:param lines: A vector of the lines that needed to be drawn.
:type lines: UMat
:rtype: UMat
````

````{py:method} compareSegments(size, lines1, lines2[, image]) -> retval, image

Draws two groups of lines in blue and red, counting the non overlapping (mismatching) pixels.




:param self: 
:type self: 
:param size: The size of the image, where lines1 and lines2 were found.
:type size: cv2.typing.Size
:param lines1: The first group of lines that needs to be drawn. It is visualized in blue color.
:type lines1: cv2.typing.MatLike
:param lines2: The second group of lines. They visualized in red color.
:type lines2: cv2.typing.MatLike
:param image: Optional image, where the lines will be drawn. The image should be color(3-channel)in order for lines1 and lines2 to be drawn in the above mentioned colors. 
:type image: cv2.typing.MatLike | None
:rtype: tuple[int, cv2.typing.MatLike]
````

````{py:method} compareSegments(size, lines1, lines2[, image]) -> retval, image

Draws two groups of lines in blue and red, counting the non overlapping (mismatching) pixels.




:param self: 
:type self: 
:param size: The size of the image, where lines1 and lines2 were found.
:type size: cv2.typing.Size
:param lines1: The first group of lines that needs to be drawn. It is visualized in blue color.
:type lines1: UMat
:param lines2: The second group of lines. They visualized in red color.
:type lines2: UMat
:param image: Optional image, where the lines will be drawn. The image should be color(3-channel)in order for lines1 and lines2 to be drawn in the above mentioned colors. 
:type image: UMat | None
:rtype: tuple[int, UMat]
````


`````


`````{py:class} MSER




````{py:method} create([, delta[, min_area[, max_area[, max_variation[, min_diversity[, max_evolution[, area_threshold[, min_margin[, edge_blur_size]]]]]]]]]) -> retval
:classmethod:
Full constructor for %MSER detector




:param cls: 
:type cls: 
:param delta: it compares $(size_{i}-size_{i-delta})/size_{i-delta}$
:type delta: int
:param min_area: prune the area which smaller than minArea
:type min_area: int
:param max_area: prune the area which bigger than maxArea
:type max_area: int
:param max_variation: prune the area have similar size to its children
:type max_variation: float
:param min_diversity: for color image, trace back to cut off mser with diversity less than min_diversity
:type min_diversity: float
:param max_evolution: for color image, the evolution steps
:type max_evolution: int
:param area_threshold: for color image, the area threshold to cause re-initialize
:type area_threshold: float
:param min_margin: for color image, ignore too small margin
:type min_margin: float
:param edge_blur_size: for color image, the aperture size for edge blur
:type edge_blur_size: int
:rtype: MSER
````

````{py:method} detectRegions(image) -> msers, bboxes

Detect %MSER regions




:param self: 
:type self: 
:param image: input image (8UC1, 8UC3 or 8UC4, must be greater or equal than 3x3)
:type image: cv2.typing.MatLike
:param msers: resulting list of point sets
:type msers: 
:param bboxes: resulting bounding boxes
:type bboxes: 
:rtype: tuple[_typing.Sequence[_typing.Sequence[cv2.typing.Point]], _typing.Sequence[cv2.typing.Rect]]
````

````{py:method} detectRegions(image) -> msers, bboxes

Detect %MSER regions




:param self: 
:type self: 
:param image: input image (8UC1, 8UC3 or 8UC4, must be greater or equal than 3x3)
:type image: UMat
:param msers: resulting list of point sets
:type msers: 
:param bboxes: resulting bounding boxes
:type bboxes: 
:rtype: tuple[_typing.Sequence[_typing.Sequence[cv2.typing.Point]], _typing.Sequence[cv2.typing.Rect]]
````

````{py:method} setDelta(delta) -> None





:param self: 
:type self: 
:param delta: 
:type delta: int
:rtype: None
````

````{py:method} getDelta() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMinArea(minArea) -> None





:param self: 
:type self: 
:param minArea: 
:type minArea: int
:rtype: None
````

````{py:method} getMinArea() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMaxArea(maxArea) -> None





:param self: 
:type self: 
:param maxArea: 
:type maxArea: int
:rtype: None
````

````{py:method} getMaxArea() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMaxVariation(maxVariation) -> None





:param self: 
:type self: 
:param maxVariation: 
:type maxVariation: float
:rtype: None
````

````{py:method} getMaxVariation() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMinDiversity(minDiversity) -> None





:param self: 
:type self: 
:param minDiversity: 
:type minDiversity: float
:rtype: None
````

````{py:method} getMinDiversity() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMaxEvolution(maxEvolution) -> None





:param self: 
:type self: 
:param maxEvolution: 
:type maxEvolution: int
:rtype: None
````

````{py:method} getMaxEvolution() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setAreaThreshold(areaThreshold) -> None





:param self: 
:type self: 
:param areaThreshold: 
:type areaThreshold: float
:rtype: None
````

````{py:method} getAreaThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMinMargin(min_margin) -> None





:param self: 
:type self: 
:param min_margin: 
:type min_margin: float
:rtype: None
````

````{py:method} getMinMargin() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setEdgeBlurSize(edge_blur_size) -> None





:param self: 
:type self: 
:param edge_blur_size: 
:type edge_blur_size: int
:rtype: None
````

````{py:method} getEdgeBlurSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setPass2Only(f) -> None





:param self: 
:type self: 
:param f: 
:type f: bool
:rtype: None
````

````{py:method} getPass2Only() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} 





`````


`````{py:class} MergeDebevec




````{py:method} process(src, times, response[, dst]) -> dst




process(src, times[, dst]) -> dst 


:param self: 
:type self: 
:param src: 
:type src: _typing.Sequence[cv2.typing.MatLike]
:param times: 
:type times: cv2.typing.MatLike
:param response: 
:type response: cv2.typing.MatLike
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} process(src, times, response[, dst]) -> dst




process(src, times[, dst]) -> dst 


:param self: 
:type self: 
:param src: 
:type src: _typing.Sequence[UMat]
:param times: 
:type times: UMat
:param response: 
:type response: UMat
:param dst: 
:type dst: UMat | None
:rtype: UMat
````

````{py:method} process(src, times, response[, dst]) -> dst




process(src, times[, dst]) -> dst 


:param self: 
:type self: 
:param src: 
:type src: _typing.Sequence[cv2.typing.MatLike]
:param times: 
:type times: cv2.typing.MatLike
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} process(src, times, response[, dst]) -> dst




process(src, times[, dst]) -> dst 


:param self: 
:type self: 
:param src: 
:type src: _typing.Sequence[UMat]
:param times: 
:type times: UMat
:param dst: 
:type dst: UMat | None
:rtype: UMat
````


`````


`````{py:class} MergeExposures




````{py:method} process(src, times, response[, dst]) -> dst

Merges images.




:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[cv2.typing.MatLike]
:param times: vector of exposure time values for each image
:type times: cv2.typing.MatLike
:param response: 256x1 matrix with inverse camera response function for each pixel value, it shouldhave the same number of channels as images. 
:type response: cv2.typing.MatLike
:param dst: result image
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} process(src, times, response[, dst]) -> dst

Merges images.




:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[UMat]
:param times: vector of exposure time values for each image
:type times: UMat
:param response: 256x1 matrix with inverse camera response function for each pixel value, it shouldhave the same number of channels as images. 
:type response: UMat
:param dst: result image
:type dst: UMat | None
:rtype: UMat
````


`````


`````{py:class} MergeMertens




````{py:method} process(src, times, response[, dst]) -> dst

Short version of process, that doesn't take extra arguments.


process(src[, dst]) -> dst 


:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[cv2.typing.MatLike]
:param times: 
:type times: cv2.typing.MatLike
:param response: 
:type response: cv2.typing.MatLike
:param dst: result image
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} process(src, times, response[, dst]) -> dst

Short version of process, that doesn't take extra arguments.


process(src[, dst]) -> dst 


:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[UMat]
:param times: 
:type times: UMat
:param response: 
:type response: UMat
:param dst: result image
:type dst: UMat | None
:rtype: UMat
````

````{py:method} process(src, times, response[, dst]) -> dst

Short version of process, that doesn't take extra arguments.


process(src[, dst]) -> dst 


:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[cv2.typing.MatLike]
:param dst: result image
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} process(src, times, response[, dst]) -> dst

Short version of process, that doesn't take extra arguments.


process(src[, dst]) -> dst 


:param self: 
:type self: 
:param src: vector of input images
:type src: _typing.Sequence[UMat]
:param dst: result image
:type dst: UMat | None
:rtype: UMat
````

````{py:method} getContrastWeight() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setContrastWeight(contrast_weiht) -> None





:param self: 
:type self: 
:param contrast_weiht: 
:type contrast_weiht: float
:rtype: None
````

````{py:method} getSaturationWeight() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setSaturationWeight(saturation_weight) -> None





:param self: 
:type self: 
:param saturation_weight: 
:type saturation_weight: float
:rtype: None
````

````{py:method} getExposureWeight() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setExposureWeight(exposure_weight) -> None





:param self: 
:type self: 
:param exposure_weight: 
:type exposure_weight: float
:rtype: None
````


`````


`````{py:class} MergeRobertson




````{py:method} process(src, times, response[, dst]) -> dst




process(src, times[, dst]) -> dst 


:param self: 
:type self: 
:param src: 
:type src: _typing.Sequence[cv2.typing.MatLike]
:param times: 
:type times: cv2.typing.MatLike
:param response: 
:type response: cv2.typing.MatLike
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} process(src, times, response[, dst]) -> dst




process(src, times[, dst]) -> dst 


:param self: 
:type self: 
:param src: 
:type src: _typing.Sequence[UMat]
:param times: 
:type times: UMat
:param response: 
:type response: UMat
:param dst: 
:type dst: UMat | None
:rtype: UMat
````

````{py:method} process(src, times, response[, dst]) -> dst




process(src, times[, dst]) -> dst 


:param self: 
:type self: 
:param src: 
:type src: _typing.Sequence[cv2.typing.MatLike]
:param times: 
:type times: cv2.typing.MatLike
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} process(src, times, response[, dst]) -> dst




process(src, times[, dst]) -> dst 


:param self: 
:type self: 
:param src: 
:type src: _typing.Sequence[UMat]
:param times: 
:type times: UMat
:param dst: 
:type dst: UMat | None
:rtype: UMat
````


`````


`````{py:class} ORB




````{py:method} create([, nfeatures[, scaleFactor[, nlevels[, edgeThreshold[, firstLevel[, WTA_K[, scoreType[, patchSize[, fastThreshold]]]]]]]]]) -> retval
:classmethod:
The ORB constructor




:param cls: 
:type cls: 
:param nfeatures: The maximum number of features to retain.
:type nfeatures: int
:param scaleFactor: Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classicalpyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer. 
:type scaleFactor: float
:param nlevels: The number of pyramid levels. The smallest level will have linear size equal toinput_image_linear_size/pow(scaleFactor, nlevels - firstLevel). 
:type nlevels: int
:param edgeThreshold: This is size of the border where the features are not detected. It shouldroughly match the patchSize parameter. 
:type edgeThreshold: int
:param firstLevel: The level of pyramid to put source image to. Previous layers are filledwith upscaled source image. 
:type firstLevel: int
:param WTA_K: The number of points that produce each element of the oriented BRIEF descriptor. Thedefault value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3). 
:type WTA_K: int
:param scoreType: The default HARRIS_SCORE means that Harris algorithm is used to rank features(the score is written to KeyPoint::score and is used to retain best nfeatures features); FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute. 
:type scoreType: ORB_ScoreType
:param patchSize: size of the patch used by the oriented BRIEF descriptor. Of course, on smallerpyramid layers the perceived image area covered by a feature will be larger. 
:type patchSize: int
:param fastThreshold: the fast threshold
:type fastThreshold: int
:rtype: ORB
````

````{py:method} setMaxFeatures(maxFeatures) -> None





:param self: 
:type self: 
:param maxFeatures: 
:type maxFeatures: int
:rtype: None
````

````{py:method} getMaxFeatures() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setScaleFactor(scaleFactor) -> None





:param self: 
:type self: 
:param scaleFactor: 
:type scaleFactor: float
:rtype: None
````

````{py:method} getScaleFactor() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setNLevels(nlevels) -> None





:param self: 
:type self: 
:param nlevels: 
:type nlevels: int
:rtype: None
````

````{py:method} getNLevels() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setEdgeThreshold(edgeThreshold) -> None





:param self: 
:type self: 
:param edgeThreshold: 
:type edgeThreshold: int
:rtype: None
````

````{py:method} getEdgeThreshold() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setFirstLevel(firstLevel) -> None





:param self: 
:type self: 
:param firstLevel: 
:type firstLevel: int
:rtype: None
````

````{py:method} getFirstLevel() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setWTA_K(wta_k) -> None





:param self: 
:type self: 
:param wta_k: 
:type wta_k: int
:rtype: None
````

````{py:method} getWTA_K() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setScoreType(scoreType) -> None





:param self: 
:type self: 
:param scoreType: 
:type scoreType: ORB_ScoreType
:rtype: None
````

````{py:method} getScoreType() -> retval





:param self: 
:type self: 
:rtype: ORB_ScoreType
````

````{py:method} setPatchSize(patchSize) -> None





:param self: 
:type self: 
:param patchSize: 
:type patchSize: int
:rtype: None
````

````{py:method} getPatchSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setFastThreshold(fastThreshold) -> None





:param self: 
:type self: 
:param fastThreshold: 
:type fastThreshold: int
:rtype: None
````

````{py:method} getFastThreshold() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} PyRotationWarper




````{py:method} __init__(self, type: str, scale: float)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param type: 
:type type: str
:param scale: 
:type scale: float
:rtype: None
````

````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} warpPoint(pt, K, R) -> retval

Projects the image point.




:param self: 
:type self: 
:param pt: Source point
:type pt: cv2.typing.Point2f
:param K: Camera intrinsic parameters
:type K: cv2.typing.MatLike
:param R: Camera rotation matrix
:type R: cv2.typing.MatLike
:return: Projected point
:rtype: cv2.typing.Point2f
````

````{py:method} warpPoint(pt, K, R) -> retval

Projects the image point.




:param self: 
:type self: 
:param pt: Source point
:type pt: cv2.typing.Point2f
:param K: Camera intrinsic parameters
:type K: UMat
:param R: Camera rotation matrix
:type R: UMat
:return: Projected point
:rtype: cv2.typing.Point2f
````

````{py:method} warpPointBackward(pt, K, R) -> retval

Projects the image point backward.




:param self: 
:type self: 
:param pt: Projected point
:type pt: cv2.typing.Point2f
:param K: Camera intrinsic parameters
:type K: cv2.typing.MatLike
:param R: Camera rotation matrix
:type R: cv2.typing.MatLike
:return: Backward-projected point
:rtype: cv2.typing.Point2f
````

````{py:method} warpPointBackward(pt, K, R) -> retval

Projects the image point backward.




:param self: 
:type self: 
:param pt: Projected point
:type pt: cv2.typing.Point2f
:param K: Camera intrinsic parameters
:type K: UMat
:param R: Camera rotation matrix
:type R: UMat
:return: Backward-projected point
:rtype: cv2.typing.Point2f
````

````{py:method} warpPointBackward(pt, K, R) -> retval

Projects the image point backward.




:param self: 
:type self: 
:param pt: Projected point
:type pt: cv2.typing.Point2f
:param K: Camera intrinsic parameters
:type K: cv2.typing.MatLike
:param R: Camera rotation matrix
:type R: cv2.typing.MatLike
:return: Backward-projected point
:rtype: cv2.typing.Point2f
````

````{py:method} warpPointBackward(pt, K, R) -> retval

Projects the image point backward.




:param self: 
:type self: 
:param pt: Projected point
:type pt: cv2.typing.Point2f
:param K: Camera intrinsic parameters
:type K: UMat
:param R: Camera rotation matrix
:type R: UMat
:return: Backward-projected point
:rtype: cv2.typing.Point2f
````

````{py:method} buildMaps(src_size, K, R[, xmap[, ymap]]) -> retval, xmap, ymap

Builds the projection maps according to the given camera data.




:param self: 
:type self: 
:param src_size: Source image size
:type src_size: cv2.typing.Size
:param K: Camera intrinsic parameters
:type K: cv2.typing.MatLike
:param R: Camera rotation matrix
:type R: cv2.typing.MatLike
:param xmap: Projection map for the x axis
:type xmap: cv2.typing.MatLike | None
:param ymap: Projection map for the y axis
:type ymap: cv2.typing.MatLike | None
:return: Projected image minimum bounding box
:rtype: tuple[cv2.typing.Rect, cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} buildMaps(src_size, K, R[, xmap[, ymap]]) -> retval, xmap, ymap

Builds the projection maps according to the given camera data.




:param self: 
:type self: 
:param src_size: Source image size
:type src_size: cv2.typing.Size
:param K: Camera intrinsic parameters
:type K: UMat
:param R: Camera rotation matrix
:type R: UMat
:param xmap: Projection map for the x axis
:type xmap: UMat | None
:param ymap: Projection map for the y axis
:type ymap: UMat | None
:return: Projected image minimum bounding box
:rtype: tuple[cv2.typing.Rect, UMat, UMat]
````

````{py:method} warp(src, K, R, interp_mode, border_mode[, dst]) -> retval, dst

Projects the image.




:param self: 
:type self: 
:param src: Source image
:type src: cv2.typing.MatLike
:param K: Camera intrinsic parameters
:type K: cv2.typing.MatLike
:param R: Camera rotation matrix
:type R: cv2.typing.MatLike
:param interp_mode: Interpolation mode
:type interp_mode: int
:param border_mode: Border extrapolation mode
:type border_mode: int
:param dst: Projected image
:type dst: cv2.typing.MatLike | None
:return: Project image top-left corner
:rtype: tuple[cv2.typing.Point, cv2.typing.MatLike]
````

````{py:method} warp(src, K, R, interp_mode, border_mode[, dst]) -> retval, dst

Projects the image.




:param self: 
:type self: 
:param src: Source image
:type src: UMat
:param K: Camera intrinsic parameters
:type K: UMat
:param R: Camera rotation matrix
:type R: UMat
:param interp_mode: Interpolation mode
:type interp_mode: int
:param border_mode: Border extrapolation mode
:type border_mode: int
:param dst: Projected image
:type dst: UMat | None
:return: Project image top-left corner
:rtype: tuple[cv2.typing.Point, UMat]
````

````{py:method} warpBackward(src, K, R, interp_mode, border_mode, dst_size[, dst]) -> dst

Projects the image backward.




:param self: 
:type self: 
:param src: Projected image
:type src: cv2.typing.MatLike
:param K: Camera intrinsic parameters
:type K: cv2.typing.MatLike
:param R: Camera rotation matrix
:type R: cv2.typing.MatLike
:param interp_mode: Interpolation mode
:type interp_mode: int
:param border_mode: Border extrapolation mode
:type border_mode: int
:param dst_size: Backward-projected image size
:type dst_size: cv2.typing.Size
:param dst: Backward-projected image
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} warpBackward(src, K, R, interp_mode, border_mode, dst_size[, dst]) -> dst

Projects the image backward.




:param self: 
:type self: 
:param src: Projected image
:type src: UMat
:param K: Camera intrinsic parameters
:type K: UMat
:param R: Camera rotation matrix
:type R: UMat
:param interp_mode: Interpolation mode
:type interp_mode: int
:param border_mode: Border extrapolation mode
:type border_mode: int
:param dst_size: Backward-projected image size
:type dst_size: cv2.typing.Size
:param dst: Backward-projected image
:type dst: UMat | None
:rtype: UMat
````

````{py:method} warpRoi(src_size, K, R) -> retval






:param self: 
:type self: 
:param src_size: Source image bounding box
:type src_size: cv2.typing.Size
:param K: Camera intrinsic parameters
:type K: cv2.typing.MatLike
:param R: Camera rotation matrix
:type R: cv2.typing.MatLike
:return: Projected image minimum bounding box
:rtype: cv2.typing.Rect
````

````{py:method} warpRoi(src_size, K, R) -> retval






:param self: 
:type self: 
:param src_size: Source image bounding box
:type src_size: cv2.typing.Size
:param K: Camera intrinsic parameters
:type K: UMat
:param R: Camera rotation matrix
:type R: UMat
:return: Projected image minimum bounding box
:rtype: cv2.typing.Rect
````

````{py:method} getScale() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setScale(arg1) -> None





:param self: 
:type self: 
:param arg1: 
:type arg1: float
:rtype: None
````


`````


`````{py:class} QRCodeDetector




````{py:method} decodeCurved(img, points[, straight_qrcode]) -> retval, straight_qrcode

Decodes QR code on a curved surface in image once it's found by the detect() method.


Returns UTF8-encoded output string or empty string if the code cannot be decoded. 


:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing QR code.
:type img: cv2.typing.MatLike
:param points: Quadrangle vertices found by detect() method (or some other algorithm).
:type points: cv2.typing.MatLike
:param straight_qrcode: The optional output image containing rectified and binarized QR code
:type straight_qrcode: cv2.typing.MatLike | None
:rtype: tuple[str, cv2.typing.MatLike]
````

````{py:method} decodeCurved(img, points[, straight_qrcode]) -> retval, straight_qrcode

Decodes QR code on a curved surface in image once it's found by the detect() method.


Returns UTF8-encoded output string or empty string if the code cannot be decoded. 


:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing QR code.
:type img: UMat
:param points: Quadrangle vertices found by detect() method (or some other algorithm).
:type points: UMat
:param straight_qrcode: The optional output image containing rectified and binarized QR code
:type straight_qrcode: UMat | None
:rtype: tuple[str, UMat]
````

````{py:method} detectAndDecodeCurved(img[, points[, straight_qrcode]]) -> retval, points, straight_qrcode

Both detects and decodes QR code on a curved surface




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing QR code.
:type img: cv2.typing.MatLike
:param points: optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
:type points: cv2.typing.MatLike | None
:param straight_qrcode: The optional output image containing rectified and binarized QR code
:type straight_qrcode: cv2.typing.MatLike | None
:rtype: tuple[str, cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} detectAndDecodeCurved(img[, points[, straight_qrcode]]) -> retval, points, straight_qrcode

Both detects and decodes QR code on a curved surface




:param self: 
:type self: 
:param img: grayscale or color (BGR) image containing QR code.
:type img: UMat
:param points: optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
:type points: UMat | None
:param straight_qrcode: The optional output image containing rectified and binarized QR code
:type straight_qrcode: UMat | None
:rtype: tuple[str, UMat, UMat]
````

````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} setEpsX(epsX) -> retval
sets the epsilon used during the horizontal scan of QR code stop marker detection.




:param self: 
:type self: 
:param epsX: Epsilon neighborhood, which allows you to determine the horizontal patternof the scheme 1:1:3:1:1 according to QR code standard. 
:type epsX: float
:rtype: QRCodeDetector
````

````{py:method} setEpsY(epsY) -> retval
sets the epsilon used during the vertical scan of QR code stop marker detection.




:param self: 
:type self: 
:param epsY: Epsilon neighborhood, which allows you to determine the vertical patternof the scheme 1:1:3:1:1 according to QR code standard. 
:type epsY: float
:rtype: QRCodeDetector
````

````{py:method} setUseAlignmentMarkers(useAlignmentMarkers) -> retval
use markers to improve the position of the corners of the QR code


alignmentMarkers using by default 


:param self: 
:type self: 
:param useAlignmentMarkers: 
:type useAlignmentMarkers: bool
:rtype: QRCodeDetector
````


`````


`````{py:class} QRCodeDetectorAruco




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, params: QRCodeDetectorAruco.Params)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param params: 
:type params: QRCodeDetectorAruco.Params
:rtype: None
````

````{py:method} getDetectorParameters() -> retval
Detector parameters getter. See cv::QRCodeDetectorAruco::Params




:param self: 
:type self: 
:rtype: QRCodeDetectorAruco.Params
````

````{py:method} setDetectorParameters(params) -> retval
Detector parameters setter. See cv::QRCodeDetectorAruco::Params




:param self: 
:type self: 
:param params: 
:type params: QRCodeDetectorAruco.Params
:rtype: QRCodeDetectorAruco
````

````{py:method} getArucoParameters() -> retval
Aruco detector parameters are used to search for the finder patterns.




:param self: 
:type self: 
:rtype: cv2.aruco.DetectorParameters
````

````{py:method} setArucoParameters(params) -> None
Aruco detector parameters are used to search for the finder patterns.




:param self: 
:type self: 
:param params: 
:type params: cv2.aruco.DetectorParameters
:rtype: None
````


`````


`````{py:class} 





`````


`````{py:class} QRCodeEncoder




````{py:method} create([, parameters]) -> retval
:classmethod:
Constructor




:param cls: 
:type cls: 
:param parameters: QR code encoder parameters QRCodeEncoder::Params
:type parameters: QRCodeEncoder.Params
:rtype: QRCodeEncoder
````

````{py:method} encode(encoded_info[, qrcode]) -> qrcode

Generates QR code from input string.




:param self: 
:type self: 
:param encoded_info: Input string to encode.
:type encoded_info: str
:param qrcode: Generated QR code.
:type qrcode: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} encode(encoded_info[, qrcode]) -> qrcode

Generates QR code from input string.




:param self: 
:type self: 
:param encoded_info: Input string to encode.
:type encoded_info: str
:param qrcode: Generated QR code.
:type qrcode: UMat | None
:rtype: UMat
````

````{py:method} encodeStructuredAppend(encoded_info[, qrcodes]) -> qrcodes

Generates QR code from input string in Structured Append mode. The encoded message is splitting over a number of QR codes.




:param self: 
:type self: 
:param encoded_info: Input string to encode.
:type encoded_info: str
:param qrcodes: Vector of generated QR codes.
:type qrcodes: _typing.Sequence[cv2.typing.MatLike] | None
:rtype: _typing.Sequence[cv2.typing.MatLike]
````

````{py:method} encodeStructuredAppend(encoded_info[, qrcodes]) -> qrcodes

Generates QR code from input string in Structured Append mode. The encoded message is splitting over a number of QR codes.




:param self: 
:type self: 
:param encoded_info: Input string to encode.
:type encoded_info: str
:param qrcodes: Vector of generated QR codes.
:type qrcodes: _typing.Sequence[UMat] | None
:rtype: _typing.Sequence[UMat]
````


`````


`````{py:class} 





`````


`````{py:class} RotatedRect




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, center: cv2.typing.Point2f, size: cv2.typing.Size2f, angle: float)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param center: 
:type center: cv2.typing.Point2f
:param size: 
:type size: cv2.typing.Size2f
:param angle: 
:type angle: float
:rtype: None
````

````{py:method} __init__(self, point1: cv2.typing.Point2f, point2: cv2.typing.Point2f, point3: cv2.typing.Point2f)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param point1: 
:type point1: cv2.typing.Point2f
:param point2: 
:type point2: cv2.typing.Point2f
:param point3: 
:type point3: cv2.typing.Point2f
:rtype: None
````

````{py:method} points() -> pts



returns 4 vertices of the rotated rectangle 
```{note}
_Bottom_, _Top_, _Left_ and _Right_ sides refer to the original rectangle (angle is 0),so after 180 degree rotation _bottomLeft_ point will be located at the top right corner of the rectangle. 
```


:param self: 
:type self: 
:param pts: The points array for storing rectangle vertices. The order is _bottomLeft_, _topLeft_, topRight, bottomRight.
:type pts: 
:rtype: _typing.Sequence[cv2.typing.Point2f]
````

````{py:method} boundingRect() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Rect
````

```{py:attribute} center
:type: cv2.typing.Point2f
```

```{py:attribute} size
:type: cv2.typing.Size2f
```

```{py:attribute} angle
:type: float
```


`````


`````{py:class} SIFT




````{py:method} create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma[, enable_precise_upscale]]]]]]) -> retval
:classmethod:
Create SIFT with specified descriptorType.


create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, descriptorType[, enable_precise_upscale]) -> retval 
```{note}
The contrast threshold will be divided by nOctaveLayers when the filtering is applied. WhennOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09. 
```
```{note}
The contrast threshold will be divided by nOctaveLayers when the filtering is applied. WhennOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09. 
```


:param cls: 
:type cls: 
:param nfeatures: The number of best features to retain. The features are ranked by their scores(measured in SIFT algorithm as the local contrast) 
:type nfeatures: int
:param nOctaveLayers: The number of layers in each octave. 3 is the value used in D. Lowe paper. Thenumber of octaves is computed automatically from the image resolution. 
:type nOctaveLayers: int
:param contrastThreshold: The contrast threshold used to filter out weak features in semi-uniform(low-contrast) regions. The larger the threshold, the less features are produced by the detector. 
:type contrastThreshold: float
:param edgeThreshold: The threshold used to filter out edge-like features. Note that the its meaningis different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained). 
:type edgeThreshold: float
:param sigma: The sigma of the Gaussian applied to the input image at the octave \#0. If your imageis captured with a weak camera with soft lenses, you might want to reduce the number. 
:type sigma: float
:param enable_precise_upscale: Whether to enable precise upscaling in the scale pyramid, which mapsindex $\texttt{x}$ to $\texttt{2x}$. This prevents localization bias. The option is disabled by default. 
:type enable_precise_upscale: bool
:param descriptorType: The type of descriptors. Only CV_32F and CV_8U are supported.
:type descriptorType: 
:rtype: SIFT
````

````{py:method} create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma[, enable_precise_upscale]]]]]]) -> retval
:classmethod:
Create SIFT with specified descriptorType.


create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, descriptorType[, enable_precise_upscale]) -> retval 
```{note}
The contrast threshold will be divided by nOctaveLayers when the filtering is applied. WhennOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09. 
```
```{note}
The contrast threshold will be divided by nOctaveLayers when the filtering is applied. WhennOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09. 
```


:param cls: 
:type cls: 
:param nfeatures: The number of best features to retain. The features are ranked by their scores(measured in SIFT algorithm as the local contrast) 
:type nfeatures: int
:param nOctaveLayers: The number of layers in each octave. 3 is the value used in D. Lowe paper. Thenumber of octaves is computed automatically from the image resolution. 
:type nOctaveLayers: int
:param contrastThreshold: The contrast threshold used to filter out weak features in semi-uniform(low-contrast) regions. The larger the threshold, the less features are produced by the detector. 
:type contrastThreshold: float
:param edgeThreshold: The threshold used to filter out edge-like features. Note that the its meaningis different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained). 
:type edgeThreshold: float
:param sigma: The sigma of the Gaussian applied to the input image at the octave \#0. If your imageis captured with a weak camera with soft lenses, you might want to reduce the number. 
:type sigma: float
:param descriptorType: The type of descriptors. Only CV_32F and CV_8U are supported.
:type descriptorType: int
:param enable_precise_upscale: Whether to enable precise upscaling in the scale pyramid, which mapsindex $\texttt{x}$ to $\texttt{2x}$. This prevents localization bias. The option is disabled by default. 
:type enable_precise_upscale: bool
:rtype: SIFT
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````

````{py:method} setNFeatures(maxFeatures) -> None





:param self: 
:type self: 
:param maxFeatures: 
:type maxFeatures: int
:rtype: None
````

````{py:method} getNFeatures() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setNOctaveLayers(nOctaveLayers) -> None





:param self: 
:type self: 
:param nOctaveLayers: 
:type nOctaveLayers: int
:rtype: None
````

````{py:method} getNOctaveLayers() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setContrastThreshold(contrastThreshold) -> None





:param self: 
:type self: 
:param contrastThreshold: 
:type contrastThreshold: float
:rtype: None
````

````{py:method} getContrastThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setEdgeThreshold(edgeThreshold) -> None





:param self: 
:type self: 
:param edgeThreshold: 
:type edgeThreshold: float
:rtype: None
````

````{py:method} getEdgeThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setSigma(sigma) -> None





:param self: 
:type self: 
:param sigma: 
:type sigma: float
:rtype: None
````

````{py:method} getSigma() -> retval





:param self: 
:type self: 
:rtype: float
````


`````


`````{py:class} SimpleBlobDetector




````{py:method} create([, parameters]) -> retval
:classmethod:





:param cls: 
:type cls: 
:param parameters: 
:type parameters: SimpleBlobDetector.Params
:rtype: SimpleBlobDetector
````

````{py:method} setParams(params) -> None





:param self: 
:type self: 
:param params: 
:type params: SimpleBlobDetector.Params
:rtype: None
````

````{py:method} getParams() -> retval





:param self: 
:type self: 
:rtype: SimpleBlobDetector.Params
````

````{py:method} getDefaultName() -> retval





:param self: 
:type self: 
:rtype: str
````

````{py:method} getBlobContours() -> retval





:param self: 
:type self: 
:rtype: _typing.Sequence[_typing.Sequence[cv2.typing.Point]]
````


`````


`````{py:class} 





`````


`````{py:class} SparseOpticalFlow




````{py:method} calc(prevImg, nextImg, prevPts, nextPts[, status[, err]]) -> nextPts, status, err

Calculates a sparse optical flow.




:param self: 
:type self: 
:param prevImg: First input image.
:type prevImg: cv2.typing.MatLike
:param nextImg: Second input image of the same size and the same type as prevImg.
:type nextImg: cv2.typing.MatLike
:param prevPts: Vector of 2D points for which the flow needs to be found.
:type prevPts: cv2.typing.MatLike
:param nextPts: Output vector of 2D points containing the calculated new positions of input features in the second image.
:type nextPts: cv2.typing.MatLike
:param status: Output status vector. Each element of the vector is set to 1 if theflow for the corresponding features has been found. Otherwise, it is set to 0. 
:type status: cv2.typing.MatLike | None
:param err: Optional output vector that contains error response for each point (inverse confidence).
:type err: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} calc(prevImg, nextImg, prevPts, nextPts[, status[, err]]) -> nextPts, status, err

Calculates a sparse optical flow.




:param self: 
:type self: 
:param prevImg: First input image.
:type prevImg: UMat
:param nextImg: Second input image of the same size and the same type as prevImg.
:type nextImg: UMat
:param prevPts: Vector of 2D points for which the flow needs to be found.
:type prevPts: UMat
:param nextPts: Output vector of 2D points containing the calculated new positions of input features in the second image.
:type nextPts: UMat
:param status: Output status vector. Each element of the vector is set to 1 if theflow for the corresponding features has been found. Otherwise, it is set to 0. 
:type status: UMat | None
:param err: Optional output vector that contains error response for each point (inverse confidence).
:type err: UMat | None
:rtype: tuple[UMat, UMat, UMat]
````


`````


`````{py:class} SparsePyrLKOpticalFlow




````{py:method} create([, winSize[, maxLevel[, crit[, flags[, minEigThreshold]]]]]) -> retval
:classmethod:





:param cls: 
:type cls: 
:param winSize: 
:type winSize: cv2.typing.Size
:param maxLevel: 
:type maxLevel: int
:param crit: 
:type crit: cv2.typing.TermCriteria
:param flags: 
:type flags: int
:param minEigThreshold: 
:type minEigThreshold: float
:rtype: SparsePyrLKOpticalFlow
````

````{py:method} getWinSize() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} setWinSize(winSize) -> None





:param self: 
:type self: 
:param winSize: 
:type winSize: cv2.typing.Size
:rtype: None
````

````{py:method} getMaxLevel() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMaxLevel(maxLevel) -> None





:param self: 
:type self: 
:param maxLevel: 
:type maxLevel: int
:rtype: None
````

````{py:method} getTermCriteria() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.TermCriteria
````

````{py:method} setTermCriteria(crit) -> None





:param self: 
:type self: 
:param crit: 
:type crit: cv2.typing.TermCriteria
:rtype: None
````

````{py:method} getFlags() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setFlags(flags) -> None





:param self: 
:type self: 
:param flags: 
:type flags: int
:rtype: None
````

````{py:method} getMinEigThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMinEigThreshold(minEigThreshold) -> None





:param self: 
:type self: 
:param minEigThreshold: 
:type minEigThreshold: float
:rtype: None
````


`````


`````{py:class} StereoBM




````{py:method} create([, numDisparities[, blockSize]]) -> retval
:classmethod:
Creates StereoBM object


The function create StereoBM object. You can then call StereoBM::compute() to compute disparity for a specific stereo pair. 


:param cls: 
:type cls: 
:param numDisparities: the disparity search range. For each pixel algorithm will find the bestdisparity from 0 (default minimum disparity) to numDisparities. The search range can then be shifted by changing the minimum disparity. 
:type numDisparities: int
:param blockSize: the linear size of the blocks compared by the algorithm. The size should be odd(as the block is centered at the current pixel). Larger block size implies smoother, though less accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher chance for algorithm to find a wrong correspondence. 
:type blockSize: int
:rtype: StereoBM
````

````{py:method} getPreFilterType() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setPreFilterType(preFilterType) -> None





:param self: 
:type self: 
:param preFilterType: 
:type preFilterType: int
:rtype: None
````

````{py:method} getPreFilterSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setPreFilterSize(preFilterSize) -> None





:param self: 
:type self: 
:param preFilterSize: 
:type preFilterSize: int
:rtype: None
````

````{py:method} getPreFilterCap() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setPreFilterCap(preFilterCap) -> None





:param self: 
:type self: 
:param preFilterCap: 
:type preFilterCap: int
:rtype: None
````

````{py:method} getTextureThreshold() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setTextureThreshold(textureThreshold) -> None





:param self: 
:type self: 
:param textureThreshold: 
:type textureThreshold: int
:rtype: None
````

````{py:method} getUniquenessRatio() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setUniquenessRatio(uniquenessRatio) -> None





:param self: 
:type self: 
:param uniquenessRatio: 
:type uniquenessRatio: int
:rtype: None
````

````{py:method} getSmallerBlockSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setSmallerBlockSize(blockSize) -> None





:param self: 
:type self: 
:param blockSize: 
:type blockSize: int
:rtype: None
````

````{py:method} getROI1() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Rect
````

````{py:method} setROI1(roi1) -> None





:param self: 
:type self: 
:param roi1: 
:type roi1: cv2.typing.Rect
:rtype: None
````

````{py:method} getROI2() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Rect
````

````{py:method} setROI2(roi2) -> None





:param self: 
:type self: 
:param roi2: 
:type roi2: cv2.typing.Rect
:rtype: None
````


`````


`````{py:class} StereoMatcher




````{py:method} compute(left, right[, disparity]) -> disparity

Computes disparity map for the specified stereo pair




:param self: 
:type self: 
:param left: Left 8-bit single-channel image.
:type left: cv2.typing.MatLike
:param right: Right image of the same size and the same type as the left one.
:type right: cv2.typing.MatLike
:param disparity: Output disparity map. It has the same size as the input images. Some algorithms,like StereoBM or StereoSGBM compute 16-bit fixed-point disparity map (where each disparity value has 4 fractional bits), whereas other algorithms output 32-bit floating-point disparity map. 
:type disparity: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} compute(left, right[, disparity]) -> disparity

Computes disparity map for the specified stereo pair




:param self: 
:type self: 
:param left: Left 8-bit single-channel image.
:type left: UMat
:param right: Right image of the same size and the same type as the left one.
:type right: UMat
:param disparity: Output disparity map. It has the same size as the input images. Some algorithms,like StereoBM or StereoSGBM compute 16-bit fixed-point disparity map (where each disparity value has 4 fractional bits), whereas other algorithms output 32-bit floating-point disparity map. 
:type disparity: UMat | None
:rtype: UMat
````

````{py:method} getMinDisparity() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMinDisparity(minDisparity) -> None





:param self: 
:type self: 
:param minDisparity: 
:type minDisparity: int
:rtype: None
````

````{py:method} getNumDisparities() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setNumDisparities(numDisparities) -> None





:param self: 
:type self: 
:param numDisparities: 
:type numDisparities: int
:rtype: None
````

````{py:method} getBlockSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setBlockSize(blockSize) -> None





:param self: 
:type self: 
:param blockSize: 
:type blockSize: int
:rtype: None
````

````{py:method} getSpeckleWindowSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setSpeckleWindowSize(speckleWindowSize) -> None





:param self: 
:type self: 
:param speckleWindowSize: 
:type speckleWindowSize: int
:rtype: None
````

````{py:method} getSpeckleRange() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setSpeckleRange(speckleRange) -> None





:param self: 
:type self: 
:param speckleRange: 
:type speckleRange: int
:rtype: None
````

````{py:method} getDisp12MaxDiff() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setDisp12MaxDiff(disp12MaxDiff) -> None





:param self: 
:type self: 
:param disp12MaxDiff: 
:type disp12MaxDiff: int
:rtype: None
````


`````


`````{py:class} StereoSGBM




````{py:method} create([, minDisparity[, numDisparities[, blockSize[, P1[, P2[, disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]]]]) -> retval
:classmethod:
Creates StereoSGBM object


The first constructor initializes StereoSGBM with all the default parameters. So, you only have to set StereoSGBM::numDisparities at minimum. The second constructor enables you to set each parameter to a custom value. 


:param cls: 
:type cls: 
:param minDisparity: Minimum possible disparity value. Normally, it is zero but sometimesrectification algorithms can shift images, so this parameter needs to be adjusted accordingly. 
:type minDisparity: int
:param numDisparities: Maximum disparity minus minimum disparity. The value is always greater thanzero. In the current implementation, this parameter must be divisible by 16. 
:type numDisparities: int
:param blockSize: Matched block size. It must be an odd number \>=1 . Normally, it should besomewhere in the 3..11 range. 
:type blockSize: int
:param P1: The first parameter controlling the disparity smoothness. See below.
:type P1: int
:param P2: The second parameter controlling the disparity smoothness. The larger the values are,the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 \> P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8\*number_of_image_channels\*blockSize\*blockSize and 32\*number_of_image_channels\*blockSize\*blockSize , respectively). 
:type P2: int
:param disp12MaxDiff: Maximum allowed difference (in integer pixel units) in the left-rightdisparity check. Set it to a non-positive value to disable the check. 
:type disp12MaxDiff: int
:param preFilterCap: Truncation value for the prefiltered image pixels. The algorithm firstcomputes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function. 
:type preFilterCap: int
:param uniquenessRatio: Margin in percentage by which the best (minimum) computed cost functionvalue should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough. 
:type uniquenessRatio: int
:param speckleWindowSize: Maximum size of smooth disparity regions to consider their noise specklesand invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range. 
:type speckleWindowSize: int
:param speckleRange: Maximum disparity variation within each connected component. If you do specklefiltering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough. 
:type speckleRange: int
:param mode: Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programmingalgorithm. It will consume O(W\*H\*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false . 
:type mode: int
:rtype: StereoSGBM
````

````{py:method} getPreFilterCap() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setPreFilterCap(preFilterCap) -> None





:param self: 
:type self: 
:param preFilterCap: 
:type preFilterCap: int
:rtype: None
````

````{py:method} getUniquenessRatio() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setUniquenessRatio(uniquenessRatio) -> None





:param self: 
:type self: 
:param uniquenessRatio: 
:type uniquenessRatio: int
:rtype: None
````

````{py:method} getP1() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setP1(P1) -> None





:param self: 
:type self: 
:param P1: 
:type P1: int
:rtype: None
````

````{py:method} getP2() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setP2(P2) -> None





:param self: 
:type self: 
:param P2: 
:type P2: int
:rtype: None
````

````{py:method} getMode() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} setMode(mode) -> None





:param self: 
:type self: 
:param mode: 
:type mode: int
:rtype: None
````


`````


`````{py:class} Stitcher




````{py:method} create([, mode]) -> retval
:classmethod:
Creates a Stitcher configured in one of the stitching modes.




:param cls: 
:type cls: 
:param mode: Scenario for stitcher operation. This is usually determined by source of imagesto stitch and their transformation. Default parameters will be chosen for operation in given scenario. 
:type mode: Stitcher_Mode
:return: Stitcher class instance.
:rtype: Stitcher
````

````{py:method} estimateTransform(images[, masks]) -> retval

These functions try to match the given images and to estimate rotations of each camera.


```{note}
Use the functions only if you're aware of the stitching pipeline, otherwise useStitcher::stitch. 
```


:param self: 
:type self: 
:param images: Input images.
:type images: _typing.Sequence[cv2.typing.MatLike]
:param masks: Masks for each input image specifying where to look for keypoints (optional).
:type masks: _typing.Sequence[cv2.typing.MatLike] | None
:return: Status code.
:rtype: Stitcher_Status
````

````{py:method} estimateTransform(images[, masks]) -> retval

These functions try to match the given images and to estimate rotations of each camera.


```{note}
Use the functions only if you're aware of the stitching pipeline, otherwise useStitcher::stitch. 
```


:param self: 
:type self: 
:param images: Input images.
:type images: _typing.Sequence[UMat]
:param masks: Masks for each input image specifying where to look for keypoints (optional).
:type masks: _typing.Sequence[UMat] | None
:return: Status code.
:rtype: Stitcher_Status
````

````{py:method} composePanorama([, pano]) -> retval, pano

These functions try to compose the given images (or images stored internally from the other functioncalls) into the final pano under the assumption that the image transformations were estimated before. 


@overload 
composePanorama(images[, pano]) -> retval, pano 
```{note}
Use the functions only if you're aware of the stitching pipeline, otherwise useStitcher::stitch. 
```


:param self: 
:type self: 
:param pano: Final pano.
:type pano: cv2.typing.MatLike | None
:param images: Input images.
:type images: 
:return: Status code.
:rtype: tuple[Stitcher_Status, cv2.typing.MatLike]
````

````{py:method} composePanorama([, pano]) -> retval, pano

These functions try to compose the given images (or images stored internally from the other functioncalls) into the final pano under the assumption that the image transformations were estimated before. 


@overload 
composePanorama(images[, pano]) -> retval, pano 
```{note}
Use the functions only if you're aware of the stitching pipeline, otherwise useStitcher::stitch. 
```


:param self: 
:type self: 
:param pano: Final pano.
:type pano: UMat | None
:param images: Input images.
:type images: 
:return: Status code.
:rtype: tuple[Stitcher_Status, UMat]
````

````{py:method} composePanorama([, pano]) -> retval, pano

These functions try to compose the given images (or images stored internally from the other functioncalls) into the final pano under the assumption that the image transformations were estimated before. 


@overload 
composePanorama(images[, pano]) -> retval, pano 
```{note}
Use the functions only if you're aware of the stitching pipeline, otherwise useStitcher::stitch. 
```


:param self: 
:type self: 
:param images: Input images.
:type images: _typing.Sequence[cv2.typing.MatLike]
:param pano: Final pano.
:type pano: cv2.typing.MatLike | None
:return: Status code.
:rtype: tuple[Stitcher_Status, cv2.typing.MatLike]
````

````{py:method} composePanorama([, pano]) -> retval, pano

These functions try to compose the given images (or images stored internally from the other functioncalls) into the final pano under the assumption that the image transformations were estimated before. 


@overload 
composePanorama(images[, pano]) -> retval, pano 
```{note}
Use the functions only if you're aware of the stitching pipeline, otherwise useStitcher::stitch. 
```


:param self: 
:type self: 
:param images: Input images.
:type images: _typing.Sequence[UMat]
:param pano: Final pano.
:type pano: UMat | None
:return: Status code.
:rtype: tuple[Stitcher_Status, UMat]
````

````{py:method} stitch(images[, pano]) -> retval, pano

These functions try to stitch the given images.


@overload 
stitch(images, masks[, pano]) -> retval, pano 


:param self: 
:type self: 
:param images: Input images.
:type images: _typing.Sequence[cv2.typing.MatLike]
:param pano: Final pano.
:type pano: cv2.typing.MatLike | None
:param masks: Masks for each input image specifying where to look for keypoints (optional).
:type masks: 
:return: Status code.
:rtype: tuple[Stitcher_Status, cv2.typing.MatLike]
````

````{py:method} stitch(images[, pano]) -> retval, pano

These functions try to stitch the given images.


@overload 
stitch(images, masks[, pano]) -> retval, pano 


:param self: 
:type self: 
:param images: Input images.
:type images: _typing.Sequence[UMat]
:param pano: Final pano.
:type pano: UMat | None
:param masks: Masks for each input image specifying where to look for keypoints (optional).
:type masks: 
:return: Status code.
:rtype: tuple[Stitcher_Status, UMat]
````

````{py:method} stitch(images[, pano]) -> retval, pano

These functions try to stitch the given images.


@overload 
stitch(images, masks[, pano]) -> retval, pano 


:param self: 
:type self: 
:param images: Input images.
:type images: _typing.Sequence[cv2.typing.MatLike]
:param masks: Masks for each input image specifying where to look for keypoints (optional).
:type masks: _typing.Sequence[cv2.typing.MatLike]
:param pano: Final pano.
:type pano: cv2.typing.MatLike | None
:return: Status code.
:rtype: tuple[Stitcher_Status, cv2.typing.MatLike]
````

````{py:method} stitch(images[, pano]) -> retval, pano

These functions try to stitch the given images.


@overload 
stitch(images, masks[, pano]) -> retval, pano 


:param self: 
:type self: 
:param images: Input images.
:type images: _typing.Sequence[UMat]
:param masks: Masks for each input image specifying where to look for keypoints (optional).
:type masks: _typing.Sequence[UMat]
:param pano: Final pano.
:type pano: UMat | None
:return: Status code.
:rtype: tuple[Stitcher_Status, UMat]
````

````{py:method} registrationResol() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setRegistrationResol(resol_mpx) -> None





:param self: 
:type self: 
:param resol_mpx: 
:type resol_mpx: float
:rtype: None
````

````{py:method} seamEstimationResol() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setSeamEstimationResol(resol_mpx) -> None





:param self: 
:type self: 
:param resol_mpx: 
:type resol_mpx: float
:rtype: None
````

````{py:method} compositingResol() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setCompositingResol(resol_mpx) -> None





:param self: 
:type self: 
:param resol_mpx: 
:type resol_mpx: float
:rtype: None
````

````{py:method} panoConfidenceThresh() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setPanoConfidenceThresh(conf_thresh) -> None





:param self: 
:type self: 
:param conf_thresh: 
:type conf_thresh: float
:rtype: None
````

````{py:method} waveCorrection() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} setWaveCorrection(flag) -> None





:param self: 
:type self: 
:param flag: 
:type flag: bool
:rtype: None
````

````{py:method} interpolationFlags() -> retval





:param self: 
:type self: 
:rtype: InterpolationFlags
````

````{py:method} setInterpolationFlags(interp_flags) -> None





:param self: 
:type self: 
:param interp_flags: 
:type interp_flags: InterpolationFlags
:rtype: None
````

````{py:method} workScale() -> retval





:param self: 
:type self: 
:rtype: float
````


`````


`````{py:class} Subdiv2D




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, rect: cv2.typing.Rect)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param rect: 
:type rect: cv2.typing.Rect
:rtype: None
````

````{py:method} insert(pt) -> retval

Insert multiple points into a Delaunay triangulation.


The function inserts a single point into a subdivision and modifies the subdivision topology appropriately. If a point with the same coordinates exists already, no new point is added. 
insert(ptvec) -> None 
The function inserts a vector of points into a subdivision and modifies the subdivision topology appropriately. 
```{note}
If the point is outside of the triangulation specified rect a runtime error is raised.
```


:param self: 
:type self: 
:param pt: Point to insert.
:type pt: cv2.typing.Point2f
:param ptvec: Points to insert.
:type ptvec: 
:return: the ID of the point.
:rtype: int
````

````{py:method} insert(pt) -> retval

Insert multiple points into a Delaunay triangulation.


The function inserts a single point into a subdivision and modifies the subdivision topology appropriately. If a point with the same coordinates exists already, no new point is added. 
insert(ptvec) -> None 
The function inserts a vector of points into a subdivision and modifies the subdivision topology appropriately. 
```{note}
If the point is outside of the triangulation specified rect a runtime error is raised.
```


:param self: 
:type self: 
:param ptvec: Points to insert.
:type ptvec: _typing.Sequence[cv2.typing.Point2f]
:param pt: Point to insert.
:type pt: 
:return: the ID of the point.
:rtype: None
````

````{py:method} initDelaunay(rect) -> None
Creates a new empty Delaunay subdivision




:param self: 
:type self: 
:param rect: Rectangle that includes all of the 2D points that are to be added to the subdivision.
:type rect: cv2.typing.Rect
:rtype: None
````

````{py:method} locate(pt) -> retval, edge, vertex
Returns the location of a point within a Delaunay triangulation.


The function locates the input point within the subdivision and gives one of the triangle edges or vertices. 


:param self: 
:type self: 
:param pt: Point to locate.
:type pt: cv2.typing.Point2f
:param edge: Output edge that the point belongs to or is located to the right of it.
:type edge: 
:param vertex: Optional output vertex the input point coincides with.
:type vertex: 
:return: an integer which specify one of the following five cases for point location:-  The point falls into some facet. The function returns #PTLOC_INSIDE and edge will contain one of edges of the facet. -  The point falls onto the edge. The function returns #PTLOC_ON_EDGE and edge will contain this edge. -  The point coincides with one of the subdivision vertices. The function returns #PTLOC_VERTEX and vertex will contain a pointer to the vertex. -  The point is outside the subdivision reference rectangle. The function returns #PTLOC_OUTSIDE_RECT and no pointers are filled. -  One of input arguments is invalid. A runtime error is raised or, if silent or "parent" error processing mode is selected, #PTLOC_ERROR is returned. 
:rtype: tuple[int, int, int]
````

````{py:method} findNearest(pt) -> retval, nearestPt
Finds the subdivision vertex closest to the given point.


The function is another function that locates the input point within the subdivision. It finds the subdivision vertex that is the closest to the input point. It is not necessarily one of vertices of the facet containing the input point, though the facet (located using locate() ) is used as a starting point. 


:param self: 
:type self: 
:param pt: Input point.
:type pt: cv2.typing.Point2f
:param nearestPt: Output subdivision vertex point.
:type nearestPt: 
:return: vertex ID.
:rtype: tuple[int, cv2.typing.Point2f]
````

````{py:method} getEdgeList() -> edgeList
Returns a list of all edges.


The function gives each edge as a 4 numbers vector, where each two are one of the edge vertices. i.e. org_x = v[0], org_y = v[1], dst_x = v[2], dst_y = v[3]. 


:param self: 
:type self: 
:param edgeList: Output vector.
:type edgeList: 
:rtype: _typing.Sequence[cv2.typing.Vec4f]
````

````{py:method} getLeadingEdgeList() -> leadingEdgeList
Returns a list of the leading edge ID connected to each triangle.


The function gives one edge ID for each triangle. 


:param self: 
:type self: 
:param leadingEdgeList: Output vector.
:type leadingEdgeList: 
:rtype: _typing.Sequence[int]
````

````{py:method} getTriangleList() -> triangleList
Returns a list of all triangles.


The function gives each triangle as a 6 numbers vector, where each two are one of the triangle vertices. i.e. p1_x = v[0], p1_y = v[1], p2_x = v[2], p2_y = v[3], p3_x = v[4], p3_y = v[5]. 


:param self: 
:type self: 
:param triangleList: Output vector.
:type triangleList: 
:rtype: _typing.Sequence[cv2.typing.Vec6f]
````

````{py:method} getVoronoiFacetList(idx) -> facetList, facetCenters
Returns a list of all Voronoi facets.




:param self: 
:type self: 
:param idx: Vector of vertices IDs to consider. For all vertices you can pass empty vector.
:type idx: _typing.Sequence[int]
:param facetList: Output vector of the Voronoi facets.
:type facetList: 
:param facetCenters: Output vector of the Voronoi facets center points.
:type facetCenters: 
:rtype: tuple[_typing.Sequence[_typing.Sequence[cv2.typing.Point2f]], _typing.Sequence[cv2.typing.Point2f]]
````

````{py:method} getVertex(vertex) -> retval, firstEdge
Returns vertex location from vertex ID.




:param self: 
:type self: 
:param vertex: vertex ID.
:type vertex: int
:param firstEdge: Optional. The first edge ID which is connected to the vertex.
:type firstEdge: 
:return: vertex (x,y)
:rtype: tuple[cv2.typing.Point2f, int]
````

````{py:method} getEdge(edge, nextEdgeType) -> retval
Returns one of the edges related to the given edge.


![sample output](pics/quadedge.png) 


:param self: 
:type self: 
:param edge: Subdivision edge ID.
:type edge: int
:param nextEdgeType: Parameter specifying which of the related edges to return.The following values are possible: -   NEXT_AROUND_ORG next around the edge origin ( eOnext on the picture below if e is the input edge) -   NEXT_AROUND_DST next around the edge vertex ( eDnext ) -   PREV_AROUND_ORG previous around the edge origin (reversed eRnext ) -   PREV_AROUND_DST previous around the edge destination (reversed eLnext ) -   NEXT_AROUND_LEFT next around the left facet ( eLnext ) -   NEXT_AROUND_RIGHT next around the right facet ( eRnext ) -   PREV_AROUND_LEFT previous around the left facet (reversed eOnext ) -   PREV_AROUND_RIGHT previous around the right facet (reversed eDnext ) 
:type nextEdgeType: int
:return: edge ID related to the input edge.
:rtype: int
````

````{py:method} nextEdge(edge) -> retval
Returns next edge around the edge origin.




:param self: 
:type self: 
:param edge: Subdivision edge ID.
:type edge: int
:return: an integer which is next edge ID around the edge origin: eOnext on thepicture above if e is the input edge). 
:rtype: int
````

````{py:method} rotateEdge(edge, rotate) -> retval
Returns another edge of the same quad-edge.




:param self: 
:type self: 
:param edge: Subdivision edge ID.
:type edge: int
:param rotate: Parameter specifying which of the edges of the same quad-edge as the inputone to return. The following values are possible: -   0 - the input edge ( e on the picture below if e is the input edge) -   1 - the rotated edge ( eRot ) -   2 - the reversed edge (reversed e (in green)) -   3 - the reversed rotated edge (reversed eRot (in green)) 
:type rotate: int
:return: one of the edges ID of the same quad-edge as the input edge.
:rtype: int
````

````{py:method} symEdge(edge) -> retval





:param self: 
:type self: 
:param edge: 
:type edge: int
:rtype: int
````

````{py:method} edgeOrg(edge) -> retval, orgpt
Returns the edge origin.




:param self: 
:type self: 
:param edge: Subdivision edge ID.
:type edge: int
:param orgpt: Output vertex location.
:type orgpt: 
:return: vertex ID.
:rtype: tuple[int, cv2.typing.Point2f]
````

````{py:method} edgeDst(edge) -> retval, dstpt
Returns the edge destination.




:param self: 
:type self: 
:param edge: Subdivision edge ID.
:type edge: int
:param dstpt: Output vertex location.
:type dstpt: 
:return: vertex ID.
:rtype: tuple[int, cv2.typing.Point2f]
````


`````


`````{py:class} TickMeter




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} start() -> None





:param self: 
:type self: 
:rtype: None
````

````{py:method} stop() -> None





:param self: 
:type self: 
:rtype: None
````

````{py:method} getTimeTicks() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} getTimeMicro() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} getTimeMilli() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} getTimeSec() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} getCounter() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} getFPS() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} getAvgTimeSec() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} getAvgTimeMilli() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} reset() -> None





:param self: 
:type self: 
:rtype: None
````


`````


`````{py:class} Tonemap




````{py:method} process(src[, dst]) -> dst

Tonemaps image




:param self: 
:type self: 
:param src: source image - CV_32FC3 Mat (float 32 bits 3 channels)
:type src: cv2.typing.MatLike
:param dst: destination image - CV_32FC3 Mat with values in [0, 1] range
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} process(src[, dst]) -> dst

Tonemaps image




:param self: 
:type self: 
:param src: source image - CV_32FC3 Mat (float 32 bits 3 channels)
:type src: UMat
:param dst: destination image - CV_32FC3 Mat with values in [0, 1] range
:type dst: UMat | None
:rtype: UMat
````

````{py:method} getGamma() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setGamma(gamma) -> None





:param self: 
:type self: 
:param gamma: 
:type gamma: float
:rtype: None
````


`````


`````{py:class} TonemapDrago




````{py:method} getSaturation() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setSaturation(saturation) -> None





:param self: 
:type self: 
:param saturation: 
:type saturation: float
:rtype: None
````

````{py:method} getBias() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setBias(bias) -> None





:param self: 
:type self: 
:param bias: 
:type bias: float
:rtype: None
````


`````


`````{py:class} TonemapMantiuk




````{py:method} getScale() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setScale(scale) -> None





:param self: 
:type self: 
:param scale: 
:type scale: float
:rtype: None
````

````{py:method} getSaturation() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setSaturation(saturation) -> None





:param self: 
:type self: 
:param saturation: 
:type saturation: float
:rtype: None
````


`````


`````{py:class} TonemapReinhard




````{py:method} getIntensity() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setIntensity(intensity) -> None





:param self: 
:type self: 
:param intensity: 
:type intensity: float
:rtype: None
````

````{py:method} getLightAdaptation() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setLightAdaptation(light_adapt) -> None





:param self: 
:type self: 
:param light_adapt: 
:type light_adapt: float
:rtype: None
````

````{py:method} getColorAdaptation() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setColorAdaptation(color_adapt) -> None





:param self: 
:type self: 
:param color_adapt: 
:type color_adapt: float
:rtype: None
````


`````


`````{py:class} Tracker




````{py:method} init(image, boundingBox) -> None

Initialize the tracker with a known bounding box that surrounded the target




:param self: 
:type self: 
:param image: The initial frame
:type image: cv2.typing.MatLike
:param boundingBox: The initial bounding box
:type boundingBox: cv2.typing.Rect
:rtype: None
````

````{py:method} init(image, boundingBox) -> None

Initialize the tracker with a known bounding box that surrounded the target




:param self: 
:type self: 
:param image: The initial frame
:type image: UMat
:param boundingBox: The initial bounding box
:type boundingBox: cv2.typing.Rect
:rtype: None
````

````{py:method} update(image) -> retval, boundingBox

Update the tracker, find the new most likely bounding box for the target




:param self: 
:type self: 
:param image: The current frame
:type image: cv2.typing.MatLike
:param boundingBox: The bounding box that represent the new target location, if true was returned, notmodified otherwise 
:type boundingBox: 
:return: True means that target was located and false means that tracker cannot locate target incurrent frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed missing from the frame (say, out of sight) 
:rtype: tuple[bool, cv2.typing.Rect]
````

````{py:method} update(image) -> retval, boundingBox

Update the tracker, find the new most likely bounding box for the target




:param self: 
:type self: 
:param image: The current frame
:type image: UMat
:param boundingBox: The bounding box that represent the new target location, if true was returned, notmodified otherwise 
:type boundingBox: 
:return: True means that target was located and false means that tracker cannot locate target incurrent frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed missing from the frame (say, out of sight) 
:rtype: tuple[bool, cv2.typing.Rect]
````


`````


`````{py:class} TrackerDaSiamRPN




````{py:method} create([, parameters]) -> retval
:classmethod:
Constructor




:param cls: 
:type cls: 
:param parameters: DaSiamRPN parameters TrackerDaSiamRPN::Params
:type parameters: TrackerDaSiamRPN.Params
:rtype: TrackerDaSiamRPN
````

````{py:method} getTrackingScore() -> retval
Return tracking score




:param self: 
:type self: 
:rtype: float
````


`````


`````{py:class} 





`````


`````{py:class} TrackerGOTURN




````{py:method} create([, parameters]) -> retval
:classmethod:
Constructor




:param cls: 
:type cls: 
:param parameters: GOTURN parameters TrackerGOTURN::Params
:type parameters: TrackerGOTURN.Params
:rtype: TrackerGOTURN
````


`````


`````{py:class} 





`````


`````{py:class} TrackerMIL




````{py:method} create([, parameters]) -> retval
:classmethod:
Create MIL tracker instance




:param cls: 
:type cls: 
:param parameters: MIL parameters TrackerMIL::Params
:type parameters: TrackerMIL.Params
:rtype: TrackerMIL
````


`````


`````{py:class} 





`````


`````{py:class} TrackerNano




````{py:method} create([, parameters]) -> retval
:classmethod:
Constructor




:param cls: 
:type cls: 
:param parameters: NanoTrack parameters TrackerNano::Params
:type parameters: TrackerNano.Params
:rtype: TrackerNano
````

````{py:method} getTrackingScore() -> retval
Return tracking score




:param self: 
:type self: 
:rtype: float
````


`````


`````{py:class} 





`````


`````{py:class} TrackerVit




````{py:method} create([, parameters]) -> retval
:classmethod:
Constructor




:param cls: 
:type cls: 
:param parameters: vit tracker parameters TrackerVit::Params
:type parameters: TrackerVit.Params
:rtype: TrackerVit
````

````{py:method} getTrackingScore() -> retval
Return tracking score




:param self: 
:type self: 
:rtype: float
````


`````


`````{py:class} 





`````


`````{py:class} UMat




````{py:method} __init__(self, usageFlags: UMatUsageFlags=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param usageFlags: 
:type usageFlags: UMatUsageFlags
:rtype: None
````

````{py:method} __init__(self, rows: int, cols: int, type: int, usageFlags: UMatUsageFlags=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param rows: 
:type rows: int
:param cols: 
:type cols: int
:param type: 
:type type: int
:param usageFlags: 
:type usageFlags: UMatUsageFlags
:rtype: None
````

````{py:method} __init__(self, size: cv2.typing.Size, type: int, usageFlags: UMatUsageFlags=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param type: 
:type type: int
:param usageFlags: 
:type usageFlags: UMatUsageFlags
:rtype: None
````

````{py:method} __init__(self, rows: int, cols: int, type: int, s: cv2.typing.Scalar, usageFlags: UMatUsageFlags=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param rows: 
:type rows: int
:param cols: 
:type cols: int
:param type: 
:type type: int
:param s: 
:type s: cv2.typing.Scalar
:param usageFlags: 
:type usageFlags: UMatUsageFlags
:rtype: None
````

````{py:method} __init__(self, size: cv2.typing.Size, type: int, s: cv2.typing.Scalar, usageFlags: UMatUsageFlags=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param type: 
:type type: int
:param s: 
:type s: cv2.typing.Scalar
:param usageFlags: 
:type usageFlags: UMatUsageFlags
:rtype: None
````

````{py:method} __init__(self, m: UMat)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param m: 
:type m: UMat
:rtype: None
````

````{py:method} __init__(self, m: UMat, rowRange: cv2.typing.Range, colRange: cv2.typing.Range=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param m: 
:type m: UMat
:param rowRange: 
:type rowRange: cv2.typing.Range
:param colRange: 
:type colRange: cv2.typing.Range
:rtype: None
````

````{py:method} __init__(self, m: UMat, roi: cv2.typing.Rect)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param m: 
:type m: UMat
:param roi: 
:type roi: cv2.typing.Rect
:rtype: None
````

````{py:method} __init__(self, m: UMat, ranges: _typing.Sequence[cv2.typing.Range])




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param m: 
:type m: UMat
:param ranges: 
:type ranges: _typing.Sequence[cv2.typing.Range]
:rtype: None
````

````{py:method} queue() -> retval
:staticmethod:





:rtype: cv2.typing.IntPointer
````

````{py:method} context() -> retval
:staticmethod:





:rtype: cv2.typing.IntPointer
````

````{py:method} get() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.MatLike
````

````{py:method} isContinuous() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isSubmatrix() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} handle(accessFlags) -> retval





:param self: 
:type self: 
:param accessFlags: 
:type accessFlags: AccessFlag
:rtype: cv2.typing.IntPointer
````

```{py:attribute} offset
:type: int
```


`````


`````{py:class} UsacParams




````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

```{py:attribute} confidence
:type: float
```

```{py:attribute} isParallel
:type: bool
```

```{py:attribute} loIterations
:type: int
```

```{py:attribute} loMethod
:type: LocalOptimMethod
```

```{py:attribute} loSampleSize
:type: int
```

```{py:attribute} maxIterations
:type: int
```

```{py:attribute} neighborsSearch
:type: NeighborSearchMethod
```

```{py:attribute} randomGeneratorState
:type: int
```

```{py:attribute} sampler
:type: SamplingMethod
```

```{py:attribute} score
:type: ScoreMethod
```

```{py:attribute} threshold
:type: float
```

```{py:attribute} final_polisher
:type: PolishingMethod
```

```{py:attribute} final_polisher_iterations
:type: int
```


`````


`````{py:class} VariationalRefinement




````{py:method} create() -> retval
:classmethod:
Creates an instance of VariationalRefinement




:param cls: 
:type cls: 
:rtype: VariationalRefinement
````

````{py:method} calcUV(I0, I1, flow_u, flow_v) -> flow_u, flow_v

@ref calc function overload to handle separate horizontal (u) and vertical (v) flow components(to avoid extra splits/merges) 




:param self: 
:type self: 
:param I0: 
:type I0: cv2.typing.MatLike
:param I1: 
:type I1: cv2.typing.MatLike
:param flow_u: 
:type flow_u: cv2.typing.MatLike
:param flow_v: 
:type flow_v: cv2.typing.MatLike
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````

````{py:method} calcUV(I0, I1, flow_u, flow_v) -> flow_u, flow_v

@ref calc function overload to handle separate horizontal (u) and vertical (v) flow components(to avoid extra splits/merges) 




:param self: 
:type self: 
:param I0: 
:type I0: UMat
:param I1: 
:type I1: UMat
:param flow_u: 
:type flow_u: UMat
:param flow_v: 
:type flow_v: UMat
:rtype: tuple[UMat, UMat]
````

````{py:method} getFixedPointIterations() -> retval
Number of outer (fixed-point) iterations in the minimization procedure.


**See also:** setFixedPointIterations


:param self: 
:type self: 
:rtype: int
````

````{py:method} setFixedPointIterations(val) -> None



@copybrief getFixedPointIterations @see getFixedPointIterations 


:param self: 
:type self: 
:param val: 
:type val: int
:rtype: None
````

````{py:method} getSorIterations() -> retval
Number of inner successive over-relaxation (SOR) iterationsin the minimization procedure to solve the respective linear system. 


**See also:** setSorIterations


:param self: 
:type self: 
:rtype: int
````

````{py:method} setSorIterations(val) -> None



@copybrief getSorIterations @see getSorIterations 


:param self: 
:type self: 
:param val: 
:type val: int
:rtype: None
````

````{py:method} getOmega() -> retval
Relaxation factor in SOR


**See also:** setOmega


:param self: 
:type self: 
:rtype: float
````

````{py:method} setOmega(val) -> None



@copybrief getOmega @see getOmega 


:param self: 
:type self: 
:param val: 
:type val: float
:rtype: None
````

````{py:method} getAlpha() -> retval
Weight of the smoothness term


**See also:** setAlpha


:param self: 
:type self: 
:rtype: float
````

````{py:method} setAlpha(val) -> None



@copybrief getAlpha @see getAlpha 


:param self: 
:type self: 
:param val: 
:type val: float
:rtype: None
````

````{py:method} getDelta() -> retval
Weight of the color constancy term


**See also:** setDelta


:param self: 
:type self: 
:rtype: float
````

````{py:method} setDelta(val) -> None



@copybrief getDelta @see getDelta 


:param self: 
:type self: 
:param val: 
:type val: float
:rtype: None
````

````{py:method} getGamma() -> retval
Weight of the gradient constancy term


**See also:** setGamma


:param self: 
:type self: 
:rtype: float
````

````{py:method} setGamma(val) -> None



@copybrief getGamma @see getGamma 


:param self: 
:type self: 
:param val: 
:type val: float
:rtype: None
````


`````


`````{py:class} VideoCapture




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, filename: str, apiPreference: int=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param apiPreference: 
:type apiPreference: int
:rtype: None
````

````{py:method} __init__(self, filename: str, apiPreference: int, params: _typing.Sequence[int])




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param apiPreference: 
:type apiPreference: int
:param params: 
:type params: _typing.Sequence[int]
:rtype: None
````

````{py:method} __init__(self, index: int, apiPreference: int=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param index: 
:type index: int
:param apiPreference: 
:type apiPreference: int
:rtype: None
````

````{py:method} __init__(self, index: int, apiPreference: int, params: _typing.Sequence[int])




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param index: 
:type index: int
:param apiPreference: 
:type apiPreference: int
:param params: 
:type params: _typing.Sequence[int]
:rtype: None
````

````{py:method} open(filename[, apiPreference]) -> retval

 Opens a camera for video capturing with API Preference and parameters


@overload 
Parameters are same as the constructor VideoCapture(const String& filename, int apiPreference = CAP_ANY) 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(filename, apiPreference, params) -> retval 
@overload 
The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`. See cv::VideoCaptureProperties 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(index[, apiPreference]) -> retval 
@overload 
Parameters are same as the constructor VideoCapture(int index, int apiPreference = CAP_ANY) 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(index, apiPreference, params) -> retval 
@overload 
The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`. See cv::VideoCaptureProperties 
The method first calls VideoCapture::release to close the already opened file or camera. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param apiPreference: 
:type apiPreference: int
:return: `true` if the camera has been successfully opened.
:rtype: bool
````

````{py:method} open(filename[, apiPreference]) -> retval

 Opens a camera for video capturing with API Preference and parameters


@overload 
Parameters are same as the constructor VideoCapture(const String& filename, int apiPreference = CAP_ANY) 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(filename, apiPreference, params) -> retval 
@overload 
The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`. See cv::VideoCaptureProperties 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(index[, apiPreference]) -> retval 
@overload 
Parameters are same as the constructor VideoCapture(int index, int apiPreference = CAP_ANY) 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(index, apiPreference, params) -> retval 
@overload 
The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`. See cv::VideoCaptureProperties 
The method first calls VideoCapture::release to close the already opened file or camera. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param apiPreference: 
:type apiPreference: int
:param params: 
:type params: _typing.Sequence[int]
:return: `true` if the camera has been successfully opened.
:rtype: bool
````

````{py:method} open(filename[, apiPreference]) -> retval

 Opens a camera for video capturing with API Preference and parameters


@overload 
Parameters are same as the constructor VideoCapture(const String& filename, int apiPreference = CAP_ANY) 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(filename, apiPreference, params) -> retval 
@overload 
The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`. See cv::VideoCaptureProperties 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(index[, apiPreference]) -> retval 
@overload 
Parameters are same as the constructor VideoCapture(int index, int apiPreference = CAP_ANY) 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(index, apiPreference, params) -> retval 
@overload 
The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`. See cv::VideoCaptureProperties 
The method first calls VideoCapture::release to close the already opened file or camera. 


:param self: 
:type self: 
:param index: 
:type index: int
:param apiPreference: 
:type apiPreference: int
:return: `true` if the camera has been successfully opened.
:rtype: bool
````

````{py:method} open(filename[, apiPreference]) -> retval

 Opens a camera for video capturing with API Preference and parameters


@overload 
Parameters are same as the constructor VideoCapture(const String& filename, int apiPreference = CAP_ANY) 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(filename, apiPreference, params) -> retval 
@overload 
The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`. See cv::VideoCaptureProperties 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(index[, apiPreference]) -> retval 
@overload 
Parameters are same as the constructor VideoCapture(int index, int apiPreference = CAP_ANY) 
The method first calls VideoCapture::release to close the already opened file or camera. 
open(index, apiPreference, params) -> retval 
@overload 
The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`. See cv::VideoCaptureProperties 
The method first calls VideoCapture::release to close the already opened file or camera. 


:param self: 
:type self: 
:param index: 
:type index: int
:param apiPreference: 
:type apiPreference: int
:param params: 
:type params: _typing.Sequence[int]
:return: `true` if the camera has been successfully opened.
:rtype: bool
````

````{py:method} retrieve([, image[, flag]]) -> retval, image

Decodes and returns the grabbed video frame.


The method decodes and returns the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the method returns false and the function returns an empty image (with %cv::Mat, test it with Mat::empty()). 
**See also:** read()
```{note}
In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the videocapturing structure. It is not allowed to modify or release the image! You can copy the frame using cvCloneImage and then do whatever you want with the copy. 
```


:param self: 
:type self: 
:param image: 
:type image: cv2.typing.MatLike | None
:param flag: it could be a frame index or a driver specific flag
:type flag: int
:param [out]: image the video frame is returned here. If no frames has been grabbed the image will be empty.
:type [out]: 
:return: `false` if no frames has been grabbed
:rtype: tuple[bool, cv2.typing.MatLike]
````

````{py:method} retrieve([, image[, flag]]) -> retval, image

Decodes and returns the grabbed video frame.


The method decodes and returns the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the method returns false and the function returns an empty image (with %cv::Mat, test it with Mat::empty()). 
**See also:** read()
```{note}
In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the videocapturing structure. It is not allowed to modify or release the image! You can copy the frame using cvCloneImage and then do whatever you want with the copy. 
```


:param self: 
:type self: 
:param image: 
:type image: UMat | None
:param flag: it could be a frame index or a driver specific flag
:type flag: int
:param [out]: image the video frame is returned here. If no frames has been grabbed the image will be empty.
:type [out]: 
:return: `false` if no frames has been grabbed
:rtype: tuple[bool, UMat]
````

````{py:method} read([, image]) -> retval, image

Grabs, decodes and returns the next video frame.


The method/function combines VideoCapture::grab() and VideoCapture::retrieve() in one call. This is the most convenient method for reading video files or capturing data from decode and returns the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the method returns false and the function returns empty image (with %cv::Mat, test it with Mat::empty()). 
```{note}
In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the videocapturing structure. It is not allowed to modify or release the image! You can copy the frame using cvCloneImage and then do whatever you want with the copy. 
```


:param self: 
:type self: 
:param image: 
:type image: cv2.typing.MatLike | None
:param [out]: image the video frame is returned here. If no frames has been grabbed the image will be empty.
:type [out]: 
:return: `false` if no frames has been grabbed
:rtype: tuple[bool, cv2.typing.MatLike]
````

````{py:method} read([, image]) -> retval, image

Grabs, decodes and returns the next video frame.


The method/function combines VideoCapture::grab() and VideoCapture::retrieve() in one call. This is the most convenient method for reading video files or capturing data from decode and returns the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the method returns false and the function returns empty image (with %cv::Mat, test it with Mat::empty()). 
```{note}
In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the videocapturing structure. It is not allowed to modify or release the image! You can copy the frame using cvCloneImage and then do whatever you want with the copy. 
```


:param self: 
:type self: 
:param image: 
:type image: UMat | None
:param [out]: image the video frame is returned here. If no frames has been grabbed the image will be empty.
:type [out]: 
:return: `false` if no frames has been grabbed
:rtype: tuple[bool, UMat]
````

````{py:method} isOpened() -> retval
Returns true if video capturing has been initialized already.


If the previous call to VideoCapture constructor or VideoCapture::open() succeeded, the method returns true. 


:param self: 
:type self: 
:rtype: bool
````

````{py:method} release() -> None
Closes video file or capturing device.


The method is automatically called by subsequent VideoCapture::open and by VideoCapture destructor. 
The C function also deallocates memory and clears \*capture pointer. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} grab() -> retval
Grabs the next frame from video file or capturing device.


The method/function grabs the next frame from video file or camera and returns true (non-zero) in the case of success. 
The primary use of the function is in multi-camera environments, especially when the cameras do not have hardware synchronization. That is, you call VideoCapture::grab() for each camera and after that call the slower method VideoCapture::retrieve() to decode and get frame from each camera. This way the overhead on demosaicing or motion jpeg decompression etc. is eliminated and the retrieved frames from different cameras will be closer in time. 
Also, when a connected camera is multi-head (for example, a stereo camera or a Kinect device), the correct way of retrieving data from it is to call VideoCapture::grab() first and then call VideoCapture::retrieve() one or more times with different values of the channel parameter. 
@ref tutorial_kinect_openni 


:param self: 
:type self: 
:return: `true` (non-zero) in the case of success.
:rtype: bool
````

````{py:method} set(propId, value) -> retval
Sets a property in the VideoCapture.


```{note}
Even if it returns `true` this doesn't ensure that the propertyvalue has been accepted by the capture device. See note in VideoCapture::get() 
```


:param self: 
:type self: 
:param propId: Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...)or one from @ref videoio_flags_others 
:type propId: int
:param value: Value of the property.
:type value: float
:return: `true` if the property is supported by backend used by the VideoCapture instance.
:rtype: bool
````

````{py:method} get(propId) -> retval
Returns the specified VideoCapture property


```{note}
Reading / writing properties involves many layers. Some unexpected result might happensalong this chain. 
```txt
VideoCapture -> API Backend -> Operating System -> Device Driver -> Device Hardware
```
The returned value might be different from what really used by the device or it could be encoded using device dependent rules (eg. steps or percentage). Effective behaviour depends from device driver and API Backend 
```


:param self: 
:type self: 
:param propId: Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...)or one from @ref videoio_flags_others 
:type propId: int
:return: Value for the specified property. Value 0 is returned when querying a property that isnot supported by the backend used by the VideoCapture instance. 
:rtype: float
````

````{py:method} getBackendName() -> retval
Returns used backend API name


```{note}
Stream should be opened.
```


:param self: 
:type self: 
:rtype: str
````

````{py:method} setExceptionMode(enable) -> None



Switches exceptions mode 
methods raise exceptions if not successful instead of returning an error code 


:param self: 
:type self: 
:param enable: 
:type enable: bool
:rtype: None
````

````{py:method} getExceptionMode() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} waitAny(streams[, timeoutNs]) -> retval, readyIndex
:staticmethod:
Wait for ready frames from VideoCapture.


@throws Exception %Exception on stream errors (check .isOpened() to filter out malformed streams) or VideoCapture type is not supported 
The primary use of the function is in multi-camera environments. The method fills the ready state vector, grabs video frame, if camera is ready. 
After this call use VideoCapture::retrieve() to decode and fetch frame data. 


:param streams: input video streams
:type streams: _typing.Sequence[VideoCapture]
:param timeoutNs: number of nanoseconds (0 - infinite)
:type timeoutNs: int
:param readyIndex: stream indexes with grabbed frames (ready to use .retrieve() to fetch actual frame)
:type readyIndex: 
:return: `true` if streamReady is not empty
:rtype: tuple[bool, _typing.Sequence[int]]
````


`````


`````{py:class} VideoWriter




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, filename: str, fourcc: int, fps: float, frameSize: cv2.typing.Size, isColor: bool=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param fourcc: 
:type fourcc: int
:param fps: 
:type fps: float
:param frameSize: 
:type frameSize: cv2.typing.Size
:param isColor: 
:type isColor: bool
:rtype: None
````

````{py:method} __init__(self, filename: str, apiPreference: int, fourcc: int, fps: float, frameSize: cv2.typing.Size, isColor: bool=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param apiPreference: 
:type apiPreference: int
:param fourcc: 
:type fourcc: int
:param fps: 
:type fps: float
:param frameSize: 
:type frameSize: cv2.typing.Size
:param isColor: 
:type isColor: bool
:rtype: None
````

````{py:method} __init__(self, filename: str, fourcc: int, fps: float, frameSize: cv2.typing.Size, params: _typing.Sequence[int])




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param fourcc: 
:type fourcc: int
:param fps: 
:type fps: float
:param frameSize: 
:type frameSize: cv2.typing.Size
:param params: 
:type params: _typing.Sequence[int]
:rtype: None
````

````{py:method} __init__(self, filename: str, apiPreference: int, fourcc: int, fps: float, frameSize: cv2.typing.Size, params: _typing.Sequence[int])




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param apiPreference: 
:type apiPreference: int
:param fourcc: 
:type fourcc: int
:param fps: 
:type fps: float
:param frameSize: 
:type frameSize: cv2.typing.Size
:param params: 
:type params: _typing.Sequence[int]
:rtype: None
````

````{py:method} open(filename, fourcc, fps, frameSize[, isColor]) -> retval

Initializes or reinitializes video writer.


The method opens video writer. Parameters are the same as in the constructor VideoWriter::VideoWriter. 
The method first calls VideoWriter::release to close the already opened file. 
open(filename, apiPreference, fourcc, fps, frameSize[, isColor]) -> retval @overload 
open(filename, fourcc, fps, frameSize, params) -> retval @overload 
open(filename, apiPreference, fourcc, fps, frameSize, params) -> retval @overload 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param fourcc: 
:type fourcc: int
:param fps: 
:type fps: float
:param frameSize: 
:type frameSize: cv2.typing.Size
:param isColor: 
:type isColor: bool
:return: `true` if video writer has been successfully initialized
:rtype: bool
````

````{py:method} open(filename, fourcc, fps, frameSize[, isColor]) -> retval

Initializes or reinitializes video writer.


The method opens video writer. Parameters are the same as in the constructor VideoWriter::VideoWriter. 
The method first calls VideoWriter::release to close the already opened file. 
open(filename, apiPreference, fourcc, fps, frameSize[, isColor]) -> retval @overload 
open(filename, fourcc, fps, frameSize, params) -> retval @overload 
open(filename, apiPreference, fourcc, fps, frameSize, params) -> retval @overload 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param apiPreference: 
:type apiPreference: int
:param fourcc: 
:type fourcc: int
:param fps: 
:type fps: float
:param frameSize: 
:type frameSize: cv2.typing.Size
:param isColor: 
:type isColor: bool
:return: `true` if video writer has been successfully initialized
:rtype: bool
````

````{py:method} open(filename, fourcc, fps, frameSize[, isColor]) -> retval

Initializes or reinitializes video writer.


The method opens video writer. Parameters are the same as in the constructor VideoWriter::VideoWriter. 
The method first calls VideoWriter::release to close the already opened file. 
open(filename, apiPreference, fourcc, fps, frameSize[, isColor]) -> retval @overload 
open(filename, fourcc, fps, frameSize, params) -> retval @overload 
open(filename, apiPreference, fourcc, fps, frameSize, params) -> retval @overload 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param fourcc: 
:type fourcc: int
:param fps: 
:type fps: float
:param frameSize: 
:type frameSize: cv2.typing.Size
:param params: 
:type params: _typing.Sequence[int]
:return: `true` if video writer has been successfully initialized
:rtype: bool
````

````{py:method} open(filename, fourcc, fps, frameSize[, isColor]) -> retval

Initializes or reinitializes video writer.


The method opens video writer. Parameters are the same as in the constructor VideoWriter::VideoWriter. 
The method first calls VideoWriter::release to close the already opened file. 
open(filename, apiPreference, fourcc, fps, frameSize[, isColor]) -> retval @overload 
open(filename, fourcc, fps, frameSize, params) -> retval @overload 
open(filename, apiPreference, fourcc, fps, frameSize, params) -> retval @overload 


:param self: 
:type self: 
:param filename: 
:type filename: str
:param apiPreference: 
:type apiPreference: int
:param fourcc: 
:type fourcc: int
:param fps: 
:type fps: float
:param frameSize: 
:type frameSize: cv2.typing.Size
:param params: 
:type params: _typing.Sequence[int]
:return: `true` if video writer has been successfully initialized
:rtype: bool
````

````{py:method} write(image) -> None

Writes the next video frame


The function/method writes the specified image to video file. It must have the same size as has been specified when opening the video writer. 


:param self: 
:type self: 
:param image: The written frame. In general, color images are expected in BGR format.
:type image: cv2.typing.MatLike
:rtype: None
````

````{py:method} write(image) -> None

Writes the next video frame


The function/method writes the specified image to video file. It must have the same size as has been specified when opening the video writer. 


:param self: 
:type self: 
:param image: The written frame. In general, color images are expected in BGR format.
:type image: UMat
:rtype: None
````

````{py:method} isOpened() -> retval
Returns true if video writer has been successfully initialized.




:param self: 
:type self: 
:rtype: bool
````

````{py:method} release() -> None
Closes the video writer.


The method is automatically called by subsequent VideoWriter::open and by the VideoWriter destructor. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} set(propId, value) -> retval
Sets a property in the VideoWriter.




:param self: 
:type self: 
:param propId: Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY)or one of @ref videoio_flags_others 
:type propId: int
:param value: Value of the property.
:type value: float
:return:  `true` if the property is supported by the backend used by the VideoWriter instance.
:rtype: bool
````

````{py:method} get(propId) -> retval
Returns the specified VideoWriter property




:param self: 
:type self: 
:param propId: Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY)or one of @ref videoio_flags_others 
:type propId: int
:return: Value for the specified property. Value 0 is returned when querying a property that isnot supported by the backend used by the VideoWriter instance. 
:rtype: float
````

````{py:method} fourcc(c1, c2, c3, c4) -> retval
:staticmethod:
Concatenates 4 chars to a fourcc code


This static method constructs the fourcc code of the codec to be used in the constructor VideoWriter::VideoWriter or VideoWriter::open. 


:param c1: 
:type c1: str
:param c2: 
:type c2: str
:param c3: 
:type c3: str
:param c4: 
:type c4: str
:return: a fourcc code
:rtype: int
````

````{py:method} getBackendName() -> retval
Returns used backend API name


```{note}
Stream should be opened.
```


:param self: 
:type self: 
:rtype: str
````


`````


`````{py:class} WarperCreator





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} error




```{py:attribute} code
:type: int
```

```{py:attribute} err
:type: str
```

```{py:attribute} file
:type: str
```

```{py:attribute} func
:type: str
```

```{py:attribute} line
:type: int
```

```{py:attribute} msg
:type: str
```


`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````


`````{py:class} 





`````



## Functions
````{py:function} AKAZE_create([, descriptor_type[, descriptor_size[, descriptor_channels[, threshold[, nOctaves[, nOctaveLayers[, diffusivity[, max_points]]]]]]]]) -> retval

The AKAZE constructor




:param descriptor_type: Type of the extracted descriptor: DESCRIPTOR_KAZE,DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT. 
:type descriptor_type: 
:param descriptor_size: Size of the descriptor in bits. 0 -\> Full size
:type descriptor_size: 
:param descriptor_channels: Number of channels in the descriptor (1, 2, 3)
:type descriptor_channels: 
:param threshold: Detector response threshold to accept point
:type threshold: 
:param nOctaves: Maximum octave evolution of the image
:type nOctaves: 
:param nOctaveLayers: Default number of sublevels per scale level
:type nOctaveLayers: 
:param diffusivity: Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT orDIFF_CHARBONNIER 
:type diffusivity: 
:param max_points: Maximum amount of returned points. In case if image containsmore features, then the features with highest response are returned. Negative value means no limitation. 
:type max_points: 
:rtype: object
````


````{py:function} AffineFeature_create(backend[, maxTilt[, minTilt[, tiltStep[, rotateStepBase]]]]) -> retval






:param backend: The detector/extractor you want to use as backend.
:type backend: 
:param maxTilt: The highest power index of tilt factor. 5 is used in the paper as tilt sampling range n.
:type maxTilt: 
:param minTilt: The lowest power index of tilt factor. 0 is used in the paper.
:type minTilt: 
:param tiltStep: Tilt sampling step $\delta_t$ in Algorithm 1 in the paper.
:type tiltStep: 
:param rotateStepBase: Rotation sampling step factor b in Algorithm 1 in the paper.
:type rotateStepBase: 
:rtype: object
````


````{py:function} AgastFeatureDetector_create([, threshold[, nonmaxSuppression[, type]]]) -> retval






:rtype: object
````


````{py:function} BFMatcher_create([, normType[, crossCheck]]) -> retval

Brute-force matcher create method.




:param normType: One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms arepreferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description). 
:type normType: 
:param crossCheck: If it is false, this is will be default BFMatcher behaviour when it finds the knearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper. 
:type crossCheck: 
:rtype: object
````


````{py:function} BRISK_create([, thresh[, octaves[, patternScale]]]) -> retval

The BRISK constructor for a custom pattern, detection threshold and octaves


BRISK_create(radiusList, numberList[, dMax[, dMin[, indexChange]]]) -> retval 
BRISK_create(thresh, octaves, radiusList, numberList[, dMax[, dMin[, indexChange]]]) -> retval 


:param thresh: AGAST detection threshold score.
:type thresh: 
:param octaves: detection octaves. Use 0 to do single scale.
:type octaves: 
:param patternScale: apply this scale to the pattern used for sampling the neighbourhood of akeypoint. 
:type patternScale: 
:param radiusList: defines the radii (in pixels) where the samples around a keypoint are taken (forkeypoint scale 1). 
:type radiusList: 
:param numberList: defines the number of sampling points on the sampling circle. Must be the samesize as radiusList.. 
:type numberList: 
:param dMax: threshold for the short pairings used for descriptor formation (in pixels for keypointscale 1). 
:type dMax: 
:param dMin: threshold for the long pairings used for orientation determination (in pixels forkeypoint scale 1). 
:type dMin: 
:param indexChange: index remapping of the bits.
:type indexChange: 
:rtype: object
````


````{py:function} CV_16FC(channels) -> retval






:param channels: 
:type channels: int
:rtype: int
````


````{py:function} CV_16SC(channels) -> retval






:param channels: 
:type channels: int
:rtype: int
````


````{py:function} CV_16UC(channels) -> retval






:param channels: 
:type channels: int
:rtype: int
````


````{py:function} CV_32FC(channels) -> retval






:param channels: 
:type channels: int
:rtype: int
````


````{py:function} CV_32SC(channels) -> retval






:param channels: 
:type channels: int
:rtype: int
````


````{py:function} CV_64FC(channels) -> retval






:param channels: 
:type channels: int
:rtype: int
````


````{py:function} CV_8SC(channels) -> retval






:param channels: 
:type channels: int
:rtype: int
````


````{py:function} CV_8UC(channels) -> retval






:param channels: 
:type channels: int
:rtype: int
````


````{py:function} CV_MAKETYPE(depth, channels) -> retval






:param depth: 
:type depth: int
:param channels: 
:type channels: int
:rtype: int
````


````{py:function} CamShift(probImage, window, criteria) -> retval, window

Finds an object center, size, and orientation.


See the OpenCV sample camshiftdemo.c that tracks colored objects. 
@note -   (Python) A sample explaining the camshift tracking algorithm can be found at opencv_source_code/samples/python/camshift.py 


:param probImage: Back projection of the object histogram. See calcBackProject.
:type probImage: cv2.typing.MatLike
:param window: Initial search window.
:type window: cv2.typing.Rect
:param criteria: Stop criteria for the underlying meanShift.returns (in old interfaces) Number of iterations CAMSHIFT took to converge The function implements the CAMSHIFT object tracking algorithm @cite Bradski98 . First, it finds an object center using meanShift and then adjusts the window size and finds the optimal rotation. The function returns the rotated rectangle structure that includes the object position, size, and orientation. The next position of the search window can be obtained with RotatedRect::boundingRect() 
:type criteria: cv2.typing.TermCriteria
:rtype: tuple[cv2.typing.RotatedRect, cv2.typing.Rect]
````


````{py:function} Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges

Finds edges in an image using the Canny algorithm @cite Canny86 .


The function finds edges in the input image and marks them in the output map edges using the Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The largest value is used to find initial segments of strong edges. See <http://en.wikipedia.org/wiki/Canny_edge_detector> 
Canny(dx, dy, threshold1, threshold2[, edges[, L2gradient]]) -> edges \overload 
Finds edges in an image using the Canny algorithm with custom image gradient. 


:param image: 8-bit input image.
:type image: cv2.typing.MatLike
:param edges: output edge map; single channels 8-bit image, which has the same size as image .
:type edges: cv2.typing.MatLike | None
:param threshold1: first threshold for the hysteresis procedure.
:type threshold1: float
:param threshold2: second threshold for the hysteresis procedure.
:type threshold2: float
:param apertureSize: aperture size for the Sobel operator.
:type apertureSize: int
:param L2gradient: a flag, indicating whether a more accurate $L_2$ norm$=\sqrt{(dI/dx)^2 + (dI/dy)^2}$ should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default $L_1$ norm $=|dI/dx|+|dI/dy|$ is enough ( L2gradient=false ). 
:type L2gradient: bool
:param dx: 16-bit x derivative of input image (CV_16SC1 or CV_16SC3).
:type dx: 
:param dy: 16-bit y derivative of input image (same type as dx).
:type dy: 
:rtype: cv2.typing.MatLike
````


````{py:function} CascadeClassifier_convert(oldcascade, newcascade) -> retval






:rtype: object
````


````{py:function} DISOpticalFlow_create([, preset]) -> retval

Creates an instance of DISOpticalFlow




:param preset: one of PRESET_ULTRAFAST, PRESET_FAST and PRESET_MEDIUM
:type preset: 
:rtype: object
````


````{py:function} DescriptorMatcher_create(descriptorMatcherType) -> retval

Creates a descriptor matcher of a given type with the default parameters (using defaultconstructor). 


DescriptorMatcher_create(matcherType) -> retval 


:param descriptorMatcherType: Descriptor matcher type. Now the following matcher types aresupported: -   `BruteForce` (it uses L2 ) -   `BruteForce-L1` -   `BruteForce-Hamming` -   `BruteForce-Hamming(2)` -   `FlannBased` 
:type descriptorMatcherType: 
:rtype: object
````


````{py:function} EMD(signature1, signature2, distType[, cost[, lowerBound[, flow]]]) -> retval, lowerBound, flow

Computes the "minimal work" distance between two weighted point configurations.


The function computes the earth mover distance and/or a lower boundary of the distance between the two weighted point configurations. One of the applications described in @cite RubnerSept98, @cite Rubner2000 is multi-dimensional histogram comparison for image retrieval. EMD is a transportation problem that is solved using some modification of a simplex algorithm, thus the complexity is exponential in the worst case, though, on average it is much faster. In the case of a real metric the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used to determine roughly whether the two signatures are far enough so that they cannot relate to the same object. 


:param signature1: First signature, a $\texttt{size1}\times \texttt{dims}+1$ floating-point matrix.Each row stores the point weight followed by the point coordinates. The matrix is allowed to have a single column (weights only) if the user-defined cost matrix is used. The weights must be non-negative and have at least one non-zero value. 
:type signature1: cv2.typing.MatLike
:param signature2: Second signature of the same format as signature1 , though the number of rowsmay be different. The total weights may be different. In this case an extra "dummy" point is added to either signature1 or signature2. The weights must be non-negative and have at least one non-zero value. 
:type signature2: cv2.typing.MatLike
:param distType: Used metric. See #DistanceTypes.
:type distType: int
:param cost: User-defined $\texttt{size1}\times \texttt{size2}$ cost matrix. Also, if a cost matrixis used, lower boundary lowerBound cannot be calculated because it needs a metric function. 
:type cost: cv2.typing.MatLike | None
:param lowerBound: Optional input/output parameter: lower boundary of a distance between the twosignatures that is a distance between mass centers. The lower boundary may not be calculated if the user-defined cost matrix is used, the total weights of point configurations are not equal, or if the signatures consist of weights only (the signature matrices have a single column). You must** initialize \*lowerBound . If the calculated distance between mass centers is greater or equal to \*lowerBound (it means that the signatures are far enough), the function does not calculate EMD. In any case \*lowerBound is set to the calculated distance between mass centers on return. Thus, if you want to calculate both distance between mass centers and EMD, \*lowerBound should be set to 0. 
:type lowerBound: float | None
:param flow: Resultant $\texttt{size1} \times \texttt{size2}$ flow matrix: $\texttt{flow}_{i,j}$ isa flow from $i$ -th point of signature1 to $j$ -th point of signature2 . 
:type flow: cv2.typing.MatLike | None
:rtype: tuple[float, float, cv2.typing.MatLike]
````


````{py:function} FaceDetectorYN_create(model, config, input_size[, score_threshold[, nms_threshold[, top_k[, backend_id[, target_id]]]]]) -> retval

Creates an instance of face detector class with given parameters


FaceDetectorYN_create(framework, bufferModel, bufferConfig, input_size[, score_threshold[, nms_threshold[, top_k[, backend_id[, target_id]]]]]) -> retval @overload 


:param model: the path to the requested model
:type model: 
:param config: the path to the config file for compability, which is not requested for ONNX models
:type config: 
:param input_size: the size of the input image
:type input_size: 
:param score_threshold: the threshold to filter out bounding boxes of score smaller than the given value
:type score_threshold: 
:param nms_threshold: the threshold to suppress bounding boxes of IoU bigger than the given value
:type nms_threshold: 
:param top_k: keep top K bboxes before NMS
:type top_k: 
:param backend_id: the id of backend
:type backend_id: 
:param target_id: the id of target device
:type target_id: 
:param framework: Name of origin framework
:type framework: 
:param bufferModel: A buffer with a content of binary file with weights
:type bufferModel: 
:param bufferConfig: A buffer with a content of text file contains network configuration
:type bufferConfig: 
:rtype: object
````


````{py:function} FaceRecognizerSF_create(model, config[, backend_id[, target_id]]) -> retval

Creates an instance of this class with given parameters




:param model: the path of the onnx model used for face recognition
:type model: 
:param config: the path to the config file for compability, which is not requested for ONNX models
:type config: 
:param backend_id: the id of backend
:type backend_id: 
:param target_id: the id of target device
:type target_id: 
:rtype: object
````


````{py:function} FarnebackOpticalFlow_create([, numLevels[, pyrScale[, fastPyramids[, winSize[, numIters[, polyN[, polySigma[, flags]]]]]]]]) -> retval






:rtype: object
````


````{py:function} FastFeatureDetector_create([, threshold[, nonmaxSuppression[, type]]]) -> retval






:rtype: object
````


````{py:function} FlannBasedMatcher_create() -> retval






:rtype: object
````


````{py:function} GFTTDetector_create([, maxCorners[, qualityLevel[, minDistance[, blockSize[, useHarrisDetector[, k]]]]]]) -> retval




GFTTDetector_create(maxCorners, qualityLevel, minDistance, blockSize, gradiantSize[, useHarrisDetector[, k]]) -> retval 


:rtype: object
````


````{py:function} 






:rtype: object
````


````{py:function} 






:rtype: object
````


````{py:function} GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst

Blurs an image using a Gaussian filter.


The function convolves the source image with the specified Gaussian kernel. In-place filtering is supported. 
**See also:**  sepFilter2D, filter2D, blur, boxFilter, bilateralFilter, medianBlur


:param src: input image; the image can have any number of channels, which are processedindependently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F. 
:type src: cv2.typing.MatLike
:param dst: output image of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param ksize: Gaussian kernel size. ksize.width and ksize.height can differ but they both must bepositive and odd. Or, they can be zero's and then they are computed from sigma. 
:type ksize: cv2.typing.Size
:param sigmaX: Gaussian kernel standard deviation in X direction.
:type sigmaX: float
:param sigmaY: Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to beequal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively (see #getGaussianKernel for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY. 
:type sigmaY: float
:param borderType: pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} HOGDescriptor_getDaimlerPeopleDetector() -> retval

Returns coefficients of the classifier trained for people detection (for 48x96 windows).




:rtype: object
````


````{py:function} HOGDescriptor_getDefaultPeopleDetector() -> retval

Returns coefficients of the classifier trained for people detection (for 64x128 windows).




:rtype: object
````


````{py:function} HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles

Finds circles in a grayscale image using the Hough transform.


The function finds circles in a grayscale image using a modification of the Hough transform. 
Example: : @include snippets/imgproc_HoughLinesCircles.cpp 
It also helps to smooth image a bit unless it's already soft. For example, GaussianBlur() with 7x7 kernel and 1.5x1.5 sigma or similar blurring may help. 
```{note}
Usually the function detects the centers of circles well. However, it may fail to find correctradii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if you know it. Or, in the case of #HOUGH_GRADIENT method you may set maxRadius to a negative number to return centers only without radius search, and find the correct radius using an additional procedure. 
```
**See also:** fitEllipse, minEnclosingCircle


:param image: 8-bit, single-channel, grayscale input image.
:type image: cv2.typing.MatLike
:param circles: Output vector of found circles. Each vector is encoded as 3 or 4 elementfloating-point vector $(x, y, radius)$ or $(x, y, radius, votes)$ . 
:type circles: cv2.typing.MatLike | None
:param method: Detection method, see #HoughModes. The available methods are #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT.
:type method: int
:param dp: Inverse ratio of the accumulator resolution to the image resolution. For example, ifdp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height. For #HOUGH_GRADIENT_ALT the recommended value is dp=1.5, unless some small very circles need to be detected. 
:type dp: float
:param minDist: Minimum distance between the centers of the detected circles. If the parameter istoo small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed. 
:type minDist: float
:param param1: First method-specific parameter. In case of #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT,it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller). Note that #HOUGH_GRADIENT_ALT uses #Scharr algorithm to compute image derivatives, so the threshold value should normally be higher, such as 300 or normally exposed and contrasty images. 
:type param1: float
:param param2: Second method-specific parameter. In case of #HOUGH_GRADIENT, it is theaccumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first. In the case of #HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure. The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine. If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less. But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles. 
:type param2: float
:param minRadius: Minimum circle radius.
:type minRadius: int
:param maxRadius: Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, #HOUGH_GRADIENT returnscenters without finding the radius. #HOUGH_GRADIENT_ALT always computes circle radiuses. 
:type maxRadius: int
:rtype: cv2.typing.MatLike
````


````{py:function} HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]]) -> lines

Finds lines in a binary image using the standard Hough transform.


The function implements the standard or standard multi-scale Hough transform algorithm for line detection. See <http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm> for a good explanation of Hough transform. 


:param image: 8-bit, single-channel binary source image. The image may be modified by the function.
:type image: cv2.typing.MatLike
:param lines: Output vector of lines. Each line is represented by a 2 or 3 element vector$(\rho, \theta)$ or $(\rho, \theta, \textrm{votes})$, where $\rho$ is the distance from the coordinate origin $(0,0)$ (top-left corner of the image), $\theta$ is the line rotation angle in radians ( $0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}$ ), and $\textrm{votes}$ is the value of accumulator. 
:type lines: cv2.typing.MatLike | None
:param rho: Distance resolution of the accumulator in pixels.
:type rho: float
:param theta: Angle resolution of the accumulator in radians.
:type theta: float
:param threshold: %Accumulator threshold parameter. Only those lines are returned that get enoughvotes ( $>\texttt{threshold}$ ). 
:type threshold: int
:param srn: For the multi-scale Hough transform, it is a divisor for the distance resolution rho.The coarse accumulator distance resolution is rho and the accurate accumulator resolution is rho/srn. If both srn=0 and stn=0, the classical Hough transform is used. Otherwise, both these parameters should be positive. 
:type srn: float
:param stn: For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
:type stn: float
:param min_theta: For standard and multi-scale Hough transform, minimum angle to check for lines.Must fall between 0 and max_theta. 
:type min_theta: float
:param max_theta: For standard and multi-scale Hough transform, an upper bound for the angle.Must fall between min_theta and CV_PI. The actual maximum angle in the accumulator may be slightly less than max_theta, depending on the parameters min_theta and theta. 
:type max_theta: float
:rtype: cv2.typing.MatLike
````


````{py:function} HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines

Finds line segments in a binary image using the probabilistic Hough transform.


The function implements the probabilistic Hough transform algorithm for line detection, described in @cite Matas00 
See the line detection example below: @include snippets/imgproc_HoughLinesP.cpp This is a sample picture the function parameters have been tuned for: 
![image](pics/building.jpg) 
And this is the output of the above program in case of the probabilistic Hough transform: 
![image](pics/houghp.png) 
**See also:** LineSegmentDetector


:param image: 8-bit, single-channel binary source image. The image may be modified by the function.
:type image: cv2.typing.MatLike
:param lines: Output vector of lines. Each line is represented by a 4-element vector$(x_1, y_1, x_2, y_2)$ , where $(x_1,y_1)$ and $(x_2, y_2)$ are the ending points of each detected line segment. 
:type lines: cv2.typing.MatLike | None
:param rho: Distance resolution of the accumulator in pixels.
:type rho: float
:param theta: Angle resolution of the accumulator in radians.
:type theta: float
:param threshold: %Accumulator threshold parameter. Only those lines are returned that get enoughvotes ( $>\texttt{threshold}$ ). 
:type threshold: int
:param minLineLength: Minimum line length. Line segments shorter than that are rejected.
:type minLineLength: float
:param maxLineGap: Maximum allowed gap between points on the same line to link them.
:type maxLineGap: float
:rtype: cv2.typing.MatLike
````


````{py:function} HoughLinesPointSet(point, lines_max, threshold, min_rho, max_rho, rho_step, min_theta, max_theta, theta_step[, lines]) -> lines

Finds lines in a set of points using the standard Hough transform.


The function finds lines in a set of points using a modification of the Hough transform. @include snippets/imgproc_HoughLinesPointSet.cpp 


:param point: Input vector of points. Each vector must be encoded as a Point vector $(x,y)$. Type must be CV_32FC2 or CV_32SC2.
:type point: cv2.typing.MatLike
:param lines: Output vector of found lines. Each vector is encoded as a vector<Vec3d> $(votes, rho, theta)$.The larger the value of 'votes', the higher the reliability of the Hough line. 
:type lines: cv2.typing.MatLike | None
:param lines_max: Max count of Hough lines.
:type lines_max: int
:param threshold: %Accumulator threshold parameter. Only those lines are returned that get enoughvotes ( $>\texttt{threshold}$ ). 
:type threshold: int
:param min_rho: Minimum value for $\rho$ for the accumulator (Note: $\rho$ can be negative. The absolute value $|\rho|$ is the distance of a line to the origin.).
:type min_rho: float
:param max_rho: Maximum value for $\rho$ for the accumulator.
:type max_rho: float
:param rho_step: Distance resolution of the accumulator.
:type rho_step: float
:param min_theta: Minimum angle value of the accumulator in radians.
:type min_theta: float
:param max_theta: Upper bound for the angle value of the accumulator in radians. The actual maximumangle may be slightly less than max_theta, depending on the parameters min_theta and theta_step. 
:type max_theta: float
:param theta_step: Angle resolution of the accumulator in radians.
:type theta_step: float
:rtype: cv2.typing.MatLike
````


````{py:function} HoughLinesWithAccumulator(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]]) -> lines

Finds lines in a binary image using the standard Hough transform and get accumulator.


```{note}
This function is for bindings use only. Use original function in C++ code
```
**See also:** HoughLines


:param image: 
:type image: cv2.typing.MatLike
:param rho: 
:type rho: float
:param theta: 
:type theta: float
:param threshold: 
:type threshold: int
:param lines: 
:type lines: cv2.typing.MatLike | None
:param srn: 
:type srn: float
:param stn: 
:type stn: float
:param min_theta: 
:type min_theta: float
:param max_theta: 
:type max_theta: float
:rtype: cv2.typing.MatLike
````


````{py:function} HuMoments(m[, hu]) -> hu




@overload 


:param m: 
:type m: cv2.typing.Moments
:param hu: 
:type hu: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} KAZE_create([, extended[, upright[, threshold[, nOctaves[, nOctaveLayers[, diffusivity]]]]]]) -> retval

The KAZE constructor




:param extended: Set to enable extraction of extended (128-byte) descriptor.
:type extended: 
:param upright: Set to enable use of upright descriptors (non rotation-invariant).
:type upright: 
:param threshold: Detector response threshold to accept point
:type threshold: 
:param nOctaves: Maximum octave evolution of the image
:type nOctaves: 
:param nOctaveLayers: Default number of sublevels per scale level
:type nOctaveLayers: 
:param diffusivity: Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT orDIFF_CHARBONNIER 
:type diffusivity: 
:rtype: object
````


````{py:function} KeyPoint_convert(keypoints[, keypointIndexes]) -> points2f




This method converts vector of keypoints to vector of points or the reverse, where each keypoint is assigned the same size and the same orientation. 
KeyPoint_convert(points2f[, size[, response[, octave[, class_id]]]]) -> keypoints @overload 


:param keypoints: Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB
:type keypoints: 
:param points2f: Array of (x,y) coordinates of each keypoint
:type points2f: 
:param keypointIndexes: Array of indexes of keypoints to be converted to points. (Acts like a mask toconvert only specified keypoints) 
:type keypointIndexes: 
:param size: keypoint diameter
:type size: 
:param response: keypoint detector response on the keypoint (that is, strength of the keypoint)
:type response: 
:param octave: pyramid octave in which the keypoint has been detected
:type octave: 
:param class_id: object id
:type class_id: 
:rtype: object
````


````{py:function} KeyPoint_overlap(kp1, kp2) -> retval




This method computes overlap for pair of keypoints. Overlap is the ratio between area of keypoint regions' intersection and area of keypoint regions' union (considering keypoint region as circle). If they don't overlap, we get zero. If they coincide at same location with same size, we get 1. 


:param kp1: First keypoint
:type kp1: 
:param kp2: Second keypoint
:type kp2: 
:rtype: object
````


````{py:function} LUT(src, lut[, dst]) -> dst

Performs a look-up table transform of an array.


The function LUT fills the output array with values from the look-up table. Indices of the entries are taken from the input array. That is, the function processes each element of src as follows: $\texttt{dst} (I)  \leftarrow \texttt{lut(src(I) + d)}$ where $d =  \fork{0}{if \(\texttt{src}\) has depth \(\texttt{CV_8U}\)}{128}{if \(\texttt{src}\) has depth \(\texttt{CV_8S}\)}$ 
**See also:**  convertScaleAbs, Mat::convertTo


:param src: input array of 8-bit elements.
:type src: cv2.typing.MatLike
:param lut: look-up table of 256 elements; in case of multi-channel input array, the table shouldeither have a single channel (in this case the same table is used for all channels) or the same number of channels as in the input array. 
:type lut: cv2.typing.MatLike
:param dst: output array of the same size and number of channels as src, and the same depth as lut.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst

Calculates the Laplacian of an image.


The function calculates the Laplacian of the source image by adding up the second x and y derivatives calculated using the Sobel operator: 
$\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}$ 
This is done when `ksize > 1`. When `ksize == 1`, the Laplacian is computed by filtering the image with the following $3 \times 3$ aperture: 
$\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}$ 
**See also:**  Sobel, Scharr


:param src: Source image.
:type src: cv2.typing.MatLike
:param dst: Destination image of the same size and the same number of channels as src .
:type dst: cv2.typing.MatLike | None
:param ddepth: Desired depth of the destination image, see @ref filter_depths "combinations".
:type ddepth: int
:param ksize: Aperture size used to compute the second-derivative filters. See #getDerivKernels fordetails. The size must be positive and odd. 
:type ksize: int
:param scale: Optional scale factor for the computed Laplacian values. By default, no scaling isapplied. See #getDerivKernels for details. 
:type scale: float
:param delta: Optional delta value that is added to the results prior to storing them in dst .
:type delta: float
:param borderType: Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} MSER_create([, delta[, min_area[, max_area[, max_variation[, min_diversity[, max_evolution[, area_threshold[, min_margin[, edge_blur_size]]]]]]]]]) -> retval

Full constructor for %MSER detector




:param delta: it compares $(size_{i}-size_{i-delta})/size_{i-delta}$
:type delta: 
:param min_area: prune the area which smaller than minArea
:type min_area: 
:param max_area: prune the area which bigger than maxArea
:type max_area: 
:param max_variation: prune the area have similar size to its children
:type max_variation: 
:param min_diversity: for color image, trace back to cut off mser with diversity less than min_diversity
:type min_diversity: 
:param max_evolution: for color image, the evolution steps
:type max_evolution: 
:param area_threshold: for color image, the area threshold to cause re-initialize
:type area_threshold: 
:param min_margin: for color image, ignore too small margin
:type min_margin: 
:param edge_blur_size: for color image, the aperture size for edge blur
:type edge_blur_size: 
:rtype: object
````


````{py:function} Mahalanobis(v1, v2, icovar) -> retval

Calculates the Mahalanobis distance between two vectors.


The function cv::Mahalanobis calculates and returns the weighted distance between two vectors: $d( \texttt{vec1} , \texttt{vec2} )= \sqrt{\sum_{i,j}{\texttt{icovar(i,j)}\cdot(\texttt{vec1}(I)-\texttt{vec2}(I))\cdot(\texttt{vec1(j)}-\texttt{vec2(j)})} }$ The covariance matrix may be calculated using the #calcCovarMatrix function and then inverted using the invert function (preferably using the #DECOMP_SVD method, as the most accurate). 


:param v1: first 1D input vector.
:type v1: cv2.typing.MatLike
:param v2: second 1D input vector.
:type v2: cv2.typing.MatLike
:param icovar: inverse covariance matrix.
:type icovar: cv2.typing.MatLike
:rtype: float
````


````{py:function} ORB_create([, nfeatures[, scaleFactor[, nlevels[, edgeThreshold[, firstLevel[, WTA_K[, scoreType[, patchSize[, fastThreshold]]]]]]]]]) -> retval

The ORB constructor




:param nfeatures: The maximum number of features to retain.
:type nfeatures: 
:param scaleFactor: Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classicalpyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer. 
:type scaleFactor: 
:param nlevels: The number of pyramid levels. The smallest level will have linear size equal toinput_image_linear_size/pow(scaleFactor, nlevels - firstLevel). 
:type nlevels: 
:param edgeThreshold: This is size of the border where the features are not detected. It shouldroughly match the patchSize parameter. 
:type edgeThreshold: 
:param firstLevel: The level of pyramid to put source image to. Previous layers are filledwith upscaled source image. 
:type firstLevel: 
:param WTA_K: The number of points that produce each element of the oriented BRIEF descriptor. Thedefault value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3). 
:type WTA_K: 
:param scoreType: The default HARRIS_SCORE means that Harris algorithm is used to rank features(the score is written to KeyPoint::score and is used to retain best nfeatures features); FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute. 
:type scoreType: 
:param patchSize: size of the patch used by the oriented BRIEF descriptor. Of course, on smallerpyramid layers the perceived image area covered by a feature will be larger. 
:type patchSize: 
:param fastThreshold: the fast threshold
:type fastThreshold: 
:rtype: object
````


````{py:function} PCABackProject(data, mean, eigenvectors[, result]) -> result




wrap PCA::backProject 


:param data: 
:type data: cv2.typing.MatLike
:param mean: 
:type mean: cv2.typing.MatLike
:param eigenvectors: 
:type eigenvectors: cv2.typing.MatLike
:param result: 
:type result: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} PCACompute(data, mean[, eigenvectors[, maxComponents]]) -> mean, eigenvectors




wrap PCA::operator() 
PCACompute(data, mean, retainedVariance[, eigenvectors]) -> mean, eigenvectors wrap PCA::operator() 


:param data: 
:type data: cv2.typing.MatLike
:param mean: 
:type mean: cv2.typing.MatLike
:param eigenvectors: 
:type eigenvectors: cv2.typing.MatLike | None
:param maxComponents: 
:type maxComponents: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} PCACompute2(data, mean[, eigenvectors[, eigenvalues[, maxComponents]]]) -> mean, eigenvectors, eigenvalues




wrap PCA::operator() and add eigenvalues output parameter 
PCACompute2(data, mean, retainedVariance[, eigenvectors[, eigenvalues]]) -> mean, eigenvectors, eigenvalues wrap PCA::operator() and add eigenvalues output parameter 


:param data: 
:type data: cv2.typing.MatLike
:param mean: 
:type mean: cv2.typing.MatLike
:param eigenvectors: 
:type eigenvectors: cv2.typing.MatLike | None
:param eigenvalues: 
:type eigenvalues: cv2.typing.MatLike | None
:param maxComponents: 
:type maxComponents: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} PCAProject(data, mean, eigenvectors[, result]) -> result




wrap PCA::project 


:param data: 
:type data: cv2.typing.MatLike
:param mean: 
:type mean: cv2.typing.MatLike
:param eigenvectors: 
:type eigenvectors: cv2.typing.MatLike
:param result: 
:type result: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} PSNR(src1, src2[, R]) -> retval

Computes the Peak Signal-to-Noise Ratio (PSNR) image quality metric.


This function calculates the Peak Signal-to-Noise Ratio (PSNR) image quality metric in decibels (dB), between two input arrays src1 and src2. The arrays must have the same type. 
The PSNR is calculated as follows: 
$ \texttt{PSNR} = 10 \cdot \log_{10}{\left( \frac{R^2}{MSE} \right) } $ 
where R is the maximum integer value of depth (e.g. 255 in the case of CV_8U data) and MSE is the mean squared error between the two arrays. 


:param src1: first input array.
:type src1: cv2.typing.MatLike
:param src2: second input array of the same size as src1.
:type src2: cv2.typing.MatLike
:param R: the maximum pixel value (255 by default)
:type R: float
:rtype: float
````


````{py:function} QRCodeEncoder_create([, parameters]) -> retval

Constructor




:param parameters: QR code encoder parameters QRCodeEncoder::Params
:type parameters: 
:rtype: object
````


````{py:function} RQDecomp3x3(src[, mtxR[, mtxQ[, Qx[, Qy[, Qz]]]]]) -> retval, mtxR, mtxQ, Qx, Qy, Qz

Computes an RQ decomposition of 3x3 matrices.


The function computes a RQ decomposition using the given rotations. This function is used in #decomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera and a rotation matrix. 
It optionally returns three rotation matrices, one for each axis, and the three Euler angles in degrees (as the return value) that could be used in OpenGL. Note, there is always more than one sequence of rotations about the three principal axes that results in the same orientation of an object, e.g. see @cite Slabaugh . Returned three rotation matrices and corresponding three Euler angles are only one of the possible solutions. 


:param src: 3x3 input matrix.
:type src: cv2.typing.MatLike
:param mtxR: Output 3x3 upper-triangular matrix.
:type mtxR: cv2.typing.MatLike | None
:param mtxQ: Output 3x3 orthogonal matrix.
:type mtxQ: cv2.typing.MatLike | None
:param Qx: Optional output 3x3 rotation matrix around x-axis.
:type Qx: cv2.typing.MatLike | None
:param Qy: Optional output 3x3 rotation matrix around y-axis.
:type Qy: cv2.typing.MatLike | None
:param Qz: Optional output 3x3 rotation matrix around z-axis.
:type Qz: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.Vec3d, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} Rodrigues(src[, dst[, jacobian]]) -> dst, jacobian

Converts a rotation matrix to a rotation vector or vice versa.


$\begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos(\theta) I + (1- \cos{\theta} ) r r^T +  \sin(\theta) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array}$ 
Inverse transformation can be also done easily, since 
$\sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2}$ 
A rotation vector is a convenient and most compact representation of a rotation matrix (since any rotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry optimization procedures like @ref calibrateCamera, @ref stereoCalibrate, or @ref solvePnP . 
```{note}
More information about the computation of the derivative of a 3D rotation matrix with respect to its exponential coordinatecan be found in: - A Compact Formula for the Derivative of a 3-D Rotation in Exponential Coordinates, Guillermo Gallego, Anthony J. Yezzi @cite Gallego2014ACF 
```
```{note}
Useful information on SE(3) and Lie Groups can be found in:- A tutorial on SE(3) transformation parameterizations and on-manifold optimization, Jose-Luis Blanco @cite blanco2010tutorial - Lie Groups for 2D and 3D Transformation, Ethan Eade @cite Eade17 - A micro Lie theory for state estimation in robotics, Joan Sol&#224;, J&#233;r&#233;mie Deray, Dinesh Atchuthan @cite Sol2018AML 
```


:param src: Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).
:type src: cv2.typing.MatLike
:param dst: Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.
:type dst: cv2.typing.MatLike | None
:param jacobian: Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partialderivatives of the output array components with respect to the input array components. 
:type jacobian: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma[, enable_precise_upscale]]]]]]) -> retval

Create SIFT with specified descriptorType.


SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, descriptorType[, enable_precise_upscale]) -> retval 
```{note}
The contrast threshold will be divided by nOctaveLayers when the filtering is applied. WhennOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09. 
```
```{note}
The contrast threshold will be divided by nOctaveLayers when the filtering is applied. WhennOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09. 
```


:param nfeatures: The number of best features to retain. The features are ranked by their scores(measured in SIFT algorithm as the local contrast) 
:type nfeatures: 
:param nOctaveLayers: The number of layers in each octave. 3 is the value used in D. Lowe paper. Thenumber of octaves is computed automatically from the image resolution. 
:type nOctaveLayers: 
:param contrastThreshold: The contrast threshold used to filter out weak features in semi-uniform(low-contrast) regions. The larger the threshold, the less features are produced by the detector. 
:type contrastThreshold: 
:param edgeThreshold: The threshold used to filter out edge-like features. Note that the its meaningis different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained). 
:type edgeThreshold: 
:param sigma: The sigma of the Gaussian applied to the input image at the octave \#0. If your imageis captured with a weak camera with soft lenses, you might want to reduce the number. 
:type sigma: 
:param enable_precise_upscale: Whether to enable precise upscaling in the scale pyramid, which mapsindex $\texttt{x}$ to $\texttt{2x}$. This prevents localization bias. The option is disabled by default. 
:type enable_precise_upscale: 
:param descriptorType: The type of descriptors. Only CV_32F and CV_8U are supported.
:type descriptorType: 
:rtype: object
````


````{py:function} SVBackSubst(w, u, vt, rhs[, dst]) -> dst




wrap SVD::backSubst 


:param w: 
:type w: cv2.typing.MatLike
:param u: 
:type u: cv2.typing.MatLike
:param vt: 
:type vt: cv2.typing.MatLike
:param rhs: 
:type rhs: cv2.typing.MatLike
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} SVDecomp(src[, w[, u[, vt[, flags]]]]) -> w, u, vt




wrap SVD::compute 


:param src: 
:type src: cv2.typing.MatLike
:param w: 
:type w: cv2.typing.MatLike | None
:param u: 
:type u: cv2.typing.MatLike | None
:param vt: 
:type vt: cv2.typing.MatLike | None
:param flags: 
:type flags: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) -> dst

Calculates the first x- or y- image derivative using Scharr operator.


The function computes the first x- or y- spatial image derivative using the Scharr operator. The call 
$\texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}$ 
is equivalent to 
$\texttt{Sobel(src, dst, ddepth, dx, dy, FILTER_SCHARR, scale, delta, borderType)} .$ 
**See also:**  cartToPolar


:param src: input image.
:type src: cv2.typing.MatLike
:param dst: output image of the same size and the same number of channels as src.
:type dst: cv2.typing.MatLike | None
:param ddepth: output image depth, see @ref filter_depths "combinations"
:type ddepth: int
:param dx: order of the derivative x.
:type dx: int
:param dy: order of the derivative y.
:type dy: int
:param scale: optional scale factor for the computed derivative values; by default, no scaling isapplied (see #getDerivKernels for details). 
:type scale: float
:param delta: optional delta value that is added to the results prior to storing them in dst.
:type delta: float
:param borderType: pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} SimpleBlobDetector_create([, parameters]) -> retval






:rtype: object
````


````{py:function} Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst

Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.


In all cases except one, the $\texttt{ksize} \times \texttt{ksize}$ separable kernel is used to calculate the derivative. When $\texttt{ksize = 1}$, the $3 \times 1$ or $1 \times 3$ kernel is used (that is, no Gaussian smoothing is done). `ksize = 1` can only be used for the first or the second x- or y- derivatives. 
There is also the special value `ksize = #FILTER_SCHARR (-1)` that corresponds to the $3\times3$ Scharr filter that may give more accurate results than the $3\times3$ Sobel. The Scharr aperture is 
$\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}$ 
for the x-derivative, or transposed for the y-derivative. 
The function calculates an image derivative by convolving the image with the appropriate kernel: 
$\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}$ 
The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3) or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first case corresponds to a kernel of: 
$\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}$ 
The second case corresponds to a kernel of: 
$\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}$ 
**See also:**  Scharr, Laplacian, sepFilter2D, filter2D, GaussianBlur, cartToPolar


:param src: input image.
:type src: cv2.typing.MatLike
:param dst: output image of the same size and the same number of channels as src .
:type dst: cv2.typing.MatLike | None
:param ddepth: output image depth, see @ref filter_depths "combinations"; in the case of8-bit input images it will result in truncated derivatives. 
:type ddepth: int
:param dx: order of the derivative x.
:type dx: int
:param dy: order of the derivative y.
:type dy: int
:param ksize: size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
:type ksize: int
:param scale: optional scale factor for the computed derivative values; by default, no scaling isapplied (see #getDerivKernels for details). 
:type scale: float
:param delta: optional delta value that is added to the results prior to storing them in dst.
:type delta: float
:param borderType: pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} SparsePyrLKOpticalFlow_create([, winSize[, maxLevel[, crit[, flags[, minEigThreshold]]]]]) -> retval






:rtype: object
````


````{py:function} StereoBM_create([, numDisparities[, blockSize]]) -> retval

Creates StereoBM object


The function create StereoBM object. You can then call StereoBM::compute() to compute disparity for a specific stereo pair. 


:param numDisparities: the disparity search range. For each pixel algorithm will find the bestdisparity from 0 (default minimum disparity) to numDisparities. The search range can then be shifted by changing the minimum disparity. 
:type numDisparities: 
:param blockSize: the linear size of the blocks compared by the algorithm. The size should be odd(as the block is centered at the current pixel). Larger block size implies smoother, though less accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher chance for algorithm to find a wrong correspondence. 
:type blockSize: 
:rtype: object
````


````{py:function} StereoSGBM_create([, minDisparity[, numDisparities[, blockSize[, P1[, P2[, disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]]]]) -> retval

Creates StereoSGBM object


The first constructor initializes StereoSGBM with all the default parameters. So, you only have to set StereoSGBM::numDisparities at minimum. The second constructor enables you to set each parameter to a custom value. 


:param minDisparity: Minimum possible disparity value. Normally, it is zero but sometimesrectification algorithms can shift images, so this parameter needs to be adjusted accordingly. 
:type minDisparity: 
:param numDisparities: Maximum disparity minus minimum disparity. The value is always greater thanzero. In the current implementation, this parameter must be divisible by 16. 
:type numDisparities: 
:param blockSize: Matched block size. It must be an odd number \>=1 . Normally, it should besomewhere in the 3..11 range. 
:type blockSize: 
:param P1: The first parameter controlling the disparity smoothness. See below.
:type P1: 
:param P2: The second parameter controlling the disparity smoothness. The larger the values are,the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 \> P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8\*number_of_image_channels\*blockSize\*blockSize and 32\*number_of_image_channels\*blockSize\*blockSize , respectively). 
:type P2: 
:param disp12MaxDiff: Maximum allowed difference (in integer pixel units) in the left-rightdisparity check. Set it to a non-positive value to disable the check. 
:type disp12MaxDiff: 
:param preFilterCap: Truncation value for the prefiltered image pixels. The algorithm firstcomputes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function. 
:type preFilterCap: 
:param uniquenessRatio: Margin in percentage by which the best (minimum) computed cost functionvalue should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough. 
:type uniquenessRatio: 
:param speckleWindowSize: Maximum size of smooth disparity regions to consider their noise specklesand invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range. 
:type speckleWindowSize: 
:param speckleRange: Maximum disparity variation within each connected component. If you do specklefiltering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough. 
:type speckleRange: 
:param mode: Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programmingalgorithm. It will consume O(W\*H\*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false . 
:type mode: 
:rtype: object
````


````{py:function} Stitcher_create([, mode]) -> retval

Creates a Stitcher configured in one of the stitching modes.




:param mode: Scenario for stitcher operation. This is usually determined by source of imagesto stitch and their transformation. Default parameters will be chosen for operation in given scenario. 
:type mode: 
:return: Stitcher class instance.
:rtype: object
````


````{py:function} TrackerDaSiamRPN_create([, parameters]) -> retval

Constructor




:param parameters: DaSiamRPN parameters TrackerDaSiamRPN::Params
:type parameters: 
:rtype: object
````


````{py:function} TrackerGOTURN_create([, parameters]) -> retval

Constructor




:param parameters: GOTURN parameters TrackerGOTURN::Params
:type parameters: 
:rtype: object
````


````{py:function} TrackerMIL_create([, parameters]) -> retval

Create MIL tracker instance




:param parameters: MIL parameters TrackerMIL::Params
:type parameters: 
:rtype: object
````


````{py:function} TrackerNano_create([, parameters]) -> retval

Constructor




:param parameters: NanoTrack parameters TrackerNano::Params
:type parameters: 
:rtype: object
````


````{py:function} TrackerVit_create([, parameters]) -> retval

Constructor




:param parameters: vit tracker parameters TrackerVit::Params
:type parameters: 
:rtype: object
````


````{py:function} UMat_context() -> retval






:rtype: object
````


````{py:function} UMat_queue() -> retval






:rtype: object
````


````{py:function} VariationalRefinement_create() -> retval

Creates an instance of VariationalRefinement




:rtype: object
````


````{py:function} VideoCapture_waitAny(streams[, timeoutNs]) -> retval, readyIndex

Wait for ready frames from VideoCapture.


@throws Exception %Exception on stream errors (check .isOpened() to filter out malformed streams) or VideoCapture type is not supported 
The primary use of the function is in multi-camera environments. The method fills the ready state vector, grabs video frame, if camera is ready. 
After this call use VideoCapture::retrieve() to decode and fetch frame data. 


:param streams: input video streams
:type streams: 
:param readyIndex: stream indexes with grabbed frames (ready to use .retrieve() to fetch actual frame)
:type readyIndex: 
:param timeoutNs: number of nanoseconds (0 - infinite)
:type timeoutNs: 
:return: `true` if streamReady is not empty
:rtype: object
````


````{py:function} VideoWriter_fourcc(c1, c2, c3, c4) -> retval

Concatenates 4 chars to a fourcc code


This static method constructs the fourcc code of the codec to be used in the constructor VideoWriter::VideoWriter or VideoWriter::open. 


:return: a fourcc code
:rtype: object
````


````{py:function} 






:rtype: object
````


````{py:function} 






:rtype: object
````


````{py:function} _registerMatType(cv.Mat) -> None (Internal)






:rtype: object
````


````{py:function} absdiff(src1, src2[, dst]) -> dst

Calculates the per-element absolute difference between two arrays or between an array and a scalar.


The function cv::absdiff calculates: Absolute difference between two arrays when they have the same size and type: $\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2}(I)|)$ Absolute difference between an array and a scalar when the second array is constructed from Scalar or has as many elements as the number of channels in `src1`: $\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2} |)$ Absolute difference between a scalar and an array when the first array is constructed from Scalar or has as many elements as the number of channels in `src2`: $\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1} -  \texttt{src2}(I) |)$ where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each channel is processed independently. 
```{note}
Saturation is not applied when the arrays have the depth CV_32S.You may even get a negative value in the case of overflow. 
```
```{note}
(Python) Be careful to difference behaviour between src1/src2 are single number and they are tuple/array.`absdiff(src,X)` means `absdiff(src,(X,X,X,X))`. `absdiff(src,(X,))` means `absdiff(src,(X,0,0,0))`. 
```
**See also:** cv::abs(const Mat&)


:param src1: first input array or a scalar.
:type src1: cv2.typing.MatLike
:param src2: second input array or a scalar.
:type src2: cv2.typing.MatLike
:param dst: output array that has the same size and type as input arrays.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} accumulate(src, dst[, mask]) -> dst

Adds an image to the accumulator image.


The function adds src or some of its elements to dst : 
$\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0$ 
The function supports multi-channel images. Each channel is processed independently. 
The function cv::accumulate can be used, for example, to collect statistics of a scene background viewed by a still camera and for the further foreground-background segmentation. 
**See also:**  accumulateSquare, accumulateProduct, accumulateWeighted


:param src: Input image of type CV_8UC(n), CV_16UC(n), CV_32FC(n) or CV_64FC(n), where n is a positive integer.
:type src: cv2.typing.MatLike
:param dst: %Accumulator image with the same number of channels as input image, and a depth of CV_32F or CV_64F.
:type dst: cv2.typing.MatLike
:param mask: Optional operation mask.
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} accumulateProduct(src1, src2, dst[, mask]) -> dst

Adds the per-element product of two input images to the accumulator image.


The function adds the product of two images or their selected regions to the accumulator dst : 
$\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0$ 
The function supports multi-channel images. Each channel is processed independently. 
**See also:**  accumulate, accumulateSquare, accumulateWeighted


:param src1: First input image, 1- or 3-channel, 8-bit or 32-bit floating point.
:type src1: cv2.typing.MatLike
:param src2: Second input image of the same type and the same size as src1 .
:type src2: cv2.typing.MatLike
:param dst: %Accumulator image with the same number of channels as input images, 32-bit or 64-bitfloating-point. 
:type dst: cv2.typing.MatLike
:param mask: Optional operation mask.
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} accumulateSquare(src, dst[, mask]) -> dst

Adds the square of a source image to the accumulator image.


The function adds the input image src or its selected region, raised to a power of 2, to the accumulator dst : 
$\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0$ 
The function supports multi-channel images. Each channel is processed independently. 
**See also:**  accumulateSquare, accumulateProduct, accumulateWeighted


:param src: Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
:type src: cv2.typing.MatLike
:param dst: %Accumulator image with the same number of channels as input image, 32-bit or 64-bitfloating-point. 
:type dst: cv2.typing.MatLike
:param mask: Optional operation mask.
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} accumulateWeighted(src, dst, alpha[, mask]) -> dst

Updates a running average.


The function calculates the weighted sum of the input image src and the accumulator dst so that dst becomes a running average of a frame sequence: 
$\texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0$ 
That is, alpha regulates the update speed (how fast the accumulator "forgets" about earlier images). The function supports multi-channel images. Each channel is processed independently. 
**See also:**  accumulate, accumulateSquare, accumulateProduct


:param src: Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
:type src: cv2.typing.MatLike
:param dst: %Accumulator image with the same number of channels as input image, 32-bit or 64-bitfloating-point. 
:type dst: cv2.typing.MatLike
:param alpha: Weight of the input image.
:type alpha: float
:param mask: Optional operation mask.
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst

Applies an adaptive threshold to an array.


The function transforms a grayscale image to a binary image according to the formulae: -   **THRESH_BINARY** $dst(x,y) =  \fork{\texttt{maxValue}}{if \(src(x,y) > T(x,y)\)}{0}{otherwise}$ -   **THRESH_BINARY_INV** $dst(x,y) =  \fork{0}{if \(src(x,y) > T(x,y)\)}{\texttt{maxValue}}{otherwise}$ where $T(x,y)$ is a threshold calculated individually for each pixel (see adaptiveMethod parameter). 
The function can process the image in-place. 
**See also:**  threshold, blur, GaussianBlur


:param src: Source 8-bit single-channel image.
:type src: cv2.typing.MatLike
:param dst: Destination image of the same size and the same type as src.
:type dst: cv2.typing.MatLike | None
:param maxValue: Non-zero value assigned to the pixels for which the condition is satisfied
:type maxValue: float
:param adaptiveMethod: Adaptive thresholding algorithm to use, see #AdaptiveThresholdTypes.The #BORDER_REPLICATE | #BORDER_ISOLATED is used to process boundaries. 
:type adaptiveMethod: int
:param thresholdType: Thresholding type that must be either #THRESH_BINARY or #THRESH_BINARY_INV,see #ThresholdTypes. 
:type thresholdType: int
:param blockSize: Size of a pixel neighborhood that is used to calculate a threshold value for thepixel: 3, 5, 7, and so on. 
:type blockSize: int
:param C: Constant subtracted from the mean or weighted mean (see the details below). Normally, itis positive but may be zero or negative as well. 
:type C: float
:rtype: cv2.typing.MatLike
````


````{py:function} add(src1, src2[, dst[, mask[, dtype]]]) -> dst

Calculates the per-element sum of two arrays or an array and a scalar.


The function add calculates: - Sum of two arrays when both input arrays have the same size and the same number of channels: $\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0$ - Sum of an array and a scalar when src2 is constructed from Scalar or has the same number of elements as `src1.channels()`: $\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0$ - Sum of a scalar and an array when src1 is constructed from Scalar or has the same number of elements as `src2.channels()`: $\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0$ where `I` is a multi-dimensional index of array elements. In case of multi-channel arrays, each channel is processed independently. 
The first function in the list above can be replaced with matrix expressions: 
```cpp
dst = src1 + src2;
dst += src1; // equivalent to add(dst, src1, dst);
```
The input arrays and the output array can all have the same or different depths. For example, you can add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit floating-point array. Depth of the output array is determined by the dtype parameter. In the second and third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this case, the output array will have the same depth as the input array, be it src1, src2 or both. 
```{note}
Saturation is not applied when the output array has the depth CV_32S. You may even getresult of an incorrect sign in the case of overflow. 
```
```{note}
(Python) Be careful to difference behaviour between src1/src2 are single number and they are tuple/array.`add(src,X)` means `add(src,(X,X,X,X))`. `add(src,(X,))` means `add(src,(X,0,0,0))`. 
```
**See also:** subtract, addWeighted, scaleAdd, Mat::convertTo


:param src1: first input array or a scalar.
:type src1: cv2.typing.MatLike
:param src2: second input array or a scalar.
:type src2: cv2.typing.MatLike
:param dst: output array that has the same size and number of channels as the input array(s); thedepth is defined by dtype or src1/src2. 
:type dst: cv2.typing.MatLike | None
:param mask: optional operation mask - 8-bit single channel array, that specifies elements of theoutput array to be changed. 
:type mask: cv2.typing.MatLike | None
:param dtype: optional depth of the output array (see the discussion below).
:type dtype: int
:rtype: cv2.typing.MatLike
````


````{py:function} addText(img, text, org, nameFont[, pointSize[, color[, weight[, style[, spacing]]]]]) -> None

Draws a text on the image.




:param img: 8-bit 3-channel image where the text should be drawn.
:type img: cv2.typing.MatLike
:param text: Text to write on an image.
:type text: str
:param org: Point(x,y) where the text should start on an image.
:type org: cv2.typing.Point
:param nameFont: Name of the font. The name should match the name of a system font (such asTimes*). If the font is not found, a default one is used. 
:type nameFont: str
:param pointSize: Size of the font. If not specified, equal zero or negative, the point size of thefont is set to a system-dependent default value. Generally, this is 12 points. 
:type pointSize: int
:param color: Color of the font in BGRA where A = 255 is fully transparent.
:type color: cv2.typing.Scalar
:param weight: Font weight. Available operation flags are : cv::QtFontWeights You can also specify a positive integer for better control.
:type weight: int
:param style: Font style. Available operation flags are : cv::QtFontStyles
:type style: int
:param spacing: Spacing between characters. It can be negative or positive.
:type spacing: int
:rtype: None
````


````{py:function} addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst

Calculates the weighted sum of two arrays.


The function addWeighted calculates the weighted sum of two arrays as follows: $\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )$ where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each channel is processed independently. The function can be replaced with a matrix expression: 
```cpp
dst = src1*alpha + src2*beta + gamma;
```

```{note}
Saturation is not applied when the output array has the depth CV_32S. You may even getresult of an incorrect sign in the case of overflow. 
```
**See also:**  add, subtract, scaleAdd, Mat::convertTo


:param src1: first input array.
:type src1: cv2.typing.MatLike
:param alpha: weight of the first array elements.
:type alpha: float
:param src2: second input array of the same size and channel number as src1.
:type src2: cv2.typing.MatLike
:param beta: weight of the second array elements.
:type beta: float
:param gamma: scalar added to each sum.
:type gamma: float
:param dst: output array that has the same size and number of channels as the input arrays.
:type dst: cv2.typing.MatLike | None
:param dtype: optional depth of the output array; when both input arrays have the same depth, dtypecan be set to -1, which will be equivalent to src1.depth(). 
:type dtype: int
:rtype: cv2.typing.MatLike
````


````{py:function} applyColorMap(src, colormap[, dst]) -> dst

Applies a user colormap on a given image.


applyColorMap(src, userColor[, dst]) -> dst 


:param src: The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.
:type src: cv2.typing.MatLike
:param dst: The result is the colormapped source image. Note: Mat::create is called on dst.
:type dst: cv2.typing.MatLike | None
:param colormap: The colormap to apply, see #ColormapTypes
:type colormap: int
:param userColor: The colormap to apply of type CV_8UC1 or CV_8UC3 and size 256
:type userColor: 
:rtype: cv2.typing.MatLike
````


````{py:function} approxPolyDP(curve, epsilon, closed[, approxCurve]) -> approxCurve

Approximates a polygonal curve(s) with the specified precision.


The function cv::approxPolyDP approximates a curve or a polygon with another curve/polygon with less vertices so that the distance between them is less or equal to the specified precision. It uses the Douglas-Peucker algorithm <http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm> 


:param curve: Input vector of a 2D point stored in std::vector or Mat
:type curve: cv2.typing.MatLike
:param approxCurve: Result of the approximation. The type should match the type of the input curve.
:type approxCurve: cv2.typing.MatLike | None
:param epsilon: Parameter specifying the approximation accuracy. This is the maximum distancebetween the original curve and its approximation. 
:type epsilon: float
:param closed: If true, the approximated curve is closed (its first and last vertices areconnected). Otherwise, it is not closed. 
:type closed: bool
:rtype: cv2.typing.MatLike
````


````{py:function} arcLength(curve, closed) -> retval

Calculates a contour perimeter or a curve length.


The function computes a curve length or a closed contour perimeter. 


:param curve: Input vector of 2D points, stored in std::vector or Mat.
:type curve: cv2.typing.MatLike
:param closed: Flag indicating whether the curve is closed or not.
:type closed: bool
:rtype: float
````


````{py:function} arrowedLine(img, pt1, pt2, color[, thickness[, line_type[, shift[, tipLength]]]]) -> img

Draws an arrow segment pointing from the first point to the second one.


The function cv::arrowedLine draws an arrow between pt1 and pt2 points in the image. See also #line. 


:param img: Image.
:type img: cv2.typing.MatLike
:param pt1: The point the arrow starts from.
:type pt1: cv2.typing.Point
:param pt2: The point the arrow points to.
:type pt2: cv2.typing.Point
:param color: Line color.
:type color: cv2.typing.Scalar
:param thickness: Line thickness.
:type thickness: int
:param line_type: Type of the line. See #LineTypes
:type line_type: int
:param shift: Number of fractional bits in the point coordinates.
:type shift: int
:param tipLength: The length of the arrow tip in relation to the arrow length
:type tipLength: float
:rtype: cv2.typing.MatLike
````


````{py:function} batchDistance(src1, src2, dtype[, dist[, nidx[, normType[, K[, mask[, update[, crosscheck]]]]]]]) -> dist, nidx

naive nearest neighbor finder


see http://en.wikipedia.org/wiki/Nearest_neighbor_search @todo document 


:param src1: 
:type src1: cv2.typing.MatLike
:param src2: 
:type src2: cv2.typing.MatLike
:param dtype: 
:type dtype: int
:param dist: 
:type dist: cv2.typing.MatLike | None
:param nidx: 
:type nidx: cv2.typing.MatLike | None
:param normType: 
:type normType: int
:param K: 
:type K: int
:param mask: 
:type mask: cv2.typing.MatLike | None
:param update: 
:type update: int
:param crosscheck: 
:type crosscheck: bool
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst

Applies the bilateral filter to an image.


The function applies bilateral filtering to the input image, as described in http://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is very slow compared to most filters. 
_Sigma values_: For simplicity, you can set the 2 sigma values to be the same. If they are small (\< 10), the filter will not have much effect, whereas if they are large (\> 150), they will have a very strong effect, making the image look "cartoonish". 
_Filter size_: Large filters (d \> 5) are very slow, so it is recommended to use d=5 for real-time applications, and perhaps d=9 for offline applications that need heavy noise filtering. 
This filter does not work inplace. 


:param src: Source 8-bit or floating-point, 1-channel or 3-channel image.
:type src: cv2.typing.MatLike
:param dst: Destination image of the same size and type as src .
:type dst: cv2.typing.MatLike | None
:param d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,it is computed from sigmaSpace. 
:type d: int
:param sigmaColor: Filter sigma in the color space. A larger value of the parameter means thatfarther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color. 
:type sigmaColor: float
:param sigmaSpace: Filter sigma in the coordinate space. A larger value of the parameter means thatfarther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d\>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace. 
:type sigmaSpace: float
:param borderType: border mode used to extrapolate pixels outside of the image, see #BorderTypes
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} bitwise_and(src1, src2[, dst[, mask]]) -> dst

computes bitwise conjunction of the two arrays (dst = src1 & src2)Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar. 


The function cv::bitwise_and calculates the per-element bit-wise logical conjunction for: Two arrays when src1 and src2 have the same size: $\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0$ An array and a scalar when src2 is constructed from Scalar or has the same number of elements as `src1.channels()`: $\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} \quad \texttt{if mask} (I) \ne0$ A scalar and an array when src1 is constructed from Scalar or has the same number of elements as `src2.channels()`: $\texttt{dst} (I) =  \texttt{src1}  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0$ In case of floating-point arrays, their machine-specific bit representations (usually IEEE754-compliant) are used for the operation. In case of multi-channel arrays, each channel is processed independently. In the second and third cases above, the scalar is first converted to the array type. 


:param src1: first input array or a scalar.
:type src1: cv2.typing.MatLike
:param src2: second input array or a scalar.
:type src2: cv2.typing.MatLike
:param dst: output array that has the same size and type as the inputarrays. 
:type dst: cv2.typing.MatLike | None
:param mask: optional operation mask, 8-bit single channel array, thatspecifies elements of the output array to be changed. 
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} bitwise_not(src[, dst[, mask]]) -> dst

 Inverts every bit of an array.


The function cv::bitwise_not calculates per-element bit-wise inversion of the input array: $\texttt{dst} (I) =  \neg \texttt{src} (I)$ In case of a floating-point input array, its machine-specific bit representation (usually IEEE754-compliant) is used for the operation. In case of multi-channel arrays, each channel is processed independently. 


:param src: input array.
:type src: cv2.typing.MatLike
:param dst: output array that has the same size and type as the inputarray. 
:type dst: cv2.typing.MatLike | None
:param mask: optional operation mask, 8-bit single channel array, thatspecifies elements of the output array to be changed. 
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} bitwise_or(src1, src2[, dst[, mask]]) -> dst

Calculates the per-element bit-wise disjunction of two arrays or anarray and a scalar. 


The function cv::bitwise_or calculates the per-element bit-wise logical disjunction for: Two arrays when src1 and src2 have the same size: $\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0$ An array and a scalar when src2 is constructed from Scalar or has the same number of elements as `src1.channels()`: $\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} \quad \texttt{if mask} (I) \ne0$ A scalar and an array when src1 is constructed from Scalar or has the same number of elements as `src2.channels()`: $\texttt{dst} (I) =  \texttt{src1}  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0$ In case of floating-point arrays, their machine-specific bit representations (usually IEEE754-compliant) are used for the operation. In case of multi-channel arrays, each channel is processed independently. In the second and third cases above, the scalar is first converted to the array type. 


:param src1: first input array or a scalar.
:type src1: cv2.typing.MatLike
:param src2: second input array or a scalar.
:type src2: cv2.typing.MatLike
:param dst: output array that has the same size and type as the inputarrays. 
:type dst: cv2.typing.MatLike | None
:param mask: optional operation mask, 8-bit single channel array, thatspecifies elements of the output array to be changed. 
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} bitwise_xor(src1, src2[, dst[, mask]]) -> dst

Calculates the per-element bit-wise "exclusive or" operation on twoarrays or an array and a scalar. 


The function cv::bitwise_xor calculates the per-element bit-wise logical "exclusive-or" operation for: Two arrays when src1 and src2 have the same size: $\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0$ An array and a scalar when src2 is constructed from Scalar or has the same number of elements as `src1.channels()`: $\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} \quad \texttt{if mask} (I) \ne0$ A scalar and an array when src1 is constructed from Scalar or has the same number of elements as `src2.channels()`: $\texttt{dst} (I) =  \texttt{src1}  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0$ In case of floating-point arrays, their machine-specific bit representations (usually IEEE754-compliant) are used for the operation. In case of multi-channel arrays, each channel is processed independently. In the 2nd and 3rd cases above, the scalar is first converted to the array type. 


:param src1: first input array or a scalar.
:type src1: cv2.typing.MatLike
:param src2: second input array or a scalar.
:type src2: cv2.typing.MatLike
:param dst: output array that has the same size and type as the inputarrays. 
:type dst: cv2.typing.MatLike | None
:param mask: optional operation mask, 8-bit single channel array, thatspecifies elements of the output array to be changed. 
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} blendLinear(src1, src2, weights1, weights2[, dst]) -> dst




@overload 
variant without `mask` parameter 


:param src1: 
:type src1: cv2.typing.MatLike
:param src2: 
:type src2: cv2.typing.MatLike
:param weights1: 
:type weights1: cv2.typing.MatLike
:param weights2: 
:type weights2: cv2.typing.MatLike
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} blur(src, ksize[, dst[, anchor[, borderType]]]) -> dst

Blurs an image using the normalized box filter.


The function smooths an image using the kernel: 
$\texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \end{bmatrix}$ 
The call `blur(src, dst, ksize, anchor, borderType)` is equivalent to `boxFilter(src, dst, src.type(), ksize, anchor, true, borderType)`. 
**See also:**  boxFilter, bilateralFilter, GaussianBlur, medianBlur


:param src: input image; it can have any number of channels, which are processed independently, butthe depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F. 
:type src: cv2.typing.MatLike
:param dst: output image of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param ksize: blurring kernel size.
:type ksize: cv2.typing.Size
:param anchor: anchor point; default value Point(-1,-1) means that the anchor is at the kernelcenter. 
:type anchor: cv2.typing.Point
:param borderType: border mode used to extrapolate pixels outside of the image, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} 






:rtype: object
````


````{py:function} borderInterpolate(p, len, borderType) -> retval

Computes the source location of an extrapolated pixel.


The function computes and returns the coordinate of a donor pixel corresponding to the specified extrapolated pixel when using the specified extrapolation border mode. For example, if you use cv::BORDER_WRAP mode in the horizontal direction, cv::BORDER_REFLECT_101 in the vertical direction and want to compute value of the "virtual" pixel Point(-5, 100) in a floating-point image img , it looks like: 
```cpp
float val = img.at<float>(borderInterpolate(100, img.rows, cv::BORDER_REFLECT_101),
borderInterpolate(-5, img.cols, cv::BORDER_WRAP));
```
Normally, the function is not called directly. It is used inside filtering functions and also in copyMakeBorder. 
**See also:** copyMakeBorder


:param p: 0-based coordinate of the extrapolated pixel along one of the axes, likely \<0 or \>= len
:type p: int
:param len: Length of the array along the corresponding axis.
:type len: int
:param borderType: Border type, one of the #BorderTypes, except for #BORDER_TRANSPARENT and#BORDER_ISOLATED . When borderType==#BORDER_CONSTANT , the function always returns -1, regardless of p and len. 
:type borderType: int
:rtype: int
````


````{py:function} boundingRect(array) -> retval

Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.


The function calculates and returns the minimal up-right bounding rectangle for the specified point set or non-zero pixels of gray-scale image. 


:param array: Input gray-scale image or 2D point set, stored in std::vector or Mat.
:type array: cv2.typing.MatLike
:rtype: cv2.typing.Rect
````


````{py:function} boxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]) -> dst

Blurs an image using the box filter.


The function smooths an image using the kernel: 
$\texttt{K} =  \alpha \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1 \end{bmatrix}$ 
where 
$\alpha = \begin{cases} \frac{1}{\texttt{ksize.width*ksize.height}} & \texttt{when } \texttt{normalize=true}  \\1 & \texttt{otherwise}\end{cases}$ 
Unnormalized box filter is useful for computing various integral characteristics over each pixel neighborhood, such as covariance matrices of image derivatives (used in dense optical flow algorithms, and so on). If you need to compute pixel sums over variable-size windows, use #integral. 
**See also:**  blur, bilateralFilter, GaussianBlur, medianBlur, integral


:param src: input image.
:type src: cv2.typing.MatLike
:param dst: output image of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param ddepth: the output image depth (-1 to use src.depth()).
:type ddepth: int
:param ksize: blurring kernel size.
:type ksize: cv2.typing.Size
:param anchor: anchor point; default value Point(-1,-1) means that the anchor is at the kernelcenter. 
:type anchor: cv2.typing.Point
:param normalize: flag, specifying whether the kernel is normalized by its area or not.
:type normalize: bool
:param borderType: border mode used to extrapolate pixels outside of the image, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} boxPoints(box[, points]) -> points

Finds the four vertices of a rotated rect. Useful to draw the rotated rectangle.


The function finds the four vertices of a rotated rectangle. This function is useful to draw the rectangle. In C++, instead of using this function, you can directly use RotatedRect::points method. Please visit the @ref tutorial_bounding_rotated_ellipses "tutorial on Creating Bounding rotated boxes and ellipses for contours" for more information. 


:param box: The input rotated rectangle. It may be the output of @ref minAreaRect.
:type box: cv2.typing.RotatedRect
:param points: The output array of four vertices of rectangles.
:type points: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} broadcast(src, shape[, dst]) -> dst

Broadcast the given Mat to the given shape.




:param src: input array
:type src: cv2.typing.MatLike
:param shape: target shape. Should be a list of CV_32S numbers. Note that negative values are not supported.
:type shape: cv2.typing.MatLike
:param dst: output array that has the given shape
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} buildOpticalFlowPyramid(img, winSize, maxLevel[, pyramid[, withDerivatives[, pyrBorder[, derivBorder[, tryReuseInputImage]]]]]) -> retval, pyramid

Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.




:param img: 8-bit input image.
:type img: cv2.typing.MatLike
:param pyramid: output pyramid.
:type pyramid: _typing.Sequence[cv2.typing.MatLike] | None
:param winSize: window size of optical flow algorithm. Must be not less than winSize argument ofcalcOpticalFlowPyrLK. It is needed to calculate required padding for pyramid levels. 
:type winSize: cv2.typing.Size
:param maxLevel: 0-based maximal pyramid level number.
:type maxLevel: int
:param withDerivatives: set to precompute gradients for the every pyramid level. If pyramid isconstructed without the gradients then calcOpticalFlowPyrLK will calculate them internally. 
:type withDerivatives: bool
:param pyrBorder: the border mode for pyramid layers.
:type pyrBorder: int
:param derivBorder: the border mode for gradients.
:type derivBorder: int
:param tryReuseInputImage: put ROI of input image into the pyramid if possible. You can pass falseto force data copying. 
:type tryReuseInputImage: bool
:return: number of levels in constructed pyramid. Can be less than maxLevel.
:rtype: tuple[int, _typing.Sequence[cv2.typing.MatLike]]
````


````{py:function} calcBackProject(images, channels, hist, ranges, scale[, dst]) -> dst




@overload 


:param images: 
:type images: _typing.Sequence[cv2.typing.MatLike]
:param channels: 
:type channels: _typing.Sequence[int]
:param hist: 
:type hist: cv2.typing.MatLike
:param ranges: 
:type ranges: _typing.Sequence[float]
:param scale: 
:type scale: float
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} calcCovarMatrix(samples, mean, flags[, covar[, ctype]]) -> covar, mean




@overload 
```{note}
use #COVAR_ROWS or #COVAR_COLS flag
```


:param samples: samples stored as rows/columns of a single matrix.
:type samples: cv2.typing.MatLike
:param covar: output covariance matrix of the type ctype and square size.
:type covar: cv2.typing.MatLike | None
:param mean: input or output (depending on the flags) array as the average value of the input vectors.
:type mean: cv2.typing.MatLike
:param flags: operation flags as a combination of #CovarFlags
:type flags: int
:param ctype: type of the matrixl; it equals 'CV_64F' by default.
:type ctype: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist




@overload 
this variant supports only uniform histograms. 
ranges argument is either empty vector or a flattened vector of histSize.size()*2 elements (histSize.size() element pairs). The first and second elements of each pair specify the lower and upper boundaries. 


:param images: 
:type images: _typing.Sequence[cv2.typing.MatLike]
:param channels: 
:type channels: _typing.Sequence[int]
:param mask: 
:type mask: cv2.typing.MatLike | None
:param histSize: 
:type histSize: _typing.Sequence[int]
:param ranges: 
:type ranges: _typing.Sequence[float]
:param hist: 
:type hist: cv2.typing.MatLike | None
:param accumulate: 
:type accumulate: bool
:rtype: cv2.typing.MatLike
````


````{py:function} calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) -> flow

Computes a dense optical flow using the Gunnar Farneback's algorithm.


The function finds an optical flow for each prev pixel using the @cite Farneback2003 algorithm so that 
$\texttt{prev} (y,x)  \sim \texttt{next} ( y + \texttt{flow} (y,x)[1],  x + \texttt{flow} (y,x)[0])$ 
-   An example using the optical flow algorithm described by Gunnar Farneback can be found at opencv_source_code/samples/cpp/fback.cpp -   (Python) An example using the optical flow algorithm described by Gunnar Farneback can be found at opencv_source_code/samples/python/opt_flow.py 
```{note}
Some examples:
```


:param prev: first 8-bit single-channel input image.
:type prev: cv2.typing.MatLike
:param next: second input image of the same size and the same type as prev.
:type next: cv2.typing.MatLike
:param flow: computed flow image that has the same size as prev and type CV_32FC2.
:type flow: cv2.typing.MatLike
:param pyr_scale: parameter, specifying the image scale (\<1) to build pyramids for each image;pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one. 
:type pyr_scale: float
:param levels: number of pyramid layers including the initial image; levels=1 means that no extralayers are created and only the original images are used. 
:type levels: int
:param winsize: averaging window size; larger values increase the algorithm robustness to imagenoise and give more chances for fast motion detection, but yield more blurred motion field. 
:type winsize: int
:param iterations: number of iterations the algorithm does at each pyramid level.
:type iterations: int
:param poly_n: size of the pixel neighborhood used to find polynomial expansion in each pixel;larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7. 
:type poly_n: int
:param poly_sigma: standard deviation of the Gaussian that is used to smooth derivatives used as abasis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5. 
:type poly_sigma: float
:param flags: operation flags that can be a combination of the following:-   **OPTFLOW_USE_INITIAL_FLOW** uses the input flow as an initial flow approximation. -   **OPTFLOW_FARNEBACK_GAUSSIAN** uses the Gaussian $\texttt{winsize}\times\texttt{winsize}$ filter instead of a box filter of the same size for optical flow estimation; usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness. 
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]) -> nextPts, status, err

Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method withpyramids. 


The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See @cite Bouguet00 . The function is parallelized with the TBB library. 
-   An example using the Lucas-Kanade optical flow algorithm can be found at opencv_source_code/samples/cpp/lkdemo.cpp -   (Python) An example using the Lucas-Kanade optical flow algorithm can be found at opencv_source_code/samples/python/lk_track.py -   (Python) An example using the Lucas-Kanade tracker for homography matching can be found at opencv_source_code/samples/python/lk_homography.py 
```{note}
Some examples:
```


:param prevImg: first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
:type prevImg: cv2.typing.MatLike
:param nextImg: second input image or pyramid of the same size and the same type as prevImg.
:type nextImg: cv2.typing.MatLike
:param prevPts: vector of 2D points for which the flow needs to be found; point coordinates must besingle-precision floating-point numbers. 
:type prevPts: cv2.typing.MatLike
:param nextPts: output vector of 2D points (with single-precision floating-point coordinates)containing the calculated new positions of input features in the second image; when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input. 
:type nextPts: cv2.typing.MatLike
:param status: output status vector (of unsigned chars); each element of the vector is set to 1 ifthe flow for the corresponding features has been found, otherwise, it is set to 0. 
:type status: cv2.typing.MatLike | None
:param err: output vector of errors; each element of the vector is set to an error for thecorresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't found then the error is not defined (use the status parameter to find such cases). 
:type err: cv2.typing.MatLike | None
:param winSize: size of the search window at each pyramid level.
:type winSize: cv2.typing.Size
:param maxLevel: 0-based maximal pyramid level number; if set to 0, pyramids are not used (singlelevel), if set to 1, two levels are used, and so on; if pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than maxLevel. 
:type maxLevel: int
:param criteria: parameter, specifying the termination criteria of the iterative search algorithm(after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon. 
:type criteria: cv2.typing.TermCriteria
:param flags: operation flags:-   **OPTFLOW_USE_INITIAL_FLOW** uses initial estimations, stored in nextPts; if the flag is not set, then prevPts is copied to nextPts and is considered the initial estimate. -   **OPTFLOW_LK_GET_MIN_EIGENVALS** use minimum eigen values as an error measure (see minEigThreshold description); if the flag is not set, then L1 distance between patches around the original and a moved point, divided by number of pixels in a window, is used as a error measure. 
:type flags: int
:param minEigThreshold: the algorithm calculates the minimum eigen value of a 2x2 normal matrix ofoptical flow equations (this matrix is called a spatial gradient matrix in @cite Bouguet00), divided by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding feature is filtered out and its flow is not processed, so it allows to remove bad points and get a performance boost. 
:type minEigThreshold: float
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs




@overload 


:param objectPoints: 
:type objectPoints: _typing.Sequence[cv2.typing.MatLike]
:param imagePoints: 
:type imagePoints: _typing.Sequence[cv2.typing.MatLike]
:param imageSize: 
:type imageSize: cv2.typing.Size
:param cameraMatrix: 
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: 
:type distCoeffs: cv2.typing.MatLike
:param rvecs: 
:type rvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param tvecs: 
:type tvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param flags: 
:type flags: int
:param criteria: 
:type criteria: cv2.typing.TermCriteria
:rtype: tuple[float, cv2.typing.MatLike, cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike]]
````


````{py:function} calibrateCameraExtended(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, perViewErrors[, flags[, criteria]]]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors

Finds the camera intrinsic and extrinsic parameters from several views of a calibrationpattern. 


The function estimates the intrinsic camera parameters and extrinsic parameters for each of the views. The algorithm is based on @cite Zhang2000 and @cite BouguetMCT . The coordinates of 3D object points and their corresponding 2D projections in each view must be specified. That may be achieved by using an object with known geometry and easily detectable feature points. Such an object is called a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as a calibration rig (see @ref findChessboardCorners). Currently, initialization of intrinsic parameters (when @ref CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration patterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also be used as long as initial cameraMatrix is provided. 
The algorithm performs the following steps: 
-   Compute the initial intrinsic parameters (the option only available for planar calibration patterns) or read them from the input parameters. The distortion coefficients are all set to zeros initially unless some of CALIB_FIX_K? are specified. 
-   Estimate the initial camera pose as if the intrinsic parameters have been already known. This is done using @ref solvePnP . 
-   Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error, that is, the total sum of squared distances between the observed feature points imagePoints and the projected (using the current estimates for camera parameters and the poses) object points objectPoints. See @ref projectPoints for details. 
@note If you use a non-square (i.e. non-N-by-N) grid and @ref findChessboardCorners for calibration, and @ref calibrateCamera returns bad values (zero distortion coefficients, $c_x$ and $c_y$ very far from the image center, and/or large differences between $f_x$ and $f_y$ (ratios of 10:1 or more)), then you are probably using patternSize=cvSize(rows,cols) instead of using patternSize=cvSize(cols,rows) in @ref findChessboardCorners. 
@note The function may throw exceptions, if unsupported combination of parameters is provided or the system is underconstrained. 
@sa calibrateCameraRO, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort 


:param objectPoints: In the new interface it is a vector of vectors of calibration pattern points inthe calibration pattern coordinate space (e.g. std::vector<std::vector<cv::Vec3f>>). The outer vector contains as many elements as the number of pattern views. If the same calibration pattern is shown in each view and it is fully visible, all the vectors will be the same. Although, it is possible to use partially occluded patterns or even different patterns in different views. Then, the vectors will be different. Although the points are 3D, they all lie in the calibration pattern's XY coordinate plane (thus 0 in the Z-coordinate), if the used calibration pattern is a planar rig. In the old interface all the vectors of object points from different views are concatenated together. 
:type objectPoints: _typing.Sequence[cv2.typing.MatLike]
:param imagePoints: In the new interface it is a vector of vectors of the projections of calibrationpattern points (e.g. std::vector<std::vector<cv::Vec2f>>). imagePoints.size() and objectPoints.size(), and imagePoints[i].size() and objectPoints[i].size() for each i, must be equal, respectively. In the old interface all the vectors of object points from different views are concatenated together. 
:type imagePoints: _typing.Sequence[cv2.typing.MatLike]
:param imageSize: Size of the image used only to initialize the camera intrinsic matrix.
:type imageSize: cv2.typing.Size
:param cameraMatrix: Input/output 3x3 floating-point camera intrinsic matrix$\cameramatrix{A}$ . If @ref CALIB_USE_INTRINSIC_GUESS and/or @ref CALIB_FIX_ASPECT_RATIO, @ref CALIB_FIX_PRINCIPAL_POINT or @ref CALIB_FIX_FOCAL_LENGTH are specified, some or all of fx, fy, cx, cy must be initialized before calling the function. 
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input/output vector of distortion coefficients$\distcoeffs$. 
:type distCoeffs: cv2.typing.MatLike
:param rvecs: Output vector of rotation vectors (@ref Rodrigues ) estimated for each pattern view(e.g. std::vector<cv::Mat>>). That is, each i-th rotation vector together with the corresponding i-th translation vector (see the next output parameter description) brings the calibration pattern from the object coordinate space (in which object points are specified) to the camera coordinate space. In more technical terms, the tuple of the i-th rotation and translation vector performs a change of basis from object coordinate space to camera coordinate space. Due to its duality, this tuple is equivalent to the position of the calibration pattern with respect to the camera coordinate space. 
:type rvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param tvecs: Output vector of translation vectors estimated for each pattern view, see parameterdescribtion above. 
:type tvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param stdDeviationsIntrinsics: Output vector of standard deviations estimated for intrinsicparameters. Order of deviations values: $(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3, s_4, \tau_x, \tau_y)$ If one of parameters is not estimated, it's deviation is equals to zero. 
:type stdDeviationsIntrinsics: cv2.typing.MatLike | None
:param stdDeviationsExtrinsics: Output vector of standard deviations estimated for extrinsicparameters. Order of deviations values: $(R_0, T_0, \dotsc , R_{M - 1}, T_{M - 1})$ where M is the number of pattern views. $R_i, T_i$ are concatenated 1x3 vectors. 
:type stdDeviationsExtrinsics: cv2.typing.MatLike | None
:param perViewErrors: Output vector of the RMS re-projection error estimated for each pattern view.
:type perViewErrors: cv2.typing.MatLike | None
:param flags: Different flags that may be zero or a combination of the following values:-   @ref CALIB_USE_INTRINSIC_GUESS cameraMatrix contains valid initial values of fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image center ( imageSize is used), and focal distances are computed in a least-squares fashion. Note, that if intrinsic parameters are known, there is no need to use this function just to estimate extrinsic parameters. Use @ref solvePnP instead. -   @ref CALIB_FIX_PRINCIPAL_POINT The principal point is not changed during the global optimization. It stays at the center or at a different location specified when @ref CALIB_USE_INTRINSIC_GUESS is set too. -   @ref CALIB_FIX_ASPECT_RATIO The functions consider only fy as a free parameter. The ratio fx/fy stays the same as in the input cameraMatrix . When @ref CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are ignored, only their ratio is computed and used further. -   @ref CALIB_ZERO_TANGENT_DIST Tangential distortion coefficients $(p_1, p_2)$ are set to zeros and stay zero. -   @ref CALIB_FIX_FOCAL_LENGTH The focal length is not changed during the global optimization if @ref CALIB_USE_INTRINSIC_GUESS is set. -   @ref CALIB_FIX_K1,..., @ref CALIB_FIX_K6 The corresponding radial distortion coefficient is not changed during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0. -   @ref CALIB_RATIONAL_MODEL Coefficients k4, k5, and k6 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the rational model and return 8 coefficients or more. -   @ref CALIB_THIN_PRISM_MODEL Coefficients s1, s2, s3 and s4 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the thin prism model and return 12 coefficients or more. -   @ref CALIB_FIX_S1_S2_S3_S4 The thin prism distortion coefficients are not changed during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0. -   @ref CALIB_TILTED_MODEL Coefficients tauX and tauY are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the tilted sensor model and return 14 coefficients. -   @ref CALIB_FIX_TAUX_TAUY The coefficients of the tilted sensor model are not changed during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0. 
:type flags: int
:param criteria: Termination criteria for the iterative optimization algorithm.
:type criteria: cv2.typing.TermCriteria
:return: the overall RMS re-projection error.
:rtype: tuple[float, cv2.typing.MatLike, cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint, cameraMatrix, distCoeffs[, rvecs[, tvecs[, newObjPoints[, flags[, criteria]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints




@overload 


:param objectPoints: 
:type objectPoints: _typing.Sequence[cv2.typing.MatLike]
:param imagePoints: 
:type imagePoints: _typing.Sequence[cv2.typing.MatLike]
:param imageSize: 
:type imageSize: cv2.typing.Size
:param iFixedPoint: 
:type iFixedPoint: int
:param cameraMatrix: 
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: 
:type distCoeffs: cv2.typing.MatLike
:param rvecs: 
:type rvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param tvecs: 
:type tvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param newObjPoints: 
:type newObjPoints: cv2.typing.MatLike | None
:param flags: 
:type flags: int
:param criteria: 
:type criteria: cv2.typing.TermCriteria
:rtype: tuple[float, cv2.typing.MatLike, cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike]
````


````{py:function} calibrateCameraROExtended(objectPoints, imagePoints, imageSize, iFixedPoint, cameraMatrix, distCoeffs[, rvecs[, tvecs[, newObjPoints[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, stdDeviationsObjPoints[, perViewErrors[, flags[, criteria]]]]]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints, stdDeviationsIntrinsics, stdDeviationsExtrinsics, stdDeviationsObjPoints, perViewErrors

Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.


This function is an extension of #calibrateCamera with the method of releasing object which was proposed in @cite strobl2011iccv. In many common cases with inaccurate, unmeasured, roughly planar targets (calibration plates), this method can dramatically improve the precision of the estimated camera parameters. Both the object-releasing method and standard method are supported by this function. Use the parameter **iFixedPoint** for method selection. In the internal implementation, #calibrateCamera is a wrapper for this function. 
The function estimates the intrinsic camera parameters and extrinsic parameters for each of the views. The algorithm is based on @cite Zhang2000, @cite BouguetMCT and @cite strobl2011iccv. See #calibrateCamera for other detailed explanations. @sa calibrateCamera, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort 


:param objectPoints: Vector of vectors of calibration pattern points in the calibration patterncoordinate space. See #calibrateCamera for details. If the method of releasing object to be used, the identical calibration board must be used in each view and it must be fully visible, and all objectPoints[i] must be the same and all points should be roughly close to a plane. **The calibration target has to be rigid, or at least static if the camera (rather than the calibration target) is shifted for grabbing images.** 
:type objectPoints: _typing.Sequence[cv2.typing.MatLike]
:param imagePoints: Vector of vectors of the projections of calibration pattern points. See#calibrateCamera for details. 
:type imagePoints: _typing.Sequence[cv2.typing.MatLike]
:param imageSize: Size of the image used only to initialize the intrinsic camera matrix.
:type imageSize: cv2.typing.Size
:param iFixedPoint: The index of the 3D object point in objectPoints[0] to be fixed. It also acts asa switch for calibration method selection. If object-releasing method to be used, pass in the parameter in the range of [1, objectPoints[0].size()-2], otherwise a value out of this range will make standard calibration method selected. Usually the top-right corner point of the calibration board grid is recommended to be fixed when object-releasing method being utilized. According to \cite strobl2011iccv, two other points are also fixed. In this implementation, objectPoints[0].front and objectPoints[0].back.z are used. With object-releasing method, accurate rvecs, tvecs and newObjPoints are only possible if coordinates of these three fixed points are accurate enough. 
:type iFixedPoint: int
:param cameraMatrix: Output 3x3 floating-point camera matrix. See #calibrateCamera for details.
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Output vector of distortion coefficients. See #calibrateCamera for details.
:type distCoeffs: cv2.typing.MatLike
:param rvecs: Output vector of rotation vectors estimated for each pattern view. See #calibrateCamerafor details. 
:type rvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param tvecs: Output vector of translation vectors estimated for each pattern view.
:type tvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param newObjPoints: The updated output vector of calibration pattern points. The coordinates mightbe scaled based on three fixed points. The returned coordinates are accurate only if the above mentioned three fixed points are accurate. If not needed, noArray() can be passed in. This parameter is ignored with standard calibration method. 
:type newObjPoints: cv2.typing.MatLike | None
:param stdDeviationsIntrinsics: Output vector of standard deviations estimated for intrinsic parameters.See #calibrateCamera for details. 
:type stdDeviationsIntrinsics: cv2.typing.MatLike | None
:param stdDeviationsExtrinsics: Output vector of standard deviations estimated for extrinsic parameters.See #calibrateCamera for details. 
:type stdDeviationsExtrinsics: cv2.typing.MatLike | None
:param stdDeviationsObjPoints: Output vector of standard deviations estimated for refined coordinatesof calibration pattern points. It has the same size and order as objectPoints[0] vector. This parameter is ignored with standard calibration method. 
:type stdDeviationsObjPoints: cv2.typing.MatLike | None
:param perViewErrors: Output vector of the RMS re-projection error estimated for each pattern view.
:type perViewErrors: cv2.typing.MatLike | None
:param flags: Different flags that may be zero or a combination of some predefined values. See#calibrateCamera for details. If the method of releasing object is used, the calibration time may be much longer. CALIB_USE_QR or CALIB_USE_LU could be used for faster calibration with potentially less precise and less stable in some rare cases. 
:type flags: int
:param criteria: Termination criteria for the iterative optimization algorithm.
:type criteria: cv2.typing.TermCriteria
:return: the overall RMS re-projection error.
:rtype: tuple[float, cv2.typing.MatLike, cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam[, R_cam2gripper[, t_cam2gripper[, method]]]) -> R_cam2gripper, t_cam2gripper

Computes Hand-Eye calibration: $_{}^{g}\textrm{T}_c$


The function performs the Hand-Eye calibration using various methods. One approach consists in estimating the rotation then the translation (separable solutions) and the following methods are implemented: - R. Tsai, R. Lenz A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/EyeCalibration \cite Tsai89 - F. Park, B. Martin Robot Sensor Calibration: Solving AX = XB on the Euclidean Group \cite Park94 - R. Horaud, F. Dornaika Hand-Eye Calibration \cite Horaud95 
Another approach consists in estimating simultaneously the rotation and the translation (simultaneous solutions), with the following implemented methods: - N. Andreff, R. Horaud, B. Espiau On-line Hand-Eye Calibration \cite Andreff99 - K. Daniilidis Hand-Eye Calibration Using Dual Quaternions \cite Daniilidis98 
The following picture describes the Hand-Eye calibration problem where the transformation between a camera ("eye") mounted on a robot gripper ("hand") has to be estimated. This configuration is called eye-in-hand. 
The eye-to-hand configuration consists in a static camera observing a calibration pattern mounted on the robot end-effector. The transformation from the camera to the robot base frame can then be estimated by inputting the suitable transformations to the function, see below. 
![](pics/hand-eye_figure.png) 
The calibration procedure is the following: - a static calibration pattern is used to estimate the transformation between the target frame and the camera frame - the robot gripper is moved in order to acquire several poses - for each pose, the homogeneous transformation between the gripper frame and the robot base frame is recorded using for instance the robot kinematics $ \begin{bmatrix} X_b\\ Y_b\\ Z_b\\ 1 \end{bmatrix} = \begin{bmatrix} _{}^{b}\textrm{R}_g & _{}^{b}\textrm{t}_g \\ 0_{1 \times 3} & 1 \end{bmatrix} \begin{bmatrix} X_g\\ Y_g\\ Z_g\\ 1 \end{bmatrix} $ - for each pose, the homogeneous transformation between the calibration target frame and the camera frame is recorded using for instance a pose estimation method (PnP) from 2D-3D point correspondences $ \begin{bmatrix} X_c\\ Y_c\\ Z_c\\ 1 \end{bmatrix} = \begin{bmatrix} _{}^{c}\textrm{R}_t & _{}^{c}\textrm{t}_t \\ 0_{1 \times 3} & 1 \end{bmatrix} \begin{bmatrix} X_t\\ Y_t\\ Z_t\\ 1 \end{bmatrix} $ 
The Hand-Eye calibration procedure returns the following homogeneous transformation $ \begin{bmatrix} X_g\\ Y_g\\ Z_g\\ 1 \end{bmatrix} = \begin{bmatrix} _{}^{g}\textrm{R}_c & _{}^{g}\textrm{t}_c \\ 0_{1 \times 3} & 1 \end{bmatrix} \begin{bmatrix} X_c\\ Y_c\\ Z_c\\ 1 \end{bmatrix} $ 
This problem is also known as solving the $\mathbf{A}\mathbf{X}=\mathbf{X}\mathbf{B}$ equation: - for an eye-in-hand configuration $ \begin{align*} ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(1)} &= \hspace{0.1em} ^{b}{\textrm{T}_g}^{(2)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} \\ 
(^{b}{\textrm{T}_g}^{(2)})^{-1} \hspace{0.2em} ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c &= \hspace{0.1em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} (^{c}{\textrm{T}_t}^{(1)})^{-1} \\ 
\textrm{A}_i \textrm{X} &= \textrm{X} \textrm{B}_i \\ \end{align*} $ 
- for an eye-to-hand configuration $ \begin{align*} ^{g}{\textrm{T}_b}^{(1)} \hspace{0.2em} ^{b}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(1)} &= \hspace{0.1em} ^{g}{\textrm{T}_b}^{(2)} \hspace{0.2em} ^{b}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} \\ 
(^{g}{\textrm{T}_b}^{(2)})^{-1} \hspace{0.2em} ^{g}{\textrm{T}_b}^{(1)} \hspace{0.2em} ^{b}\textrm{T}_c &= \hspace{0.1em} ^{b}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} (^{c}{\textrm{T}_t}^{(1)})^{-1} \\ 
\textrm{A}_i \textrm{X} &= \textrm{X} \textrm{B}_i \\ \end{align*} $ 
\note Additional information can be found on this [website](http://campar.in.tum.de/Chair/HandEyeCalibration). \note A minimum of 2 motions with non parallel rotation axes are necessary to determine the hand-eye transformation. So at least 3 different poses are required, but it is strongly recommended to use many more poses. 


:param R_gripper2base: [in] Rotation part extracted from the homogeneous matrix that transforms a pointexpressed in the gripper frame to the robot base frame ($_{}^{b}\textrm{T}_g$). This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors, for all the transformations from gripper frame to robot base frame. 
:type R_gripper2base: _typing.Sequence[cv2.typing.MatLike]
:param t_gripper2base: [in] Translation part extracted from the homogeneous matrix that transforms a pointexpressed in the gripper frame to the robot base frame ($_{}^{b}\textrm{T}_g$). This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations from gripper frame to robot base frame. 
:type t_gripper2base: _typing.Sequence[cv2.typing.MatLike]
:param R_target2cam: [in] Rotation part extracted from the homogeneous matrix that transforms a pointexpressed in the target frame to the camera frame ($_{}^{c}\textrm{T}_t$). This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors, for all the transformations from calibration target frame to camera frame. 
:type R_target2cam: _typing.Sequence[cv2.typing.MatLike]
:param t_target2cam: [in] Rotation part extracted from the homogeneous matrix that transforms a pointexpressed in the target frame to the camera frame ($_{}^{c}\textrm{T}_t$). This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations from calibration target frame to camera frame. 
:type t_target2cam: _typing.Sequence[cv2.typing.MatLike]
:param R_cam2gripper: [out] Estimated `(3x3)` rotation part extracted from the homogeneous matrix that transforms a pointexpressed in the camera frame to the gripper frame ($_{}^{g}\textrm{T}_c$). 
:type R_cam2gripper: cv2.typing.MatLike | None
:param t_cam2gripper: [out] Estimated `(3x1)` translation part extracted from the homogeneous matrix that transforms a pointexpressed in the camera frame to the gripper frame ($_{}^{g}\textrm{T}_c$). 
:type t_cam2gripper: cv2.typing.MatLike | None
:param method: [in] One of the implemented Hand-Eye calibration method, see cv::HandEyeCalibrationMethod
:type method: HandEyeCalibrationMethod
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} calibrateRobotWorldHandEye(R_world2cam, t_world2cam, R_base2gripper, t_base2gripper[, R_base2world[, t_base2world[, R_gripper2cam[, t_gripper2cam[, method]]]]]) -> R_base2world, t_base2world, R_gripper2cam, t_gripper2cam

Computes Robot-World/Hand-Eye calibration: $_{}^{w}\textrm{T}_b$ and $_{}^{c}\textrm{T}_g$


The function performs the Robot-World/Hand-Eye calibration using various methods. One approach consists in estimating the rotation then the translation (separable solutions): - M. Shah, Solving the robot-world/hand-eye calibration problem using the kronecker product \cite Shah2013SolvingTR 
Another approach consists in estimating simultaneously the rotation and the translation (simultaneous solutions), with the following implemented method: - A. Li, L. Wang, and D. Wu, Simultaneous robot-world and hand-eye calibration using dual-quaternions and kronecker product \cite Li2010SimultaneousRA 
The following picture describes the Robot-World/Hand-Eye calibration problem where the transformations between a robot and a world frame and between a robot gripper ("hand") and a camera ("eye") mounted at the robot end-effector have to be estimated. 
![](pics/robot-world_hand-eye_figure.png) 
The calibration procedure is the following: - a static calibration pattern is used to estimate the transformation between the target frame and the camera frame - the robot gripper is moved in order to acquire several poses - for each pose, the homogeneous transformation between the gripper frame and the robot base frame is recorded using for instance the robot kinematics $ \begin{bmatrix} X_g\\ Y_g\\ Z_g\\ 1 \end{bmatrix} = \begin{bmatrix} _{}^{g}\textrm{R}_b & _{}^{g}\textrm{t}_b \\ 0_{1 \times 3} & 1 \end{bmatrix} \begin{bmatrix} X_b\\ Y_b\\ Z_b\\ 1 \end{bmatrix} $ - for each pose, the homogeneous transformation between the calibration target frame (the world frame) and the camera frame is recorded using for instance a pose estimation method (PnP) from 2D-3D point correspondences $ \begin{bmatrix} X_c\\ Y_c\\ Z_c\\ 1 \end{bmatrix} = \begin{bmatrix} _{}^{c}\textrm{R}_w & _{}^{c}\textrm{t}_w \\ 0_{1 \times 3} & 1 \end{bmatrix} \begin{bmatrix} X_w\\ Y_w\\ Z_w\\ 1 \end{bmatrix} $ 
The Robot-World/Hand-Eye calibration procedure returns the following homogeneous transformations $ \begin{bmatrix} X_w\\ Y_w\\ Z_w\\ 1 \end{bmatrix} = \begin{bmatrix} _{}^{w}\textrm{R}_b & _{}^{w}\textrm{t}_b \\ 0_{1 \times 3} & 1 \end{bmatrix} \begin{bmatrix} X_b\\ Y_b\\ Z_b\\ 1 \end{bmatrix} $ $ \begin{bmatrix} X_c\\ Y_c\\ Z_c\\ 1 \end{bmatrix} = \begin{bmatrix} _{}^{c}\textrm{R}_g & _{}^{c}\textrm{t}_g \\ 0_{1 \times 3} & 1 \end{bmatrix} \begin{bmatrix} X_g\\ Y_g\\ Z_g\\ 1 \end{bmatrix} $ 
This problem is also known as solving the $\mathbf{A}\mathbf{X}=\mathbf{Z}\mathbf{B}$ equation, with: - $\mathbf{A} \Leftrightarrow \hspace{0.1em} _{}^{c}\textrm{T}_w$ - $\mathbf{X} \Leftrightarrow \hspace{0.1em} _{}^{w}\textrm{T}_b$ - $\mathbf{Z} \Leftrightarrow \hspace{0.1em} _{}^{c}\textrm{T}_g$ - $\mathbf{B} \Leftrightarrow \hspace{0.1em} _{}^{g}\textrm{T}_b$ 
\note At least 3 measurements are required (input vectors size must be greater or equal to 3). 


:param R_world2cam: [in] Rotation part extracted from the homogeneous matrix that transforms a pointexpressed in the world frame to the camera frame ($_{}^{c}\textrm{T}_w$). This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors, for all the transformations from world frame to the camera frame. 
:type R_world2cam: _typing.Sequence[cv2.typing.MatLike]
:param t_world2cam: [in] Translation part extracted from the homogeneous matrix that transforms a pointexpressed in the world frame to the camera frame ($_{}^{c}\textrm{T}_w$). This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations from world frame to the camera frame. 
:type t_world2cam: _typing.Sequence[cv2.typing.MatLike]
:param R_base2gripper: [in] Rotation part extracted from the homogeneous matrix that transforms a pointexpressed in the robot base frame to the gripper frame ($_{}^{g}\textrm{T}_b$). This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors, for all the transformations from robot base frame to the gripper frame. 
:type R_base2gripper: _typing.Sequence[cv2.typing.MatLike]
:param t_base2gripper: [in] Rotation part extracted from the homogeneous matrix that transforms a pointexpressed in the robot base frame to the gripper frame ($_{}^{g}\textrm{T}_b$). This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations from robot base frame to the gripper frame. 
:type t_base2gripper: _typing.Sequence[cv2.typing.MatLike]
:param R_base2world: [out] Estimated `(3x3)` rotation part extracted from the homogeneous matrix that transforms a pointexpressed in the robot base frame to the world frame ($_{}^{w}\textrm{T}_b$). 
:type R_base2world: cv2.typing.MatLike | None
:param t_base2world: [out] Estimated `(3x1)` translation part extracted from the homogeneous matrix that transforms a pointexpressed in the robot base frame to the world frame ($_{}^{w}\textrm{T}_b$). 
:type t_base2world: cv2.typing.MatLike | None
:param R_gripper2cam: [out] Estimated `(3x3)` rotation part extracted from the homogeneous matrix that transforms a pointexpressed in the gripper frame to the camera frame ($_{}^{c}\textrm{T}_g$). 
:type R_gripper2cam: cv2.typing.MatLike | None
:param t_gripper2cam: [out] Estimated `(3x1)` translation part extracted from the homogeneous matrix that transforms a pointexpressed in the gripper frame to the camera frame ($_{}^{c}\textrm{T}_g$). 
:type t_gripper2cam: cv2.typing.MatLike | None
:param method: [in] One of the implemented Robot-World/Hand-Eye calibration method, see cv::RobotWorldHandEyeCalibrationMethod
:type method: RobotWorldHandEyeCalibrationMethod
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight) -> fovx, fovy, focalLength, principalPoint, aspectRatio

Computes useful camera characteristics from the camera intrinsic matrix.


The function computes various useful camera characteristics from the previously estimated camera matrix. 
@note Do keep in mind that the unity measure 'mm' stands for whatever unit of measure one chooses for the chessboard pitch (it can thus be any value). 


:param cameraMatrix: Input camera intrinsic matrix that can be estimated by #calibrateCamera or#stereoCalibrate . 
:type cameraMatrix: cv2.typing.MatLike
:param imageSize: Input image size in pixels.
:type imageSize: cv2.typing.Size
:param apertureWidth: Physical width in mm of the sensor.
:type apertureWidth: float
:param apertureHeight: Physical height in mm of the sensor.
:type apertureHeight: float
:param fovx: Output field of view in degrees along the horizontal sensor axis.
:type fovx: 
:param fovy: Output field of view in degrees along the vertical sensor axis.
:type fovy: 
:param focalLength: Focal length of the lens in mm.
:type focalLength: 
:param principalPoint: Principal point in mm.
:type principalPoint: 
:param aspectRatio: $f_y/f_x$
:type aspectRatio: 
:rtype: tuple[float, float, float, cv2.typing.Point2d, float]
````


````{py:function} cartToPolar(x, y[, magnitude[, angle[, angleInDegrees]]]) -> magnitude, angle

Calculates the magnitude and angle of 2D vectors.


The function cv::cartToPolar calculates either the magnitude, angle, or both for every 2D vector (x(I),y(I)): $\begin{array}{l} \texttt{magnitude} (I)= \sqrt{\texttt{x}(I)^2+\texttt{y}(I)^2} , \\ \texttt{angle} (I)= \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))[ \cdot180 / \pi ] \end{array}$ 
The angles are calculated with accuracy about 0.3 degrees. For the point (0,0), the angle is set to 0. 
**See also:** Sobel, Scharr


:param x: array of x-coordinates; this must be a single-precision ordouble-precision floating-point array. 
:type x: cv2.typing.MatLike
:param y: array of y-coordinates, that must have the same size and same type as x.
:type y: cv2.typing.MatLike
:param magnitude: output array of magnitudes of the same size and type as x.
:type magnitude: cv2.typing.MatLike | None
:param angle: output array of angles that has the same size and type asx; the angles are measured in radians (from 0 to 2\*Pi) or in degrees (0 to 360 degrees). 
:type angle: cv2.typing.MatLike | None
:param angleInDegrees: a flag, indicating whether the angles are measuredin radians (which is by default), or in degrees. 
:type angleInDegrees: bool
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} checkChessboard(img, size) -> retval






:param img: 
:type img: cv2.typing.MatLike
:param size: 
:type size: cv2.typing.Size
:rtype: bool
````


````{py:function} checkHardwareSupport(feature) -> retval

Returns true if the specified feature is supported by the host hardware.


The function returns true if the host hardware supports the specified feature. When user calls setUseOptimized(false), the subsequent calls to checkHardwareSupport() will return false until setUseOptimized(true) is called. This way user can dynamically switch on and off the optimized code in OpenCV. 


:param feature: The feature of interest, one of cv::CpuFeatures
:type feature: int
:rtype: bool
````


````{py:function} checkRange(a[, quiet[, minVal[, maxVal]]]) -> retval, pos

Checks every element of an input array for invalid values.


The function cv::checkRange checks that every array element is neither NaN nor infinite. When minVal \> -DBL_MAX and maxVal \< DBL_MAX, the function also checks that each value is between minVal and maxVal. In case of multi-channel arrays, each channel is processed independently. If some values are out of range, position of the first outlier is stored in pos (when pos != NULL). Then, the function either returns false (when quiet=true) or throws an exception. 


:param a: input array.
:type a: cv2.typing.MatLike
:param quiet: a flag, indicating whether the functions quietly return false when the array elementsare out of range or they throw an exception. 
:type quiet: bool
:param pos: optional output parameter, when not NULL, must be a pointer to array of src.dimselements. 
:type pos: 
:param minVal: inclusive lower boundary of valid values range.
:type minVal: float
:param maxVal: exclusive upper boundary of valid values range.
:type maxVal: float
:rtype: tuple[bool, cv2.typing.Point]
````


````{py:function} circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img

Draws a circle.


The function cv::circle draws a simple or filled circle with a given center and radius. 


:param img: Image where the circle is drawn.
:type img: cv2.typing.MatLike
:param center: Center of the circle.
:type center: cv2.typing.Point
:param radius: Radius of the circle.
:type radius: int
:param color: Circle color.
:type color: cv2.typing.Scalar
:param thickness: Thickness of the circle outline, if positive. Negative values, like #FILLED,mean that a filled circle is to be drawn. 
:type thickness: int
:param lineType: Type of the circle boundary. See #LineTypes
:type lineType: int
:param shift: Number of fractional bits in the coordinates of the center and in the radius value.
:type shift: int
:rtype: cv2.typing.MatLike
````


````{py:function} clipLine(imgRect, pt1, pt2) -> retval, pt1, pt2




@overload 


:param imgRect: Image rectangle.
:type imgRect: cv2.typing.Rect
:param pt1: First line point.
:type pt1: cv2.typing.Point
:param pt2: Second line point.
:type pt2: cv2.typing.Point
:rtype: tuple[bool, cv2.typing.Point, cv2.typing.Point]
````


````{py:function} colorChange(src, mask[, dst[, red_mul[, green_mul[, blue_mul]]]]) -> dst

Given an original color image, two differently colored versions of this image can be mixedseamlessly. 


Multiplication factor is between .5 to 2.5. 


:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param mask: Input 8-bit 1 or 3-channel image.
:type mask: cv2.typing.MatLike
:param dst: Output image with the same size and type as src .
:type dst: cv2.typing.MatLike | None
:param red_mul: R-channel multiply factor.
:type red_mul: float
:param green_mul: G-channel multiply factor.
:type green_mul: float
:param blue_mul: B-channel multiply factor.
:type blue_mul: float
:rtype: cv2.typing.MatLike
````


````{py:function} compare(src1, src2, cmpop[, dst]) -> dst

Performs the per-element comparison of two arrays or an array and scalar value.


The function compares: Elements of two arrays when src1 and src2 have the same size: $\texttt{dst} (I) =  \texttt{src1} (I)  \,\texttt{cmpop}\, \texttt{src2} (I)$ Elements of src1 with a scalar src2 when src2 is constructed from Scalar or has a single element: $\texttt{dst} (I) =  \texttt{src1}(I) \,\texttt{cmpop}\,  \texttt{src2}$ src1 with elements of src2 when src1 is constructed from Scalar or has a single element: $\texttt{dst} (I) =  \texttt{src1}  \,\texttt{cmpop}\, \texttt{src2} (I)$ When the comparison result is true, the corresponding element of output array is set to 255. The comparison operations can be replaced with the equivalent matrix expressions: 
```cpp
Mat dst1 = src1 >= src2;
Mat dst2 = src1 < 8;

```

**See also:** checkRange, min, max, threshold


:param src1: first input array or a scalar; when it is an array, it must have a single channel.
:type src1: cv2.typing.MatLike
:param src2: second input array or a scalar; when it is an array, it must have a single channel.
:type src2: cv2.typing.MatLike
:param dst: output array of type ref CV_8U that has the same size and the same number of channels asthe input arrays. 
:type dst: cv2.typing.MatLike | None
:param cmpop: a flag, that specifies correspondence between the arrays (cv::CmpTypes)
:type cmpop: int
:rtype: cv2.typing.MatLike
````


````{py:function} compareHist(H1, H2, method) -> retval

Compares two histograms.


The function cv::compareHist compares two dense or two sparse histograms using the specified method. 
The function returns $d(H_1, H_2)$ . 
While the function works well with 1-, 2-, 3-dimensional dense histograms, it may not be suitable for high-dimensional sparse histograms. In such histograms, because of aliasing and sampling problems, the coordinates of non-zero histogram bins can slightly shift. To compare such histograms or more general sparse configurations of weighted points, consider using the #EMD function. 


:param H1: First compared histogram.
:type H1: cv2.typing.MatLike
:param H2: Second compared histogram of the same size as H1 .
:type H2: cv2.typing.MatLike
:param method: Comparison method, see #HistCompMethods
:type method: int
:rtype: float
````


````{py:function} completeSymm(m[, lowerToUpper]) -> m

Copies the lower or the upper half of a square matrix to its another half.


The function cv::completeSymm copies the lower or the upper half of a square matrix to its another half. The matrix diagonal remains unchanged: - $\texttt{m}_{ij}=\texttt{m}_{ji}$ for $i > j$ if lowerToUpper=false - $\texttt{m}_{ij}=\texttt{m}_{ji}$ for $i < j$ if lowerToUpper=true 
**See also:** flip, transpose


:param m: input-output floating-point square matrix.
:type m: cv2.typing.MatLike
:param lowerToUpper: operation flag; if true, the lower half is copied tothe upper half. Otherwise, the upper half is copied to the lower half. 
:type lowerToUpper: bool
:rtype: cv2.typing.MatLike
````


````{py:function} composeRT(rvec1, tvec1, rvec2, tvec2[, rvec3[, tvec3[, dr3dr1[, dr3dt1[, dr3dr2[, dr3dt2[, dt3dr1[, dt3dt1[, dt3dr2[, dt3dt2]]]]]]]]]]) -> rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2

Combines two rotation-and-shift transformations.


The functions compute: 
$\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,$ 
where $\mathrm{rodrigues}$ denotes a rotation vector to a rotation matrix transformation, and $\mathrm{rodrigues}^{-1}$ denotes the inverse transformation. See #Rodrigues for details. 
Also, the functions can compute the derivatives of the output vectors with regards to the input vectors (see #matMulDeriv ). The functions are used inside #stereoCalibrate but can also be used in your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a function that contains a matrix multiplication. 


:param rvec1: First rotation vector.
:type rvec1: cv2.typing.MatLike
:param tvec1: First translation vector.
:type tvec1: cv2.typing.MatLike
:param rvec2: Second rotation vector.
:type rvec2: cv2.typing.MatLike
:param tvec2: Second translation vector.
:type tvec2: cv2.typing.MatLike
:param rvec3: Output rotation vector of the superposition.
:type rvec3: cv2.typing.MatLike | None
:param tvec3: Output translation vector of the superposition.
:type tvec3: cv2.typing.MatLike | None
:param dr3dr1: Optional output derivative of rvec3 with regard to rvec1
:type dr3dr1: cv2.typing.MatLike | None
:param dr3dt1: Optional output derivative of rvec3 with regard to tvec1
:type dr3dt1: cv2.typing.MatLike | None
:param dr3dr2: Optional output derivative of rvec3 with regard to rvec2
:type dr3dr2: cv2.typing.MatLike | None
:param dr3dt2: Optional output derivative of rvec3 with regard to tvec2
:type dr3dt2: cv2.typing.MatLike | None
:param dt3dr1: Optional output derivative of tvec3 with regard to rvec1
:type dt3dr1: cv2.typing.MatLike | None
:param dt3dt1: Optional output derivative of tvec3 with regard to tvec1
:type dt3dt1: cv2.typing.MatLike | None
:param dt3dr2: Optional output derivative of tvec3 with regard to rvec2
:type dt3dr2: cv2.typing.MatLike | None
:param dt3dt2: Optional output derivative of tvec3 with regard to tvec2
:type dt3dt2: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} computeCorrespondEpilines(points, whichImage, F[, lines]) -> lines

For points in an image of a stereo pair, computes the corresponding epilines in the other image.


For every point in one of the two images of a stereo pair, the function finds the equation of the corresponding epipolar line in the other image. 
From the fundamental matrix definition (see #findFundamentalMat ), line $l^{(2)}_i$ in the second image for the point $p^{(1)}_i$ in the first image (when whichImage=1 ) is computed as: 
$l^{(2)}_i = F p^{(1)}_i$ 
And vice versa, when whichImage=2, $l^{(1)}_i$ is computed from $p^{(2)}_i$ as: 
$l^{(1)}_i = F^T p^{(2)}_i$ 
Line coefficients are defined up to a scale. They are normalized so that $a_i^2+b_i^2=1$ . 


:param points: Input points. $N \times 1$ or $1 \times N$ matrix of type CV_32FC2 orvector\<Point2f\> . 
:type points: cv2.typing.MatLike
:param whichImage: Index of the image (1 or 2) that contains the points .
:type whichImage: int
:param F: Fundamental matrix that can be estimated using #findFundamentalMat or #stereoRectify .
:type F: cv2.typing.MatLike
:param lines: Output vector of the epipolar lines corresponding to the points in the other image.Each line $ax + by + c=0$ is encoded by 3 numbers $(a, b, c)$ . 
:type lines: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} computeECC(templateImage, inputImage[, inputMask]) -> retval

Computes the Enhanced Correlation Coefficient value between two images @cite EP08 .


@sa findTransformECC 


:param templateImage: single-channel template image; CV_8U or CV_32F array.
:type templateImage: cv2.typing.MatLike
:param inputImage: single-channel input image to be warped to provide an image similar totemplateImage, same type as templateImage. 
:type inputImage: cv2.typing.MatLike
:param inputMask: An optional mask to indicate valid values of inputImage.
:type inputMask: cv2.typing.MatLike | None
:rtype: float
````


````{py:function} connectedComponents(image[, labels[, connectivity[, ltype]]]) -> retval, labels




@overload 


:param image: the 8-bit single-channel image to be labeled
:type image: cv2.typing.MatLike
:param labels: destination labeled image
:type labels: cv2.typing.MatLike | None
:param connectivity: 8 or 4 for 8-way or 4-way connectivity respectively
:type connectivity: int
:param ltype: output image label type. Currently CV_32S and CV_16U are supported.
:type ltype: int
:rtype: tuple[int, cv2.typing.MatLike]
````


````{py:function} connectedComponentsWithAlgorithm(image, connectivity, ltype, ccltype[, labels]) -> retval, labels

computes the connected components labeled image of boolean image


image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0 represents the background label. ltype specifies the output label image type, an important consideration based on the total number of labels or alternatively the total number of pixels in the source image. ccltype specifies the connected components labeling algorithm to use, currently Bolelli (Spaghetti) @cite Bolelli2019, Grana (BBDT) @cite Grana2010 and Wu's (SAUF) @cite Wu2009 algorithms are supported, see the #ConnectedComponentsAlgorithmsTypes for details. Note that SAUF algorithm forces a row major ordering of labels while Spaghetti and BBDT do not. This function uses parallel version of the algorithms if at least one allowed parallel framework is enabled and if the rows of the image are at least twice the number returned by #getNumberOfCPUs. 


:param image: the 8-bit single-channel image to be labeled
:type image: cv2.typing.MatLike
:param labels: destination labeled image
:type labels: cv2.typing.MatLike | None
:param connectivity: 8 or 4 for 8-way or 4-way connectivity respectively
:type connectivity: int
:param ltype: output image label type. Currently CV_32S and CV_16U are supported.
:type ltype: int
:param ccltype: connected components algorithm type (see the #ConnectedComponentsAlgorithmsTypes).
:type ccltype: int
:rtype: tuple[int, cv2.typing.MatLike]
````


````{py:function} connectedComponentsWithStats(image[, labels[, stats[, centroids[, connectivity[, ltype]]]]]) -> retval, labels, stats, centroids




@overload 


:param image: the 8-bit single-channel image to be labeled
:type image: cv2.typing.MatLike
:param labels: destination labeled image
:type labels: cv2.typing.MatLike | None
:param stats: statistics output for each label, including the background label.Statistics are accessed via stats(label, COLUMN) where COLUMN is one of #ConnectedComponentsTypes, selecting the statistic. The data type is CV_32S. 
:type stats: cv2.typing.MatLike | None
:param centroids: centroid output for each label, including the background label. Centroids areaccessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F. 
:type centroids: cv2.typing.MatLike | None
:param connectivity: 8 or 4 for 8-way or 4-way connectivity respectively
:type connectivity: int
:param ltype: output image label type. Currently CV_32S and CV_16U are supported.
:type ltype: int
:rtype: tuple[int, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} connectedComponentsWithStatsWithAlgorithm(image, connectivity, ltype, ccltype[, labels[, stats[, centroids]]]) -> retval, labels, stats, centroids

computes the connected components labeled image of boolean image and also produces a statistics output for each label


image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0 represents the background label. ltype specifies the output label image type, an important consideration based on the total number of labels or alternatively the total number of pixels in the source image. ccltype specifies the connected components labeling algorithm to use, currently Bolelli (Spaghetti) @cite Bolelli2019, Grana (BBDT) @cite Grana2010 and Wu's (SAUF) @cite Wu2009 algorithms are supported, see the #ConnectedComponentsAlgorithmsTypes for details. Note that SAUF algorithm forces a row major ordering of labels while Spaghetti and BBDT do not. This function uses parallel version of the algorithms (statistics included) if at least one allowed parallel framework is enabled and if the rows of the image are at least twice the number returned by #getNumberOfCPUs. 


:param image: the 8-bit single-channel image to be labeled
:type image: cv2.typing.MatLike
:param labels: destination labeled image
:type labels: cv2.typing.MatLike | None
:param stats: statistics output for each label, including the background label.Statistics are accessed via stats(label, COLUMN) where COLUMN is one of #ConnectedComponentsTypes, selecting the statistic. The data type is CV_32S. 
:type stats: cv2.typing.MatLike | None
:param centroids: centroid output for each label, including the background label. Centroids areaccessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F. 
:type centroids: cv2.typing.MatLike | None
:param connectivity: 8 or 4 for 8-way or 4-way connectivity respectively
:type connectivity: int
:param ltype: output image label type. Currently CV_32S and CV_16U are supported.
:type ltype: int
:param ccltype: connected components algorithm type (see #ConnectedComponentsAlgorithmsTypes).
:type ccltype: int
:rtype: tuple[int, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} contourArea(contour[, oriented]) -> retval

Calculates a contour area.


The function computes a contour area. Similarly to moments , the area is computed using the Green formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using #drawContours or #fillPoly , can be different. Also, the function will most certainly give a wrong results for contours with self-intersections. 
Example: 
```c++
vector<Point> contour;
contour.push_back(Point2f(0, 0));
contour.push_back(Point2f(10, 0));
contour.push_back(Point2f(10, 10));
contour.push_back(Point2f(5, 4));

double area0 = contourArea(contour);
vector<Point> approx;
approxPolyDP(contour, approx, 5, true);
double area1 = contourArea(approx);

cout << "area0 =" << area0 << endl <<
"area1 =" << area1 << endl <<
"approx poly vertices" << approx.size() << endl;
```



:param contour: Input vector of 2D points (contour vertices), stored in std::vector or Mat.
:type contour: cv2.typing.MatLike
:param oriented: Oriented area flag. If it is true, the function returns a signed area value,depending on the contour orientation (clockwise or counter-clockwise). Using this feature you can determine orientation of a contour by taking the sign of an area. By default, the parameter is false, which means that the absolute value is returned. 
:type oriented: bool
:rtype: float
````


````{py:function} convertFp16(src[, dst]) -> dst

Converts an array to half precision floating number.


This function converts FP32 (single precision floating point) from/to FP16 (half precision floating point). CV_16S format is used to represent FP16 data. There are two use modes (src -> dst): CV_32F -> CV_16S and CV_16S -> CV_32F. The input array has to have type of CV_32F or CV_16S to represent the bit depth. If the input array is neither of them, the function will raise an error. The format of half precision floating point is defined in IEEE 754-2008. 


:param src: input array.
:type src: cv2.typing.MatLike
:param dst: output array.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} convertMaps(map1, map2, dstmap1type[, dstmap1[, dstmap2[, nninterpolation]]]) -> dstmap1, dstmap2

Converts image transformation maps from one representation to another.


The function converts a pair of maps for remap from one representation to another. The following options ( (map1.type(), map2.type()) $\rightarrow$ (dstmap1.type(), dstmap2.type()) ) are supported: 
- $\texttt{(CV_32FC1, CV_32FC1)} \rightarrow \texttt{(CV_16SC2, CV_16UC1)}$. This is the most frequently used conversion operation, in which the original floating-point maps (see #remap) are converted to a more compact and much faster fixed-point representation. The first output array contains the rounded coordinates and the second array (created only when nninterpolation=false ) contains indices in the interpolation tables. 
- $\texttt{(CV_32FC2)} \rightarrow \texttt{(CV_16SC2, CV_16UC1)}$. The same as above but the original maps are stored in one 2-channel matrix. 
- Reverse conversion. Obviously, the reconstructed floating-point maps will not be exactly the same as the originals. 
**See also:**  remap, undistort, initUndistortRectifyMap


:param map1: The first input map of type CV_16SC2, CV_32FC1, or CV_32FC2 .
:type map1: cv2.typing.MatLike
:param map2: The second input map of type CV_16UC1, CV_32FC1, or none (empty matrix),respectively. 
:type map2: cv2.typing.MatLike
:param dstmap1: The first output map that has the type dstmap1type and the same size as src .
:type dstmap1: cv2.typing.MatLike | None
:param dstmap2: The second output map.
:type dstmap2: cv2.typing.MatLike | None
:param dstmap1type: Type of the first output map that should be CV_16SC2, CV_32FC1, orCV_32FC2 . 
:type dstmap1type: int
:param nninterpolation: Flag indicating whether the fixed-point maps are used for thenearest-neighbor or for a more complex interpolation. 
:type nninterpolation: bool
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} convertPointsFromHomogeneous(src[, dst]) -> dst

Converts points from homogeneous to Euclidean space.


The function converts points homogeneous to Euclidean space using perspective projection. That is, each point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn). When xn=0, the output point coordinates will be (0,0,0,...). 


:param src: Input vector of N-dimensional points.
:type src: cv2.typing.MatLike
:param dst: Output vector of N-1-dimensional points.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} convertPointsToHomogeneous(src[, dst]) -> dst

Converts points from Euclidean to homogeneous space.


The function converts points from Euclidean to homogeneous space by appending 1's to the tuple of point coordinates. That is, each point (x1, x2, ..., xn) is converted to (x1, x2, ..., xn, 1). 


:param src: Input vector of N-dimensional points.
:type src: cv2.typing.MatLike
:param dst: Output vector of N+1-dimensional points.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst

Scales, calculates absolute values, and converts the result to 8-bit.


On each element of the input array, the function convertScaleAbs performs three operations sequentially: scaling, taking an absolute value, conversion to an unsigned 8-bit type: $\texttt{dst} (I)= \texttt{saturate\_cast<uchar>} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)$ In case of multi-channel arrays, the function processes each channel independently. When the output is not 8-bit, the operation can be emulated by calling the Mat::convertTo method (or by using matrix expressions) and then by calculating an absolute value of the result. For example: 
```cpp
Mat_<float> A(30,30);
randu(A, Scalar(-100), Scalar(100));
Mat_<float> B = A*5 + 3;
B = abs(B);
// Mat_<float> B = abs(A*5+3) will also do the job,
// but it will allocate a temporary matrix
```

**See also:**  Mat::convertTo, cv::abs(const Mat&)


:param src: input array.
:type src: cv2.typing.MatLike
:param dst: output array.
:type dst: cv2.typing.MatLike | None
:param alpha: optional scale factor.
:type alpha: float
:param beta: optional delta added to the scaled values.
:type beta: float
:rtype: cv2.typing.MatLike
````


````{py:function} convexHull(points[, hull[, clockwise[, returnPoints]]]) -> hull

Finds the convex hull of a point set.


The function cv::convexHull finds the convex hull of a 2D point set using the Sklansky's algorithm @cite Sklansky82 that has *O(N logN)* complexity in the current implementation. 
Check @ref tutorial_hull "the corresponding tutorial" for more details. 
useful links: 
https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/ 
```{note}
`points` and `hull` should be different arrays, inplace processing isn't supported.
```


:param points: Input 2D point set, stored in std::vector or Mat.
:type points: cv2.typing.MatLike
:param hull: Output convex hull. It is either an integer vector of indices or vector of points. Inthe first case, the hull elements are 0-based indices of the convex hull points in the original array (since the set of convex hull points is a subset of the original point set). In the second case, hull elements are the convex hull points themselves. 
:type hull: cv2.typing.MatLike | None
:param clockwise: Orientation flag. If it is true, the output convex hull is oriented clockwise.Otherwise, it is oriented counter-clockwise. The assumed coordinate system has its X axis pointing to the right, and its Y axis pointing upwards. 
:type clockwise: bool
:param returnPoints: Operation flag. In case of a matrix, when the flag is true, the functionreturns convex hull points. Otherwise, it returns indices of the convex hull points. When the output array is std::vector, the flag is ignored, and the output depends on the type of the vector: std::vector\<int\> implies returnPoints=false, std::vector\<Point\> implies returnPoints=true. 
:type returnPoints: bool
:rtype: cv2.typing.MatLike
````


````{py:function} convexityDefects(contour, convexhull[, convexityDefects]) -> convexityDefects

Finds the convexity defects of a contour.


The figure below displays convexity defects of a hand contour: 
![image](pics/defects.png) 


:param contour: Input contour.
:type contour: cv2.typing.MatLike
:param convexhull: Convex hull obtained using convexHull that should contain indices of the contourpoints that make the hull. 
:type convexhull: cv2.typing.MatLike
:param convexityDefects: The output vector of convexity defects. In C++ and the new Python/Javainterface each convexity defect is represented as 4-element integer vector (a.k.a. #Vec4i): (start_index, end_index, farthest_pt_index, fixpt_depth), where indices are 0-based indices in the original contour of the convexity defect beginning, end and the farthest point, and fixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the farthest contour point and the hull. That is, to get the floating-point value of the depth will be fixpt_depth/256.0. 
:type convexityDefects: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) -> dst

Forms a border around an image.


The function copies the source image into the middle of the destination image. The areas to the left, to the right, above and below the copied source image will be filled with extrapolated pixels. This is not what filtering functions based on it do (they extrapolate pixels on-fly), but what other more complex functions, including your own, may do to simplify image boundary handling. 
The function supports the mode when src is already in the middle of dst . In this case, the function does not copy src itself but simply constructs the border, for example: 

```cpp
// let border be the same in all directions
int border=2;
// constructs a larger image to fit both the image and the border
Mat gray_buf(rgb.rows + border*2, rgb.cols + border*2, rgb.depth());
// select the middle part of it w/o copying data
Mat gray(gray_canvas, Rect(border, border, rgb.cols, rgb.rows));
// convert image from RGB to grayscale
cvtColor(rgb, gray, COLOR_RGB2GRAY);
// form a border in-place
copyMakeBorder(gray, gray_buf, border, border,
border, border, BORDER_REPLICATE);
// now do some custom filtering ...

```

```{note}
When the source image is a part (ROI) of a bigger image, the function will try to use thepixels outside of the ROI to form a border. To disable this feature and always do extrapolation, as if src was not a ROI, use borderType | #BORDER_ISOLATED. 
```
**See also:**  borderInterpolate


:param src: Source image.
:type src: cv2.typing.MatLike
:param dst: Destination image of the same type as src and the size Size(src.cols+left+right,src.rows+top+bottom) . 
:type dst: cv2.typing.MatLike | None
:param top: the top pixels
:type top: int
:param bottom: the bottom pixels
:type bottom: int
:param left: the left pixels
:type left: int
:param right: Parameter specifying how many pixels in each direction from the source image rectangleto extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs to be built. 
:type right: int
:param borderType: Border type. See borderInterpolate for details.
:type borderType: int
:param value: Border value if borderType==BORDER_CONSTANT .
:type value: cv2.typing.Scalar
:rtype: cv2.typing.MatLike
````


````{py:function} copyTo(src, mask[, dst]) -> dst

 This is an overloaded member function, provided for convenience (python)Copies the matrix to another one. When the operation mask is specified, if the Mat::create call shown above reallocates the matrix, the newly allocated matrix is initialized with all zeros before copying the data. 




:param src: source matrix.
:type src: cv2.typing.MatLike
:param dst: Destination matrix. If it does not have a proper size or type before the operation, it isreallocated. 
:type dst: cv2.typing.MatLike | None
:param mask: Operation mask of the same size as \*this. Its non-zero elements indicate which matrixelements need to be copied. The mask has to be of type CV_8U and can have 1 or multiple channels. 
:type mask: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````


````{py:function} cornerEigenValsAndVecs(src, blockSize, ksize[, dst[, borderType]]) -> dst

Calculates eigenvalues and eigenvectors of image blocks for corner detection.


For every pixel $p$ , the function cornerEigenValsAndVecs considers a blockSize $\times$ blockSize neighborhood $S(p)$ . It calculates the covariation matrix of derivatives over the neighborhood as: 
$M =  \begin{bmatrix} \sum _{S(p)}(dI/dx)^2 &  \sum _{S(p)}dI/dx dI/dy  \\ \sum _{S(p)}dI/dx dI/dy &  \sum _{S(p)}(dI/dy)^2 \end{bmatrix}$ 
where the derivatives are computed using the Sobel operator. 
After that, it finds eigenvectors and eigenvalues of $M$ and stores them in the destination image as $(\lambda_1, \lambda_2, x_1, y_1, x_2, y_2)$ where 
-   $\lambda_1, \lambda_2$ are the non-sorted eigenvalues of $M$ -   $x_1, y_1$ are the eigenvectors corresponding to $\lambda_1$ -   $x_2, y_2$ are the eigenvectors corresponding to $\lambda_2$ 
The output of the function can be used for robust edge or corner detection. 
**See also:**  cornerMinEigenVal, cornerHarris, preCornerDetect


:param src: Input single-channel 8-bit or floating-point image.
:type src: cv2.typing.MatLike
:param dst: Image to store the results. It has the same size as src and the type CV_32FC(6) .
:type dst: cv2.typing.MatLike | None
:param blockSize: Neighborhood size (see details below).
:type blockSize: int
:param ksize: Aperture parameter for the Sobel operator.
:type ksize: int
:param borderType: Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]) -> dst

Harris corner detector.


The function runs the Harris corner detector on the image. Similarly to cornerMinEigenVal and cornerEigenValsAndVecs , for each pixel $(x, y)$ it calculates a $2\times2$ gradient covariance matrix $M^{(x,y)}$ over a $\texttt{blockSize} \times \texttt{blockSize}$ neighborhood. Then, it computes the following characteristic: 
$\texttt{dst} (x,y) =  \mathrm{det} M^{(x,y)} - k  \cdot \left ( \mathrm{tr} M^{(x,y)} \right )^2$ 
Corners in the image can be found as the local maxima of this response map. 


:param src: Input single-channel 8-bit or floating-point image.
:type src: cv2.typing.MatLike
:param dst: Image to store the Harris detector responses. It has the type CV_32FC1 and the samesize as src . 
:type dst: cv2.typing.MatLike | None
:param blockSize: Neighborhood size (see the details on #cornerEigenValsAndVecs ).
:type blockSize: int
:param ksize: Aperture parameter for the Sobel operator.
:type ksize: int
:param k: Harris detector free parameter. See the formula above.
:type k: float
:param borderType: Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} cornerMinEigenVal(src, blockSize[, dst[, ksize[, borderType]]]) -> dst

Calculates the minimal eigenvalue of gradient matrices for corner detection.


The function is similar to cornerEigenValsAndVecs but it calculates and stores only the minimal eigenvalue of the covariance matrix of derivatives, that is, $\min(\lambda_1, \lambda_2)$ in terms of the formulae in the cornerEigenValsAndVecs description. 


:param src: Input single-channel 8-bit or floating-point image.
:type src: cv2.typing.MatLike
:param dst: Image to store the minimal eigenvalues. It has the type CV_32FC1 and the same size assrc . 
:type dst: cv2.typing.MatLike | None
:param blockSize: Neighborhood size (see the details on #cornerEigenValsAndVecs ).
:type blockSize: int
:param ksize: Aperture parameter for the Sobel operator.
:type ksize: int
:param borderType: Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} cornerSubPix(image, corners, winSize, zeroZone, criteria) -> corners

Refines the corner locations.


The function iterates to find the sub-pixel accurate location of corners or radial saddle points as described in @cite forstner1987fast, and as shown on the figure below. 
![image](pics/cornersubpix.png) 
Sub-pixel accurate corner locator is based on the observation that every vector from the center $q$ to a point $p$ located within a neighborhood of $q$ is orthogonal to the image gradient at $p$ subject to image and measurement noise. Consider the expression: 
$\epsilon _i = {DI_{p_i}}^T  \cdot (q - p_i)$ 
where ${DI_{p_i}}$ is an image gradient at one of the points $p_i$ in a neighborhood of $q$ . The value of $q$ is to be found so that $\epsilon_i$ is minimized. A system of equations may be set up with $\epsilon_i$ set to zero: 
$\sum _i(DI_{p_i}  \cdot {DI_{p_i}}^T) \cdot q -  \sum _i(DI_{p_i}  \cdot {DI_{p_i}}^T  \cdot p_i)$ 
where the gradients are summed within a neighborhood ("search window") of $q$ . Calling the first gradient term $G$ and the second gradient term $b$ gives: 
$q = G^{-1}  \cdot b$ 
The algorithm sets the center of the neighborhood window at this new center $q$ and then iterates until the center stays within a set threshold. 


:param image: Input single-channel, 8-bit or float image.
:type image: cv2.typing.MatLike
:param corners: Initial coordinates of the input corners and refined coordinates provided foroutput. 
:type corners: cv2.typing.MatLike
:param winSize: Half of the side length of the search window. For example, if winSize=Size(5,5) ,then a $(5*2+1) \times (5*2+1) = 11 \times 11$ search window is used. 
:type winSize: cv2.typing.Size
:param zeroZone: Half of the size of the dead region in the middle of the search zone over whichthe summation in the formula below is not done. It is used sometimes to avoid possible singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size. 
:type zeroZone: cv2.typing.Size
:param criteria: Criteria for termination of the iterative process of corner refinement. That is,the process of corner position refinement stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration. 
:type criteria: cv2.typing.TermCriteria
:rtype: cv2.typing.MatLike
````


````{py:function} correctMatches(F, points1, points2[, newPoints1[, newPoints2]]) -> newPoints1, newPoints2

Refines coordinates of corresponding points.


The function implements the Optimal Triangulation Method (see Multiple View Geometry @cite HartleyZ00 for details). For each given point correspondence points1[i] \<-\> points2[i], and a fundamental matrix F, it computes the corrected correspondences newPoints1[i] \<-\> newPoints2[i] that minimize the geometric error $d(points1[i], newPoints1[i])^2 + d(points2[i],newPoints2[i])^2$ (where $d(a,b)$ is the geometric distance between points $a$ and $b$ ) subject to the epipolar constraint $newPoints2^T \cdot F \cdot newPoints1 = 0$ . 


:param F: 3x3 fundamental matrix.
:type F: cv2.typing.MatLike
:param points1: 1xN array containing the first set of points.
:type points1: cv2.typing.MatLike
:param points2: 1xN array containing the second set of points.
:type points2: cv2.typing.MatLike
:param newPoints1: The optimized points1.
:type newPoints1: cv2.typing.MatLike | None
:param newPoints2: The optimized points2.
:type newPoints2: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} countNonZero(src) -> retval

Counts non-zero array elements.


The function returns the number of non-zero elements in src : $\sum _{I: \; \texttt{src} (I) \ne0 } 1$ 
**See also:**  mean, meanStdDev, norm, minMaxLoc, calcCovarMatrix


:param src: single-channel array.
:type src: cv2.typing.MatLike
:rtype: int
````


````{py:function} createAlignMTB([, max_bits[, exclude_range[, cut]]]) -> retval

Creates AlignMTB object




:param max_bits: logarithm to the base 2 of maximal shift in each dimension. Values of 5 and 6 areusually good enough (31 and 63 pixels shift respectively). 
:type max_bits: int
:param exclude_range: range for exclusion bitmap that is constructed to suppress noise around themedian value. 
:type exclude_range: int
:param cut: if true cuts images, otherwise fills the new regions with zeros.
:type cut: bool
:rtype: AlignMTB
````


````{py:function} createBackgroundSubtractorKNN([, history[, dist2Threshold[, detectShadows]]]) -> retval

Creates KNN Background Subtractor




:param history: Length of the history.
:type history: int
:param dist2Threshold: Threshold on the squared distance between the pixel and the sample to decidewhether a pixel is close to that sample. This parameter does not affect the background update. 
:type dist2Threshold: float
:param detectShadows: If true, the algorithm will detect shadows and mark them. It decreases thespeed a bit, so if you do not need this feature, set the parameter to false. 
:type detectShadows: bool
:rtype: BackgroundSubtractorKNN
````


````{py:function} createBackgroundSubtractorMOG2([, history[, varThreshold[, detectShadows]]]) -> retval

Creates MOG2 Background Subtractor




:param history: Length of the history.
:type history: int
:param varThreshold: Threshold on the squared Mahalanobis distance between the pixel and the modelto decide whether a pixel is well described by the background model. This parameter does not affect the background update. 
:type varThreshold: float
:param detectShadows: If true, the algorithm will detect shadows and mark them. It decreases thespeed a bit, so if you do not need this feature, set the parameter to false. 
:type detectShadows: bool
:rtype: BackgroundSubtractorMOG2
````


````{py:function} createButton(buttonName, onChange [, userData, buttonType, initialButtonState]) -> None






:param buttonName: 
:type buttonName: str
:param onChange: 
:type onChange: _typing.Callable[[tuple[int] | tuple[int, _typing.Any]], None]
:param userData: 
:type userData: _typing.Any | None
:param buttonType: 
:type buttonType: int
:param initialButtonState: 
:type initialButtonState: int
:rtype: None
````


````{py:function} createCLAHE([, clipLimit[, tileGridSize]]) -> retval

Creates a smart pointer to a cv::CLAHE class and initializes it.




:param clipLimit: Threshold for contrast limiting.
:type clipLimit: float
:param tileGridSize: Size of grid for histogram equalization. Input image will be divided intoequally sized rectangular tiles. tileGridSize defines the number of tiles in row and column. 
:type tileGridSize: cv2.typing.Size
:rtype: CLAHE
````


````{py:function} createCalibrateDebevec([, samples[, lambda_[, random]]]) -> retval

Creates CalibrateDebevec object




:param samples: number of pixel locations to use
:type samples: int
:param lambda: smoothness term weight. Greater values produce smoother results, but can alter theresponse. 
:type lambda: 
:param random: if true sample pixel locations are chosen at random, otherwise they form arectangular grid. 
:type random: bool
:param lambda_: 
:type lambda_: float
:rtype: CalibrateDebevec
````


````{py:function} createCalibrateRobertson([, max_iter[, threshold]]) -> retval

Creates CalibrateRobertson object




:param max_iter: maximal number of Gauss-Seidel solver iterations.
:type max_iter: int
:param threshold: target difference between results of two successive steps of the minimization.
:type threshold: float
:rtype: CalibrateRobertson
````


````{py:function} createGeneralizedHoughBallard() -> retval

Creates a smart pointer to a cv::GeneralizedHoughBallard class and initializes it.




:rtype: GeneralizedHoughBallard
````


````{py:function} createGeneralizedHoughGuil() -> retval

Creates a smart pointer to a cv::GeneralizedHoughGuil class and initializes it.




:rtype: GeneralizedHoughGuil
````


````{py:function} createHanningWindow(winSize, type[, dst]) -> dst

This function computes a Hanning window coefficients in two dimensions.


See (http://en.wikipedia.org/wiki/Hann_function) and (http://en.wikipedia.org/wiki/Window_function) for more information. 
An example is shown below: 
```c++
// create hanning window of size 100x100 and type CV_32F
Mat hann;
createHanningWindow(hann, Size(100, 100), CV_32F);
```



:param dst: Destination array to place Hann coefficients in
:type dst: cv2.typing.MatLike | None
:param winSize: The window size specifications (both width and height must be > 1)
:type winSize: cv2.typing.Size
:param type: Created array type
:type type: int
:rtype: cv2.typing.MatLike
````


````{py:function} createLineSegmentDetector([, refine[, scale[, sigma_scale[, quant[, ang_th[, log_eps[, density_th[, n_bins]]]]]]]]) -> retval

Creates a smart pointer to a LineSegmentDetector object and initializes it.


The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want to edit those, as to tailor it for their own application. 


:param refine: The way found lines will be refined, see #LineSegmentDetectorModes
:type refine: int
:param scale: The scale of the image that will be used to find the lines. Range (0..1].
:type scale: float
:param sigma_scale: Sigma for Gaussian filter. It is computed as sigma = sigma_scale/scale.
:type sigma_scale: float
:param quant: Bound to the quantization error on the gradient norm.
:type quant: float
:param ang_th: Gradient angle tolerance in degrees.
:type ang_th: float
:param log_eps: Detection threshold: -log10(NFA) \> log_eps. Used only when advance refinement is chosen.
:type log_eps: float
:param density_th: Minimal density of aligned region points in the enclosing rectangle.
:type density_th: float
:param n_bins: Number of bins in pseudo-ordering of gradient modulus.
:type n_bins: int
:rtype: LineSegmentDetector
````


````{py:function} createMergeDebevec() -> retval

Creates MergeDebevec object




:rtype: MergeDebevec
````


````{py:function} createMergeMertens([, contrast_weight[, saturation_weight[, exposure_weight]]]) -> retval

Creates MergeMertens object




:param contrast_weight: contrast measure weight. See MergeMertens.
:type contrast_weight: float
:param saturation_weight: saturation measure weight
:type saturation_weight: float
:param exposure_weight: well-exposedness measure weight
:type exposure_weight: float
:rtype: MergeMertens
````


````{py:function} createMergeRobertson() -> retval

Creates MergeRobertson object




:rtype: MergeRobertson
````


````{py:function} createTonemap([, gamma]) -> retval

Creates simple linear mapper with gamma correction




:param gamma: positive value for gamma correction. Gamma value of 1.0 implies no correction, gammaequal to 2.2f is suitable for most displays. Generally gamma \> 1 brightens the image and gamma \< 1 darkens it. 
:type gamma: float
:rtype: Tonemap
````


````{py:function} createTonemapDrago([, gamma[, saturation[, bias]]]) -> retval

Creates TonemapDrago object




:param gamma: gamma value for gamma correction. See createTonemap
:type gamma: float
:param saturation: positive saturation enhancement value. 1.0 preserves saturation, values greaterthan 1 increase saturation and values less than 1 decrease it. 
:type saturation: float
:param bias: value for bias function in [0, 1] range. Values from 0.7 to 0.9 usually give bestresults, default value is 0.85. 
:type bias: float
:rtype: TonemapDrago
````


````{py:function} createTonemapMantiuk([, gamma[, scale[, saturation]]]) -> retval

Creates TonemapMantiuk object




:param gamma: gamma value for gamma correction. See createTonemap
:type gamma: float
:param scale: contrast scale factor. HVS response is multiplied by this parameter, thus compressingdynamic range. Values from 0.6 to 0.9 produce best results. 
:type scale: float
:param saturation: saturation enhancement value. See createTonemapDrago
:type saturation: float
:rtype: TonemapMantiuk
````


````{py:function} createTonemapReinhard([, gamma[, intensity[, light_adapt[, color_adapt]]]]) -> retval

Creates TonemapReinhard object




:param gamma: gamma value for gamma correction. See createTonemap
:type gamma: float
:param intensity: result intensity in [-8, 8] range. Greater intensity produces brighter results.
:type intensity: float
:param light_adapt: light adaptation in [0, 1] range. If 1 adaptation is based only on pixelvalue, if 0 it's global, otherwise it's a weighted mean of this two cases. 
:type light_adapt: float
:param color_adapt: chromatic adaptation in [0, 1] range. If 1 channels are treated independently,if 0 adaptation level is the same for each channel. 
:type color_adapt: float
:rtype: TonemapReinhard
````


````{py:function} createTrackbar(trackbarName, windowName, value, count, onChange) -> None






:param trackbarName: 
:type trackbarName: str
:param windowName: 
:type windowName: str
:param value: 
:type value: int
:param count: 
:type count: int
:param onChange: 
:type onChange: _typing.Callable[[int], None]
:rtype: None
````


````{py:function} cubeRoot(val) -> retval

Computes the cube root of an argument.


The function cubeRoot computes $\sqrt[3]{\texttt{val}}$. Negative arguments are handled correctly. NaN and Inf are not handled. The accuracy approaches the maximum possible accuracy for single-precision data. 


:param val: A function argument.
:type val: float
:rtype: float
````


````{py:function} cvtColor(src, code[, dst[, dstCn]]) -> dst

Converts an image from one color space to another.


The function converts an input image from one color space to another. In case of a transformation to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on. 
The conventional ranges for R, G, and B channel values are: -   0 to 255 for CV_8U images -   0 to 65535 for CV_16U images -   0 to 1 for CV_32F images 
In case of linear transformations, the range does not matter. But in case of a non-linear transformation, an input RGB image should be normalized to the proper value range to get the correct results, for example, for RGB $\rightarrow$ L\*u\*v\* transformation. For example, if you have a 32-bit floating-point image directly converted from an 8-bit image without any scaling, then it will have the 0..255 value range instead of 0..1 assumed by the function. So, before calling #cvtColor , you need first to scale the image down: 
```c++
img *= 1./255;
cvtColor(img, img, COLOR_BGR2Luv);
```
If you use #cvtColor with 8-bit images, the conversion will have some information lost. For many applications, this will not be noticeable but it is recommended to use 32-bit images in applications that need the full range of colors or that convert an image before an operation and then convert back. 
If conversion adds the alpha channel, its value will set to the maximum of corresponding channel range: 255 for CV_8U, 65535 for CV_16U, 1 for CV_32F. 
**See also:** @ref imgproc_color_conversions


:param src: input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precisionfloating-point. 
:type src: cv2.typing.MatLike
:param dst: output image of the same size and depth as src.
:type dst: cv2.typing.MatLike | None
:param code: color space conversion code (see #ColorConversionCodes).
:type code: int
:param dstCn: number of channels in the destination image; if the parameter is 0, the number of thechannels is derived automatically from src and code. 
:type dstCn: int
:rtype: cv2.typing.MatLike
````


````{py:function} cvtColorTwoPlane(src1, src2, code[, dst]) -> dst

Converts an image from one color space to another where the source image isstored in two planes. 


This function only supports YUV420 to RGB conversion as of now. 


:param src1: 8-bit image (#CV_8U) of the Y plane.
:type src1: cv2.typing.MatLike
:param src2: image containing interleaved U/V plane.
:type src2: cv2.typing.MatLike
:param dst: output image.
:type dst: cv2.typing.MatLike | None
:param code: Specifies the type of conversion. It can take any of the following values:- #COLOR_YUV2BGR_NV12 - #COLOR_YUV2RGB_NV12 - #COLOR_YUV2BGRA_NV12 - #COLOR_YUV2RGBA_NV12 - #COLOR_YUV2BGR_NV21 - #COLOR_YUV2RGB_NV21 - #COLOR_YUV2BGRA_NV21 - #COLOR_YUV2RGBA_NV21 
:type code: int
:rtype: cv2.typing.MatLike
````


````{py:function} dct(src[, dst[, flags]]) -> dst

Performs a forward or inverse discrete Cosine transform of 1D or 2D array.


The function cv::dct performs a forward or inverse discrete Cosine transform (DCT) of a 1D or 2D floating-point array: -   Forward Cosine transform of a 1D vector of N elements: $Y = C^{(N)}  \cdot X$ where $C^{(N)}_{jk}= \sqrt{\alpha_j/N} \cos \left ( \frac{\pi(2k+1)j}{2N} \right )$ and $\alpha_0=1$, $\alpha_j=2$ for *j \> 0*. -   Inverse Cosine transform of a 1D vector of N elements: $X =  \left (C^{(N)} \right )^{-1}  \cdot Y =  \left (C^{(N)} \right )^T  \cdot Y$ (since $C^{(N)}$ is an orthogonal matrix, $C^{(N)} \cdot \left(C^{(N)}\right)^T = I$ ) -   Forward 2D Cosine transform of M x N matrix: $Y = C^{(N)}  \cdot X  \cdot \left (C^{(N)} \right )^T$ -   Inverse 2D Cosine transform of M x N matrix: $X =  \left (C^{(N)} \right )^T  \cdot X  \cdot C^{(N)}$ 
The function chooses the mode of operation by looking at the flags and size of the input array: -   If (flags & #DCT_INVERSE) == 0 , the function does a forward 1D or 2D transform. Otherwise, it is an inverse 1D or 2D transform. -   If (flags & #DCT_ROWS) != 0 , the function performs a 1D transform of each row. -   If the array is a single column or a single row, the function performs a 1D transform. -   If none of the above is true, the function performs a 2D transform. 
```{note}
Currently dct supports even-size arrays (2, 4, 6 ...). For data analysis and approximation, youcan pad the array when necessary. Also, the function performance depends very much, and not monotonically, on the array size (see getOptimalDFTSize ). In the current implementation DCT of a vector of size N is calculated via DFT of a vector of size N/2 . Thus, the optimal DCT size N1 \>= N can be calculated as: 
```c++
size_t getOptimalDCTSize(size_t N) { return 2*getOptimalDFTSize((N+1)/2); }
N1 = getOptimalDCTSize(N);
```

```
**See also:** dft , getOptimalDFTSize , idct


:param src: input floating-point array.
:type src: cv2.typing.MatLike
:param dst: output array of the same size and type as src .
:type dst: cv2.typing.MatLike | None
:param flags: transformation flags as a combination of cv::DftFlags (DCT_*)
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} decolor(src[, grayscale[, color_boost]]) -> grayscale, color_boost

Transforms a color image to a grayscale image. It is a basic tool in digital printing, stylizedblack-and-white photograph rendering, and in many single channel image processing applications @cite CL12 . 


This function is to be applied on color images. 


:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param grayscale: Output 8-bit 1-channel image.
:type grayscale: cv2.typing.MatLike | None
:param color_boost: Output 8-bit 3-channel image.
:type color_boost: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} decomposeEssentialMat(E[, R1[, R2[, t]]]) -> R1, R2, t

Decompose an essential matrix to possible rotations and translation.


This function decomposes the essential matrix E using svd decomposition @cite HartleyZ00. In general, four possible poses exist for the decomposition of E. They are $[R_1, t]$, $[R_1, -t]$, $[R_2, t]$, $[R_2, -t]$. 
If E gives the epipolar constraint $[p_2; 1]^T A^{-T} E A^{-1} [p_1; 1] = 0$ between the image points $p_1$ in the first image and $p_2$ in second image, then any of the tuples $[R_1, t]$, $[R_1, -t]$, $[R_2, t]$, $[R_2, -t]$ is a change of basis from the first camera's coordinate system to the second camera's coordinate system. However, by decomposing E, one can only get the direction of the translation. For this reason, the translation t is returned with unit length. 


:param E: The input essential matrix.
:type E: cv2.typing.MatLike
:param R1: One possible rotation matrix.
:type R1: cv2.typing.MatLike | None
:param R2: Another possible rotation matrix.
:type R2: cv2.typing.MatLike | None
:param t: One possible translation.
:type t: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} decomposeHomographyMat(H, K[, rotations[, translations[, normals]]]) -> retval, rotations, translations, normals

Decompose a homography matrix to rotation(s), translation(s) and plane normal(s).


This function extracts relative camera motion between two views of a planar object and returns up to four mathematical solution tuples of rotation, translation, and plane normal. The decomposition of the homography matrix H is described in detail in @cite Malis2007. 
If the homography H, induced by the plane, gives the constraint $s_i \vecthree{x'_i}{y'_i}{1} \sim H \vecthree{x_i}{y_i}{1}$ on the source image points $p_i$ and the destination image points $p'_i$, then the tuple of rotations[k] and translations[k] is a change of basis from the source camera's coordinate system to the destination camera's coordinate system. However, by decomposing H, one can only get the translation normalized by the (typically unknown) depth of the scene, i.e. its direction but with normalized length. 
If point correspondences are available, at least two solutions may further be invalidated, by applying positive depth constraint, i.e. all points must be in front of the camera. 


:param H: The input homography matrix between two images.
:type H: cv2.typing.MatLike
:param K: The input camera intrinsic matrix.
:type K: cv2.typing.MatLike
:param rotations: Array of rotation matrices.
:type rotations: _typing.Sequence[cv2.typing.MatLike] | None
:param translations: Array of translation matrices.
:type translations: _typing.Sequence[cv2.typing.MatLike] | None
:param normals: Array of plane normal matrices.
:type normals: _typing.Sequence[cv2.typing.MatLike] | None
:rtype: tuple[int, _typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike]]
````


````{py:function} decomposeProjectionMatrix(projMatrix[, cameraMatrix[, rotMatrix[, transVect[, rotMatrixX[, rotMatrixY[, rotMatrixZ[, eulerAngles]]]]]]]) -> cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles

Decomposes a projection matrix into a rotation matrix and a camera intrinsic matrix.


The function computes a decomposition of a projection matrix into a calibration and a rotation matrix and the position of a camera. 
It optionally returns three rotation matrices, one for each axis, and three Euler angles that could be used in OpenGL. Note, there is always more than one sequence of rotations about the three principal axes that results in the same orientation of an object, e.g. see @cite Slabaugh . Returned three rotation matrices and corresponding three Euler angles are only one of the possible solutions. 
The function is based on #RQDecomp3x3 . 


:param projMatrix: 3x4 input projection matrix P.
:type projMatrix: cv2.typing.MatLike
:param cameraMatrix: Output 3x3 camera intrinsic matrix $\cameramatrix{A}$.
:type cameraMatrix: cv2.typing.MatLike | None
:param rotMatrix: Output 3x3 external rotation matrix R.
:type rotMatrix: cv2.typing.MatLike | None
:param transVect: Output 4x1 translation vector T.
:type transVect: cv2.typing.MatLike | None
:param rotMatrixX: Optional 3x3 rotation matrix around x-axis.
:type rotMatrixX: cv2.typing.MatLike | None
:param rotMatrixY: Optional 3x3 rotation matrix around y-axis.
:type rotMatrixY: cv2.typing.MatLike | None
:param rotMatrixZ: Optional 3x3 rotation matrix around z-axis.
:type rotMatrixZ: cv2.typing.MatLike | None
:param eulerAngles: Optional three-element vector containing three Euler angles of rotation indegrees. 
:type eulerAngles: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} demosaicing(src, code[, dst[, dstCn]]) -> dst

main function for all demosaicing processes


The function can do the following transformations: 
-   Demosaicing using bilinear interpolation 
#COLOR_BayerBG2BGR , #COLOR_BayerGB2BGR , #COLOR_BayerRG2BGR , #COLOR_BayerGR2BGR 
#COLOR_BayerBG2GRAY , #COLOR_BayerGB2GRAY , #COLOR_BayerRG2GRAY , #COLOR_BayerGR2GRAY 
-   Demosaicing using Variable Number of Gradients. 
#COLOR_BayerBG2BGR_VNG , #COLOR_BayerGB2BGR_VNG , #COLOR_BayerRG2BGR_VNG , #COLOR_BayerGR2BGR_VNG 
-   Edge-Aware Demosaicing. 
#COLOR_BayerBG2BGR_EA , #COLOR_BayerGB2BGR_EA , #COLOR_BayerRG2BGR_EA , #COLOR_BayerGR2BGR_EA 
-   Demosaicing with alpha channel 
#COLOR_BayerBG2BGRA , #COLOR_BayerGB2BGRA , #COLOR_BayerRG2BGRA , #COLOR_BayerGR2BGRA 
**See also:** cvtColor


:param src: input image: 8-bit unsigned or 16-bit unsigned.
:type src: cv2.typing.MatLike
:param dst: output image of the same size and depth as src.
:type dst: cv2.typing.MatLike | None
:param code: Color space conversion code (see the description below).
:type code: int
:param dstCn: number of channels in the destination image; if the parameter is 0, the number of thechannels is derived automatically from src and code. 
:type dstCn: int
:rtype: cv2.typing.MatLike
````


````{py:function} denoise_TVL1(observations, result[, lambda_[, niters]]) -> None

Primal-dual algorithm is an algorithm for solving special types of variational problems (that is,finding a function to minimize some functional). As the image denoising, in particular, may be seen as the variational problem, primal-dual algorithm then can be used to perform denoising and this is exactly what is implemented. 


It should be noted, that this implementation was taken from the July 2013 blog entry @cite MA13 , which also contained (slightly more general) ready-to-use source code on Python. Subsequently, that code was rewritten on C++ with the usage of openCV by Vadim Pisarevsky at the end of July 2013 and finally it was slightly adapted by later authors. 
Although the thorough discussion and justification of the algorithm involved may be found in @cite ChambolleEtAl, it might make sense to skim over it here, following @cite MA13 . To begin with, we consider the 1-byte gray-level images as the functions from the rectangular domain of pixels (it may be seen as set $\left\{(x,y)\in\mathbb{N}\times\mathbb{N}\mid 1\leq x\leq n,\;1\leq y\leq m\right\}$ for some $m,\;n\in\mathbb{N}$) into $\{0,1,\dots,255\}$. We shall denote the noised images as $f_i$ and with this view, given some image $x$ of the same size, we may measure how bad it is by the formula 
$\left\|\left\|\nabla x\right\|\right\| + \lambda\sum_i\left\|\left\|x-f_i\right\|\right\|$ 
$\|\|\cdot\|\|$ here denotes $L_2$-norm and as you see, the first addend states that we want our image to be smooth (ideally, having zero gradient, thus being constant) and the second states that we want our result to be close to the observations we've got. If we treat $x$ as a function, this is exactly the functional what we seek to minimize and here the Primal-Dual algorithm comes into play. 


:param observations: This array should contain one or more noised versions of the image that is tobe restored. 
:type observations: _typing.Sequence[cv2.typing.MatLike]
:param result: Here the denoised image will be stored. There is no need to do pre-allocation ofstorage space, as it will be automatically allocated, if necessary. 
:type result: cv2.typing.MatLike
:param lambda: Corresponds to $\lambda$ in the formulas above. As it is enlarged, the smooth(blurred) images are treated more favorably than detailed (but maybe more noised) ones. Roughly speaking, as it becomes smaller, the result will be more blur but more sever outliers will be removed. 
:type lambda: 
:param niters: Number of iterations that the algorithm will run. Of course, as more iterations asbetter, but it is hard to quantitatively refine this statement, so just use the default and increase it if the results are poor. 
:type niters: int
:param lambda_: 
:type lambda_: float
:rtype: None
````


````{py:function} destroyAllWindows() -> None

Destroys all of the HighGUI windows.


The function destroyAllWindows destroys all of the opened HighGUI windows. 


:rtype: None
````


````{py:function} destroyWindow(winname) -> None

Destroys the specified window.


The function destroyWindow destroys the window with the given name. 


:param winname: Name of the window to be destroyed.
:type winname: str
:rtype: None
````


````{py:function} detailEnhance(src[, dst[, sigma_s[, sigma_r]]]) -> dst

This filter enhances the details of a particular image.




:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param dst: Output image with the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param sigma_s: %Range between 0 to 200.
:type sigma_s: float
:param sigma_r: %Range between 0 to 1.
:type sigma_r: float
:rtype: cv2.typing.MatLike
````


````{py:function} determinant(mtx) -> retval

Returns the determinant of a square floating-point matrix.


The function cv::determinant calculates and returns the determinant of the specified matrix. For small matrices ( mtx.cols=mtx.rows\<=3 ), the direct method is used. For larger matrices, the function uses LU factorization with partial pivoting. 
For symmetric positively-determined matrices, it is also possible to use eigen decomposition to calculate the determinant. 
**See also:** trace, invert, solve, eigen, @ref MatrixExpressions


:param mtx: input matrix that must have CV_32FC1 or CV_64FC1 type andsquare size. 
:type mtx: cv2.typing.MatLike
:rtype: float
````


````{py:function} dft(src[, dst[, flags[, nonzeroRows]]]) -> dst

Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.


The function cv::dft performs one of the following: -   Forward the Fourier transform of a 1D vector of N elements: $Y = F^{(N)}  \cdot X,$ where $F^{(N)}_{jk}=\exp(-2\pi i j k/N)$ and $i=\sqrt{-1}$ -   Inverse the Fourier transform of a 1D vector of N elements: $\begin{array}{l} X'=  \left (F^{(N)} \right )^{-1}  \cdot Y =  \left (F^{(N)} \right )^*  \cdot y  \\ X = (1/N)  \cdot X, \end{array}$ where $F^*=\left(\textrm{Re}(F^{(N)})-\textrm{Im}(F^{(N)})\right)^T$ -   Forward the 2D Fourier transform of a M x N matrix: $Y = F^{(M)}  \cdot X  \cdot F^{(N)}$ -   Inverse the 2D Fourier transform of a M x N matrix: $\begin{array}{l} X'=  \left (F^{(M)} \right )^*  \cdot Y  \cdot \left (F^{(N)} \right )^* \\ X =  \frac{1}{M \cdot N} \cdot X' \end{array}$ 
In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input spectrum of the inverse Fourier transform can be represented in a packed format called *CCS* (complex-conjugate-symmetrical). It was borrowed from IPL (Intel\* Image Processing Library). Here is how 2D *CCS* spectrum looks: $\begin{bmatrix} Re Y_{0,0} & Re Y_{0,1} & Im Y_{0,1} & Re Y_{0,2} & Im Y_{0,2} &  \cdots & Re Y_{0,N/2-1} & Im Y_{0,N/2-1} & Re Y_{0,N/2}  \\ Re Y_{1,0} & Re Y_{1,1} & Im Y_{1,1} & Re Y_{1,2} & Im Y_{1,2} &  \cdots & Re Y_{1,N/2-1} & Im Y_{1,N/2-1} & Re Y_{1,N/2}  \\ Im Y_{1,0} & Re Y_{2,1} & Im Y_{2,1} & Re Y_{2,2} & Im Y_{2,2} &  \cdots & Re Y_{2,N/2-1} & Im Y_{2,N/2-1} & Im Y_{1,N/2}  \\ \hdotsfor{9} \\ Re Y_{M/2-1,0} &  Re Y_{M-3,1}  & Im Y_{M-3,1} &  \hdotsfor{3} & Re Y_{M-3,N/2-1} & Im Y_{M-3,N/2-1}& Re Y_{M/2-1,N/2}  \\ Im Y_{M/2-1,0} &  Re Y_{M-2,1}  & Im Y_{M-2,1} &  \hdotsfor{3} & Re Y_{M-2,N/2-1} & Im Y_{M-2,N/2-1}& Im Y_{M/2-1,N/2}  \\ Re Y_{M/2,0}  &  Re Y_{M-1,1} &  Im Y_{M-1,1} &  \hdotsfor{3} & Re Y_{M-1,N/2-1} & Im Y_{M-1,N/2-1}& Re Y_{M/2,N/2} \end{bmatrix}$ 
In case of 1D transform of a real vector, the output looks like the first row of the matrix above. 
So, the function chooses an operation mode depending on the flags and size of the input array: -   If #DFT_ROWS is set or the input array has a single row or single column, the function performs a 1D forward or inverse transform of each row of a matrix when #DFT_ROWS is set. Otherwise, it performs a 2D transform. -   If the input array is real and #DFT_INVERSE is not set, the function performs a forward 1D or 2D transform: -   When #DFT_COMPLEX_OUTPUT is set, the output is a complex matrix of the same size as input. -   When #DFT_COMPLEX_OUTPUT is not set, the output is a real matrix of the same size as input. In case of 2D transform, it uses the packed format as shown above. In case of a single 1D transform, it looks like the first row of the matrix above. In case of multiple 1D transforms (when using the #DFT_ROWS flag), each row of the output matrix looks like the first row of the matrix above. -   If the input array is complex and either #DFT_INVERSE or #DFT_REAL_OUTPUT are not set, the output is a complex array of the same size as input. The function performs a forward or inverse 1D or 2D transform of the whole input array or each row of the input array independently, depending on the flags DFT_INVERSE and DFT_ROWS. -   When #DFT_INVERSE is set and the input array is real, or it is complex but #DFT_REAL_OUTPUT is set, the output is a real array of the same size as input. The function performs a 1D or 2D inverse transformation of the whole input array or each individual row, depending on the flags #DFT_INVERSE and #DFT_ROWS. 
If #DFT_SCALE is set, the scaling is done after the transformation. 
Unlike dct , the function supports arrays of arbitrary size. But only those arrays are processed efficiently, whose sizes can be factorized in a product of small prime numbers (2, 3, and 5 in the current implementation). Such an efficient DFT size can be calculated using the getOptimalDFTSize method. 
The sample below illustrates how to calculate a DFT-based convolution of two 2D real arrays: 
```c++
void convolveDFT(InputArray A, InputArray B, OutputArray C)
{
// reallocate the output array if needed
C.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());
Size dftSize;
// calculate the size of DFT transform
dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

// allocate temporary buffers and initialize them with 0's
Mat tempA(dftSize, A.type(), Scalar::all(0));
Mat tempB(dftSize, B.type(), Scalar::all(0));

// copy A and B to the top-left corners of tempA and tempB, respectively
Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
A.copyTo(roiA);
Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
B.copyTo(roiB);

// now transform the padded A & B in-place;
// use "nonzeroRows" hint for faster processing
dft(tempA, tempA, 0, A.rows);
dft(tempB, tempB, 0, B.rows);

// multiply the spectrums;
// the function handles packed spectrum representations well
mulSpectrums(tempA, tempB, tempA);

// transform the product back from the frequency domain.
// Even though all the result rows will be non-zero,
// you need only the first C.rows of them, and thus you
// pass nonzeroRows == C.rows
dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

// now copy the result back to C.
tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);

// all the temporary buffers will be deallocated automatically
}
```
To optimize this sample, consider the following approaches: -   Since nonzeroRows != 0 is passed to the forward transform calls and since A and B are copied to the top-left corners of tempA and tempB, respectively, it is not necessary to clear the whole tempA and tempB. It is only necessary to clear the tempA.cols - A.cols ( tempB.cols - B.cols) rightmost columns of the matrices. -   This DFT-based convolution does not have to be applied to the whole big arrays, especially if B is significantly smaller than A or vice versa. Instead, you can calculate convolution by parts. To do this, you need to split the output array C into multiple tiles. For each tile, estimate which parts of A and B are required to calculate convolution in this tile. If the tiles in C are too small, the speed will decrease a lot because of repeated work. In the ultimate case, when each tile in C is a single pixel, the algorithm becomes equivalent to the naive convolution algorithm. If the tiles are too big, the temporary arrays tempA and tempB become too big and there is also a slowdown because of bad cache locality. So, there is an optimal tile size somewhere in the middle. -   If different tiles in C can be calculated in parallel and, thus, the convolution is done by parts, the loop can be threaded. 
All of the above improvements have been implemented in #matchTemplate and #filter2D . Therefore, by using them, you can get the performance even better than with the above theoretically optimal implementation. Though, those two functions actually calculate cross-correlation, not convolution, so you need to "flip" the second convolution operand B vertically and horizontally using flip . @note -   An example using the discrete fourier transform can be found at opencv_source_code/samples/cpp/dft.cpp -   (Python) An example using the dft functionality to perform Wiener deconvolution can be found at opencv_source/samples/python/deconvolution.py -   (Python) An example rearranging the quadrants of a Fourier image can be found at opencv_source/samples/python/dft.py 
**See also:** dct , getOptimalDFTSize , mulSpectrums, filter2D , matchTemplate , flip , cartToPolar ,magnitude , phase 


:param src: input array that could be real or complex.
:type src: cv2.typing.MatLike
:param dst: output array whose size and type depends on the flags .
:type dst: cv2.typing.MatLike | None
:param flags: transformation flags, representing a combination of the #DftFlags
:type flags: int
:param nonzeroRows: when the parameter is not zero, the function assumes that only the firstnonzeroRows rows of the input array (#DFT_INVERSE is not set) or only the first nonzeroRows of the output array (#DFT_INVERSE is set) contain non-zeros, thus, the function can handle the rest of the rows more efficiently and save some time; this technique is very useful for calculating array cross-correlation or convolution using DFT. 
:type nonzeroRows: int
:rtype: cv2.typing.MatLike
````


````{py:function} dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst

Dilates an image by using a specific structuring element.


The function dilates the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the maximum is taken: $\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')$ 
The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In case of multi-channel images, each channel is processed independently. 
**See also:**  erode, morphologyEx, getStructuringElement


:param src: input image; the number of channels can be arbitrary, but the depth should be one ofCV_8U, CV_16U, CV_16S, CV_32F or CV_64F. 
:type src: cv2.typing.MatLike
:param dst: output image of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param kernel: structuring element used for dilation; if element=Mat(), a 3 x 3 rectangularstructuring element is used. Kernel can be created using #getStructuringElement 
:type kernel: cv2.typing.MatLike
:param anchor: position of the anchor within the element; default value (-1, -1) means that theanchor is at the element center. 
:type anchor: cv2.typing.Point
:param iterations: number of times dilation is applied.
:type iterations: int
:param borderType: pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not suported.
:type borderType: int
:param borderValue: border value in case of a constant border
:type borderValue: cv2.typing.Scalar
:rtype: cv2.typing.MatLike
````


````{py:function} displayOverlay(winname, text[, delayms]) -> None

Displays a text on a window image as an overlay for a specified duration.


The function displayOverlay displays useful information/tips on top of the window for a certain amount of time *delayms*. The function does not modify the image, displayed in the window, that is, after the specified delay the original content of the window is restored. 


:param winname: Name of the window.
:type winname: str
:param text: Overlay text to write on a window image.
:type text: str
:param delayms: The period (in milliseconds), during which the overlay text is displayed. If thisfunction is called before the previous overlay text timed out, the timer is restarted and the text is updated. If this value is zero, the text never disappears. 
:type delayms: int
:rtype: None
````


````{py:function} displayStatusBar(winname, text[, delayms]) -> None

Displays a text on the window statusbar during the specified period of time.


The function displayStatusBar displays useful information/tips on top of the window for a certain amount of time *delayms* . This information is displayed on the window statusbar (the window must be created with the CV_GUI_EXPANDED flags). 


:param winname: Name of the window.
:type winname: str
:param text: Text to write on the window statusbar.
:type text: str
:param delayms: Duration (in milliseconds) to display the text. If this function is called beforethe previous text timed out, the timer is restarted and the text is updated. If this value is zero, the text never disappears. 
:type delayms: int
:rtype: None
````


````{py:function} distanceTransform(src, distanceType, maskSize[, dst[, dstType]]) -> dst




@overload 


:param src: 8-bit, single-channel (binary) source image.
:type src: cv2.typing.MatLike
:param dst: Output image with calculated distances. It is a 8-bit or 32-bit floating-point,single-channel image of the same size as src . 
:type dst: cv2.typing.MatLike | None
:param distanceType: Type of distance, see #DistanceTypes
:type distanceType: int
:param maskSize: Size of the distance transform mask, see #DistanceTransformMasks. In case of the#DIST_L1 or #DIST_C distance type, the parameter is forced to 3 because a $3\times 3$ mask gives the same result as $5\times 5$ or any larger aperture. 
:type maskSize: int
:param dstType: Type of output image. It can be CV_8U or CV_32F. Type CV_8U can be used only forthe first variant of the function and distanceType == #DIST_L1. 
:type dstType: int
:rtype: cv2.typing.MatLike
````


````{py:function} distanceTransformWithLabels(src, distanceType, maskSize[, dst[, labels[, labelType]]]) -> dst, labels

Calculates the distance to the closest zero pixel for each pixel of the source image.


The function cv::distanceTransform calculates the approximate or precise distance from every binary image pixel to the nearest zero pixel. For zero image pixels, the distance will obviously be zero. 
When maskSize == #DIST_MASK_PRECISE and distanceType == #DIST_L2 , the function runs the algorithm described in @cite Felzenszwalb04 . This algorithm is parallelized with the TBB library. 
In other cases, the algorithm @cite Borgefors86 is used. This means that for a pixel the function finds the shortest path to the nearest zero pixel consisting of basic shifts: horizontal, vertical, diagonal, or knight's move (the latest is available for a $5\times 5$ mask). The overall distance is calculated as a sum of these basic distances. Since the distance function should be symmetric, all of the horizontal and vertical shifts must have the same cost (denoted as a ), all the diagonal shifts must have the same cost (denoted as `b`), and all knight's moves must have the same cost (denoted as `c`). For the #DIST_C and #DIST_L1 types, the distance is calculated precisely, whereas for #DIST_L2 (Euclidean distance) the distance can be calculated only with a relative error (a $5\times 5$ mask gives more accurate results). For `a`,`b`, and `c`, OpenCV uses the values suggested in the original paper: - DIST_L1: `a = 1, b = 2` - DIST_L2: - `3 x 3`: `a=0.955, b=1.3693` - `5 x 5`: `a=1, b=1.4, c=2.1969` - DIST_C: `a = 1, b = 1` 
Typically, for a fast, coarse distance estimation #DIST_L2, a $3\times 3$ mask is used. For a more accurate distance estimation #DIST_L2, a $5\times 5$ mask or the precise algorithm is used. Note that both the precise and the approximate algorithms are linear on the number of pixels. 
This variant of the function does not only compute the minimum distance for each pixel $(x, y)$ but also identifies the nearest connected component consisting of zero pixels (labelType==#DIST_LABEL_CCOMP) or the nearest zero pixel (labelType==#DIST_LABEL_PIXEL). Index of the component/pixel is stored in `labels(x, y)`. When labelType==#DIST_LABEL_CCOMP, the function automatically finds connected components of zero pixels in the input image and marks them with distinct labels. When labelType==#DIST_LABEL_PIXEL, the function scans through the input image and marks all the zero pixels with distinct labels. 
In this mode, the complexity is still linear. That is, the function provides a very fast way to compute the Voronoi diagram for a binary image. Currently, the second variant can use only the approximate distance transform algorithm, i.e. maskSize=#DIST_MASK_PRECISE is not supported yet. 


:param src: 8-bit, single-channel (binary) source image.
:type src: cv2.typing.MatLike
:param dst: Output image with calculated distances. It is a 8-bit or 32-bit floating-point,single-channel image of the same size as src. 
:type dst: cv2.typing.MatLike | None
:param labels: Output 2D array of labels (the discrete Voronoi diagram). It has the typeCV_32SC1 and the same size as src. 
:type labels: cv2.typing.MatLike | None
:param distanceType: Type of distance, see #DistanceTypes
:type distanceType: int
:param maskSize: Size of the distance transform mask, see #DistanceTransformMasks.#DIST_MASK_PRECISE is not supported by this variant. In case of the #DIST_L1 or #DIST_C distance type, the parameter is forced to 3 because a $3\times 3$ mask gives the same result as $5\times 5$ or any larger aperture. 
:type maskSize: int
:param labelType: Type of the label array to build, see #DistanceTransformLabelTypes.
:type labelType: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} divSpectrums(a, b, flags[, c[, conjB]]) -> c

Performs the per-element division of the first Fourier spectrum by the second Fourier spectrum.


The function cv::divSpectrums performs the per-element division of the first array by the second array. The arrays are CCS-packed or complex matrices that are results of a real or complex Fourier transform. 


:param a: first input array.
:type a: cv2.typing.MatLike
:param b: second input array of the same size and type as src1 .
:type b: cv2.typing.MatLike
:param c: output array of the same size and type as src1 .
:type c: cv2.typing.MatLike | None
:param flags: operation flags; currently, the only supported flag is cv::DFT_ROWS, which indicates thateach row of src1 and src2 is an independent 1D Fourier spectrum. If you do not want to use this flag, then simply add a `0` as value. 
:type flags: int
:param conjB: optional flag that conjugates the second input array before the multiplication (true)or not (false). 
:type conjB: bool
:rtype: cv2.typing.MatLike
````


````{py:function} divide(src1, src2[, dst[, scale[, dtype]]]) -> dst

Performs per-element division of two arrays or a scalar by an array.


The function cv::divide divides one array by another: $\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}$ or a scalar by an array when there is no src1 : $\texttt{dst(I) = saturate(scale/src2(I))}$ 
Different channels of multi-channel arrays are processed independently. 
For integer types when src2(I) is zero, dst(I) will also be zero. 
divide(scale, src2[, dst[, dtype]]) -> dst @overload 
```{note}
In case of floating point data there is no special defined behavior for zero src2(I) values.Regular floating-point division is used. Expect correct IEEE-754 behaviour for floating-point data (with NaN, Inf result values). 
```
```{note}
Saturation is not applied when the output array has the depth CV_32S. You may even getresult of an incorrect sign in the case of overflow. 
```
```{note}
(Python) Be careful to difference behaviour between src1/src2 are single number and they are tuple/array.`divide(src,X)` means `divide(src,(X,X,X,X))`. `divide(src,(X,))` means `divide(src,(X,0,0,0))`. 
```
**See also:**  multiply, add, subtract


:param src1: first input array.
:type src1: cv2.typing.MatLike
:param src2: second input array of the same size and type as src1.
:type src2: cv2.typing.MatLike
:param scale: scalar factor.
:type scale: float
:param dst: output array of the same size and type as src2.
:type dst: cv2.typing.MatLike | None
:param dtype: optional depth of the output array; if -1, dst will have depth src2.depth(), but incase of an array-by-array division, you can only pass -1 when src1.depth()==src2.depth(). 
:type dtype: int
:rtype: cv2.typing.MatLike
````


````{py:function} registerLayer(type, class) -> None






:param layerTypeName: 
:type layerTypeName: str
:param layerClass: 
:type layerClass: _typing.Type[cv2.dnn.LayerProtocol]
:rtype: None
````


````{py:function} unregisterLayer(type) -> None






:param layerTypeName: 
:type layerTypeName: str
:rtype: None
````


````{py:function} drawChessboardCorners(image, patternSize, corners, patternWasFound) -> image

Renders the detected chessboard corners.


The function draws individual chessboard corners detected either as red circles if the board was not found, or as colored corners connected with lines if the board was found. 


:param image: Destination image. It must be an 8-bit color image.
:type image: cv2.typing.MatLike
:param patternSize: Number of inner corners per a chessboard row and column(patternSize = cv::Size(points_per_row,points_per_column)). 
:type patternSize: cv2.typing.Size
:param corners: Array of detected corners, the output of #findChessboardCorners.
:type corners: cv2.typing.MatLike
:param patternWasFound: Parameter indicating whether the complete board was found or not. Thereturn value of #findChessboardCorners should be passed here. 
:type patternWasFound: bool
:rtype: cv2.typing.MatLike
````


````{py:function} drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image

Draws contours outlines or filled contours.


The function draws contour outlines in the image if $\texttt{thickness} \ge 0$ or fills the area bounded by the contours if $\texttt{thickness}<0$ . The example below shows how to retrieve connected components from the binary image and label them: : @include snippets/imgproc_drawContours.cpp 
```{note}
When thickness=#FILLED, the function is designed to handle connected components with holes correctlyeven when no hierarchy data is provided. This is done by analyzing all the outlines together using even-odd rule. This may give incorrect results if you have a joint collection of separately retrieved contours. In order to solve this problem, you need to call #drawContours separately for each sub-group of contours, or iterate over the collection using contourIdx parameter. 
```


:param image: Destination image.
:type image: cv2.typing.MatLike
:param contours: All the input contours. Each contour is stored as a point vector.
:type contours: _typing.Sequence[cv2.typing.MatLike]
:param contourIdx: Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
:type contourIdx: int
:param color: Color of the contours.
:type color: cv2.typing.Scalar
:param thickness: Thickness of lines the contours are drawn with. If it is negative (for example,thickness=#FILLED ), the contour interiors are drawn. 
:type thickness: int
:param lineType: Line connectivity. See #LineTypes
:type lineType: int
:param hierarchy: Optional information about hierarchy. It is only needed if you want to draw onlysome of the contours (see maxLevel ). 
:type hierarchy: cv2.typing.MatLike | None
:param maxLevel: Maximal level for drawn contours. If it is 0, only the specified contour is drawn.If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This parameter is only taken into account when there is hierarchy available. 
:type maxLevel: int
:param offset: Optional contour shift parameter. Shift all the drawn contours by the specified$\texttt{offset}=(dx,dy)$ . 
:type offset: cv2.typing.Point
:rtype: cv2.typing.MatLike
````


````{py:function} drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, length[, thickness]) -> image

Draw axes of the world/object coordinate system from pose estimation. @sa solvePnP


This function draws the axes of the world/object coordinate system w.r.t. to the camera frame. OX is drawn in red, OY in green and OZ in blue. 


:param image: Input/output image. It must have 1 or 3 channels. The number of channels is not altered.
:type image: cv2.typing.MatLike
:param cameraMatrix: Input 3x3 floating-point matrix of camera intrinsic parameters.$\cameramatrix{A}$ 
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$\distcoeffs$. If the vector is empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param rvec: Rotation vector (see @ref Rodrigues ) that, together with tvec, brings points fromthe model coordinate system to the camera coordinate system. 
:type rvec: cv2.typing.MatLike
:param tvec: Translation vector.
:type tvec: cv2.typing.MatLike
:param length: Length of the painted axes in the same unit than tvec (usually in meters).
:type length: float
:param thickness: Line thickness of the painted axes.
:type thickness: int
:rtype: cv2.typing.MatLike
````


````{py:function} drawKeypoints(image, keypoints, outImage[, color[, flags]]) -> outImage

Draws keypoints.


@note For Python API, flags are modified as cv.DRAW_MATCHES_FLAGS_DEFAULT, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG, cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS 


:param image: Source image.
:type image: cv2.typing.MatLike
:param keypoints: Keypoints from the source image.
:type keypoints: _typing.Sequence[KeyPoint]
:param outImage: Output image. Its content depends on the flags value defining what is drawn in theoutput image. See possible flags bit values below. 
:type outImage: cv2.typing.MatLike
:param color: Color of keypoints.
:type color: cv2.typing.Scalar
:param flags: Flags setting drawing features. Possible flags bit values are defined byDrawMatchesFlags. See details above in drawMatches . 
:type flags: DrawMatchesFlags
:rtype: cv2.typing.MatLike
````


````{py:function} drawMarker(img, position, color[, markerType[, markerSize[, thickness[, line_type]]]]) -> img

Draws a marker on a predefined position in an image.


The function cv::drawMarker draws a marker on a given position in the image. For the moment several marker types are supported, see #MarkerTypes for more information. 


:param img: Image.
:type img: cv2.typing.MatLike
:param position: The point where the crosshair is positioned.
:type position: cv2.typing.Point
:param color: Line color.
:type color: cv2.typing.Scalar
:param markerType: The specific type of marker you want to use, see #MarkerTypes
:type markerType: int
:param thickness: Line thickness.
:type thickness: int
:param line_type: Type of the line, See #LineTypes
:type line_type: int
:param markerSize: The length of the marker axis [default = 20 pixels]
:type markerSize: int
:rtype: cv2.typing.MatLike
````


````{py:function} drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]) -> outImg

Draws the found matches of keypoints from two images.


This function draws matches of keypoints from two images in the output image. Match is a line connecting two keypoints (circles). See cv::DrawMatchesFlags. 
drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchesThickness[, matchColor[, singlePointColor[, matchesMask[, flags]]]]) -> outImg @overload 


:param img1: First source image.
:type img1: cv2.typing.MatLike
:param keypoints1: Keypoints from the first source image.
:type keypoints1: _typing.Sequence[KeyPoint]
:param img2: Second source image.
:type img2: cv2.typing.MatLike
:param keypoints2: Keypoints from the second source image.
:type keypoints2: _typing.Sequence[KeyPoint]
:param matches1to2: Matches from the first image to the second one, which means that keypoints1[i]has a corresponding point in keypoints2[matches[i]] . 
:type matches1to2: _typing.Sequence[DMatch]
:param outImg: Output image. Its content depends on the flags value defining what is drawn in theoutput image. See possible flags bit values below. 
:type outImg: cv2.typing.MatLike
:param matchColor: Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1), the color is generated randomly. 
:type matchColor: cv2.typing.Scalar
:param singlePointColor: Color of single keypoints (circles), which means that keypoints do nothave the matches. If singlePointColor==Scalar::all(-1) , the color is generated randomly. 
:type singlePointColor: cv2.typing.Scalar
:param matchesMask: Mask determining which matches are drawn. If the mask is empty, all matches aredrawn. 
:type matchesMask: _typing.Sequence[str]
:param flags: Flags setting drawing features. Possible flags bit values are defined byDrawMatchesFlags. 
:type flags: DrawMatchesFlags
:rtype: cv2.typing.MatLike
````


````{py:function} drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]) -> outImg






:param img1: 
:type img1: cv2.typing.MatLike
:param keypoints1: 
:type keypoints1: _typing.Sequence[KeyPoint]
:param img2: 
:type img2: cv2.typing.MatLike
:param keypoints2: 
:type keypoints2: _typing.Sequence[KeyPoint]
:param matches1to2: 
:type matches1to2: _typing.Sequence[_typing.Sequence[DMatch]]
:param outImg: 
:type outImg: cv2.typing.MatLike
:param matchColor: 
:type matchColor: cv2.typing.Scalar
:param singlePointColor: 
:type singlePointColor: cv2.typing.Scalar
:param matchesMask: 
:type matchesMask: _typing.Sequence[_typing.Sequence[str]]
:param flags: 
:type flags: DrawMatchesFlags
:rtype: cv2.typing.MatLike
````


````{py:function} edgePreservingFilter(src[, dst[, flags[, sigma_s[, sigma_r]]]]) -> dst

Filtering is the fundamental operation in image and video processing. Edge-preserving smoothingfilters are used in many different applications @cite EM11 . 




:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param dst: Output 8-bit 3-channel image.
:type dst: cv2.typing.MatLike | None
:param flags: Edge preserving filters: cv::RECURS_FILTER or cv::NORMCONV_FILTER
:type flags: int
:param sigma_s: %Range between 0 to 200.
:type sigma_s: float
:param sigma_r: %Range between 0 to 1.
:type sigma_r: float
:rtype: cv2.typing.MatLike
````


````{py:function} eigen(src[, eigenvalues[, eigenvectors]]) -> retval, eigenvalues, eigenvectors

Calculates eigenvalues and eigenvectors of a symmetric matrix.


The function cv::eigen calculates just eigenvalues, or eigenvalues and eigenvectors of the symmetric matrix src: 
```c++
src*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()
```

```{note}
Use cv::eigenNonSymmetric for calculation of real eigenvalues and eigenvectors of non-symmetric matrix.
```
**See also:** eigenNonSymmetric, completeSymm , PCA


:param src: input matrix that must have CV_32FC1 or CV_64FC1 type, square size and be symmetrical(src ^T^ == src). 
:type src: cv2.typing.MatLike
:param eigenvalues: output vector of eigenvalues of the same type as src; the eigenvalues are storedin the descending order. 
:type eigenvalues: cv2.typing.MatLike | None
:param eigenvectors: output matrix of eigenvectors; it has the same size and type as src; theeigenvectors are stored as subsequent matrix rows, in the same order as the corresponding eigenvalues. 
:type eigenvectors: cv2.typing.MatLike | None
:rtype: tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} eigenNonSymmetric(src[, eigenvalues[, eigenvectors]]) -> eigenvalues, eigenvectors

Calculates eigenvalues and eigenvectors of a non-symmetric matrix (real eigenvalues only).


The function calculates eigenvalues and eigenvectors (optional) of the square matrix src: 
```c++
src*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()
```

```{note}
Assumes real eigenvalues.
```
**See also:** eigen


:param src: input matrix (CV_32FC1 or CV_64FC1 type).
:type src: cv2.typing.MatLike
:param eigenvalues: output vector of eigenvalues (type is the same type as src).
:type eigenvalues: cv2.typing.MatLike | None
:param eigenvectors: output matrix of eigenvectors (type is the same type as src). The eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding eigenvalues.
:type eigenvectors: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]) -> img

Draws a simple or thick elliptic arc or fills an ellipse sector.


The function cv::ellipse with more parameters draws an ellipse outline, a filled ellipse, an elliptic arc, or a filled ellipse sector. The drawing code uses general parametric form. A piecewise-linear curve is used to approximate the elliptic arc boundary. If you need more control of the ellipse rendering, you can retrieve the curve using #ellipse2Poly and then render it with #polylines or fill it with #fillPoly. If you use the first variant of the function and want to draw the whole ellipse, not an arc, pass `startAngle=0` and `endAngle=360`. If `startAngle` is greater than `endAngle`, they are swapped. The figure below explains the meaning of the parameters to draw the blue arc. 
![Parameters of Elliptic Arc](pics/ellipse.svg) 
ellipse(img, box, color[, thickness[, lineType]]) -> img @overload 


:param img: Image.
:type img: cv2.typing.MatLike
:param center: Center of the ellipse.
:type center: cv2.typing.Point
:param axes: Half of the size of the ellipse main axes.
:type axes: cv2.typing.Size
:param angle: Ellipse rotation angle in degrees.
:type angle: float
:param startAngle: Starting angle of the elliptic arc in degrees.
:type startAngle: float
:param endAngle: Ending angle of the elliptic arc in degrees.
:type endAngle: float
:param color: Ellipse color.
:type color: cv2.typing.Scalar
:param thickness: Thickness of the ellipse arc outline, if positive. Otherwise, this indicates thata filled ellipse sector is to be drawn. 
:type thickness: int
:param lineType: Type of the ellipse boundary. See #LineTypes
:type lineType: int
:param shift: Number of fractional bits in the coordinates of the center and values of axes.
:type shift: int
:param box: Alternative ellipse representation via RotatedRect. This means that the function drawsan ellipse inscribed in the rotated rectangle. 
:type box: 
:rtype: cv2.typing.MatLike
````


````{py:function} ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta) -> pts

Approximates an elliptic arc with a polyline.


The function ellipse2Poly computes the vertices of a polyline that approximates the specified elliptic arc. It is used by #ellipse. If `arcStart` is greater than `arcEnd`, they are swapped. 


:param center: Center of the arc.
:type center: cv2.typing.Point
:param axes: Half of the size of the ellipse main axes. See #ellipse for details.
:type axes: cv2.typing.Size
:param angle: Rotation angle of the ellipse in degrees. See #ellipse for details.
:type angle: int
:param arcStart: Starting angle of the elliptic arc in degrees.
:type arcStart: int
:param arcEnd: Ending angle of the elliptic arc in degrees.
:type arcEnd: int
:param delta: Angle between the subsequent polyline vertices. It defines the approximationaccuracy. 
:type delta: int
:param pts: Output vector of polyline vertices.
:type pts: 
:rtype: _typing.Sequence[cv2.typing.Point]
````


````{py:function} empty_array_desc() -> retval






:rtype: GArrayDesc
````


````{py:function} empty_gopaque_desc() -> retval






:rtype: GOpaqueDesc
````


````{py:function} empty_scalar_desc() -> retval






:rtype: GScalarDesc
````


````{py:function} equalizeHist(src[, dst]) -> dst

Equalizes the histogram of a grayscale image.


The function equalizes the histogram of the input image using the following algorithm: 
- Calculate the histogram $H$ for src . - Normalize the histogram so that the sum of histogram bins is 255. - Compute the integral of the histogram: $H'_i =  \sum _{0  \le j < i} H(j)$ - Transform the image using $H'$ as a look-up table: $\texttt{dst}(x,y) = H'(\texttt{src}(x,y))$ 
The algorithm normalizes the brightness and increases the contrast of the image. 


:param src: Source 8-bit single channel image.
:type src: cv2.typing.MatLike
:param dst: Destination image of the same size and type as src .
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst

Erodes an image by using a specific structuring element.


The function erodes the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the minimum is taken: 
$\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')$ 
The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In case of multi-channel images, each channel is processed independently. 
**See also:**  dilate, morphologyEx, getStructuringElement


:param src: input image; the number of channels can be arbitrary, but the depth should be one ofCV_8U, CV_16U, CV_16S, CV_32F or CV_64F. 
:type src: cv2.typing.MatLike
:param dst: output image of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param kernel: structuring element used for erosion; if `element=Mat()`, a `3 x 3` rectangularstructuring element is used. Kernel can be created using #getStructuringElement. 
:type kernel: cv2.typing.MatLike
:param anchor: position of the anchor within the element; default value (-1, -1) means that theanchor is at the element center. 
:type anchor: cv2.typing.Point
:param iterations: number of times erosion is applied.
:type iterations: int
:param borderType: pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:param borderValue: border value in case of a constant border
:type borderValue: cv2.typing.Scalar
:rtype: cv2.typing.MatLike
````


````{py:function} estimateAffine2D(from_, to[, inliers[, method[, ransacReprojThreshold[, maxIters[, confidence[, refineIters]]]]]]) -> retval, inliers

Computes an optimal affine transformation between two 2D point sets.


It computes $ \begin{bmatrix} x\\ y\\ \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12}\\ a_{21} & a_{22}\\ \end{bmatrix} \begin{bmatrix} X\\ Y\\ \end{bmatrix} + \begin{bmatrix} b_1\\ b_2\\ \end{bmatrix} $ 
The function estimates an optimal 2D affine transformation between two 2D point sets using the selected robust algorithm. 
The computed transformation is then refined further (using only inliers) with the Levenberg-Marquardt method to reduce the re-projection error even more. 
@note The RANSAC method can handle practically any ratio of outliers but needs a threshold to distinguish inliers from outliers. The method LMeDS does not need any threshold but it works correctly only when there are more than 50% of inliers. 
estimateAffine2D(pts1, pts2, params[, inliers]) -> retval, inliers 
**See also:** estimateAffinePartial2D, getAffineTransform


:param from: First input 2D point set containing $(X,Y)$.
:type from: 
:param to: Second input 2D point set containing $(x,y)$.
:type to: cv2.typing.MatLike
:param inliers: Output vector indicating which points are inliers (1-inlier, 0-outlier).
:type inliers: cv2.typing.MatLike | None
:param method: Robust method used to compute transformation. The following methods are possible:-   @ref RANSAC - RANSAC-based robust method -   @ref LMEDS - Least-Median robust method RANSAC is the default method. 
:type method: int
:param ransacReprojThreshold: Maximum reprojection error in the RANSAC algorithm to considera point as an inlier. Applies only to RANSAC. 
:type ransacReprojThreshold: float
:param maxIters: The maximum number of robust method iterations.
:type maxIters: int
:param confidence: Confidence level, between 0 and 1, for the estimated transformation. Anythingbetween 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation. 
:type confidence: float
:param refineIters: Maximum number of iterations of refining algorithm (Levenberg-Marquardt).Passing 0 will disable refining, so the output matrix will be output of robust method. 
:type refineIters: int
:param from_: 
:type from_: cv2.typing.MatLike
:return: Output 2D affine transformation matrix $2 \times 3$ or empty matrix if transformationcould not be estimated. The returned matrix has the following form: $ \begin{bmatrix} a_{11} & a_{12} & b_1\\ a_{21} & a_{22} & b_2\\ \end{bmatrix} $ 
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} estimateAffine3D(src, dst[, out[, inliers[, ransacThreshold[, confidence]]]]) -> retval, out, inliers

Computes an optimal affine transformation between two 3D point sets.


It computes $ \begin{bmatrix} x\\ y\\ z\\ \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & a_{13}\\ a_{21} & a_{22} & a_{23}\\ a_{31} & a_{32} & a_{33}\\ \end{bmatrix} \begin{bmatrix} X\\ Y\\ Z\\ \end{bmatrix} + \begin{bmatrix} b_1\\ b_2\\ b_3\\ \end{bmatrix} $ 
The function estimates an optimal 3D affine transformation between two 3D point sets using the RANSAC algorithm. 
estimateAffine3D(src, dst[, force_rotation]) -> retval, scale 
It computes $R,s,t$ minimizing $\sum{i} dst_i - c \cdot R \cdot src_i $ where $R$ is a 3x3 rotation matrix, $t$ is a 3x1 translation vector and $s$ is a scalar size value. This is an implementation of the algorithm by Umeyama \cite umeyama1991least . The estimated affine transform has a homogeneous scale which is a subclass of affine transformations with 7 degrees of freedom. The paired point sets need to comprise at least 3 points each. 


:param src: First input 3D point set.
:type src: cv2.typing.MatLike
:param dst: Second input 3D point set.
:type dst: cv2.typing.MatLike
:param out: Output 3D affine transformation matrix $3 \times 4$ of the form$ \begin{bmatrix} a_{11} & a_{12} & a_{13} & b_1\\ a_{21} & a_{22} & a_{23} & b_2\\ a_{31} & a_{32} & a_{33} & b_3\\ \end{bmatrix} $ 
:type out: cv2.typing.MatLike | None
:param inliers: Output vector indicating which points are inliers (1-inlier, 0-outlier).
:type inliers: cv2.typing.MatLike | None
:param ransacThreshold: Maximum reprojection error in the RANSAC algorithm to consider a point asan inlier. 
:type ransacThreshold: float
:param confidence: Confidence level, between 0 and 1, for the estimated transformation. Anythingbetween 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation. 
:type confidence: float
:param scale: If null is passed, the scale parameter c will be assumed to be 1.0.Else the pointed-to variable will be set to the optimal scale. 
:type scale: 
:param force_rotation: If true, the returned rotation will never be a reflection.This might be unwanted, e.g. when optimizing a transform between a right- and a left-handed coordinate system. 
:type force_rotation: 
:return: 3D affine transformation matrix $3 \times 4$ of the form$T = \begin{bmatrix} R & t\\ \end{bmatrix} $ 
:rtype: tuple[int, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} estimateAffinePartial2D(from_, to[, inliers[, method[, ransacReprojThreshold[, maxIters[, confidence[, refineIters]]]]]]) -> retval, inliers

Computes an optimal limited affine transformation with 4 degrees of freedom betweentwo 2D point sets. 


The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust estimation. 
The computed transformation is then refined further (using only inliers) with the Levenberg-Marquardt method to reduce the re-projection error even more. 
Estimated transformation matrix is: $ \begin{bmatrix} \cos(\theta) \cdot s & -\sin(\theta) \cdot s & t_x \\ \sin(\theta) \cdot s & \cos(\theta) \cdot s & t_y \end{bmatrix} $ Where $ \theta $ is the rotation angle, $ s $ the scaling factor and $ t_x, t_y $ are translations in $ x, y $ axes respectively. 
@note The RANSAC method can handle practically any ratio of outliers but need a threshold to distinguish inliers from outliers. The method LMeDS does not need any threshold but it works correctly only when there are more than 50% of inliers. 
**See also:** estimateAffine2D, getAffineTransform


:param from: First input 2D point set.
:type from: 
:param to: Second input 2D point set.
:type to: cv2.typing.MatLike
:param inliers: Output vector indicating which points are inliers.
:type inliers: cv2.typing.MatLike | None
:param method: Robust method used to compute transformation. The following methods are possible:-   @ref RANSAC - RANSAC-based robust method -   @ref LMEDS - Least-Median robust method RANSAC is the default method. 
:type method: int
:param ransacReprojThreshold: Maximum reprojection error in the RANSAC algorithm to considera point as an inlier. Applies only to RANSAC. 
:type ransacReprojThreshold: float
:param maxIters: The maximum number of robust method iterations.
:type maxIters: int
:param confidence: Confidence level, between 0 and 1, for the estimated transformation. Anythingbetween 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation. 
:type confidence: float
:param refineIters: Maximum number of iterations of refining algorithm (Levenberg-Marquardt).Passing 0 will disable refining, so the output matrix will be output of robust method. 
:type refineIters: int
:param from_: 
:type from_: cv2.typing.MatLike
:return: Output 2D affine transformation (4 degrees of freedom) matrix $2 \times 3$ orempty matrix if transformation could not be estimated. 
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} estimateChessboardSharpness(image, patternSize, corners[, rise_distance[, vertical[, sharpness]]]) -> retval, sharpness

Estimates the sharpness of a detected chessboard.


Image sharpness, as well as brightness, are a critical parameter for accuracte camera calibration. For accessing these parameters for filtering out problematic calibraiton images, this method calculates edge profiles by traveling from black to white chessboard cell centers. Based on this, the number of pixels is calculated required to transit from black to white. This width of the transition area is a good indication of how sharp the chessboard is imaged and should be below ~3.0 pixels. 
The optional sharpness array is of type CV_32FC1 and has for each calculated profile one row with the following five entries: 0 = x coordinate of the underlying edge in the image 1 = y coordinate of the underlying edge in the image 2 = width of the transition area (sharpness) 3 = signal strength in the black cell (min brightness) 4 = signal strength in the white cell (max brightness) 


:param image: Gray image used to find chessboard corners
:type image: cv2.typing.MatLike
:param patternSize: Size of a found chessboard pattern
:type patternSize: cv2.typing.Size
:param corners: Corners found by #findChessboardCornersSB
:type corners: cv2.typing.MatLike
:param rise_distance: Rise distance 0.8 means 10% ... 90% of the final signal strength
:type rise_distance: float
:param vertical: By default edge responses for horizontal lines are calculated
:type vertical: bool
:param sharpness: Optional output array with a sharpness value for calculated edge responses (see description)
:type sharpness: cv2.typing.MatLike | None
:return: Scalar(average sharpness, average min brightness, average max brightness,0)
:rtype: tuple[cv2.typing.Scalar, cv2.typing.MatLike]
````


````{py:function} estimateTranslation3D(src, dst[, out[, inliers[, ransacThreshold[, confidence]]]]) -> retval, out, inliers

Computes an optimal translation between two 3D point sets.


It computes $ \begin{bmatrix} x\\ y\\ z\\ \end{bmatrix} = \begin{bmatrix} X\\ Y\\ Z\\ \end{bmatrix} + \begin{bmatrix} b_1\\ b_2\\ b_3\\ \end{bmatrix} $ 
The function estimates an optimal 3D translation between two 3D point sets using the RANSAC algorithm. 


:param src: First input 3D point set containing $(X,Y,Z)$.
:type src: cv2.typing.MatLike
:param dst: Second input 3D point set containing $(x,y,z)$.
:type dst: cv2.typing.MatLike
:param out: Output 3D translation vector $3 \times 1$ of the form$ \begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ \end{bmatrix} $ 
:type out: cv2.typing.MatLike | None
:param inliers: Output vector indicating which points are inliers (1-inlier, 0-outlier).
:type inliers: cv2.typing.MatLike | None
:param ransacThreshold: Maximum reprojection error in the RANSAC algorithm to consider a point asan inlier. 
:type ransacThreshold: float
:param confidence: Confidence level, between 0 and 1, for the estimated transformation. Anythingbetween 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation. 
:type confidence: float
:rtype: tuple[int, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} exp(src[, dst]) -> dst

Calculates the exponent of every array element.


The function cv::exp calculates the exponent of every element of the input array: $\texttt{dst} [I] = e^{ src(I) }$ 
The maximum relative error is about 7e-6 for single-precision input and less than 1e-10 for double-precision input. Currently, the function converts denormalized values to zeros on output. Special values (NaN, Inf) are not handled. 
**See also:** log , cartToPolar , polarToCart , phase , pow , sqrt , magnitude


:param src: input array.
:type src: cv2.typing.MatLike
:param dst: output array of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} extractChannel(src, coi[, dst]) -> dst

Extracts a single channel from src (coi is 0-based index)


**See also:** mixChannels, split


:param src: input array
:type src: cv2.typing.MatLike
:param dst: output array
:type dst: cv2.typing.MatLike | None
:param coi: index of channel to extract
:type coi: int
:rtype: cv2.typing.MatLike
````


````{py:function} fastAtan2(y, x) -> retval

Calculates the angle of a 2D vector in degrees.


The function fastAtan2 calculates the full-range angle of an input 2D vector. The angle is measured in degrees and varies from 0 to 360 degrees. The accuracy is about 0.3 degrees. 


:param x: x-coordinate of the vector.
:type x: float
:param y: y-coordinate of the vector.
:type y: float
:rtype: float
````


````{py:function} fastNlMeansDenoising(src[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst

Perform image denoising using Non-local Means Denoising algorithm<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/> with several computational optimizations. Noise expected to be a gaussian white noise 


This function expected to be applied to grayscale images. For colored images look at fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting image to CIELAB colorspace and then separately denoise L and AB components with different h parameter. 
fastNlMeansDenoising(src, h[, dst[, templateWindowSize[, searchWindowSize[, normType]]]]) -> dst 
This function expected to be applied to grayscale images. For colored images look at fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting image to CIELAB colorspace and then separately denoise L and AB components with different h parameter. 


:param src: Input 8-bit or 16-bit (only with NORM_L1) 1-channel,2-channel, 3-channel or 4-channel image. 
:type src: cv2.typing.MatLike
:param dst: Output image with the same size and type as src .
:type dst: cv2.typing.MatLike | None
:param templateWindowSize: Size in pixels of the template patch that is used to compute weights.Should be odd. Recommended value 7 pixels 
:type templateWindowSize: int
:param searchWindowSize: Size in pixels of the window that is used to compute weighted average forgiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels 
:type searchWindowSize: int
:param h: Array of parameters regulating filter strength, either oneparameter applied to all channels or one per channel in dst. Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise 
:type h: float
:param normType: Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1
:type normType: 
:rtype: cv2.typing.MatLike
````


````{py:function} fastNlMeansDenoisingColored(src[, dst[, h[, hColor[, templateWindowSize[, searchWindowSize]]]]]) -> dst

Modification of fastNlMeansDenoising function for colored images


The function converts image to CIELAB colorspace and then separately denoise L and AB components with given h parameters using fastNlMeansDenoising function. 


:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param dst: Output image with the same size and type as src .
:type dst: cv2.typing.MatLike | None
:param templateWindowSize: Size in pixels of the template patch that is used to compute weights.Should be odd. Recommended value 7 pixels 
:type templateWindowSize: int
:param searchWindowSize: Size in pixels of the window that is used to compute weighted average forgiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels 
:type searchWindowSize: int
:param h: Parameter regulating filter strength for luminance component. Bigger h value perfectlyremoves noise but also removes image details, smaller h value preserves details but also preserves some noise 
:type h: float
:param hColor: The same as h but for color components. For most images value equals 10will be enough to remove colored noise and do not distort colors 
:type hColor: float
:rtype: cv2.typing.MatLike
````


````{py:function} fastNlMeansDenoisingColoredMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize[, dst[, h[, hColor[, templateWindowSize[, searchWindowSize]]]]]) -> dst

Modification of fastNlMeansDenoisingMulti function for colored images sequences


The function converts images to CIELAB colorspace and then separately denoise L and AB components with given h parameters using fastNlMeansDenoisingMulti function. 


:param srcImgs: Input 8-bit 3-channel images sequence. All images should have the same type andsize. 
:type srcImgs: _typing.Sequence[cv2.typing.MatLike]
:param imgToDenoiseIndex: Target image to denoise index in srcImgs sequence
:type imgToDenoiseIndex: int
:param temporalWindowSize: Number of surrounding images to use for target image denoising. Shouldbe odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise srcImgs[imgToDenoiseIndex] image. 
:type temporalWindowSize: int
:param dst: Output image with the same size and type as srcImgs images.
:type dst: cv2.typing.MatLike | None
:param templateWindowSize: Size in pixels of the template patch that is used to compute weights.Should be odd. Recommended value 7 pixels 
:type templateWindowSize: int
:param searchWindowSize: Size in pixels of the window that is used to compute weighted average forgiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels 
:type searchWindowSize: int
:param h: Parameter regulating filter strength for luminance component. Bigger h value perfectlyremoves noise but also removes image details, smaller h value preserves details but also preserves some noise. 
:type h: float
:param hColor: The same as h but for color components.
:type hColor: float
:rtype: cv2.typing.MatLike
````


````{py:function} fastNlMeansDenoisingMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst

Modification of fastNlMeansDenoising function for images sequence where consecutive images have beencaptured in small period of time. For example video. This version of the function is for grayscale images or for manual manipulation with colorspaces. See @cite Buades2005DenoisingIS for more details (open access [here](https://static.aminer.org/pdf/PDF/000/317/196/spatio_temporal_wiener_filtering_of_image_sequences_using_a_parametric.pdf)). 


fastNlMeansDenoisingMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize, h[, dst[, templateWindowSize[, searchWindowSize[, normType]]]]) -> dst 


:param srcImgs: Input 8-bit or 16-bit (only with NORM_L1) 1-channel,2-channel, 3-channel or 4-channel images sequence. All images should have the same type and size. 
:type srcImgs: _typing.Sequence[cv2.typing.MatLike]
:param imgToDenoiseIndex: Target image to denoise index in srcImgs sequence
:type imgToDenoiseIndex: int
:param temporalWindowSize: Number of surrounding images to use for target image denoising. Shouldbe odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise srcImgs[imgToDenoiseIndex] image. 
:type temporalWindowSize: int
:param dst: Output image with the same size and type as srcImgs images.
:type dst: cv2.typing.MatLike | None
:param templateWindowSize: Size in pixels of the template patch that is used to compute weights.Should be odd. Recommended value 7 pixels 
:type templateWindowSize: int
:param searchWindowSize: Size in pixels of the window that is used to compute weighted average forgiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels 
:type searchWindowSize: int
:param h: Array of parameters regulating filter strength, either oneparameter applied to all channels or one per channel in dst. Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise 
:type h: float
:param normType: Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1
:type normType: 
:rtype: cv2.typing.MatLike
````


````{py:function} fillConvexPoly(img, points, color[, lineType[, shift]]) -> img

Fills a convex polygon.


The function cv::fillConvexPoly draws a filled convex polygon. This function is much faster than the function #fillPoly . It can fill not only convex polygons but any monotonic polygon without self-intersections, that is, a polygon whose contour intersects every horizontal line (scan line) twice at the most (though, its top-most and/or the bottom edge could be horizontal). 


:param img: Image.
:type img: cv2.typing.MatLike
:param points: Polygon vertices.
:type points: cv2.typing.MatLike
:param color: Polygon color.
:type color: cv2.typing.Scalar
:param lineType: Type of the polygon boundaries. See #LineTypes
:type lineType: int
:param shift: Number of fractional bits in the vertex coordinates.
:type shift: int
:rtype: cv2.typing.MatLike
````


````{py:function} fillPoly(img, pts, color[, lineType[, shift[, offset]]]) -> img

Fills the area bounded by one or more polygons.


The function cv::fillPoly fills an area bounded by several polygonal contours. The function can fill complex areas, for example, areas with holes, contours with self-intersections (some of their parts), and so forth. 


:param img: Image.
:type img: cv2.typing.MatLike
:param pts: Array of polygons where each polygon is represented as an array of points.
:type pts: _typing.Sequence[cv2.typing.MatLike]
:param color: Polygon color.
:type color: cv2.typing.Scalar
:param lineType: Type of the polygon boundaries. See #LineTypes
:type lineType: int
:param shift: Number of fractional bits in the vertex coordinates.
:type shift: int
:param offset: Optional offset of all points of the contours.
:type offset: cv2.typing.Point
:rtype: cv2.typing.MatLike
````


````{py:function} filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst

Convolves an image with the kernel.


The function applies an arbitrary linear filter to an image. In-place operation is supported. When the aperture is partially outside the image, the function interpolates outlier pixel values according to the specified border mode. 
The function does actually compute correlation, not the convolution: 
$\texttt{dst} (x,y) =  \sum _{ \substack{0\leq x' < \texttt{kernel.cols}\\{0\leq y' < \texttt{kernel.rows}}}}  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )$ 
That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip the kernel using #flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows - anchor.y - 1)`. 
The function uses the DFT-based algorithm in case of sufficiently large kernels (~`11 x 11` or larger) and the direct algorithm for small kernels. 
**See also:**  sepFilter2D, dft, matchTemplate


:param src: input image.
:type src: cv2.typing.MatLike
:param dst: output image of the same size and the same number of channels as src.
:type dst: cv2.typing.MatLike | None
:param ddepth: desired depth of the destination image, see @ref filter_depths "combinations"
:type ddepth: int
:param kernel: convolution kernel (or rather a correlation kernel), a single-channel floating pointmatrix; if you want to apply different kernels to different channels, split the image into separate color planes using split and process them individually. 
:type kernel: cv2.typing.MatLike
:param anchor: anchor of the kernel that indicates the relative position of a filtered point withinthe kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor is at the kernel center. 
:type anchor: cv2.typing.Point
:param delta: optional value added to the filtered pixels before storing them in dst.
:type delta: float
:param borderType: pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} filterHomographyDecompByVisibleRefpoints(rotations, normals, beforePoints, afterPoints[, possibleSolutions[, pointsMask]]) -> possibleSolutions

Filters homography decompositions based on additional information.


This function is intended to filter the output of the #decomposeHomographyMat based on additional information as described in @cite Malis2007 . The summary of the method: the #decomposeHomographyMat function returns 2 unique solutions and their "opposites" for a total of 4 solutions. If we have access to the sets of points visible in the camera frame before and after the homography transformation is applied, we can determine which are the true potential solutions and which are the opposites by verifying which homographies are consistent with all visible reference points being in front of the camera. The inputs are left unchanged; the filtered solution set is returned as indices into the existing one. 


:param rotations: Vector of rotation matrices.
:type rotations: _typing.Sequence[cv2.typing.MatLike]
:param normals: Vector of plane normal matrices.
:type normals: _typing.Sequence[cv2.typing.MatLike]
:param beforePoints: Vector of (rectified) visible reference points before the homography is applied
:type beforePoints: cv2.typing.MatLike
:param afterPoints: Vector of (rectified) visible reference points after the homography is applied
:type afterPoints: cv2.typing.MatLike
:param possibleSolutions: Vector of int indices representing the viable solution set after filtering
:type possibleSolutions: cv2.typing.MatLike | None
:param pointsMask: optional Mat/Vector of 8u type representing the mask for the inliers as given by the #findHomography function
:type pointsMask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} filterSpeckles(img, newVal, maxSpeckleSize, maxDiff[, buf]) -> img, buf

Filters off small noise blobs (speckles) in the disparity map




:param img: The input 16-bit signed disparity image
:type img: cv2.typing.MatLike
:param newVal: The disparity value used to paint-off the speckles
:type newVal: float
:param maxSpeckleSize: The maximum speckle size to consider it a speckle. Larger blobs are notaffected by the algorithm 
:type maxSpeckleSize: int
:param maxDiff: Maximum difference between neighbor disparity pixels to put them into the sameblob. Note that since StereoBM, StereoSGBM and may be other algorithms return a fixed-point disparity map, where disparity values are multiplied by 16, this scale factor should be taken into account when specifying this parameter value. 
:type maxDiff: float
:param buf: The optional temporary buffer to avoid memory allocation within the function.
:type buf: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} find4QuadCornerSubpix(img, corners, region_size) -> retval, corners






:param img: 
:type img: cv2.typing.MatLike
:param corners: 
:type corners: cv2.typing.MatLike
:param region_size: 
:type region_size: cv2.typing.Size
:rtype: tuple[bool, cv2.typing.MatLike]
````


````{py:function} findChessboardCorners(image, patternSize[, corners[, flags]]) -> retval, corners

Finds the positions of internal corners of the chessboard.


The function attempts to determine whether the input image is a view of the chessboard pattern and locate the internal chessboard corners. The function returns a non-zero value if all of the corners are found and they are placed in a certain order (row by row, left to right in every row). Otherwise, if the function fails to find all the corners or reorder them, it returns 0. For example, a regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black squares touch each other. The detected coordinates are approximate, and to determine their positions more accurately, the function calls #cornerSubPix. You also may use the function #cornerSubPix with different parameters if returned coordinates are not accurate enough. 
Sample usage of detecting and drawing chessboard corners: : 
```c++
Size patternsize(8,6); //interior number of corners
Mat gray = ....; //source image
vector<Point2f> corners; //this will be filled by the detected corners

//CALIB_CB_FAST_CHECK saves a lot of time on images
//that do not contain any chessboard corners
bool patternfound = findChessboardCorners(gray, patternsize, corners,
CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
+ CALIB_CB_FAST_CHECK);

if(patternfound)
cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
```

Use gen_pattern.py (@ref tutorial_camera_calibration_pattern) to create checkerboard. 
```{note}
The function requires white space (like a square-thick border, the wider the better) aroundthe board to make the detection more robust in various environments. Otherwise, if there is no border and the background is dark, the outer black squares cannot be segmented properly and so the square grouping and ordering algorithm fails. 
```


:param image: Source chessboard view. It must be an 8-bit grayscale or color image.
:type image: cv2.typing.MatLike
:param patternSize: Number of inner corners per a chessboard row and column( patternSize = cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ). 
:type patternSize: cv2.typing.Size
:param corners: Output array of detected corners.
:type corners: cv2.typing.MatLike | None
:param flags: Various operation flags that can be zero or a combination of the following values:-   @ref CALIB_CB_ADAPTIVE_THRESH Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness). -   @ref CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with #equalizeHist before applying fixed or adaptive thresholding. -   @ref CALIB_CB_FILTER_QUADS Use additional criteria (like contour area, perimeter, square-like shape) to filter out false quads extracted at the contour retrieval stage. -   @ref CALIB_CB_FAST_CHECK Run a fast check on the image that looks for chessboard corners, and shortcut the call if none is found. This can drastically speed up the call in the degenerate condition when no chessboard is observed. -   @ref CALIB_CB_PLAIN All other flags are ignored. The input image is taken as is. No image processing is done to improve to find the checkerboard. This has the effect of speeding up the execution of the function but could lead to not recognizing the checkerboard if the image is not previously binarized in the appropriate manner. 
:type flags: int
:rtype: tuple[bool, cv2.typing.MatLike]
````


````{py:function} findChessboardCornersSB(image, patternSize[, corners[, flags]]) -> retval, corners




@overload 


:param image: 
:type image: cv2.typing.MatLike
:param patternSize: 
:type patternSize: cv2.typing.Size
:param corners: 
:type corners: cv2.typing.MatLike | None
:param flags: 
:type flags: int
:rtype: tuple[bool, cv2.typing.MatLike]
````


````{py:function} findChessboardCornersSBWithMeta(image, patternSize, flags[, corners[, meta]]) -> retval, corners, meta

Finds the positions of internal corners of the chessboard using a sector based approach.


The function is analog to #findChessboardCorners but uses a localized radon transformation approximated by box filters being more robust to all sort of noise, faster on larger images and is able to directly return the sub-pixel position of the internal chessboard corners. The Method is based on the paper @cite duda2018 "Accurate Detection and Localization of Checkerboard Corners for Calibration" demonstrating that the returned sub-pixel positions are more accurate than the one returned by cornerSubPix allowing a precise camera calibration for demanding applications. 
In the case, the flags @ref CALIB_CB_LARGER or @ref CALIB_CB_MARKER are given, the result can be recovered from the optional meta array. Both flags are helpful to use calibration patterns exceeding the field of view of the camera. These oversized patterns allow more accurate calibrations as corners can be utilized, which are as close as possible to the image borders.  For a consistent coordinate system across all images, the optional marker (see image below) can be used to move the origin of the board to the location where the black circle is located. 
Use gen_pattern.py (@ref tutorial_camera_calibration_pattern) to create checkerboard. ![Checkerboard](pics/checkerboard_radon.png) 
```{note}
The function requires a white boarder with roughly the same width as oneof the checkerboard fields around the whole board to improve the detection in various environments. In addition, because of the localized radon transformation it is beneficial to use round corners for the field corners which are located on the outside of the board. The following figure illustrates a sample checkerboard optimized for the detection. However, any other checkerboard can be used as well. 
```


:param image: Source chessboard view. It must be an 8-bit grayscale or color image.
:type image: cv2.typing.MatLike
:param patternSize: Number of inner corners per a chessboard row and column( patternSize = cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ). 
:type patternSize: cv2.typing.Size
:param corners: Output array of detected corners.
:type corners: cv2.typing.MatLike | None
:param flags: Various operation flags that can be zero or a combination of the following values:-   @ref CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with equalizeHist before detection. -   @ref CALIB_CB_EXHAUSTIVE Run an exhaustive search to improve detection rate. -   @ref CALIB_CB_ACCURACY Up sample input image to improve sub-pixel accuracy due to aliasing effects. -   @ref CALIB_CB_LARGER The detected pattern is allowed to be larger than patternSize (see description). -   @ref CALIB_CB_MARKER The detected pattern must have a marker (see description). This should be used if an accurate camera calibration is required. 
:type flags: int
:param meta: Optional output arrray of detected corners (CV_8UC1 and size = cv::Size(columns,rows)).Each entry stands for one corner of the pattern and can have one of the following values: -   0 = no meta data attached -   1 = left-top corner of a black cell -   2 = left-top corner of a white cell -   3 = left-top corner of a black cell with a white marker dot -   4 = left-top corner of a white cell with a black marker dot (pattern origin in case of markers otherwise first corner) 
:type meta: cv2.typing.MatLike | None
:rtype: tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} findCirclesGrid(image, patternSize, flags, blobDetector, parameters[, centers]) -> retval, centers

Finds centers in the grid of circles.


The function attempts to determine whether the input image contains a grid of circles. If it is, the function locates centers of the circles. The function returns a non-zero value if all of the centers have been found and they have been placed in a certain order (row by row, left to right in every row). Otherwise, if the function fails to find all the corners or reorder them, it returns 0. 
Sample usage of detecting and drawing the centers of circles: : 
```c++
Size patternsize(7,7); //number of centers
Mat gray = ...; //source image
vector<Point2f> centers; //this will be filled by the detected centers

bool patternfound = findCirclesGrid(gray, patternsize, centers);

drawChessboardCorners(img, patternsize, Mat(centers), patternfound);
```

findCirclesGrid(image, patternSize[, centers[, flags[, blobDetector]]]) -> retval, centers @overload 
```{note}
The function requires white space (like a square-thick border, the wider the better) aroundthe board to make the detection more robust in various environments. 
```


:param image: grid view of input circles; it must be an 8-bit grayscale or color image.
:type image: cv2.typing.MatLike
:param patternSize: number of circles per row and column( patternSize = Size(points_per_row, points_per_colum) ). 
:type patternSize: cv2.typing.Size
:param centers: output array of detected centers.
:type centers: cv2.typing.MatLike | None
:param flags: various operation flags that can be one of the following values:-   @ref CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles. -   @ref CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles. -   @ref CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more robust to perspective distortions but much more sensitive to background clutter. 
:type flags: int
:param blobDetector: feature detector that finds blobs like dark circles on light background.If `blobDetector` is NULL then `image` represents Point2f array of candidates. 
:type blobDetector: cv2.typing.FeatureDetector
:param parameters: struct for finding circles in a grid pattern.
:type parameters: CirclesGridFinderParameters
:rtype: tuple[bool, cv2.typing.MatLike]
````


````{py:function} findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy

Finds contours in a binary image.


The function retrieves contours from the binary image using the algorithm @cite Suzuki85 . The contours are a useful tool for shape analysis and object detection and recognition. See squares.cpp in the OpenCV sample directory. 
```{note}
Since opencv 3.2 source image is not modified by this function.
```
```{note}
In Python, hierarchy is nested inside a top level array. Use hierarchy[0][i] to access hierarchical elements of i-th contour.
```


:param image: Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zeropixels remain 0's, so the image is treated as binary . You can use #compare, #inRange, #threshold , #adaptiveThreshold, #Canny, and others to create a binary image out of a grayscale or color one. If mode equals to #RETR_CCOMP or #RETR_FLOODFILL, the input can also be a 32-bit integer image of labels (CV_32SC1). 
:type image: cv2.typing.MatLike
:param contours: Detected contours. Each contour is stored as a vector of points (e.g.std::vector<std::vector<cv::Point> >). 
:type contours: _typing.Sequence[cv2.typing.MatLike] | None
:param hierarchy: Optional output vector (e.g. std::vector<cv::Vec4i>), containing information about the image topology. It hasas many elements as the number of contours. For each i-th contour contours[i], the elements hierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , and hierarchy[i][3] are set to 0-based indices in contours of the next and previous contours at the same hierarchical level, the first child contour and the parent contour, respectively. If for the contour i there are no next, previous, parent, or nested contours, the corresponding elements of hierarchy[i] will be negative. 
:type hierarchy: cv2.typing.MatLike | None
:param mode: Contour retrieval mode, see #RetrievalModes
:type mode: int
:param method: Contour approximation method, see #ContourApproximationModes
:type method: int
:param offset: Optional offset by which every contour point is shifted. This is useful if thecontours are extracted from the image ROI and then they should be analyzed in the whole image context. 
:type offset: cv2.typing.Point
:rtype: tuple[_typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike]
````


````{py:function} findEssentialMat(points1, points2, cameraMatrix[, method[, prob[, threshold[, maxIters[, mask]]]]]) -> retval, mask

Calculates an essential matrix from the corresponding points in two images from potentially two different cameras.


This function estimates essential matrix based on the five-point algorithm solver in @cite Nister03 . @cite SteweniusCFS is also a related. The epipolar geometry is described by the following equation: 
$[p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0$ 
where $E$ is an essential matrix, $p_1$ and $p_2$ are corresponding points in the first and the second images, respectively. The result of this function may be passed further to #decomposeEssentialMat or #recoverPose to recover the relative pose between cameras. 
findEssentialMat(points1, points2[, focal[, pp[, method[, prob[, threshold[, maxIters[, mask]]]]]]]) -> retval, mask @overload 
This function differs from the one above that it computes camera intrinsic matrix from focal length and principal point: 
$A = \begin{bmatrix} f & 0 & x_{pp}  \\ 0 & f & y_{pp}  \\ 0 & 0 & 1 \end{bmatrix}$ 
findEssentialMat(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2[, method[, prob[, threshold[, mask]]]]) -> retval, mask 
This function estimates essential matrix based on the five-point algorithm solver in @cite Nister03 . @cite SteweniusCFS is also a related. The epipolar geometry is described by the following equation: 
$[p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0$ 
where $E$ is an essential matrix, $p_1$ and $p_2$ are corresponding points in the first and the second images, respectively. The result of this function may be passed further to #decomposeEssentialMat or  #recoverPose to recover the relative pose between cameras. 
findEssentialMat(points1, points2, cameraMatrix1, cameraMatrix2, dist_coeff1, dist_coeff2, params[, mask]) -> retval, mask 


:param points1: Array of N (N \>= 5) 2D points from the first image. The point coordinates shouldbe floating-point (single or double precision). 
:type points1: cv2.typing.MatLike
:param points2: Array of the second image points of the same size and format as points1 .
:type points2: cv2.typing.MatLike
:param cameraMatrix: Camera intrinsic matrix $\cameramatrix{A}$ .Note that this function assumes that points1 and points2 are feature points from cameras with the same camera intrinsic matrix. If this assumption does not hold for your use case, use #undistortPoints with `P = cv::NoArray()` for both cameras to transform image points to normalized image coordinates, which are valid for the identity camera intrinsic matrix. When passing these coordinates, pass the identity matrix for this parameter. 
:type cameraMatrix: cv2.typing.MatLike
:param method: Method for computing an essential matrix.-   @ref RANSAC for the RANSAC algorithm. -   @ref LMEDS for the LMedS algorithm. 
:type method: int
:param prob: Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level ofconfidence (probability) that the estimated matrix is correct. 
:type prob: float
:param threshold: Parameter used for RANSAC. It is the maximum distance from a point to an epipolarline in pixels, beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point localization, image resolution, and the image noise. 
:type threshold: float
:param mask: Output array of N elements, every element of which is set to 0 for outliers and to 1for the other points. The array is computed only in the RANSAC and LMedS methods. 
:type mask: cv2.typing.MatLike | None
:param maxIters: The maximum number of robust method iterations.
:type maxIters: int
:param focal: focal length of the camera. Note that this function assumes that points1 and points2are feature points from cameras with same focal length and principal point. 
:type focal: 
:param pp: principal point of the camera.
:type pp: 
:param cameraMatrix1: Camera matrix $K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .Note that this function assumes that points1 and points2 are feature points from cameras with the same camera matrix. If this assumption does not hold for your use case, use #undistortPoints with `P = cv::NoArray()` for both cameras to transform image points to normalized image coordinates, which are valid for the identity camera matrix. When passing these coordinates, pass the identity matrix for this parameter. 
:type cameraMatrix1: 
:param cameraMatrix2: Camera matrix $K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .Note that this function assumes that points1 and points2 are feature points from cameras with the same camera matrix. If this assumption does not hold for your use case, use #undistortPoints with `P = cv::NoArray()` for both cameras to transform image points to normalized image coordinates, which are valid for the identity camera matrix. When passing these coordinates, pass the identity matrix for this parameter. 
:type cameraMatrix2: 
:param distCoeffs1: Input vector of distortion coefficients$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs1: 
:param distCoeffs2: Input vector of distortion coefficients$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs2: 
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} findFundamentalMat(points1, points2, method, ransacReprojThreshold, confidence, maxIters[, mask]) -> retval, mask

Calculates a fundamental matrix from the corresponding points in two images.


The epipolar geometry is described by the following equation: 
$[p_2; 1]^T F [p_1; 1] = 0$ 
where $F$ is a fundamental matrix, $p_1$ and $p_2$ are corresponding points in the first and the second images, respectively. 
The function calculates the fundamental matrix using one of four methods listed above and returns the found fundamental matrix. Normally just one matrix is found. But in case of the 7-point algorithm, the function may return up to 3 solutions ( $9 \times 3$ matrix that stores all 3 matrices sequentially). 
The calculated fundamental matrix may be passed further to #computeCorrespondEpilines that finds the epipolar lines corresponding to the specified points. It can also be passed to #stereoRectifyUncalibrated to compute the rectification transformation. : 
```c++
// Example. Estimation of fundamental matrix using the RANSAC algorithm
int point_count = 100;
vector<Point2f> points1(point_count);
vector<Point2f> points2(point_count);

// initialize the points here ...
for( int i = 0; i < point_count; i++ )
{
points1[i] = ...;
points2[i] = ...;
}

Mat fundamental_matrix =
findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
```

findFundamentalMat(points1, points2[, method[, ransacReprojThreshold[, confidence[, mask]]]]) -> retval, mask @overload 
findFundamentalMat(points1, points2, params[, mask]) -> retval, mask @overload 


:param points1: Array of N points from the first image. The point coordinates should befloating-point (single or double precision). 
:type points1: cv2.typing.MatLike
:param points2: Array of the second image points of the same size and format as points1 .
:type points2: cv2.typing.MatLike
:param method: Method for computing a fundamental matrix.-   @ref FM_7POINT for a 7-point algorithm. $N = 7$ -   @ref FM_8POINT for an 8-point algorithm. $N \ge 8$ -   @ref FM_RANSAC for the RANSAC algorithm. $N \ge 8$ -   @ref FM_LMEDS for the LMedS algorithm. $N \ge 8$ 
:type method: int
:param ransacReprojThreshold: Parameter used only for RANSAC. It is the maximum distance from a point to an epipolarline in pixels, beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point localization, image resolution, and the image noise. 
:type ransacReprojThreshold: float
:param confidence: Parameter used for the RANSAC and LMedS methods only. It specifies a desirable levelof confidence (probability) that the estimated matrix is correct. 
:type confidence: float
:param mask: [out] optional output mask
:type mask: cv2.typing.MatLike | None
:param maxIters: The maximum number of robust method iterations.
:type maxIters: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask[, maxIters[, confidence]]]]]) -> retval, mask

Finds a perspective transformation between two planes.


The function finds and returns the perspective transformation $H$ between the source and the destination planes: 
$s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}$ 
so that the back-projection error 
$\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2$ 
is minimized. If the parameter method is set to the default value 0, the function uses all the point pairs to compute an initial homography estimate with a simple least-squares scheme. 
However, if not all of the point pairs ( $srcPoints_i$, $dstPoints_i$ ) fit the rigid perspective transformation (that is, there are some outliers), this initial estimate will be poor. In this case, you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix using this subset and a simple least-squares algorithm, and then compute the quality/goodness of the computed homography (which is the number of inliers for RANSAC or the least median re-projection error for LMeDS). The best subset is then used to produce the initial estimate of the homography matrix and the mask of inliers/outliers. 
Regardless of the method, robust or not, the computed homography matrix is refined further (using inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the re-projection error even more. 
The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to distinguish inliers from outliers. The method LMeDS does not need any threshold but it works correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the noise is rather small, use the default method (method=0). 
The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is determined up to a scale. Thus, it is normalized so that $h_{33}=1$. Note that whenever an $H$ matrix cannot be estimated, an empty one will be returned. 
@sa getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective, perspectiveTransform 
findHomography(srcPoints, dstPoints, params[, mask]) -> retval, mask @overload 


:param srcPoints: Coordinates of the points in the original plane, a matrix of the type CV_32FC2or vector\<Point2f\> . 
:type srcPoints: cv2.typing.MatLike
:param dstPoints: Coordinates of the points in the target plane, a matrix of the type CV_32FC2 ora vector\<Point2f\> . 
:type dstPoints: cv2.typing.MatLike
:param method: Method used to compute a homography matrix. The following methods are possible:-   **0** - a regular method using all the points, i.e., the least squares method -   @ref RANSAC - RANSAC-based robust method -   @ref LMEDS - Least-Median robust method -   @ref RHO - PROSAC-based robust method 
:type method: int
:param ransacReprojThreshold: Maximum allowed reprojection error to treat a point pair as an inlier(used in the RANSAC and RHO methods only). That is, if $\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} \cdot \texttt{srcPoints} _i) \|_2  >  \texttt{ransacReprojThreshold}$ then the point $i$ is considered as an outlier. If srcPoints and dstPoints are measured in pixels, it usually makes sense to set this parameter somewhere in the range of 1 to 10. 
:type ransacReprojThreshold: float
:param mask: Optional output mask set by a robust method ( RANSAC or LMeDS ). Note that the inputmask values are ignored. 
:type mask: cv2.typing.MatLike | None
:param maxIters: The maximum number of RANSAC iterations.
:type maxIters: int
:param confidence: Confidence level, between 0 and 1.
:type confidence: float
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} findNonZero(src[, idx]) -> idx

Returns the list of locations of non-zero pixels


Given a binary matrix (likely returned from an operation such as threshold(), compare(), >, ==, etc, return all of the non-zero indices as a cv::Mat or std::vector<cv::Point> (x,y) For example: 
```cpp
cv::Mat binaryImage; // input, binary image
cv::Mat locations;   // output, locations of non-zero pixels
cv::findNonZero(binaryImage, locations);

// access pixel coordinates
Point pnt = locations.at<Point>(i);
```
or 
```cpp
cv::Mat binaryImage; // input, binary image
vector<Point> locations;   // output, locations of non-zero pixels
cv::findNonZero(binaryImage, locations);

// access pixel coordinates
Point pnt = locations[i];
```



:param src: single-channel array
:type src: cv2.typing.MatLike
:param idx: the output array, type of cv::Mat or std::vector<Point>, corresponding to non-zero indices in the input
:type idx: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} findTransformECC(templateImage, inputImage, warpMatrix, motionType, criteria, inputMask, gaussFiltSize) -> retval, warpMatrix

Finds the geometric transform (warp) between two images in terms of the ECC criterion @cite EP08 .


The function estimates the optimum transformation (warpMatrix) with respect to ECC criterion (@cite EP08), that is 
$\texttt{warpMatrix} = \arg\max_{W} \texttt{ECC}(\texttt{templateImage}(x,y),\texttt{inputImage}(x',y'))$ 
where 
$\begin{bmatrix} x' \\ y' \end{bmatrix} = W \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$ 
(the equation holds with homogeneous coordinates for homography). It returns the final enhanced correlation coefficient, that is the correlation coefficient between the template image and the final warped input image. When a $3\times 3$ matrix is given with motionType =0, 1 or 2, the third row is ignored. 
Unlike findHomography and estimateRigidTransform, the function findTransformECC implements an area-based alignment that builds on intensity similarities. In essence, the function updates the initial transformation that roughly aligns the images. If this information is missing, the identity warp (unity matrix) is used as an initialization. Note that if images undergo strong displacements/rotations, an initial transformation that roughly aligns the images is necessary (e.g., a simple euclidean/similarity transform that allows for the images showing the same image content approximately). Use inverse warping in the second image to take an image close to the first one, i.e. use the flag WARP_INVERSE_MAP with warpAffine or warpPerspective. See also the OpenCV sample image_alignment.cpp that demonstrates the use of the function. Note that the function throws an exception if algorithm does not converges. 
@sa computeECC, estimateAffine2D, estimateAffinePartial2D, findHomography 
findTransformECC(templateImage, inputImage, warpMatrix[, motionType[, criteria[, inputMask]]]) -> retval, warpMatrix @overload 


:param templateImage: single-channel template image; CV_8U or CV_32F array.
:type templateImage: cv2.typing.MatLike
:param inputImage: single-channel input image which should be warped with the final warpMatrix inorder to provide an image similar to templateImage, same type as templateImage. 
:type inputImage: cv2.typing.MatLike
:param warpMatrix: floating-point $2\times 3$ or $3\times 3$ mapping matrix (warp).
:type warpMatrix: cv2.typing.MatLike
:param motionType: parameter, specifying the type of motion:-   **MOTION_TRANSLATION** sets a translational motion model; warpMatrix is $2\times 3$ with the first $2\times 2$ part being the unity matrix and the rest two parameters being estimated. -   **MOTION_EUCLIDEAN** sets a Euclidean (rigid) transformation as motion model; three parameters are estimated; warpMatrix is $2\times 3$. -   **MOTION_AFFINE** sets an affine motion model (DEFAULT); six parameters are estimated; warpMatrix is $2\times 3$. -   **MOTION_HOMOGRAPHY** sets a homography as a motion model; eight parameters are estimated;\`warpMatrix\` is $3\times 3$. 
:type motionType: int
:param criteria: parameter, specifying the termination criteria of the ECC algorithm;criteria.epsilon defines the threshold of the increment in the correlation coefficient between two iterations (a negative criteria.epsilon makes criteria.maxcount the only termination criterion). Default values are shown in the declaration above. 
:type criteria: cv2.typing.TermCriteria
:param inputMask: An optional mask to indicate valid values of inputImage.
:type inputMask: cv2.typing.MatLike
:param gaussFiltSize: An optional value indicating size of gaussian blur filter; (DEFAULT: 5)
:type gaussFiltSize: int
:rtype: tuple[float, cv2.typing.MatLike]
````


````{py:function} fitEllipse(points) -> retval

Fits an ellipse around a set of 2D points.


The function calculates the ellipse that fits (in a least-squares sense) a set of 2D points best of all. It returns the rotated rectangle in which the ellipse is inscribed. The first algorithm described by @cite Fitzgibbon95 is used. Developer should keep in mind that it is possible that the returned ellipse/rotatedRect data contains negative indices, due to the data points being close to the border of the containing Mat element. 


:param points: Input 2D point set, stored in std::vector\<\> or Mat
:type points: cv2.typing.MatLike
:rtype: cv2.typing.RotatedRect
````


````{py:function} fitEllipseAMS(points) -> retval

Fits an ellipse around a set of 2D points.


The function calculates the ellipse that fits a set of 2D points. It returns the rotated rectangle in which the ellipse is inscribed. The Approximate Mean Square (AMS) proposed by @cite Taubin1991 is used. 
For an ellipse, this basis set is $ \chi= \left(x^2, x y, y^2, x, y, 1\right) $, which is a set of six free coefficients $ A^T=\left\{A_{\text{xx}},A_{\text{xy}},A_{\text{yy}},A_x,A_y,A_0\right\} $. However, to specify an ellipse, all that is needed is five numbers; the major and minor axes lengths $ (a,b) $, the position $ (x_0,y_0) $, and the orientation $ \theta $. This is because the basis set includes lines, quadratics, parabolic and hyperbolic functions as well as elliptical functions as possible fits. If the fit is found to be a parabolic or hyperbolic function then the standard #fitEllipse method is used. The AMS method restricts the fit to parabolic, hyperbolic and elliptical curves by imposing the condition that $ A^T ( D_x^T D_x  +   D_y^T D_y) A = 1 $ where the matrices $ Dx $ and $ Dy $ are the partial derivatives of the design matrix $ D $ with respect to x and y. The matrices are formed row by row applying the following to each of the points in the set: \f{align*}{ D(i,:)&=\left\{x_i^2, x_i y_i, y_i^2, x_i, y_i, 1\right\} & D_x(i,:)&=\left\{2 x_i,y_i,0,1,0,0\right\} & D_y(i,:)&=\left\{0,x_i,2 y_i,0,1,0\right\} \f} The AMS method minimizes the cost function \f{equation*}{ \epsilon ^2=\frac{ A^T D^T D A }{ A^T (D_x^T D_x +  D_y^T D_y) A^T } \f} 
The minimum cost is found by solving the generalized eigenvalue problem. 
\f{equation*}{ D^T D A = \lambda  \left( D_x^T D_x +  D_y^T D_y\right) A \f} 


:param points: Input 2D point set, stored in std::vector\<\> or Mat
:type points: cv2.typing.MatLike
:rtype: cv2.typing.RotatedRect
````


````{py:function} fitEllipseDirect(points) -> retval

Fits an ellipse around a set of 2D points.


The function calculates the ellipse that fits a set of 2D points. It returns the rotated rectangle in which the ellipse is inscribed. The Direct least square (Direct) method by @cite Fitzgibbon1999 is used. 
For an ellipse, this basis set is $ \chi= \left(x^2, x y, y^2, x, y, 1\right) $, which is a set of six free coefficients $ A^T=\left\{A_{\text{xx}},A_{\text{xy}},A_{\text{yy}},A_x,A_y,A_0\right\} $. However, to specify an ellipse, all that is needed is five numbers; the major and minor axes lengths $ (a,b) $, the position $ (x_0,y_0) $, and the orientation $ \theta $. This is because the basis set includes lines, quadratics, parabolic and hyperbolic functions as well as elliptical functions as possible fits. The Direct method confines the fit to ellipses by ensuring that $ 4 A_{xx} A_{yy}- A_{xy}^2 > 0 $. The condition imposed is that $ 4 A_{xx} A_{yy}- A_{xy}^2=1 $ which satisfies the inequality and as the coefficients can be arbitrarily scaled is not overly restrictive. 
\f{equation*}{ \epsilon ^2= A^T D^T D A \quad \text{with} \quad A^T C A =1 \quad \text{and} \quad C=\left(\begin{matrix} 0 & 0  & 2  & 0  & 0  &  0  \\ 0 & -1  & 0  & 0  & 0  &  0 \\ 2 & 0  & 0  & 0  & 0  &  0 \\ 0 & 0  & 0  & 0  & 0  &  0 \\ 0 & 0  & 0  & 0  & 0  &  0 \\ 0 & 0  & 0  & 0  & 0  &  0 \end{matrix} \right) \f} 
The minimum cost is found by solving the generalized eigenvalue problem. 
\f{equation*}{ D^T D A = \lambda  \left( C\right) A \f} 
The system produces only one positive eigenvalue $ \lambda$ which is chosen as the solution with its eigenvector $\mathbf{u}$. These are used to find the coefficients 
\f{equation*}{ A = \sqrt{\frac{1}{\mathbf{u}^T C \mathbf{u}}}  \mathbf{u} \f} The scaling factor guarantees that  $A^T C A =1$. 


:param points: Input 2D point set, stored in std::vector\<\> or Mat
:type points: cv2.typing.MatLike
:rtype: cv2.typing.RotatedRect
````


````{py:function} fitLine(points, distType, param, reps, aeps[, line]) -> line

Fits a line to a 2D or 3D point set.


The function fitLine fits a line to a 2D or 3D point set by minimizing $\sum_i \rho(r_i)$ where $r_i$ is a distance between the $i^{th}$ point, the line and $\rho(r)$ is a distance function, one of the following: -  DIST_L2 $\rho (r) = r^2/2  \quad \text{(the simplest and the fastest least-squares method)}$ - DIST_L1 $\rho (r) = r$ - DIST_L12 $\rho (r) = 2  \cdot ( \sqrt{1 + \frac{r^2}{2}} - 1)$ - DIST_FAIR $\rho \left (r \right ) = C^2  \cdot \left (  \frac{r}{C} -  \log{\left(1 + \frac{r}{C}\right)} \right )  \quad \text{where} \quad C=1.3998$ - DIST_WELSCH $\rho \left (r \right ) =  \frac{C^2}{2} \cdot \left ( 1 -  \exp{\left(-\left(\frac{r}{C}\right)^2\right)} \right )  \quad \text{where} \quad C=2.9846$ - DIST_HUBER $\rho (r) =  \fork{r^2/2}{if \(r < C\)}{C \cdot (r-C/2)}{otherwise} \quad \text{where} \quad C=1.345$ 
The algorithm is based on the M-estimator ( <http://en.wikipedia.org/wiki/M-estimator> ) technique that iteratively fits the line using the weighted least-squares algorithm. After each iteration the weights $w_i$ are adjusted to be inversely proportional to $\rho(r_i)$ . 


:param points: Input vector of 2D or 3D points, stored in std::vector\<\> or Mat.
:type points: cv2.typing.MatLike
:param line: Output line parameters. In case of 2D fitting, it should be a vector of 4 elements(like Vec4f) - (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line. In case of 3D fitting, it should be a vector of 6 elements (like Vec6f) - (vx, vy, vz, x0, y0, z0), where (vx, vy, vz) is a normalized vector collinear to the line and (x0, y0, z0) is a point on the line. 
:type line: cv2.typing.MatLike | None
:param distType: Distance used by the M-estimator, see #DistanceTypes
:type distType: int
:param param: Numerical parameter ( C ) for some types of distances. If it is 0, an optimal valueis chosen. 
:type param: float
:param reps: Sufficient accuracy for the radius (distance between the coordinate origin and the line).
:type reps: float
:param aeps: Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps.
:type aeps: float
:rtype: cv2.typing.MatLike
````


````{py:function} flip(src, flipCode[, dst]) -> dst

Flips a 2D array around vertical, horizontal, or both axes.


The function cv::flip flips the array in one of three different ways (row and column indices are 0-based): $\texttt{dst} _{ij} = \left\{ \begin{array}{l l} \texttt{src} _{\texttt{src.rows}-i-1,j} & if\;  \texttt{flipCode} = 0 \\ \texttt{src} _{i, \texttt{src.cols} -j-1} & if\;  \texttt{flipCode} > 0 \\ \texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1} & if\; \texttt{flipCode} < 0 \\ \end{array} \right.$ The example scenarios of using the function are the following: Vertical flipping of the image (flipCode == 0) to switch between top-left and bottom-left image origin. This is a typical operation in video processing on Microsoft Windows\* OS. Horizontal flipping of the image with the subsequent horizontal shift and absolute difference calculation to check for a vertical-axis symmetry (flipCode \> 0). Simultaneous horizontal and vertical flipping of the image with the subsequent shift and absolute difference calculation to check for a central symmetry (flipCode \< 0). Reversing the order of point arrays (flipCode \> 0 or flipCode == 0). 
**See also:** transpose , repeat , completeSymm


:param src: input array.
:type src: cv2.typing.MatLike
:param dst: output array of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param flipCode: a flag to specify how to flip the array; 0 meansflipping around the x-axis and positive value (for example, 1) means flipping around y-axis. Negative value (for example, -1) means flipping around both axes. 
:type flipCode: int
:rtype: cv2.typing.MatLike
````


````{py:function} flipND(src, axis[, dst]) -> dst

Flips a n-dimensional at given axis




:param src: input array
:type src: cv2.typing.MatLike
:param dst: output array that has the same shape of src
:type dst: cv2.typing.MatLike | None
:param axis: axis that performs a flip on. 0 <= axis < src.dims.
:type axis: int
:rtype: cv2.typing.MatLike
````


````{py:function} floodFill(image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]]) -> retval, image, mask, rect

Fills a connected component with the given color.


The function cv::floodFill fills a connected component starting from the seed point with the specified color. The connectivity is determined by the color/brightness closeness of the neighbor pixels. The pixel at $(x,y)$ is considered to belong to the repainted domain if: 
- in case of a grayscale image and floating range $\texttt{src} (x',y')- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} (x',y')+ \texttt{upDiff}$ 
- in case of a grayscale image and fixed range $\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)+ \texttt{upDiff}$ 
- in case of a color image and floating range $\texttt{src} (x',y')_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} (x',y')_r+ \texttt{upDiff} _r,$ $\texttt{src} (x',y')_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} (x',y')_g+ \texttt{upDiff} _g$ and $\texttt{src} (x',y')_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} (x',y')_b+ \texttt{upDiff} _b$ 
- in case of a color image and fixed range $\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r+ \texttt{upDiff} _r,$ $\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g+ \texttt{upDiff} _g$ and $\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b+ \texttt{upDiff} _b$ 
where $src(x',y')$ is the value of one of pixel neighbors that is already known to belong to the component. That is, to be added to the connected component, a color/brightness of the pixel should be close enough to: - Color/brightness of one of its neighbors that already belong to the connected component in case of a floating range. - Color/brightness of the seed point in case of a fixed range. 
Use these functions to either mark a connected component with the specified color in-place, or build a mask and then extract the contour, or copy the region to another image, and so on. 
```{note}
Since the mask is larger than the filled image, a pixel $(x, y)$ in image corresponds to thepixel $(x+1, y+1)$ in the mask . 
```
**See also:** findContours


:param image: Input/output 1- or 3-channel, 8-bit, or floating-point image. It is modified by thefunction unless the #FLOODFILL_MASK_ONLY flag is set in the second variant of the function. See the details below. 
:type image: cv2.typing.MatLike
:param mask: Operation mask that should be a single-channel 8-bit image, 2 pixels wider and 2 pixelstaller than image. If an empty Mat is passed it will be created automatically. Since this is both an input and output parameter, you must take responsibility of initializing it. Flood-filling cannot go across non-zero pixels in the input mask. For example, an edge detector output can be used as a mask to stop filling at edges. On output, pixels in the mask corresponding to filled pixels in the image are set to 1 or to the specified value in flags as described below. Additionally, the function fills the border of the mask with ones to simplify internal processing. It is therefore possible to use the same mask in multiple calls to the function to make sure the filled areas do not overlap. 
:type mask: cv2.typing.MatLike
:param seedPoint: Starting point.
:type seedPoint: cv2.typing.Point
:param newVal: New value of the repainted domain pixels.
:type newVal: cv2.typing.Scalar
:param loDiff: Maximal lower brightness/color difference between the currently observed pixel andone of its neighbors belonging to the component, or a seed pixel being added to the component. 
:type loDiff: cv2.typing.Scalar
:param upDiff: Maximal upper brightness/color difference between the currently observed pixel andone of its neighbors belonging to the component, or a seed pixel being added to the component. 
:type upDiff: cv2.typing.Scalar
:param rect: Optional output parameter set by the function to the minimum bounding rectangle of therepainted domain. 
:type rect: 
:param flags: Operation flags. The first 8 bits contain a connectivity value. The default value of4 means that only the four nearest neighbor pixels (those that share an edge) are considered. A connectivity value of 8 means that the eight nearest neighbor pixels (those that share a corner) will be considered. The next 8 bits (8-16) contain a value between 1 and 255 with which to fill the mask (the default value is 1). For example, 4 | ( 255 \<\< 8 ) will consider 4 nearest neighbours and fill the mask with a value of 255. The following additional options occupy higher bits and therefore may be further combined with the connectivity and mask fill values using bit-wise or (|), see #FloodFillFlags. 
:type flags: int
:rtype: tuple[int, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.Rect]
````


````{py:function} gemm(src1, src2, alpha, src3, beta[, dst[, flags]]) -> dst

Performs generalized matrix multiplication.


The function cv::gemm performs generalized matrix multiplication similar to the gemm functions in BLAS level 3. For example, `gemm(src1, src2, alpha, src3, beta, dst, GEMM_1_T + GEMM_3_T)` corresponds to $\texttt{dst} =  \texttt{alpha} \cdot \texttt{src1} ^T  \cdot \texttt{src2} +  \texttt{beta} \cdot \texttt{src3} ^T$ 
In case of complex (two-channel) data, performed a complex matrix multiplication. 
The function can be replaced with a matrix expression. For example, the above call can be replaced with: 
```cpp
dst = alpha*src1.t()*src2 + beta*src3.t();
```

**See also:** mulTransposed , transform


:param src1: first multiplied input matrix that could be real(CV_32FC1,CV_64FC1) or complex(CV_32FC2, CV_64FC2). 
:type src1: cv2.typing.MatLike
:param src2: second multiplied input matrix of the same type as src1.
:type src2: cv2.typing.MatLike
:param alpha: weight of the matrix product.
:type alpha: float
:param src3: third optional delta matrix added to the matrix product; itshould have the same type as src1 and src2. 
:type src3: cv2.typing.MatLike
:param beta: weight of src3.
:type beta: float
:param dst: output matrix; it has the proper size and the same type asinput matrices. 
:type dst: cv2.typing.MatLike | None
:param flags: operation flags (cv::GemmFlags)
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} getAffineTransform(src, dst) -> retval




@overload 


:param src: 
:type src: cv2.typing.MatLike
:param dst: 
:type dst: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````


````{py:function} getBuildInformation() -> retval

Returns full configuration time cmake output.


Returned value is raw cmake output including version control system revision, compiler version, compiler flags, enabled modules and third party libraries, etc. Output format depends on target architecture. 


:rtype: str
````


````{py:function} getCPUFeaturesLine() -> retval

Returns list of CPU features enabled during compilation.


Returned value is a string containing space separated list of CPU features with following markers: 
- no markers - baseline features - prefix `*` - features enabled in dispatcher - suffix `?` - features enabled but not available in HW 
Example: `SSE SSE2 SSE3 *SSE4.1 *SSE4.2 *FP16 *AVX *AVX2 *AVX512-SKX?` 


:rtype: str
````


````{py:function} getCPUTickCount() -> retval

Returns the number of CPU ticks.


The function returns the current number of CPU ticks on some architectures (such as x86, x64, PowerPC). On other platforms the function is equivalent to getTickCount. It can also be used for very accurate time measurements, as well as for RNG initialization. Note that in case of multi-CPU systems a thread, from which getCPUTickCount is called, can be suspended and resumed at another CPU with its own counter. So, theoretically (and practically) the subsequent calls to the function do not necessary return the monotonously increasing values. Also, since a modern CPU varies the CPU frequency depending on the load, the number of CPU clocks spent in some code cannot be directly converted to time units. Therefore, getTickCount is generally a preferable solution for measuring execution time. 


:rtype: int
````


````{py:function} getDefaultNewCameraMatrix(cameraMatrix[, imgsize[, centerPrincipalPoint]]) -> retval

Returns the default new camera matrix.


The function returns the camera matrix that is either an exact copy of the input cameraMatrix (when centerPrinicipalPoint=false ), or the modified one (when centerPrincipalPoint=true). 
In the latter case, the new camera matrix will be: 
$\begin{bmatrix} f_x && 0 && ( \texttt{imgSize.width} -1)*0.5  \\ 0 && f_y && ( \texttt{imgSize.height} -1)*0.5  \\ 0 && 0 && 1 \end{bmatrix} ,$ 
where $f_x$ and $f_y$ are $(0,0)$ and $(1,1)$ elements of cameraMatrix, respectively. 
By default, the undistortion functions in OpenCV (see #initUndistortRectifyMap, #undistort) do not move the principal point. However, when you work with stereo, it is important to move the principal points in both views to the same y-coordinate (which is required by most of stereo correspondence algorithms), and may be to the same x-coordinate too. So, you can form the new camera matrix for each view where the principal points are located at the center. 


:param cameraMatrix: Input camera matrix.
:type cameraMatrix: cv2.typing.MatLike
:param imgsize: Camera view image size in pixels.
:type imgsize: cv2.typing.Size
:param centerPrincipalPoint: Location of the principal point in the new camera matrix. Theparameter indicates whether this location should be at the image center or not. 
:type centerPrincipalPoint: bool
:rtype: cv2.typing.MatLike
````


````{py:function} getDerivKernels(dx, dy, ksize[, kx[, ky[, normalize[, ktype]]]]) -> kx, ky

Returns filter coefficients for computing spatial image derivatives.


The function computes and returns the filter coefficients for spatial image derivatives. When `ksize=FILTER_SCHARR`, the Scharr $3 \times 3$ kernels are generated (see #Scharr). Otherwise, Sobel kernels are generated (see #Sobel). The filters are normally passed to #sepFilter2D or to 


:param kx: Output matrix of row filter coefficients. It has the type ktype .
:type kx: cv2.typing.MatLike | None
:param ky: Output matrix of column filter coefficients. It has the type ktype .
:type ky: cv2.typing.MatLike | None
:param dx: Derivative order in respect of x.
:type dx: int
:param dy: Derivative order in respect of y.
:type dy: int
:param ksize: Aperture size. It can be FILTER_SCHARR, 1, 3, 5, or 7.
:type ksize: int
:param normalize: Flag indicating whether to normalize (scale down) the filter coefficients or not.Theoretically, the coefficients should have the denominator $=2^{ksize*2-dx-dy-2}$. If you are going to filter floating-point images, you are likely to use the normalized kernels. But if you compute derivatives of an 8-bit image, store the results in a 16-bit image, and wish to preserve all the fractional bits, you may want to set normalize=false . 
:type normalize: bool
:param ktype: Type of filter coefficients. It can be CV_32f or CV_64F .
:type ktype: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} getFontScaleFromHeight(fontFace, pixelHeight[, thickness]) -> retval

Calculates the font-specific size to use to achieve a given height in pixels.


**See also:** cv::putText


:param fontFace: Font to use, see cv::HersheyFonts.
:type fontFace: int
:param pixelHeight: Pixel height to compute the fontScale for
:type pixelHeight: int
:param thickness: Thickness of lines used to render the text.See putText for details.
:type thickness: int
:return: The fontSize to use for cv::putText
:rtype: float
````


````{py:function} getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]]) -> retval

Returns Gabor filter coefficients.


For more details about gabor filter equations and parameters, see: [Gabor Filter](http://en.wikipedia.org/wiki/Gabor_filter). 


:param ksize: Size of the filter returned.
:type ksize: cv2.typing.Size
:param sigma: Standard deviation of the gaussian envelope.
:type sigma: float
:param theta: Orientation of the normal to the parallel stripes of a Gabor function.
:type theta: float
:param lambd: Wavelength of the sinusoidal factor.
:type lambd: float
:param gamma: Spatial aspect ratio.
:type gamma: float
:param psi: Phase offset.
:type psi: float
:param ktype: Type of filter coefficients. It can be CV_32F or CV_64F .
:type ktype: int
:rtype: cv2.typing.MatLike
````


````{py:function} getGaussianKernel(ksize, sigma[, ktype]) -> retval

Returns Gaussian filter coefficients.


The function computes and returns the $\texttt{ksize} \times 1$ matrix of Gaussian filter coefficients: 
$G_i= \alpha *e^{-(i-( \texttt{ksize} -1)/2)^2/(2* \texttt{sigma}^2)},$ 
where $i=0..\texttt{ksize}-1$ and $\alpha$ is the scale factor chosen so that $\sum_i G_i=1$. 
Two of such generated kernels can be passed to sepFilter2D. Those functions automatically recognize smoothing kernels (a symmetrical kernel with sum of weights equal to 1) and handle them accordingly. You may also use the higher-level GaussianBlur. 
**See also:**  sepFilter2D, getDerivKernels, getStructuringElement, GaussianBlur


:param ksize: Aperture size. It should be odd ( $\texttt{ksize} \mod 2 = 1$ ) and positive.
:type ksize: int
:param sigma: Gaussian standard deviation. If it is non-positive, it is computed from ksize as`sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. 
:type sigma: float
:param ktype: Type of filter coefficients. It can be CV_32F or CV_64F .
:type ktype: int
:rtype: cv2.typing.MatLike
````


````{py:function} getHardwareFeatureName(feature) -> retval

Returns feature name by ID


Returns empty string if feature is not defined 


:param feature: 
:type feature: int
:rtype: str
````


````{py:function} getLogLevel() -> retval






:rtype: int
````


````{py:function} getNumThreads() -> retval

Returns the number of threads used by OpenCV for parallel regions.


Always returns 1 if OpenCV is built without threading support. 
The exact meaning of return value depends on the threading framework used by OpenCV library: - `TBB` - The number of threads, that OpenCV will try to use for parallel regions. If there is any tbb::thread_scheduler_init in user code conflicting with OpenCV, then function returns default number of threads used by TBB library. - `OpenMP` - An upper bound on the number of threads that could be used to form a new team. - `Concurrency` - The number of threads, that OpenCV will try to use for parallel regions. - `GCD` - Unsupported; returns the GCD thread pool limit (512) for compatibility. - `C=` - The number of threads, that OpenCV will try to use for parallel regions, if before called setNumThreads with threads \> 0, otherwise returns the number of logical CPUs, available for the process. 
**See also:** setNumThreads, getThreadNum


:rtype: int
````


````{py:function} getNumberOfCPUs() -> retval

Returns the number of logical CPUs available for the process.




:rtype: int
````


````{py:function} getOptimalDFTSize(vecsize) -> retval

Returns the optimal DFT size for a given vector size.


DFT performance is not a monotonic function of a vector size. Therefore, when you calculate convolution of two arrays or perform the spectral analysis of an array, it usually makes sense to pad the input data with zeros to get a bit larger array that can be transformed much faster than the original one. Arrays whose size is a power-of-two (2, 4, 8, 16, 32, ...) are the fastest to process. Though, the arrays whose size is a product of 2's, 3's, and 5's (for example, 300 = 5\*5\*3\*2\*2) are also processed quite efficiently. 
The function cv::getOptimalDFTSize returns the minimum number N that is greater than or equal to vecsize so that the DFT of a vector of size N can be processed efficiently. In the current implementation N = 2 ^p^ \* 3 ^q^ \* 5 ^r^ for some integer p, q, r. 
The function returns a negative number if vecsize is too large (very close to INT_MAX ). 
While the function cannot be used directly to estimate the optimal vector size for DCT transform (since the current DCT implementation supports only even-size vectors), it can be easily processed as getOptimalDFTSize((vecsize+1)/2)\*2. 
**See also:** dft , dct , idft , idct , mulSpectrums


:param vecsize: vector size.
:type vecsize: int
:rtype: int
````


````{py:function} getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha[, newImgSize[, centerPrincipalPoint]]) -> retval, validPixROI

Returns the new camera intrinsic matrix based on the free scaling parameter.


The function computes and returns the optimal new camera intrinsic matrix based on the free scaling parameter. By varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original image pixels if there is valuable information in the corners alpha=1 , or get something in between. When alpha\>0 , the undistorted result is likely to have some black pixels corresponding to "virtual" pixels outside of the captured distorted image. The original camera intrinsic matrix, distortion coefficients, the computed new camera intrinsic matrix, and newImageSize should be passed to #initUndistortRectifyMap to produce the maps for #remap . 


:param cameraMatrix: Input camera intrinsic matrix.
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$\distcoeffs$. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param imageSize: Original image size.
:type imageSize: cv2.typing.Size
:param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image arevalid) and 1 (when all the source image pixels are retained in the undistorted image). See #stereoRectify for details. 
:type alpha: float
:param newImgSize: Image size after rectification. By default, it is set to imageSize .
:type newImgSize: cv2.typing.Size
:param validPixROI: Optional output rectangle that outlines all-good-pixels region in theundistorted image. See roi1, roi2 description in #stereoRectify . 
:type validPixROI: 
:param centerPrincipalPoint: Optional flag that indicates whether in the new camera intrinsic matrix theprincipal point should be at the image center or not. By default, the principal point is chosen to best fit a subset of the source image (determined by alpha) to the corrected image. 
:type centerPrincipalPoint: bool
:return: new_camera_matrix Output new camera intrinsic matrix.
:rtype: tuple[cv2.typing.MatLike, cv2.typing.Rect]
````


````{py:function} getPerspectiveTransform(src, dst[, solveMethod]) -> retval

Calculates a perspective transform from four pairs of the corresponding points.


The function calculates the $3 \times 3$ matrix of a perspective transform so that: 
$\begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i \end{bmatrix} = \texttt{map_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$ 
where 
$dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2,3$ 
**See also:**  findHomography, warpPerspective, perspectiveTransform


:param src: Coordinates of quadrangle vertices in the source image.
:type src: cv2.typing.MatLike
:param dst: Coordinates of the corresponding quadrangle vertices in the destination image.
:type dst: cv2.typing.MatLike
:param solveMethod: method passed to cv::solve (#DecompTypes)
:type solveMethod: int
:rtype: cv2.typing.MatLike
````


````{py:function} getRectSubPix(image, patchSize, center[, patch[, patchType]]) -> patch

Retrieves a pixel rectangle from an image with sub-pixel accuracy.


The function getRectSubPix extracts pixels from src: 
$patch(x, y) = src(x +  \texttt{center.x} - ( \texttt{dst.cols} -1)*0.5, y +  \texttt{center.y} - ( \texttt{dst.rows} -1)*0.5)$ 
where the values of the pixels at non-integer coordinates are retrieved using bilinear interpolation. Every channel of multi-channel images is processed independently. Also the image should be a single channel or three channel image. While the center of the rectangle must be inside the image, parts of the rectangle may be outside. 
**See also:**  warpAffine, warpPerspective


:param image: Source image.
:type image: cv2.typing.MatLike
:param patchSize: Size of the extracted patch.
:type patchSize: cv2.typing.Size
:param center: Floating point coordinates of the center of the extracted rectangle within thesource image. The center must be inside the image. 
:type center: cv2.typing.Point2f
:param patch: Extracted patch that has the size patchSize and the same number of channels as src .
:type patch: cv2.typing.MatLike | None
:param patchType: Depth of the extracted pixels. By default, they have the same depth as src .
:type patchType: int
:rtype: cv2.typing.MatLike
````


````{py:function} getRotationMatrix2D(center, angle, scale) -> retval

Calculates an affine matrix of 2D rotation.


The function calculates the following matrix: 
$\begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot \texttt{center.x} -  \beta \cdot \texttt{center.y} \\ - \beta &  \alpha &  \beta \cdot \texttt{center.x} + (1- \alpha )  \cdot \texttt{center.y} \end{bmatrix}$ 
where 
$\begin{array}{l} \alpha =  \texttt{scale} \cdot \cos \texttt{angle} , \\ \beta =  \texttt{scale} \cdot \sin \texttt{angle} \end{array}$ 
The transformation maps the rotation center to itself. If this is not the target, adjust the shift. 
**See also:**  getAffineTransform, warpAffine, transform


:param center: Center of the rotation in the source image.
:type center: cv2.typing.Point2f
:param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (thecoordinate origin is assumed to be the top-left corner). 
:type angle: float
:param scale: Isotropic scale factor.
:type scale: float
:rtype: cv2.typing.MatLike
````


````{py:function} getStructuringElement(shape, ksize[, anchor]) -> retval

Returns a structuring element of the specified size and shape for morphological operations.


The function constructs and returns the structuring element that can be further passed to #erode, #dilate or #morphologyEx. But you can also construct an arbitrary binary mask yourself and use it as the structuring element. 


:param shape: Element shape that could be one of #MorphShapes
:type shape: int
:param ksize: Size of the structuring element.
:type ksize: cv2.typing.Size
:param anchor: Anchor position within the element. The default value $(-1, -1)$ means that theanchor is at the center. Note that only the shape of a cross-shaped element depends on the anchor position. In other cases the anchor just regulates how much the result of the morphological operation is shifted. 
:type anchor: cv2.typing.Point
:rtype: cv2.typing.MatLike
````


````{py:function} getTextSize(text, fontFace, fontScale, thickness) -> retval, baseLine

Calculates the width and height of a text string.


The function cv::getTextSize calculates and returns the size of a box that contains the specified text. That is, the following code renders some text, the tight box surrounding it, and the baseline: : 
```c++
String text = "Funny text inside the box";
int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
double fontScale = 2;
int thickness = 3;

Mat img(600, 800, CV_8UC3, Scalar::all(0));

int baseline=0;
Size textSize = getTextSize(text, fontFace,
fontScale, thickness, &baseline);
baseline += thickness;

// center the text
Point textOrg((img.cols - textSize.width)/2,
(img.rows + textSize.height)/2);

// draw the box
rectangle(img, textOrg + Point(0, baseline),
textOrg + Point(textSize.width, -textSize.height),
Scalar(0,0,255));
// ... and the baseline first
line(img, textOrg + Point(0, thickness),
textOrg + Point(textSize.width, thickness),
Scalar(0, 0, 255));

// then put the text itself
putText(img, text, textOrg, fontFace, fontScale,
Scalar::all(255), thickness, 8);
```

**See also:** putText


:param text: Input text string.
:type text: str
:param fontFace: Font to use, see #HersheyFonts.
:type fontFace: int
:param fontScale: Font scale factor that is multiplied by the font-specific base size.
:type fontScale: float
:param thickness: Thickness of lines used to render the text. See #putText for details.
:type thickness: int
:param baseLine: [out] y-coordinate of the baseline relative to the bottom-most textpoint. 
:type baseLine: 
:return: The size of a box that contains the specified text.
:rtype: tuple[cv2.typing.Size, int]
````


````{py:function} getThreadNum() -> retval

Returns the index of the currently executed thread within the current parallel region. Alwaysreturns 0 if called outside of parallel region. 


The exact meaning of the return value depends on the threading framework used by OpenCV library: - `TBB` - Unsupported with current 4.1 TBB release. Maybe will be supported in future. - `OpenMP` - The thread number, within the current team, of the calling thread. - `Concurrency` - An ID for the virtual processor that the current context is executing on (0 for master thread and unique number for others, but not necessary 1,2,3,...). - `GCD` - System calling thread's ID. Never returns 0 inside parallel region. - `C=` - The index of the current parallel task. 
```{deprecated} unknown
Current implementation doesn't corresponding to this documentation.
```
**See also:** setNumThreads, getNumThreads


:rtype: int
````


````{py:function} getTickCount() -> retval

Returns the number of ticks.


The function returns the number of ticks after the certain event (for example, when the machine was turned on). It can be used to initialize RNG or to measure a function execution time by reading the tick count before and after the function call. 
**See also:** getTickFrequency, TickMeter


:rtype: int
````


````{py:function} getTickFrequency() -> retval

Returns the number of ticks per second.


The function returns the number of ticks per second. That is, the following code computes the execution time in seconds: 
```c++
double t = (double)getTickCount();
// do something ...
t = ((double)getTickCount() - t)/getTickFrequency();
```

**See also:** getTickCount, TickMeter


:rtype: float
````


````{py:function} getTrackbarPos(trackbarname, winname) -> retval

Returns the trackbar position.


The function returns the current position of the specified trackbar. 
```{note}
[__Qt Backend Only__] winname can be empty if the trackbar is attached to the controlpanel. 
```


:param trackbarname: Name of the trackbar.
:type trackbarname: str
:param winname: Name of the window that is the parent of the trackbar.
:type winname: str
:rtype: int
````


````{py:function} getValidDisparityROI(roi1, roi2, minDisparity, numberOfDisparities, blockSize) -> retval






:param roi1: 
:type roi1: cv2.typing.Rect
:param roi2: 
:type roi2: cv2.typing.Rect
:param minDisparity: 
:type minDisparity: int
:param numberOfDisparities: 
:type numberOfDisparities: int
:param blockSize: 
:type blockSize: int
:rtype: cv2.typing.Rect
````


````{py:function} getVersionMajor() -> retval

Returns major library version




:rtype: int
````


````{py:function} getVersionMinor() -> retval

Returns minor library version




:rtype: int
````


````{py:function} getVersionRevision() -> retval

Returns revision field of the library version




:rtype: int
````


````{py:function} getVersionString() -> retval

Returns library version string


For example "3.4.1-dev". 
**See also:** getMajorVersion, getMinorVersion, getRevisionVersion


:rtype: str
````


````{py:function} getWindowImageRect(winname) -> retval

Provides rectangle of image in the window.


The function getWindowImageRect returns the client screen coordinates, width and height of the image rendering area. 
**See also:** resizeWindow moveWindow


:param winname: Name of the window.
:type winname: str
:rtype: cv2.typing.Rect
````


````{py:function} getWindowProperty(winname, prop_id) -> retval

Provides parameters of a window.


The function getWindowProperty returns properties of a window. 
**See also:** setWindowProperty


:param winname: Name of the window.
:type winname: str
:param prop_id: Window property to retrieve. The following operation flags are available: (cv::WindowPropertyFlags)
:type prop_id: int
:rtype: float
````


````{py:function} 






:rtype: object
````


````{py:function} goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) -> corners

Determines strong corners on an image.


The function finds the most prominent corners in the image or in the specified image region, as described in @cite Shi94 
-   Function calculates the corner quality measure at every source image pixel using the #cornerMinEigenVal or #cornerHarris . -   Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are retained). -   The corners with the minimal eigenvalue less than $\texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y)$ are rejected. -   The remaining corners are sorted by the quality measure in the descending order. -   Function throws away each corner for which there is a stronger corner at a distance less than maxDistance. 
The function can be used to initialize a point-based tracker of an object. 
goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize[, corners[, useHarrisDetector[, k]]]) -> corners 
```{note}
If the function is called with different values A and B of the parameter qualityLevel , andA \> B, the vector of returned corners with qualityLevel=A will be the prefix of the output vector with qualityLevel=B . 
```
**See also:**  cornerMinEigenVal, cornerHarris, calcOpticalFlowPyrLK, estimateRigidTransform,


:param image: Input 8-bit or floating-point 32-bit, single-channel image.
:type image: cv2.typing.MatLike
:param corners: Output vector of detected corners.
:type corners: cv2.typing.MatLike | None
:param maxCorners: Maximum number of corners to return. If there are more corners than are found,the strongest of them is returned. `maxCorners <= 0` implies that no limit on the maximum is set and all detected corners are returned. 
:type maxCorners: int
:param qualityLevel: Parameter characterizing the minimal accepted quality of image corners. Theparameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected. 
:type qualityLevel: float
:param minDistance: Minimum possible Euclidean distance between the returned corners.
:type minDistance: float
:param mask: Optional region of interest. If the image is not empty (it needs to have the typeCV_8UC1 and the same size as image ), it specifies the region in which the corners are detected. 
:type mask: cv2.typing.MatLike | None
:param blockSize: Size of an average block for computing a derivative covariation matrix over eachpixel neighborhood. See cornerEigenValsAndVecs . 
:type blockSize: int
:param useHarrisDetector: Parameter indicating whether to use a Harris detector (see #cornerHarris)or #cornerMinEigenVal. 
:type useHarrisDetector: bool
:param k: Free parameter of the Harris detector.
:type k: float
:rtype: cv2.typing.MatLike
````


````{py:function} goodFeaturesToTrackWithQuality(image, maxCorners, qualityLevel, minDistance, mask[, corners[, cornersQuality[, blockSize[, gradientSize[, useHarrisDetector[, k]]]]]]) -> corners, cornersQuality

Same as above, but returns also quality measure of the detected corners.




:param image: Input 8-bit or floating-point 32-bit, single-channel image.
:type image: cv2.typing.MatLike
:param corners: Output vector of detected corners.
:type corners: cv2.typing.MatLike | None
:param maxCorners: Maximum number of corners to return. If there are more corners than are found,the strongest of them is returned. `maxCorners <= 0` implies that no limit on the maximum is set and all detected corners are returned. 
:type maxCorners: int
:param qualityLevel: Parameter characterizing the minimal accepted quality of image corners. Theparameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected. 
:type qualityLevel: float
:param minDistance: Minimum possible Euclidean distance between the returned corners.
:type minDistance: float
:param mask: Region of interest. If the image is not empty (it needs to have the typeCV_8UC1 and the same size as image ), it specifies the region in which the corners are detected. 
:type mask: cv2.typing.MatLike
:param cornersQuality: Output vector of quality measure of the detected corners.
:type cornersQuality: cv2.typing.MatLike | None
:param blockSize: Size of an average block for computing a derivative covariation matrix over eachpixel neighborhood. See cornerEigenValsAndVecs . 
:type blockSize: int
:param gradientSize: Aperture parameter for the Sobel operator used for derivatives computation.See cornerEigenValsAndVecs . 
:type gradientSize: int
:param useHarrisDetector: Parameter indicating whether to use a Harris detector (see #cornerHarris)or #cornerMinEigenVal. 
:type useHarrisDetector: bool
:param k: Free parameter of the Harris detector.
:type k: float
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode]) -> mask, bgdModel, fgdModel

Runs the GrabCut algorithm.


The function implements the [GrabCut image segmentation algorithm](http://en.wikipedia.org/wiki/GrabCut). 


:param img: Input 8-bit 3-channel image.
:type img: cv2.typing.MatLike
:param mask: Input/output 8-bit single-channel mask. The mask is initialized by the function whenmode is set to #GC_INIT_WITH_RECT. Its elements may have one of the #GrabCutClasses. 
:type mask: cv2.typing.MatLike
:param rect: ROI containing a segmented object. The pixels outside of the ROI are marked as"obvious background". The parameter is only used when mode==#GC_INIT_WITH_RECT . 
:type rect: cv2.typing.Rect
:param bgdModel: Temporary array for the background model. Do not modify it while you areprocessing the same image. 
:type bgdModel: cv2.typing.MatLike
:param fgdModel: Temporary arrays for the foreground model. Do not modify it while you areprocessing the same image. 
:type fgdModel: cv2.typing.MatLike
:param iterCount: Number of iterations the algorithm should make before returning the result. Notethat the result can be refined with further calls with mode==#GC_INIT_WITH_MASK or mode==GC_EVAL . 
:type iterCount: int
:param mode: Operation mode that could be one of the #GrabCutModes
:type mode: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} groupRectangles(rectList, groupThreshold[, eps]) -> rectList, weights




@overload 


:param rectList: 
:type rectList: _typing.Sequence[cv2.typing.Rect]
:param groupThreshold: 
:type groupThreshold: int
:param eps: 
:type eps: float
:rtype: tuple[_typing.Sequence[cv2.typing.Rect], _typing.Sequence[int]]
````


````{py:function} hasNonZero(src) -> retval

Checks for the presence of at least one non-zero array element.


The function returns whether there are non-zero elements in src 
**See also:**  mean, meanStdDev, norm, minMaxLoc, calcCovarMatrix


:param src: single-channel array.
:type src: cv2.typing.MatLike
:rtype: bool
````


````{py:function} haveImageReader(filename) -> retval

Returns true if the specified image can be decoded by OpenCV




:param filename: File name of the image
:type filename: str
:rtype: bool
````


````{py:function} haveImageWriter(filename) -> retval

Returns true if an image with the specified filename can be encoded by OpenCV




:param filename: File name of the image
:type filename: str
:rtype: bool
````


````{py:function} haveOpenVX() -> retval






:rtype: bool
````


````{py:function} hconcat(src[, dst]) -> dst




@overload 
```cpp
std::vector<cv::Mat> matrices = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),
cv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),
cv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};

cv::Mat out;
cv::hconcat( matrices, out );
//out:
//[1, 2, 3;
// 1, 2, 3;
// 1, 2, 3;
// 1, 2, 3]
```



:param src: input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.
:type src: _typing.Sequence[cv2.typing.MatLike]
:param dst: output array. It has the same number of rows and depth as the src, and the sum of cols of the src.same depth. 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} idct(src[, dst[, flags]]) -> dst

Calculates the inverse Discrete Cosine Transform of a 1D or 2D array.


idct(src, dst, flags) is equivalent to dct(src, dst, flags | DCT_INVERSE). 
**See also:**  dct, dft, idft, getOptimalDFTSize


:param src: input floating-point single-channel array.
:type src: cv2.typing.MatLike
:param dst: output array of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param flags: operation flags.
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} idft(src[, dst[, flags[, nonzeroRows]]]) -> dst

Calculates the inverse Discrete Fourier Transform of a 1D or 2D array.


idft(src, dst, flags) is equivalent to dft(src, dst, flags | #DFT_INVERSE) . 
```{note}
None of dft and idft scales the result by default. So, you should pass #DFT_SCALE to one ofdft or idft explicitly to make these transforms mutually inverse. 
```
**See also:** dft, dct, idct, mulSpectrums, getOptimalDFTSize


:param src: input floating-point real or complex array.
:type src: cv2.typing.MatLike
:param dst: output array whose size and type depend on the flags.
:type dst: cv2.typing.MatLike | None
:param flags: operation flags (see dft and #DftFlags).
:type flags: int
:param nonzeroRows: number of dst rows to process; the rest of the rows have undefined content (seethe convolution sample in dft description. 
:type nonzeroRows: int
:rtype: cv2.typing.MatLike
````


````{py:function} illuminationChange(src, mask[, dst[, alpha[, beta]]]) -> dst

Applying an appropriate non-linear transformation to the gradient field inside the selection andthen integrating back with a Poisson solver, modifies locally the apparent illumination of an image. 


This is useful to highlight under-exposed foreground objects or to reduce specular reflections. 


:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param mask: Input 8-bit 1 or 3-channel image.
:type mask: cv2.typing.MatLike
:param dst: Output image with the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param alpha: Value ranges between 0-2.
:type alpha: float
:param beta: Value ranges between 0-2.
:type beta: float
:rtype: cv2.typing.MatLike
````


````{py:function} imcount(filename[, flags]) -> retval

Returns the number of images inside the give file


The function imcount will return the number of pages in a multi-page image, or 1 for single-page images 


:param filename: Name of file to be loaded.
:type filename: str
:param flags: Flag that can take values of cv::ImreadModes, default with cv::IMREAD_ANYCOLOR.
:type flags: int
:rtype: int
````


````{py:function} imdecode(buf, flags) -> retval

Reads an image from a buffer in memory.


The function imdecode reads an image from the specified buffer in the memory. If the buffer is too short or contains invalid data, the function returns an empty matrix ( Mat::data==NULL ). 
See cv::imread for the list of supported formats and flags description. 
```{note}
In the case of color images, the decoded images will have the channels stored in **B G R** order.
```


:param buf: Input array or vector of bytes.
:type buf: cv2.typing.MatLike
:param flags: The same flags as in cv::imread, see cv::ImreadModes.
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} imdecodemulti(buf, flags[, mats[, range]]) -> retval, mats

Reads a multi-page image from a buffer in memory.


The function imdecodemulti reads a multi-page image from the specified buffer in the memory. If the buffer is too short or contains invalid data, the function returns false. 
See cv::imreadmulti for the list of supported formats and flags description. 
```{note}
In the case of color images, the decoded images will have the channels stored in **B G R** order.
```


:param buf: Input array or vector of bytes.
:type buf: cv2.typing.MatLike
:param flags: The same flags as in cv::imread, see cv::ImreadModes.
:type flags: int
:param mats: A vector of Mat objects holding each page, if more than one.
:type mats: _typing.Sequence[cv2.typing.MatLike] | None
:param range: A continuous selection of pages.
:type range: cv2.typing.Range
:rtype: tuple[bool, _typing.Sequence[cv2.typing.MatLike]]
````


````{py:function} imencode(ext, img[, params]) -> retval, buf

Encodes an image into a memory buffer.


The function imencode compresses the image and stores it in the memory buffer that is resized to fit the result. See cv::imwrite for the list of supported formats and flags description. 


:param ext: File extension that defines the output format. Must include a leading period.
:type ext: str
:param img: Image to be written.
:type img: cv2.typing.MatLike
:param buf: Output buffer resized to fit the compressed image.
:type buf: 
:param params: Format-specific parameters. See cv::imwrite and cv::ImwriteFlags.
:type params: _typing.Sequence[int]
:rtype: tuple[bool, numpy.ndarray[_typing.Any, numpy.dtype[numpy.uint8]]]
````


````{py:function} imread(filename[, flags]) -> retval

Loads an image from a file.


@anchor imread 
The function imread loads an image from the specified file and returns it. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format), the function returns an empty matrix ( Mat::data==NULL ). 
Currently, the following file formats are supported: 
-   Windows bitmaps - \*.bmp, \*.dib (always supported) -   JPEG files - \*.jpeg, \*.jpg, \*.jpe (see the *Note* section) -   JPEG 2000 files - \*.jp2 (see the *Note* section) -   Portable Network Graphics - \*.png (see the *Note* section) -   WebP - \*.webp (see the *Note* section) -   AVIF - \*.avif (see the *Note* section) -   Portable image format - \*.pbm, \*.pgm, \*.ppm \*.pxm, \*.pnm (always supported) -   PFM files - \*.pfm (see the *Note* section) -   Sun rasters - \*.sr, \*.ras (always supported) -   TIFF files - \*.tiff, \*.tif (see the *Note* section) -   OpenEXR Image files - \*.exr (see the *Note* section) -   Radiance HDR - \*.hdr, \*.pic (always supported) -   Raster and Vector geospatial data supported by GDAL (see the *Note* section) 
@note -   The function determines the type of an image by the content, not by the file extension. -   In the case of color images, the decoded images will have the channels stored in **B G R** order. -   When using IMREAD_GRAYSCALE, the codec's internal grayscale conversion will be used, if available. Results may differ to the output of cvtColor() -   On Microsoft Windows\* OS and MacOSX\*, the codecs shipped with an OpenCV image (libjpeg, libpng, libtiff, and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs, and TIFFs. On MacOSX, there is also an option to use native MacOSX image readers. But beware that currently these native image loaders give images with different pixel values because of the color management embedded into MacOSX. -   On Linux\*, BSD flavors and other Unix-like open-source operating systems, OpenCV looks for codecs supplied with an OS image. Install the relevant packages (do not forget the development files, for example, "libjpeg-dev", in Debian\* and Ubuntu\*) to get the codec support or turn on the OPENCV_BUILD_3RDPARTY_LIBS flag in CMake. -   In the case you set *WITH_GDAL* flag to true in CMake and @ref IMREAD_LOAD_GDAL to load the image, then the [GDAL](http://www.gdal.org) driver will be used in order to decode the image, supporting the following formats: [Raster](http://www.gdal.org/formats_list.html), [Vector](http://www.gdal.org/ogr_formats.html). -   If EXIF information is embedded in the image file, the EXIF orientation will be taken into account and thus the image will be rotated accordingly except if the flags @ref IMREAD_IGNORE_ORIENTATION or @ref IMREAD_UNCHANGED are passed. -   Use the IMREAD_UNCHANGED flag to keep the floating point values from PFM image. -   By default number of pixels must be less than 2^30. Limit can be set using system variable OPENCV_IO_MAX_IMAGE_PIXELS 


:param filename: Name of file to be loaded.
:type filename: str
:param flags: Flag that can take values of cv::ImreadModes
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} imreadmulti(filename[, mats[, flags]]) -> retval, mats

Loads a of images of a multi-page image from a file.


The function imreadmulti loads a multi-page image from the specified file into a vector of Mat objects. 
imreadmulti(filename, start, count[, mats[, flags]]) -> retval, mats 
The function imreadmulti loads a specified range from a multi-page image from the specified file into a vector of Mat objects. 
**See also:** cv::imread
**See also:** cv::imread


:param filename: Name of file to be loaded.
:type filename: str
:param mats: A vector of Mat objects holding each page.
:type mats: _typing.Sequence[cv2.typing.MatLike] | None
:param flags: Flag that can take values of cv::ImreadModes, default with cv::IMREAD_ANYCOLOR.
:type flags: int
:param start: Start index of the image to load
:type start: 
:param count: Count number of images to load
:type count: 
:rtype: tuple[bool, _typing.Sequence[cv2.typing.MatLike]]
````


````{py:function} imshow(winname, mat) -> None

Displays an image in the specified window.


The function imshow displays an image in the specified window. If the window was created with the cv::WINDOW_AUTOSIZE flag, the image is shown with its original size, however it is still limited by the screen resolution. Otherwise, the image is scaled to fit the window. The function may scale the image, depending on its depth: 
-   If the image is 8-bit unsigned, it is displayed as is. -   If the image is 16-bit unsigned, the pixels are divided by 256. That is, the value range [0,255\*256] is mapped to [0,255]. -   If the image is 32-bit or 64-bit floating-point, the pixel values are multiplied by 255. That is, the value range [0,1] is mapped to [0,255]. -   32-bit integer images are not processed anymore due to ambiguouty of required transform. Convert to 8-bit unsigned matrix using a custom preprocessing specific to image's context. 
If window was created with OpenGL support, cv::imshow also support ogl::Buffer , ogl::Texture2D and cuda::GpuMat as input. 
If the window was not created before this function, it is assumed creating a window with cv::WINDOW_AUTOSIZE. 
If you need to show an image that is bigger than the screen resolution, you will need to call namedWindow("", WINDOW_NORMAL) before the imshow. 
```{note}
This function should be followed by a call to cv::waitKey or cv::pollKey to perform GUIhousekeeping tasks that are necessary to actually show the given image and make the window respond to mouse and keyboard events. Otherwise, it won't display the image and the window might lock up. For example, **waitKey(0)** will display the window infinitely until any keypress (it is suitable for image display). **waitKey(25)** will display a frame and wait approximately 25 ms for a key press (suitable for displaying a video frame-by-frame). To remove the window, use cv::destroyWindow. 
```
```{note}
[__Windows Backend Only__] Pressing Ctrl+C will copy the image to the clipboard. Pressing Ctrl+S will show a dialog to save the image.
```


:param winname: Name of the window.
:type winname: str
:param mat: Image to be shown.
:type mat: cv2.typing.MatLike
:rtype: None
````


````{py:function} imwrite(filename, img[, params]) -> retval

Saves an image to a specified file.


The function imwrite saves the image to the specified file. The image format is chosen based on the filename extension (see cv::imread for the list of extensions). In general, only 8-bit unsigned (CV_8U) single-channel or 3-channel (with 'BGR' channel order) images can be saved using this function, with these exceptions: 
- With OpenEXR encoder, only 32-bit float (CV_32F) images can be saved. - 8-bit unsigned (CV_8U) images are not supported. - With Radiance HDR encoder, non 64-bit float (CV_64F) images can be saved. - All images will be converted to 32-bit float (CV_32F). - With JPEG 2000 encoder, 8-bit unsigned (CV_8U) and 16-bit unsigned (CV_16U) images can be saved. - With PAM encoder, 8-bit unsigned (CV_8U) and 16-bit unsigned (CV_16U) images can be saved. - With PNG encoder, 8-bit unsigned (CV_8U) and 16-bit unsigned (CV_16U) images can be saved. - PNG images with an alpha channel can be saved using this function. To do this, create 8-bit (or 16-bit) 4-channel image BGRA, where the alpha channel goes last. Fully transparent pixels should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535 (see the code sample below). - With PGM/PPM encoder, 8-bit unsigned (CV_8U) and 16-bit unsigned (CV_16U) images can be saved. - With TIFF encoder, 8-bit unsigned (CV_8U), 16-bit unsigned (CV_16U), 32-bit float (CV_32F) and 64-bit float (CV_64F) images can be saved. - Multiple images (vector of Mat) can be saved in TIFF format (see the code sample below). - 32-bit float 3-channel (CV_32FC3) TIFF images will be saved using the LogLuv high dynamic range encoding (4 bytes per pixel) 
If the image format is not supported, the image will be converted to 8-bit unsigned (CV_8U) and saved that way. 
If the format, depth or channel order is different, use Mat::convertTo and cv::cvtColor to convert it before saving. Or, use the universal FileStorage I/O functions to save the image to XML or YAML format. 
The sample below shows how to create a BGRA image, how to set custom compression parameters and save it to a PNG file. It also demonstrates how to save multiple images in a TIFF file: @include snippets/imgcodecs_imwrite.cpp 


:param filename: Name of the file.
:type filename: str
:param img: (Mat or vector of Mat) Image or Images to be saved.
:type img: cv2.typing.MatLike
:param params: Format-specific parameters encoded as pairs (paramId_1, paramValue_1, paramId_2, paramValue_2, ... .) see cv::ImwriteFlags
:type params: _typing.Sequence[int]
:rtype: bool
````


````{py:function} imwritemulti(filename, img[, params]) -> retval






:param filename: 
:type filename: str
:param img: 
:type img: _typing.Sequence[cv2.typing.MatLike]
:param params: 
:type params: _typing.Sequence[int]
:rtype: bool
````


````{py:function} inRange(src, lowerb, upperb[, dst]) -> dst

 Checks if array elements lie between the elements of two other arrays.


The function checks the range as follows: -   For every element of a single-channel input array: $\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0$ -   For two-channel arrays: $\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0  \land \texttt{lowerb} (I)_1  \leq \texttt{src} (I)_1 \leq  \texttt{upperb} (I)_1$ -   and so forth. 
That is, dst (I) is set to 255 (all 1 -bits) if src (I) is within the specified 1D, 2D, 3D, ... box and 0 otherwise. 
When the lower and/or upper boundary parameters are scalars, the indexes (I) at lowerb and upperb in the above formulas should be omitted. 


:param src: first input array.
:type src: cv2.typing.MatLike
:param lowerb: inclusive lower boundary array or a scalar.
:type lowerb: cv2.typing.MatLike
:param upperb: inclusive upper boundary array or a scalar.
:type upperb: cv2.typing.MatLike
:param dst: output array of the same size as src and CV_8U type.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} initCameraMatrix2D(objectPoints, imagePoints, imageSize[, aspectRatio]) -> retval

Finds an initial camera intrinsic matrix from 3D-2D point correspondences.


The function estimates and returns an initial camera intrinsic matrix for the camera calibration process. Currently, the function only supports planar calibration patterns, which are patterns where each object point has z-coordinate =0. 


:param objectPoints: Vector of vectors of the calibration pattern points in the calibration patterncoordinate space. In the old interface all the per-view vectors are concatenated. See #calibrateCamera for details. 
:type objectPoints: _typing.Sequence[cv2.typing.MatLike]
:param imagePoints: Vector of vectors of the projections of the calibration pattern points. In theold interface all the per-view vectors are concatenated. 
:type imagePoints: _typing.Sequence[cv2.typing.MatLike]
:param imageSize: Image size in pixels used to initialize the principal point.
:type imageSize: cv2.typing.Size
:param aspectRatio: If it is zero or negative, both $f_x$ and $f_y$ are estimated independently.Otherwise, $f_x = f_y \cdot \texttt{aspectRatio}$ . 
:type aspectRatio: float
:rtype: cv2.typing.MatLike
````


````{py:function} initInverseRectificationMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]) -> map1, map2

Computes the projection and inverse-rectification transformation map. In essense, this is the inverse of#initUndistortRectifyMap to accomodate stereo-rectification of projectors ('inverse-cameras') in projector-camera pairs. 


The function computes the joint projection and inverse rectification transformation and represents the result in the form of maps for #remap. The projected image looks like a distorted version of the original which, once projected by a projector, should visually match the original. In case of a monocular camera, newCameraMatrix is usually equal to cameraMatrix, or it can be computed by #getOptimalNewCameraMatrix for a better control over scaling. In case of a projector-camera pair, newCameraMatrix is normally set to P1 or P2 computed by #stereoRectify . 
The projector is oriented differently in the coordinate space, according to R. In case of projector-camera pairs, this helps align the projector (in the same manner as #initUndistortRectifyMap for the camera) to create a stereo-rectified pair. This allows epipolar lines on both images to become horizontal and have the same y-coordinate (in case of a horizontally aligned projector-camera pair). 
The function builds the maps for the inverse mapping algorithm that is used by #remap. That is, for each pixel $(u, v)$ in the destination (projected and inverse-rectified) image, the function computes the corresponding coordinates in the source image (that is, in the original digital image). The following process is applied: 
$ \begin{array}{l} \text{newCameraMatrix}\\ x  \leftarrow (u - {c'}_x)/{f'}_x  \\ y  \leftarrow (v - {c'}_y)/{f'}_y  \\ 
\\\text{Undistortion} \\\scriptsize{\textit{though equation shown is for radial undistortion, function implements cv::undistortPoints()}}\\ r^2  \leftarrow x^2 + y^2 \\ \theta \leftarrow \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}\\ x' \leftarrow \frac{x}{\theta} \\ y'  \leftarrow \frac{y}{\theta} \\ 
\\\text{Rectification}\\ {[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\ x''  \leftarrow X/W  \\ y''  \leftarrow Y/W  \\ 
\\\text{cameraMatrix}\\ map_x(u,v)  \leftarrow x'' f_x + c_x  \\ map_y(u,v)  \leftarrow y'' f_y + c_y \end{array} $ where $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ are the distortion coefficients vector distCoeffs. 
In case of a stereo-rectified projector-camera pair, this function is called for the projector while #initUndistortRectifyMap is called for the camera head. This is done after #stereoRectify, which in turn is called after #stereoCalibrate. If the projector-camera pair is not calibrated, it is still possible to compute the rectification transformations directly from the fundamental matrix using #stereoRectifyUncalibrated. For the projector and camera, the function computes homography H as the rectification transformation in a pixel domain, not a rotation matrix R in 3D space. R can be computed from H as $\texttt{R} = \texttt{cameraMatrix} ^{-1} \cdot \texttt{H} \cdot \texttt{cameraMatrix}$ where cameraMatrix can be chosen arbitrarily. 


:param cameraMatrix: Input camera matrix $A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param R: Optional rectification transformation in the object space (3x3 matrix). R1 or R2,computed by #stereoRectify can be passed here. If the matrix is empty, the identity transformation is assumed. 
:type R: cv2.typing.MatLike
:param newCameraMatrix: New camera matrix $A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}$.
:type newCameraMatrix: cv2.typing.MatLike
:param size: Distorted image size.
:type size: cv2.typing.Size
:param m1type: Type of the first output map. Can be CV_32FC1, CV_32FC2 or CV_16SC2, see #convertMaps
:type m1type: int
:param map1: The first output map for #remap.
:type map1: cv2.typing.MatLike | None
:param map2: The second output map for #remap.
:type map2: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]) -> map1, map2

Computes the undistortion and rectification transformation map.


The function computes the joint undistortion and rectification transformation and represents the result in the form of maps for #remap. The undistorted image looks like original, as if it is captured with a camera using the camera matrix =newCameraMatrix and zero distortion. In case of a monocular camera, newCameraMatrix is usually equal to cameraMatrix, or it can be computed by #getOptimalNewCameraMatrix for a better control over scaling. In case of a stereo camera, newCameraMatrix is normally set to P1 or P2 computed by #stereoRectify . 
Also, this new camera is oriented differently in the coordinate space, according to R. That, for example, helps to align two heads of a stereo camera so that the epipolar lines on both images become horizontal and have the same y- coordinate (in case of a horizontally aligned stereo camera). 
The function actually builds the maps for the inverse mapping algorithm that is used by #remap. That is, for each pixel $(u, v)$ in the destination (corrected and rectified) image, the function computes the corresponding coordinates in the source image (that is, in the original image from camera). The following process is applied: $ \begin{array}{l} x  \leftarrow (u - {c'}_x)/{f'}_x  \\ y  \leftarrow (v - {c'}_y)/{f'}_y  \\ {[X\,Y\,W]} ^T  \leftarrow R^{-1}*[x \, y \, 1]^T  \\ x'  \leftarrow X/W  \\ y'  \leftarrow Y/W  \\ r^2  \leftarrow x'^2 + y'^2 \\ x''  \leftarrow x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2p_1 x' y' + p_2(r^2 + 2 x'^2)  + s_1 r^2 + s_2 r^4\\ y''  \leftarrow y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\ s\vecthree{x'''}{y'''}{1} = \vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}((\tau_x, \tau_y)} {0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)} {0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\\ map_x(u,v)  \leftarrow x''' f_x + c_x  \\ map_y(u,v)  \leftarrow y''' f_y + c_y \end{array} $ where $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ are the distortion coefficients. 
In case of a stereo camera, this function is called twice: once for each camera head, after #stereoRectify, which in its turn is called after #stereoCalibrate. But if the stereo camera was not calibrated, it is still possible to compute the rectification transformations directly from the fundamental matrix using #stereoRectifyUncalibrated. For each camera, the function computes homography H as the rectification transformation in a pixel domain, not a rotation matrix R in 3D space. R can be computed from H as $\texttt{R} = \texttt{cameraMatrix} ^{-1} \cdot \texttt{H} \cdot \texttt{cameraMatrix}$ where cameraMatrix can be chosen arbitrarily. 


:param cameraMatrix: Input camera matrix $A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param R: Optional rectification transformation in the object space (3x3 matrix). R1 or R2 ,computed by #stereoRectify can be passed here. If the matrix is empty, the identity transformation is assumed. In #initUndistortRectifyMap R assumed to be an identity matrix. 
:type R: cv2.typing.MatLike
:param newCameraMatrix: New camera matrix $A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}$.
:type newCameraMatrix: cv2.typing.MatLike
:param size: Undistorted image size.
:type size: cv2.typing.Size
:param m1type: Type of the first output map that can be CV_32FC1, CV_32FC2 or CV_16SC2, see #convertMaps
:type m1type: int
:param map1: The first output map.
:type map1: cv2.typing.MatLike | None
:param map2: The second output map.
:type map2: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} inpaint(src, inpaintMask, inpaintRadius, flags[, dst]) -> dst

Restores the selected region in an image using the region neighborhood.


The function reconstructs the selected image area from the pixel near the area boundary. The function may be used to remove dust and scratches from a scanned photo, or to remove undesirable objects from still images or video. See <http://en.wikipedia.org/wiki/Inpainting> for more details. 
@note -   An example using the inpainting technique can be found at opencv_source_code/samples/cpp/inpaint.cpp -   (Python) An example using the inpainting technique can be found at opencv_source_code/samples/python/inpaint.py 


:param src: Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param inpaintMask: Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area thatneeds to be inpainted. 
:type inpaintMask: cv2.typing.MatLike
:param dst: Output image with the same size and type as src .
:type dst: cv2.typing.MatLike | None
:param inpaintRadius: Radius of a circular neighborhood of each point inpainted that is consideredby the algorithm. 
:type inpaintRadius: float
:param flags: Inpainting method that could be cv::INPAINT_NS or cv::INPAINT_TELEA
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} insertChannel(src, dst, coi) -> dst

Inserts a single channel to dst (coi is 0-based index)


**See also:** mixChannels, merge


:param src: input array
:type src: cv2.typing.MatLike
:param dst: output array
:type dst: cv2.typing.MatLike
:param coi: index of channel for insertion
:type coi: int
:rtype: cv2.typing.MatLike
````


````{py:function} integral(src[, sum[, sdepth]]) -> sum




@overload 


:param src: 
:type src: cv2.typing.MatLike
:param sum: 
:type sum: cv2.typing.MatLike | None
:param sdepth: 
:type sdepth: int
:rtype: cv2.typing.MatLike
````


````{py:function} integral2(src[, sum[, sqsum[, sdepth[, sqdepth]]]]) -> sum, sqsum




@overload 


:param src: 
:type src: cv2.typing.MatLike
:param sum: 
:type sum: cv2.typing.MatLike | None
:param sqsum: 
:type sqsum: cv2.typing.MatLike | None
:param sdepth: 
:type sdepth: int
:param sqdepth: 
:type sqdepth: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} integral3(src[, sum[, sqsum[, tilted[, sdepth[, sqdepth]]]]]) -> sum, sqsum, tilted

Calculates the integral of an image.


The function calculates one or more integral images for the source image as follows: 
$\texttt{sum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)$ 
$\texttt{sqsum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)^2$ 
$\texttt{tilted} (X,Y) =  \sum _{y<Y,abs(x-X+1) \leq Y-y-1}  \texttt{image} (x,y)$ 
Using these integral images, you can calculate sum, mean, and standard deviation over a specific up-right or rotated rectangular region of the image in a constant time, for example: 
$\sum _{x_1 \leq x < x_2,  \, y_1  \leq y < y_2}  \texttt{image} (x,y) =  \texttt{sum} (x_2,y_2)- \texttt{sum} (x_1,y_2)- \texttt{sum} (x_2,y_1)+ \texttt{sum} (x_1,y_1)$ 
It makes possible to do a fast blurring or fast block correlation with a variable window size, for example. In case of multi-channel images, sums for each channel are accumulated independently. 
As a practical example, the next figure shows the calculation of the integral of a straight rectangle Rect(4,4,3,2) and of a tilted rectangle Rect(5,1,2,3) . The selected pixels in the original image are shown, as well as the relative pixels in the integral images sum and tilted . 
![integral calculation example](pics/integral.png) 


:param src: input image as $W \times H$, 8-bit or floating-point (32f or 64f).
:type src: cv2.typing.MatLike
:param sum: integral image as $(W+1)\times (H+1)$ , 32-bit integer or floating-point (32f or 64f).
:type sum: cv2.typing.MatLike | None
:param sqsum: integral image for squared pixel values; it is $(W+1)\times (H+1)$, double-precisionfloating-point (64f) array. 
:type sqsum: cv2.typing.MatLike | None
:param tilted: integral for the image rotated by 45 degrees; it is $(W+1)\times (H+1)$ array withthe same data type as sum. 
:type tilted: cv2.typing.MatLike | None
:param sdepth: desired depth of the integral and the tilted integral images, CV_32S, CV_32F, orCV_64F. 
:type sdepth: int
:param sqdepth: desired depth of the integral image of squared pixel values, CV_32F or CV_64F.
:type sqdepth: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} intersectConvexConvex(p1, p2[, p12[, handleNested]]) -> retval, p12

Finds intersection of two convex polygons


```{note}
intersectConvexConvex doesn't confirm that both polygons are convex and will return invalid results if they aren't.
```


:param p1: First polygon
:type p1: cv2.typing.MatLike
:param p2: Second polygon
:type p2: cv2.typing.MatLike
:param p12: Output polygon describing the intersecting area
:type p12: cv2.typing.MatLike | None
:param handleNested: When true, an intersection is found if one of the polygons is fully enclosed in the other.When false, no intersection is found. If the polygons share a side or the vertex of one polygon lies on an edge of the other, they are not considered nested and an intersection will be found regardless of the value of handleNested. 
:type handleNested: bool
:return: Absolute value of area of intersecting polygon
:rtype: tuple[float, cv2.typing.MatLike]
````


````{py:function} invert(src[, dst[, flags]]) -> retval, dst

Finds the inverse or pseudo-inverse of a matrix.


The function cv::invert inverts the matrix src and stores the result in dst When the matrix src is singular or non-square, the function calculates the pseudo-inverse matrix (the dst matrix) so that norm(src\*dst - I) is minimal, where I is an identity matrix. 
In case of the #DECOMP_LU method, the function returns non-zero value if the inverse has been successfully calculated and 0 if src is singular. 
In case of the #DECOMP_SVD method, the function returns the inverse condition number of src (the ratio of the smallest singular value to the largest singular value) and 0 if src is singular. The SVD method calculates a pseudo-inverse matrix if src is singular. 
Similarly to #DECOMP_LU, the method #DECOMP_CHOLESKY works only with non-singular square matrices that should also be symmetrical and positively defined. In this case, the function stores the inverted matrix in dst and returns non-zero. Otherwise, it returns 0. 
**See also:** solve, SVD


:param src: input floating-point M x N matrix.
:type src: cv2.typing.MatLike
:param dst: output matrix of N x M size and the same type as src.
:type dst: cv2.typing.MatLike | None
:param flags: inversion method (cv::DecompTypes)
:type flags: int
:rtype: tuple[float, cv2.typing.MatLike]
````


````{py:function} invertAffineTransform(M[, iM]) -> iM

Inverts an affine transformation.


The function computes an inverse affine transformation represented by $2 \times 3$ matrix M: 
$\begin{bmatrix} a_{11} & a_{12} & b_1  \\ a_{21} & a_{22} & b_2 \end{bmatrix}$ 
The result is also a $2 \times 3$ matrix of the same type as M. 


:param M: Original affine transformation.
:type M: cv2.typing.MatLike
:param iM: Output reverse affine transformation.
:type iM: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} isContourConvex(contour) -> retval

Tests a contour convexity.


The function tests whether the input contour is convex or not. The contour must be simple, that is, without self-intersections. Otherwise, the function output is undefined. 


:param contour: Input vector of 2D points, stored in std::vector\<\> or Mat
:type contour: cv2.typing.MatLike
:rtype: bool
````


````{py:function} kmeans(data, K, bestLabels, criteria, attempts, flags[, centers]) -> retval, bestLabels, centers

Finds centers of clusters and groups input samples around the clusters.


The function kmeans implements a k-means algorithm that finds the centers of cluster_count clusters and groups the input samples around the clusters. As an output, $\texttt{bestLabels}_i$ contains a 0-based cluster index for the sample stored in the $i^{th}$ row of the samples matrix. 
@note -   (Python) An example on K-means clustering can be found at opencv_source_code/samples/python/kmeans.py 


:param data: Data for clustering. An array of N-Dimensional points with float coordinates is needed.Examples of this array can be: -   Mat points(count, 2, CV_32F); -   Mat points(count, 1, CV_32FC2); -   Mat points(1, count, CV_32FC2); -   std::vector\<cv::Point2f\> points(sampleCount); 
:type data: cv2.typing.MatLike
:param K: Number of clusters to split the set by.
:type K: int
:param bestLabels: Input/output integer array that stores the cluster indices for every sample.
:type bestLabels: cv2.typing.MatLike
:param criteria: The algorithm termination criteria, that is, the maximum number of iterations and/orthe desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster centers moves by less than criteria.epsilon on some iteration, the algorithm stops. 
:type criteria: cv2.typing.TermCriteria
:param attempts: Flag to specify the number of times the algorithm is executed using differentinitial labellings. The algorithm returns the labels that yield the best compactness (see the last function parameter). 
:type attempts: int
:param flags: Flag that can take values of cv::KmeansFlags
:type flags: int
:param centers: Output matrix of the cluster centers, one row per each cluster center.
:type centers: cv2.typing.MatLike | None
:return: The function returns the compactness measure that is computed as$\sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2$ after every attempt. The best (minimum) value is chosen and the corresponding labels and the compactness value are returned by the function. Basically, you can use only the core of the function, set the number of attempts to 1, initialize labels each time using a custom algorithm, pass them with the ( flags = #KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best (most-compact) clustering. 
:rtype: tuple[float, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img

Draws a line segment connecting two points.


The function line draws the line segment between pt1 and pt2 points in the image. The line is clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased lines are drawn using Gaussian filtering. 


:param img: Image.
:type img: cv2.typing.MatLike
:param pt1: First point of the line segment.
:type pt1: cv2.typing.Point
:param pt2: Second point of the line segment.
:type pt2: cv2.typing.Point
:param color: Line color.
:type color: cv2.typing.Scalar
:param thickness: Line thickness.
:type thickness: int
:param lineType: Type of the line. See #LineTypes.
:type lineType: int
:param shift: Number of fractional bits in the point coordinates.
:type shift: int
:rtype: cv2.typing.MatLike
````


````{py:function} linearPolar(src, center, maxRadius, flags[, dst]) -> dst

Remaps an image to polar coordinates space.


@internal Transform the source image using the following transformation (See @ref polar_remaps_reference_image "Polar remaps reference image c)"): $\begin{array}{l} dst( \rho , \phi ) = src(x,y) \\ dst.size() \leftarrow src.size() \end{array}$ 
where $\begin{array}{l} I = (dx,dy) = (x - center.x,y - center.y) \\ \rho = Kmag \cdot \texttt{magnitude} (I) ,\\ \phi = angle \cdot \texttt{angle} (I) \end{array}$ 
and $\begin{array}{l} Kx = src.cols / maxRadius \\ Ky = src.rows / 2\Pi \end{array}$ 
@note -   The function can not operate in-place. -   To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees. 
```{deprecated} unknown
This function produces same result as cv::warpPolar(src, dst, src.size(), center, maxRadius, flags)
```
**See also:** cv::logPolar@endinternal 


:param src: Source image
:type src: cv2.typing.MatLike
:param dst: Destination image. It will have same size and type as src.
:type dst: cv2.typing.MatLike | None
:param center: The transformation center;
:type center: cv2.typing.Point2f
:param maxRadius: The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.
:type maxRadius: float
:param flags: A combination of interpolation methods, see #InterpolationFlags
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} log(src[, dst]) -> dst

Calculates the natural logarithm of every array element.


The function cv::log calculates the natural logarithm of every element of the input array: $\texttt{dst} (I) =  \log (\texttt{src}(I)) $ 
Output on zero, negative and special (NaN, Inf) values is undefined. 
**See also:** exp, cartToPolar, polarToCart, phase, pow, sqrt, magnitude


:param src: input array.
:type src: cv2.typing.MatLike
:param dst: output array of the same size and type as src .
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} logPolar(src, center, M, flags[, dst]) -> dst

Remaps an image to semilog-polar coordinates space.


@internal Transform the source image using the following transformation (See @ref polar_remaps_reference_image "Polar remaps reference image d)"): $\begin{array}{l} dst( \rho , \phi ) = src(x,y) \\ dst.size() \leftarrow src.size() \end{array}$ 
where $\begin{array}{l} I = (dx,dy) = (x - center.x,y - center.y) \\ \rho = M \cdot log_e(\texttt{magnitude} (I)) ,\\ \phi = Kangle \cdot \texttt{angle} (I) \\ \end{array}$ 
and $\begin{array}{l} M = src.cols / log_e(maxRadius) \\ Kangle = src.rows / 2\Pi \\ \end{array}$ 
The function emulates the human "foveal" vision and can be used for fast scale and rotation-invariant template matching, for object tracking and so forth. 
@note -   The function can not operate in-place. -   To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees. 
```{deprecated} unknown
This function produces same result as cv::warpPolar(src, dst, src.size(), center, maxRadius, flags+WARP_POLAR_LOG);
```
**See also:** cv::linearPolar@endinternal 


:param src: Source image
:type src: cv2.typing.MatLike
:param dst: Destination image. It will have same size and type as src.
:type dst: cv2.typing.MatLike | None
:param center: The transformation center; where the output precision is maximal
:type center: cv2.typing.Point2f
:param M: Magnitude scale parameter. It determines the radius of the bounding circle to transform too.
:type M: float
:param flags: A combination of interpolation methods, see #InterpolationFlags
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} magnitude(x, y[, magnitude]) -> magnitude

Calculates the magnitude of 2D vectors.


The function cv::magnitude calculates the magnitude of 2D vectors formed from the corresponding elements of x and y arrays: $\texttt{dst} (I) =  \sqrt{\texttt{x}(I)^2 + \texttt{y}(I)^2}$ 
**See also:** cartToPolar, polarToCart, phase, sqrt


:param x: floating-point array of x-coordinates of the vectors.
:type x: cv2.typing.MatLike
:param y: floating-point array of y-coordinates of the vectors; it musthave the same size as x. 
:type y: cv2.typing.MatLike
:param magnitude: output array of the same size and type as x.
:type magnitude: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} matMulDeriv(A, B[, dABdA[, dABdB]]) -> dABdA, dABdB

Computes partial derivatives of the matrix product for each multiplied matrix.


The function computes partial derivatives of the elements of the matrix product $A*B$ with regard to the elements of each of the two input matrices. The function is used to compute the Jacobian matrices in #stereoCalibrate but can also be used in any other similar optimization function. 


:param A: First multiplied matrix.
:type A: cv2.typing.MatLike
:param B: Second multiplied matrix.
:type B: cv2.typing.MatLike
:param dABdA: First output derivative matrix d(A\*B)/dA of size$\texttt{A.rows*B.cols} \times {A.rows*A.cols}$ . 
:type dABdA: cv2.typing.MatLike | None
:param dABdB: Second output derivative matrix d(A\*B)/dB of size$\texttt{A.rows*B.cols} \times {B.rows*B.cols}$ . 
:type dABdB: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} matchShapes(contour1, contour2, method, parameter) -> retval

Compares two shapes.


The function compares two shapes. All three implemented methods use the Hu invariants (see #HuMoments) 


:param contour1: First contour or grayscale image.
:type contour1: cv2.typing.MatLike
:param contour2: Second contour or grayscale image.
:type contour2: cv2.typing.MatLike
:param method: Comparison method, see #ShapeMatchModes
:type method: int
:param parameter: Method-specific parameter (not supported now).
:type parameter: float
:rtype: float
````


````{py:function} matchTemplate(image, templ, method[, result[, mask]]) -> result

Compares a template against overlapped image regions.


The function slides through image , compares the overlapped patches of size $w \times h$ against templ using the specified method and stores the comparison results in result . #TemplateMatchModes describes the formulae for the available comparison methods ( $I$ denotes image, $T$ template, $R$ result, $M$ the optional mask ). The summation is done over template and/or the image patch: $x' = 0...w-1, y' = 0...h-1$ 
After the function finishes the comparison, the best matches can be found as global minimums (when #TM_SQDIFF was used) or maximums (when #TM_CCORR or #TM_CCOEFF was used) using the #minMaxLoc function. In case of a color image, template summation in the numerator and each sum in the denominator is done over all of the channels and separate mean values are used for each channel. That is, the function can take a color template and a color image. The result will still be a single-channel image, which is easier to analyze. 


:param image: Image where the search is running. It must be 8-bit or 32-bit floating-point.
:type image: cv2.typing.MatLike
:param templ: Searched template. It must be not greater than the source image and have the samedata type. 
:type templ: cv2.typing.MatLike
:param result: Map of comparison results. It must be single-channel 32-bit floating-point. If imageis $W \times H$ and templ is $w \times h$ , then result is $(W-w+1) \times (H-h+1)$ . 
:type result: cv2.typing.MatLike | None
:param method: Parameter specifying the comparison method, see #TemplateMatchModes
:type method: int
:param mask: Optional mask. It must have the same size as templ. It must either have the same numberof channels as template or only one channel, which is then used for all template and image channels. If the data type is #CV_8U, the mask is interpreted as a binary mask, meaning only elements where mask is nonzero are used and are kept unchanged independent of the actual mask value (weight equals 1). For data tpye #CV_32F, the mask values are used as weights. The exact formulas are documented in #TemplateMatchModes. 
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} max(src1, src2[, dst]) -> dst

Calculates per-element maximum of two arrays or an array and a scalar.


The function cv::max calculates the per-element maximum of two arrays: $\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))$ or array and a scalar: $\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{value} )$ 
**See also:**  min, compare, inRange, minMaxLoc, @ref MatrixExpressions


:param src1: first input array.
:type src1: cv2.typing.MatLike
:param src2: second input array of the same size and type as src1 .
:type src2: cv2.typing.MatLike
:param dst: output array of the same size and type as src1.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} mean(src[, mask]) -> retval

Calculates an average (mean) of array elements.


The function cv::mean calculates the mean value M of array elements, independently for each channel, and return it: $\begin{array}{l} N =  \sum _{I: \; \texttt{mask} (I) \ne 0} 1 \\ M_c =  \left ( \sum _{I: \; \texttt{mask} (I) \ne 0}{ \texttt{mtx} (I)_c} \right )/N \end{array}$ When all the mask elements are 0's, the function returns Scalar::all(0) 
**See also:**  countNonZero, meanStdDev, norm, minMaxLoc


:param src: input array that should have from 1 to 4 channels so that the result can be stored inScalar_ . 
:type src: cv2.typing.MatLike
:param mask: optional operation mask.
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.Scalar
````


````{py:function} meanShift(probImage, window, criteria) -> retval, window

Finds an object on a back projection image.




:param probImage: Back projection of the object histogram. See calcBackProject for details.
:type probImage: cv2.typing.MatLike
:param window: Initial search window.
:type window: cv2.typing.Rect
:param criteria: Stop criteria for the iterative search algorithm.returns :   Number of iterations CAMSHIFT took to converge. The function implements the iterative object search algorithm. It takes the input back projection of an object and the initial position. The mass center in window of the back projection image is computed and the search window center shifts to the mass center. The procedure is repeated until the specified number of iterations criteria.maxCount is done or until the window center shifts by less than criteria.epsilon. The algorithm is used inside CamShift and, unlike CamShift , the search window size or orientation do not change during the search. You can simply pass the output of calcBackProject to this function. But better results can be obtained if you pre-filter the back projection and remove the noise. For example, you can do this by retrieving connected components with findContours , throwing away contours with small area ( contourArea ), and rendering the remaining contours with drawContours. 
:type criteria: cv2.typing.TermCriteria
:rtype: tuple[int, cv2.typing.Rect]
````


````{py:function} meanStdDev(src[, mean[, stddev[, mask]]]) -> mean, stddev




Calculates a mean and standard deviation of array elements. 
The function cv::meanStdDev calculates the mean and the standard deviation M of array elements independently for each channel and returns it via the output parameters: $\begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2}{N}} \end{array}$ When all the mask elements are 0's, the function returns mean=stddev=Scalar::all(0). 
```{note}
The calculated standard deviation is only the diagonal of thecomplete normalized covariance matrix. If the full matrix is needed, you can reshape the multi-channel array M x N to the single-channel array M\*N x mtx.channels() (only possible when the matrix is continuous) and then pass the matrix to calcCovarMatrix . 
```
**See also:**  countNonZero, mean, norm, minMaxLoc, calcCovarMatrix


:param src: input array that should have from 1 to 4 channels so that the results can be stored inScalar_ 's. 
:type src: cv2.typing.MatLike
:param mean: output parameter: calculated mean value.
:type mean: cv2.typing.MatLike | None
:param stddev: output parameter: calculated standard deviation.
:type stddev: cv2.typing.MatLike | None
:param mask: optional operation mask.
:type mask: cv2.typing.MatLike | None
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} medianBlur(src, ksize[, dst]) -> dst

Blurs an image using the median filter.


The function smoothes an image using the median filter with the $\texttt{ksize} \times \texttt{ksize}$ aperture. Each channel of a multi-channel image is processed independently. In-place operation is supported. 
```{note}
The median filter uses #BORDER_REPLICATE internally to cope with border pixels, see #BorderTypes
```
**See also:**  bilateralFilter, blur, boxFilter, GaussianBlur


:param src: input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should beCV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U. 
:type src: cv2.typing.MatLike
:param dst: destination array of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param ksize: aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
:type ksize: int
:rtype: cv2.typing.MatLike
````


````{py:function} merge(mv[, dst]) -> dst




@overload 


:param mv: input vector of matrices to be merged; all the matrices in mv must have the samesize and the same depth. 
:type mv: _typing.Sequence[cv2.typing.MatLike]
:param dst: output array of the same size and the same depth as mv[0]; The number of channels willbe the total number of channels in the matrix array. 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} min(src1, src2[, dst]) -> dst

Calculates per-element minimum of two arrays or an array and a scalar.


The function cv::min calculates the per-element minimum of two arrays: $\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))$ or array and a scalar: $\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{value} )$ 
**See also:** max, compare, inRange, minMaxLoc


:param src1: first input array.
:type src1: cv2.typing.MatLike
:param src2: second input array of the same size and type as src1.
:type src2: cv2.typing.MatLike
:param dst: output array of the same size and type as src1.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} minAreaRect(points) -> retval

Finds a rotated rectangle of the minimum area enclosing the input 2D point set.


The function calculates and returns the minimum-area bounding rectangle (possibly rotated) for a specified point set. Developer should keep in mind that the returned RotatedRect can contain negative indices when data is close to the containing Mat element boundary. 


:param points: Input vector of 2D points, stored in std::vector\<\> or Mat
:type points: cv2.typing.MatLike
:rtype: cv2.typing.RotatedRect
````


````{py:function} minEnclosingCircle(points) -> center, radius

Finds a circle of the minimum area enclosing a 2D point set.


The function finds the minimal enclosing circle of a 2D point set using an iterative algorithm. 


:param points: Input vector of 2D points, stored in std::vector\<\> or Mat
:type points: cv2.typing.MatLike
:param center: Output center of the circle.
:type center: 
:param radius: Output radius of the circle.
:type radius: 
:rtype: tuple[cv2.typing.Point2f, float]
````


````{py:function} minEnclosingTriangle(points[, triangle]) -> retval, triangle

Finds a triangle of minimum area enclosing a 2D point set and returns its area.


The function finds a triangle of minimum area enclosing the given set of 2D points and returns its area. The output for a given 2D point set is shown in the image below. 2D points are depicted in red* and the enclosing triangle in *yellow*. 
![Sample output of the minimum enclosing triangle function](pics/minenclosingtriangle.png) 
The implementation of the algorithm is based on O'Rourke's @cite ORourke86 and Klee and Laskowski's @cite KleeLaskowski85 papers. O'Rourke provides a $\theta(n)$ algorithm for finding the minimal enclosing triangle of a 2D convex polygon with n vertices. Since the #minEnclosingTriangle function takes a 2D point set as input an additional preprocessing step of computing the convex hull of the 2D point set is required. The complexity of the #convexHull function is $O(n log(n))$ which is higher than $\theta(n)$. Thus the overall complexity of the function is $O(n log(n))$. 


:param points: Input vector of 2D points with depth CV_32S or CV_32F, stored in std::vector\<\> or Mat
:type points: cv2.typing.MatLike
:param triangle: Output vector of three 2D points defining the vertices of the triangle. The depthof the OutputArray must be CV_32F. 
:type triangle: cv2.typing.MatLike | None
:rtype: tuple[float, cv2.typing.MatLike]
````


````{py:function} minMaxLoc(src[, mask]) -> minVal, maxVal, minLoc, maxLoc

Finds the global minimum and maximum in an array.


The function cv::minMaxLoc finds the minimum and maximum element values and their positions. The extremums are searched across the whole array or, if mask is not an empty array, in the specified array region. 
The function do not work with multi-channel arrays. If you need to find minimum or maximum elements across all the channels, use Mat::reshape first to reinterpret the array as single-channel. Or you may extract the particular channel using either extractImageCOI , or mixChannels , or split . 
**See also:** max, min, reduceArgMin, reduceArgMax, compare, inRange, extractImageCOI, mixChannels, split, Mat::reshape


:param src: input single-channel array.
:type src: cv2.typing.MatLike
:param minVal: pointer to the returned minimum value; NULL is used if not required.
:type minVal: 
:param maxVal: pointer to the returned maximum value; NULL is used if not required.
:type maxVal: 
:param minLoc: pointer to the returned minimum location (in 2D case); NULL is used if not required.
:type minLoc: 
:param maxLoc: pointer to the returned maximum location (in 2D case); NULL is used if not required.
:type maxLoc: 
:param mask: optional mask used to select a sub-array.
:type mask: cv2.typing.MatLike | None
:rtype: tuple[float, float, cv2.typing.Point, cv2.typing.Point]
````


````{py:function} mixChannels(src, dst, fromTo) -> dst




@overload 


:param src: input array or vector of matrices; all of the matrices must have the same size and thesame depth. 
:type src: _typing.Sequence[cv2.typing.MatLike]
:param dst: output array or vector of matrices; all the matrices **must be allocated**; their size anddepth must be the same as in src[0]. 
:type dst: _typing.Sequence[cv2.typing.MatLike]
:param fromTo: array of index pairs specifying which channels are copied and where; fromTo[k\*2] isa 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to src[0].channels()-1, the second input image channels are indexed from src[0].channels() to src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is filled with zero . 
:type fromTo: _typing.Sequence[int]
:rtype: _typing.Sequence[cv2.typing.MatLike]
````


````{py:function} moments(array[, binaryImage]) -> retval

Calculates all of the moments up to the third order of a polygon or rasterized shape.


The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape. The results are returned in the structure cv::Moments. 
```{note}
Only applicable to contour moments calculations from Python bindings: Note that the numpytype for the input array should be either np.int32 or np.float32. 
```
**See also:**  contourArea, arcLength


:param array: Raster image (single-channel, 8-bit or floating-point 2D array) or an array ($1 \times N$ or $N \times 1$ ) of 2D points (Point or Point2f ). 
:type array: cv2.typing.MatLike
:param binaryImage: If it is true, all non-zero image pixels are treated as 1's. The parameter isused for images only. 
:type binaryImage: bool
:return: moments.
:rtype: cv2.typing.Moments
````


````{py:function} morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst

Performs advanced morphological transformations.


The function cv::morphologyEx can perform advanced morphological transformations using an erosion and dilation as basic operations. 
Any of the operations can be done in-place. In case of multi-channel images, each channel is processed independently. 
**See also:**  dilate, erode, getStructuringElement
```{note}
The number of iterations is the number of times erosion or dilatation operation will be applied.For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply successively: erode -> erode -> dilate -> dilate (and not erode -> dilate -> erode -> dilate). 
```


:param src: Source image. The number of channels can be arbitrary. The depth should be one ofCV_8U, CV_16U, CV_16S, CV_32F or CV_64F. 
:type src: cv2.typing.MatLike
:param dst: Destination image of the same size and type as source image.
:type dst: cv2.typing.MatLike | None
:param op: Type of a morphological operation, see #MorphTypes
:type op: int
:param kernel: Structuring element. It can be created using #getStructuringElement.
:type kernel: cv2.typing.MatLike
:param anchor: Anchor position with the kernel. Negative values mean that the anchor is at thekernel center. 
:type anchor: cv2.typing.Point
:param iterations: Number of times erosion and dilation are applied.
:type iterations: int
:param borderType: Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:param borderValue: Border value in case of a constant border. The default value has a specialmeaning. 
:type borderValue: cv2.typing.Scalar
:rtype: cv2.typing.MatLike
````


````{py:function} moveWindow(winname, x, y) -> None

Moves the window to the specified position




:param winname: Name of the window.
:type winname: str
:param x: The new x-coordinate of the window.
:type x: int
:param y: The new y-coordinate of the window.
:type y: int
:rtype: None
````


````{py:function} mulSpectrums(a, b, flags[, c[, conjB]]) -> c

Performs the per-element multiplication of two Fourier spectrums.


The function cv::mulSpectrums performs the per-element multiplication of the two CCS-packed or complex matrices that are results of a real or complex Fourier transform. 
The function, together with dft and idft , may be used to calculate convolution (pass conjB=false ) or correlation (pass conjB=true ) of two arrays rapidly. When the arrays are complex, they are simply multiplied (per element) with an optional conjugation of the second-array elements. When the arrays are real, they are assumed to be CCS-packed (see dft for details). 


:param a: first input array.
:type a: cv2.typing.MatLike
:param b: second input array of the same size and type as src1 .
:type b: cv2.typing.MatLike
:param c: output array of the same size and type as src1 .
:type c: cv2.typing.MatLike | None
:param flags: operation flags; currently, the only supported flag is cv::DFT_ROWS, which indicates thateach row of src1 and src2 is an independent 1D Fourier spectrum. If you do not want to use this flag, then simply add a `0` as value. 
:type flags: int
:param conjB: optional flag that conjugates the second input array before the multiplication (true)or not (false). 
:type conjB: bool
:rtype: cv2.typing.MatLike
````


````{py:function} mulTransposed(src, aTa[, dst[, delta[, scale[, dtype]]]]) -> dst

Calculates the product of a matrix and its transposition.


The function cv::mulTransposed calculates the product of src and its transposition: $\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} )$ if aTa=true , and $\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T$ otherwise. The function is used to calculate the covariance matrix. With zero delta, it can be used as a faster substitute for general matrix product A\*B when B=A' 
**See also:** calcCovarMatrix, gemm, repeat, reduce


:param src: input single-channel matrix. Note that unlike gemm, thefunction can multiply not only floating-point matrices. 
:type src: cv2.typing.MatLike
:param dst: output square matrix.
:type dst: cv2.typing.MatLike | None
:param aTa: Flag specifying the multiplication ordering. See thedescription below. 
:type aTa: bool
:param delta: Optional delta matrix subtracted from src before themultiplication. When the matrix is empty ( delta=noArray() ), it is assumed to be zero, that is, nothing is subtracted. If it has the same size as src , it is simply subtracted. Otherwise, it is "repeated" (see repeat ) to cover the full src and then subtracted. Type of the delta matrix, when it is not empty, must be the same as the type of created output matrix. See the dtype parameter description below. 
:type delta: cv2.typing.MatLike | None
:param scale: Optional scale factor for the matrix product.
:type scale: float
:param dtype: Optional type of the output matrix. When it is negative,the output matrix will have the same type as src . Otherwise, it will be type=CV_MAT_DEPTH(dtype) that should be either CV_32F or CV_64F . 
:type dtype: int
:rtype: cv2.typing.MatLike
````


````{py:function} multiply(src1, src2[, dst[, scale[, dtype]]]) -> dst

Calculates the per-element scaled product of two arrays.


The function multiply calculates the per-element product of two arrays: 
$\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))$ 
There is also a @ref MatrixExpressions -friendly variant of the first function. See Mat::mul . 
For a not-per-element matrix product, see gemm . 
```{note}
Saturation is not applied when the output array has the depthCV_32S. You may even get result of an incorrect sign in the case of overflow. 
```
```{note}
(Python) Be careful to difference behaviour between src1/src2 are single number and they are tuple/array.`multiply(src,X)` means `multiply(src,(X,X,X,X))`. `multiply(src,(X,))` means `multiply(src,(X,0,0,0))`. 
```
**See also:** add, subtract, divide, scaleAdd, addWeighted, accumulate, accumulateProduct, accumulateSquare,Mat::convertTo 


:param src1: first input array.
:type src1: cv2.typing.MatLike
:param src2: second input array of the same size and the same type as src1.
:type src2: cv2.typing.MatLike
:param dst: output array of the same size and type as src1.
:type dst: cv2.typing.MatLike | None
:param scale: optional scale factor.
:type scale: float
:param dtype: optional depth of the output array
:type dtype: int
:rtype: cv2.typing.MatLike
````


````{py:function} namedWindow(winname[, flags]) -> None

Creates a window.


The function namedWindow creates a window that can be used as a placeholder for images and trackbars. Created windows are referred to by their names. 
If a window with the same name already exists, the function does nothing. 
You can call cv::destroyWindow or cv::destroyAllWindows to close the window and de-allocate any associated memory usage. For a simple program, you do not really have to call these functions because all the resources and windows of the application are closed automatically by the operating system upon exit. 
```{note}
Qt backend supports additional flags:-   **WINDOW_NORMAL or WINDOW_AUTOSIZE:** WINDOW_NORMAL enables you to resize the window, whereas WINDOW_AUTOSIZE adjusts automatically the window size to fit the displayed image (see imshow ), and you cannot change the window size manually. -   **WINDOW_FREERATIO or WINDOW_KEEPRATIO:** WINDOW_FREERATIO adjusts the image with no respect to its ratio, whereas WINDOW_KEEPRATIO keeps the image ratio. -   **WINDOW_GUI_NORMAL or WINDOW_GUI_EXPANDED:** WINDOW_GUI_NORMAL is the old way to draw the window without statusbar and toolbar, whereas WINDOW_GUI_EXPANDED is a new enhanced GUI. By default, flags == WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED 
```


:param winname: Name of the window in the window caption that may be used as a window identifier.
:type winname: str
:param flags: Flags of the window. The supported flags are: (cv::WindowFlags)
:type flags: int
:rtype: None
````


````{py:function} norm(src1[, normType[, mask]]) -> retval

Calculates an absolute difference norm or a relative difference norm.


This version of #norm calculates the absolute norm of src1. The type of norm to calculate is specified using #NormTypes. 
As example for one array consider the function $r(x)= \begin{pmatrix} x \\ 1-x \end{pmatrix}, x \in [-1;1]$. The $ L_{1}, L_{2} $ and $ L_{\infty} $ norm for the sample value $r(-1) = \begin{pmatrix} -1 \\ 2 \end{pmatrix}$ is calculated as follows \f{align*} \| r(-1) \|_{L_1} &= |-1| + |2| = 3 \\ \| r(-1) \|_{L_2} &= \sqrt{(-1)^{2} + (2)^{2}} = \sqrt{5} \\ \| r(-1) \|_{L_\infty} &= \max(|-1|,|2|) = 2 \f} and for $r(0.5) = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}$ the calculation is \f{align*} \| r(0.5) \|_{L_1} &= |0.5| + |0.5| = 1 \\ \| r(0.5) \|_{L_2} &= \sqrt{(0.5)^{2} + (0.5)^{2}} = \sqrt{0.5} \\ \| r(0.5) \|_{L_\infty} &= \max(|0.5|,|0.5|) = 0.5. \f} The following graphic shows all values for the three norm functions $\| r(x) \|_{L_1}, \| r(x) \|_{L_2}$ and $\| r(x) \|_{L_\infty}$. It is notable that the $ L_{1} $ norm forms the upper and the $ L_{\infty} $ norm forms the lower border for the example function $ r(x) $. ![Graphs for the different norm functions from the above example](pics/NormTypes_OneArray_1-2-INF.png) 
When the mask parameter is specified and it is not empty, the norm is 
If normType is not specified, #NORM_L2 is used. calculated only over the region specified by the mask. 
Multi-channel input arrays are treated as single-channel arrays, that is, the results for all channels are combined. 
Hamming norms can only be calculated with CV_8U depth arrays. 
norm(src1, src2[, normType[, mask]]) -> retval 
This version of cv::norm calculates the absolute difference norm or the relative difference norm of arrays src1 and src2. The type of norm to calculate is specified using #NormTypes. 


:param src1: first input array.
:type src1: cv2.typing.MatLike
:param normType: type of the norm (see #NormTypes).
:type normType: int
:param mask: optional operation mask; it must have the same size as src1 and CV_8UC1 type.
:type mask: cv2.typing.MatLike | None
:param src2: second input array of the same size and the same type as src1.
:type src2: 
:rtype: float
````


````{py:function} normalize(src, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]) -> dst

Normalizes the norm or value range of an array.


The function cv::normalize normalizes scale and shift the input array elements so that $\| \texttt{dst} \| _{L_p}= \texttt{alpha}$ (where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that $\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}$ 
when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or min-max but modify the whole array, you can use norm and Mat::convertTo. 
In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this, the range transformation for sparse matrices is not allowed since it can shift the zero level. 
Possible usage with some positive example data: 
```cpp
vector<double> positiveData = { 2.0, 8.0, 10.0 };
vector<double> normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;

// Norm to probability (total count)
// sum(numbers) = 20.0
// 2.0      0.1     (2.0/20.0)
// 8.0      0.4     (8.0/20.0)
// 10.0     0.5     (10.0/20.0)
normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);

// Norm to unit vector: ||positiveData|| = 1.0
// 2.0      0.15
// 8.0      0.62
// 10.0     0.77
normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);

// Norm to max element
// 2.0      0.2     (2.0/10.0)
// 8.0      0.8     (8.0/10.0)
// 10.0     1.0     (10.0/10.0)
normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);

// Norm to range [0.0;1.0]
// 2.0      0.0     (shift to left border)
// 8.0      0.75    (6.0/8.0)
// 10.0     1.0     (shift to right border)
normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
```

**See also:** norm, Mat::convertTo, SparseMat::convertTo


:param src: input array.
:type src: cv2.typing.MatLike
:param dst: output array of the same size as src .
:type dst: cv2.typing.MatLike
:param alpha: norm value to normalize to or the lower range boundary in case of the rangenormalization. 
:type alpha: float
:param beta: upper range boundary in case of the range normalization; it is not used for the normnormalization. 
:type beta: float
:param norm_type: normalization type (see cv::NormTypes).
:type norm_type: int
:param dtype: when negative, the output array has the same type as src; otherwise, it has the samenumber of channels as src and the depth =CV_MAT_DEPTH(dtype). 
:type dtype: int
:param mask: optional operation mask.
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} patchNaNs(a[, val]) -> a

Replaces NaNs by given number




:param a: input/output matrix (CV_32F type).
:type a: cv2.typing.MatLike
:param val: value to convert the NaNs
:type val: float
:rtype: cv2.typing.MatLike
````


````{py:function} pencilSketch(src[, dst1[, dst2[, sigma_s[, sigma_r[, shade_factor]]]]]) -> dst1, dst2

Pencil-like non-photorealistic line drawing




:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param dst1: Output 8-bit 1-channel image.
:type dst1: cv2.typing.MatLike | None
:param dst2: Output image with the same size and type as src.
:type dst2: cv2.typing.MatLike | None
:param sigma_s: %Range between 0 to 200.
:type sigma_s: float
:param sigma_r: %Range between 0 to 1.
:type sigma_r: float
:param shade_factor: %Range between 0 to 0.1.
:type shade_factor: float
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} perspectiveTransform(src, m[, dst]) -> dst

Performs the perspective matrix transformation of vectors.


The function cv::perspectiveTransform transforms every element of src by treating it as a 2D or 3D vector, in the following way: $(x, y, z)  \rightarrow (x'/w, y'/w, z'/w)$ where $(x', y', z', w') =  \texttt{mat} \cdot \begin{bmatrix} x & y & z & 1  \end{bmatrix}$ and $w =  \fork{w'}{if \(w' \ne 0\)}{\infty}{otherwise}$ 
Here a 3D vector transformation is shown. In case of a 2D vector transformation, the z component is omitted. 
```{note}
The function transforms a sparse set of 2D or 3D vectors. If youwant to transform an image using perspective transformation, use warpPerspective . If you have an inverse problem, that is, you want to compute the most probable perspective transformation out of several pairs of corresponding points, you can use getPerspectiveTransform or findHomography . 
```
**See also:**  transform, warpPerspective, getPerspectiveTransform, findHomography


:param src: input two-channel or three-channel floating-point array; eachelement is a 2D/3D vector to be transformed. 
:type src: cv2.typing.MatLike
:param dst: output array of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param m: 3x3 or 4x4 floating-point transformation matrix.
:type m: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````


````{py:function} phase(x, y[, angle[, angleInDegrees]]) -> angle

Calculates the rotation angle of 2D vectors.


The function cv::phase calculates the rotation angle of each 2D vector that is formed from the corresponding elements of x and y : $\texttt{angle} (I) =  \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))$ 
The angle estimation accuracy is about 0.3 degrees. When x(I)=y(I)=0 , the corresponding angle(I) is set to 0. 


:param x: input floating-point array of x-coordinates of 2D vectors.
:type x: cv2.typing.MatLike
:param y: input array of y-coordinates of 2D vectors; it must have thesame size and the same type as x. 
:type y: cv2.typing.MatLike
:param angle: output array of vector angles; it has the same size andsame type as x . 
:type angle: cv2.typing.MatLike | None
:param angleInDegrees: when true, the function calculates the angle indegrees, otherwise, they are measured in radians. 
:type angleInDegrees: bool
:rtype: cv2.typing.MatLike
````


````{py:function} phaseCorrelate(src1, src2[, window]) -> retval, response

The function is used to detect translational shifts that occur between two images.


The operation takes advantage of the Fourier shift theorem for detecting the translational shift in the frequency domain. It can be used for fast image registration as well as motion estimation. For more information please see <http://en.wikipedia.org/wiki/Phase_correlation> 
Calculates the cross-power spectrum of two supplied source arrays. The arrays are padded if needed with getOptimalDFTSize. 
The function performs the following equations: - First it applies a Hanning window (see <http://en.wikipedia.org/wiki/Hann_function>) to each image to remove possible edge effects. This window is cached until the array size changes to speed up processing time. - Next it computes the forward DFTs of each source array: $\mathbf{G}_a = \mathcal{F}\{src_1\}, \; \mathbf{G}_b = \mathcal{F}\{src_2\}$ where $\mathcal{F}$ is the forward DFT. - It then computes the cross-power spectrum of each frequency domain array: $R = \frac{ \mathbf{G}_a \mathbf{G}_b^*}{|\mathbf{G}_a \mathbf{G}_b^*|}$ - Next the cross-correlation is converted back into the time domain via the inverse DFT: $r = \mathcal{F}^{-1}\{R\}$ - Finally, it computes the peak location and computes a 5x5 weighted centroid around the peak to achieve sub-pixel accuracy. $(\Delta x, \Delta y) = \texttt{weightedCentroid} \{\arg \max_{(x, y)}\{r\}\}$ - If non-zero, the response parameter is computed as the sum of the elements of r within the 5x5 centroid around the peak location. It is normalized to a maximum of 1 (meaning there is a single peak) and will be smaller when there are multiple peaks. 
**See also:** dft, getOptimalDFTSize, idft, mulSpectrums createHanningWindow


:param src1: Source floating point array (CV_32FC1 or CV_64FC1)
:type src1: cv2.typing.MatLike
:param src2: Source floating point array (CV_32FC1 or CV_64FC1)
:type src2: cv2.typing.MatLike
:param window: Floating point array with windowing coefficients to reduce edge effects (optional).
:type window: cv2.typing.MatLike | None
:param response: Signal power within the 5x5 centroid around the peak, between 0 and 1 (optional).
:type response: 
:return: detected phase shift (sub-pixel) between the two arrays.
:rtype: tuple[cv2.typing.Point2d, float]
````


````{py:function} pointPolygonTest(contour, pt, measureDist) -> retval

Performs a point-in-contour test.


The function determines whether the point is inside a contour, outside, or lies on an edge (or coincides with a vertex). It returns positive (inside), negative (outside), or zero (on an edge) value, correspondingly. When measureDist=false , the return value is +1, -1, and 0, respectively. Otherwise, the return value is a signed distance between the point and the nearest contour edge. 
See below a sample output of the function where each image pixel is tested against the contour: 
![sample output](pics/pointpolygon.png) 


:param contour: Input contour.
:type contour: cv2.typing.MatLike
:param pt: Point tested against the contour.
:type pt: cv2.typing.Point2f
:param measureDist: If true, the function estimates the signed distance from the point to thenearest contour edge. Otherwise, the function only checks if the point is inside a contour or not. 
:type measureDist: bool
:rtype: float
````


````{py:function} polarToCart(magnitude, angle[, x[, y[, angleInDegrees]]]) -> x, y

Calculates x and y coordinates of 2D vectors from their magnitude and angle.


The function cv::polarToCart calculates the Cartesian coordinates of each 2D vector represented by the corresponding elements of magnitude and angle: $\begin{array}{l} \texttt{x} (I) =  \texttt{magnitude} (I) \cos ( \texttt{angle} (I)) \\ \texttt{y} (I) =  \texttt{magnitude} (I) \sin ( \texttt{angle} (I)) \\ \end{array}$ 
The relative accuracy of the estimated coordinates is about 1e-6. 
**See also:** cartToPolar, magnitude, phase, exp, log, pow, sqrt


:param magnitude: input floating-point array of magnitudes of 2D vectors;it can be an empty matrix (=Mat()), in this case, the function assumes that all the magnitudes are =1; if it is not empty, it must have the same size and type as angle. 
:type magnitude: cv2.typing.MatLike
:param angle: input floating-point array of angles of 2D vectors.
:type angle: cv2.typing.MatLike
:param x: output array of x-coordinates of 2D vectors; it has the samesize and type as angle. 
:type x: cv2.typing.MatLike | None
:param y: output array of y-coordinates of 2D vectors; it has the samesize and type as angle. 
:type y: cv2.typing.MatLike | None
:param angleInDegrees: when true, the input angles are measured indegrees, otherwise, they are measured in radians. 
:type angleInDegrees: bool
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} pollKey() -> retval

Polls for a pressed key.


The function pollKey polls for a key event without waiting. It returns the code of the pressed key or -1 if no key was pressed since the last invocation. To wait until a key was pressed, use #waitKey. 
```{note}
The functions #waitKey and #pollKey are the only methods in HighGUI that can fetch and handleGUI events, so one of them needs to be called periodically for normal event processing unless HighGUI is used within an environment that takes care of event processing. 
```
```{note}
The function only works if there is at least one HighGUI window created and the window isactive. If there are several HighGUI windows, any of them can be active. 
```


:rtype: int
````


````{py:function} polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]]) -> img

Draws several polygonal curves.


The function cv::polylines draws one or more polygonal curves. 


:param img: Image.
:type img: cv2.typing.MatLike
:param pts: Array of polygonal curves.
:type pts: _typing.Sequence[cv2.typing.MatLike]
:param isClosed: Flag indicating whether the drawn polylines are closed or not. If they are closed,the function draws a line from the last vertex of each curve to its first vertex. 
:type isClosed: bool
:param color: Polyline color.
:type color: cv2.typing.Scalar
:param thickness: Thickness of the polyline edges.
:type thickness: int
:param lineType: Type of the line segments. See #LineTypes
:type lineType: int
:param shift: Number of fractional bits in the vertex coordinates.
:type shift: int
:rtype: cv2.typing.MatLike
````


````{py:function} pow(src, power[, dst]) -> dst

Raises every array element to a power.


The function cv::pow raises every element of the input array to power : $\texttt{dst} (I) =  \fork{\texttt{src}(I)^{power}}{if \(\texttt{power}\) is integer}{|\texttt{src}(I)|^{power}}{otherwise}$ 
So, for a non-integer power exponent, the absolute values of input array elements are used. However, it is possible to get true values for negative values using some extra operations. In the example below, computing the 5th root of array src shows: 
```cpp
Mat mask = src < 0;
pow(src, 1./5, dst);
subtract(Scalar::all(0), dst, dst, mask);
```
For some values of power, such as integer values, 0.5 and -0.5, specialized faster algorithms are used. 
Special values (NaN, Inf) are not handled. 
**See also:** sqrt, exp, log, cartToPolar, polarToCart


:param src: input array.
:type src: cv2.typing.MatLike
:param power: exponent of power.
:type power: float
:param dst: output array of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} preCornerDetect(src, ksize[, dst[, borderType]]) -> dst

Calculates a feature map for corner detection.


The function calculates the complex spatial derivative-based function of the source image 
$\texttt{dst} = (D_x  \texttt{src} )^2  \cdot D_{yy}  \texttt{src} + (D_y  \texttt{src} )^2  \cdot D_{xx}  \texttt{src} - 2 D_x  \texttt{src} \cdot D_y  \texttt{src} \cdot D_{xy}  \texttt{src}$ 
where $D_x$,$D_y$ are the first image derivatives, $D_{xx}$,$D_{yy}$ are the second image derivatives, and $D_{xy}$ is the mixed derivative. 
The corners can be found as local maximums of the functions, as shown below: 
```c++
Mat corners, dilated_corners;
preCornerDetect(image, corners, 3);
// dilation with 3x3 rectangular structuring element
dilate(corners, dilated_corners, Mat(), 1);
Mat corner_mask = corners == dilated_corners;
```



:param src: Source single-channel 8-bit of floating-point image.
:type src: cv2.typing.MatLike
:param dst: Output image that has the type CV_32F and the same size as src .
:type dst: cv2.typing.MatLike | None
:param ksize: %Aperture size of the Sobel .
:type ksize: int
:param borderType: Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs[, imagePoints[, jacobian[, aspectRatio]]]) -> imagePoints, jacobian

Projects 3D points to an image plane.


The function computes the 2D projections of 3D points to the image plane, given intrinsic and extrinsic camera parameters. Optionally, the function computes Jacobians -matrices of partial derivatives of image points coordinates (as functions of all the input parameters) with respect to the particular parameters, intrinsic and/or extrinsic. The Jacobians are used during the global optimization in @ref calibrateCamera, @ref solvePnP, and @ref stereoCalibrate. The function itself can also be used to compute a re-projection error, given the current intrinsic and extrinsic parameters. 
```{note}
By setting rvec = tvec = $[0, 0, 0]$, or by setting cameraMatrix to a 3x3 identity matrix,or by passing zero distortion coefficients, one can get various useful partial cases of the function. This means, one can compute the distorted coordinates for a sparse set of points or apply a perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup. 
```


:param objectPoints: Array of object points expressed wrt. the world coordinate frame. A 3xN/Nx31-channel or 1xN/Nx1 3-channel (or vector\<Point3f\> ), where N is the number of points in the view. 
:type objectPoints: cv2.typing.MatLike
:param rvec: The rotation vector (@ref Rodrigues) that, together with tvec, performs a change ofbasis from world to camera coordinate system, see @ref calibrateCamera for details. 
:type rvec: cv2.typing.MatLike
:param tvec: The translation vector, see parameter description above.
:type tvec: cv2.typing.MatLike
:param cameraMatrix: Camera intrinsic matrix $\cameramatrix{A}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$\distcoeffs$ . If the vector is empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param imagePoints: Output array of image points, 1xN/Nx1 2-channel, orvector\<Point2f\> . 
:type imagePoints: cv2.typing.MatLike | None
:param jacobian: Optional output 2Nx(10+\<numDistCoeffs\>) jacobian matrix of derivatives of imagepoints with respect to components of the rotation vector, translation vector, focal lengths, coordinates of the principal point and the distortion coefficients. In the old interface different components of the jacobian are returned via different output parameters. 
:type jacobian: cv2.typing.MatLike | None
:param aspectRatio: Optional "fixed aspect ratio" parameter. If the parameter is not 0, thefunction assumes that the aspect ratio ($f_x / f_y$) is fixed and correspondingly adjusts the jacobian matrix. 
:type aspectRatio: float
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img

Draws a text string.


The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered using the specified font are replaced by question marks. See #getTextSize for a text rendering code example. 


:param img: Image.
:type img: cv2.typing.MatLike
:param text: Text string to be drawn.
:type text: str
:param org: Bottom-left corner of the text string in the image.
:type org: cv2.typing.Point
:param fontFace: Font type, see #HersheyFonts.
:type fontFace: int
:param fontScale: Font scale factor that is multiplied by the font-specific base size.
:type fontScale: float
:param color: Text color.
:type color: cv2.typing.Scalar
:param thickness: Thickness of the lines used to draw a text.
:type thickness: int
:param lineType: Line type. See #LineTypes
:type lineType: int
:param bottomLeftOrigin: When true, the image data origin is at the bottom-left corner. Otherwise,it is at the top-left corner. 
:type bottomLeftOrigin: bool
:rtype: cv2.typing.MatLike
````


````{py:function} pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst

Blurs an image and downsamples it.


By default, size of the output image is computed as `Size((src.cols+1)/2, (src.rows+1)/2)`, but in any case, the following conditions should be satisfied: 
$\begin{array}{l} | \texttt{dstsize.width} *2-src.cols| \leq 2 \\ | \texttt{dstsize.height} *2-src.rows| \leq 2 \end{array}$ 
The function performs the downsampling step of the Gaussian pyramid construction. First, it convolves the source image with the kernel: 
$\frac{1}{256} \begin{bmatrix} 1 & 4 & 6 & 4 & 1  \\ 4 & 16 & 24 & 16 & 4  \\ 6 & 24 & 36 & 24 & 6  \\ 4 & 16 & 24 & 16 & 4  \\ 1 & 4 & 6 & 4 & 1 \end{bmatrix}$ 
Then, it downsamples the image by rejecting even rows and columns. 


:param src: input image.
:type src: cv2.typing.MatLike
:param dst: output image; it has the specified size and the same type as src.
:type dst: cv2.typing.MatLike | None
:param dstsize: size of the output image.
:type dstsize: cv2.typing.Size
:param borderType: Pixel extrapolation method, see #BorderTypes (#BORDER_CONSTANT isn't supported)
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]]) -> dst

Performs initial step of meanshift segmentation of an image.


The function implements the filtering stage of meanshift segmentation, that is, the output of the function is the filtered "posterized" image with color gradients and fine-grain texture flattened. At every pixel (X,Y) of the input image (or down-sized input image, see below) the function executes meanshift iterations, that is, the pixel (X,Y) neighborhood in the joint space-color hyperspace is considered: 
$(x,y): X- \texttt{sp} \le x  \le X+ \texttt{sp} , Y- \texttt{sp} \le y  \le Y+ \texttt{sp} , ||(R,G,B)-(r,g,b)||   \le \texttt{sr}$ 
where (R,G,B) and (r,g,b) are the vectors of color components at (X,Y) and (x,y), respectively (though, the algorithm does not depend on the color space used, so any 3-component color space can be used instead). Over the neighborhood the average spatial value (X',Y') and average color vector (R',G',B') are found and they act as the neighborhood center on the next iteration: 
$(X,Y)~(X',Y'), (R,G,B)~(R',G',B').$ 
After the iterations over, the color components of the initial pixel (that is, the pixel from where the iterations started) are set to the final value (average color at the last iteration): 
$I(X,Y) <- (R*,G*,B*)$ 
When maxLevel \> 0, the gaussian pyramid of maxLevel+1 levels is built, and the above procedure is run on the smallest layer first. After that, the results are propagated to the larger layer and the iterations are run again only on those pixels where the layer colors differ by more than sr from the lower-resolution layer of the pyramid. That makes boundaries of color regions sharper. Note that the results will be actually different from the ones obtained by running the meanshift procedure on the whole original image (i.e. when maxLevel==0). 


:param src: The source 8-bit, 3-channel image.
:type src: cv2.typing.MatLike
:param dst: The destination image of the same format and the same size as the source.
:type dst: cv2.typing.MatLike | None
:param sp: The spatial window radius.
:type sp: float
:param sr: The color window radius.
:type sr: float
:param maxLevel: Maximum level of the pyramid for the segmentation.
:type maxLevel: int
:param termcrit: Termination criteria: when to stop meanshift iterations.
:type termcrit: cv2.typing.TermCriteria
:rtype: cv2.typing.MatLike
````


````{py:function} pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst

Upsamples an image and then blurs it.


By default, size of the output image is computed as `Size(src.cols\*2, (src.rows\*2)`, but in any case, the following conditions should be satisfied: 
$\begin{array}{l} | \texttt{dstsize.width} -src.cols*2| \leq  ( \texttt{dstsize.width}   \mod  2)  \\ | \texttt{dstsize.height} -src.rows*2| \leq  ( \texttt{dstsize.height}   \mod  2) \end{array}$ 
The function performs the upsampling step of the Gaussian pyramid construction, though it can actually be used to construct the Laplacian pyramid. First, it upsamples the source image by injecting even zero rows and columns and then convolves the result with the same kernel as in pyrDown multiplied by 4. 


:param src: input image.
:type src: cv2.typing.MatLike
:param dst: output image. It has the specified size and the same type as src .
:type dst: cv2.typing.MatLike | None
:param dstsize: size of the output image.
:type dstsize: cv2.typing.Size
:param borderType: Pixel extrapolation method, see #BorderTypes (only #BORDER_DEFAULT is supported)
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} randShuffle(dst[, iterFactor]) -> dst

Shuffles the array elements randomly.


The function cv::randShuffle shuffles the specified 1D array by randomly choosing pairs of elements and swapping them. The number of such swap operations will be dst.rows\*dst.cols\*iterFactor . 
**See also:** RNG, sort


:param dst: input/output numerical 1D array.
:type dst: cv2.typing.MatLike
:param iterFactor: scale factor that determines the number of random swap operations (see the detailsbelow). 
:type iterFactor: float
:param rng: optional random number generator used for shuffling; if it is zero, theRNG () is usedinstead. 
:type rng: 
:rtype: cv2.typing.MatLike
````


````{py:function} randn(dst, mean, stddev) -> dst

Fills the array with normally distributed random numbers.


The function cv::randn fills the matrix dst with normally distributed random numbers with the specified mean vector and the standard deviation matrix. The generated random numbers are clipped to fit the value range of the output array data type. 
**See also:** RNG, randu


:param dst: output array of random numbers; the array must be pre-allocated and have 1 to 4 channels.
:type dst: cv2.typing.MatLike
:param mean: mean value (expectation) of the generated random numbers.
:type mean: cv2.typing.MatLike
:param stddev: standard deviation of the generated random numbers; it can be either a vector (inwhich case a diagonal standard deviation matrix is assumed) or a square matrix. 
:type stddev: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````


````{py:function} randu(dst, low, high) -> dst

Generates a single uniformly-distributed random number or an array of random numbers.


Non-template variant of the function fills the matrix dst with uniformly-distributed random numbers from the specified range: $\texttt{low} _c  \leq \texttt{dst} (I)_c <  \texttt{high} _c$ 
**See also:** RNG, randn, theRNG


:param dst: output array of random numbers; the array must be pre-allocated.
:type dst: cv2.typing.MatLike
:param low: inclusive lower boundary of the generated random numbers.
:type low: cv2.typing.MatLike
:param high: exclusive upper boundary of the generated random numbers.
:type high: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````


````{py:function} readOpticalFlow(path) -> retval

Read a .flo file


The function readOpticalFlow loads a flow field from a file and returns it as a single matrix. Resulting Mat has a type CV_32FC2 - floating-point, 2-channel. First channel corresponds to the flow in the horizontal direction (u), second - vertical (v). 


:param path: Path to the file to be loaded
:type path: str
:rtype: cv2.typing.MatLike
````


````{py:function} recoverPose(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2[, E[, R[, t[, method[, prob[, threshold[, mask]]]]]]]) -> retval, E, R, t, mask

Recovers the relative camera rotation and the translation from an estimated essentialmatrix and the corresponding points in two images, using chirality check. Returns the number of inliers that pass the check. 


This function decomposes an essential matrix using @ref decomposeEssentialMat and then verifies possible pose hypotheses by doing cheirality check. The cheirality check means that the triangulated 3D points should have positive depth. Some details can be found in @cite Nister03. 
This function can be used to process the output E and mask from @ref findEssentialMat. In this scenario, points1 and points2 are the same input for findEssentialMat.: 
```c++
// Example. Estimation of fundamental matrix using the RANSAC algorithm
int point_count = 100;
vector<Point2f> points1(point_count);
vector<Point2f> points2(point_count);

// initialize the points here ...
for( int i = 0; i < point_count; i++ )
{
points1[i] = ...;
points2[i] = ...;
}

// Input: camera calibration of both cameras, for example using intrinsic chessboard calibration.
Mat cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2;

// Output: Essential matrix, relative rotation and relative translation.
Mat E, R, t, mask;

recoverPose(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, E, R, t, mask);
```

recoverPose(E, points1, points2, cameraMatrix[, R[, t[, mask]]]) -> retval, R, t, mask 
This function decomposes an essential matrix using @ref decomposeEssentialMat and then verifies possible pose hypotheses by doing chirality check. The chirality check means that the triangulated 3D points should have positive depth. Some details can be found in @cite Nister03. 
This function can be used to process the output E and mask from @ref findEssentialMat. In this scenario, points1 and points2 are the same input for #findEssentialMat : 
```c++
// Example. Estimation of fundamental matrix using the RANSAC algorithm
int point_count = 100;
vector<Point2f> points1(point_count);
vector<Point2f> points2(point_count);

// initialize the points here ...
for( int i = 0; i < point_count; i++ )
{
points1[i] = ...;
points2[i] = ...;
}

// cametra matrix with both focal lengths = 1, and principal point = (0, 0)
Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

Mat E, R, t, mask;

E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
```

recoverPose(E, points1, points2[, R[, t[, focal[, pp[, mask]]]]]) -> retval, R, t, mask @overload 
This function differs from the one above that it computes camera intrinsic matrix from focal length and principal point: 
$A = \begin{bmatrix} f & 0 & x_{pp}  \\ 0 & f & y_{pp}  \\ 0 & 0 & 1 \end{bmatrix}$ 
recoverPose(E, points1, points2, cameraMatrix, distanceThresh[, R[, t[, mask[, triangulatedPoints]]]]) -> retval, R, t, mask, triangulatedPoints @overload 
This function differs from the one above that it outputs the triangulated 3D point that are used for the chirality check. 


:param points1: Array of N 2D points from the first image. The point coordinates should befloating-point (single or double precision). 
:type points1: cv2.typing.MatLike
:param points2: Array of the second image points of the same size and format as points1.
:type points2: cv2.typing.MatLike
:param cameraMatrix1: Input/output camera matrix for the first camera, the same as in@ref calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below. 
:type cameraMatrix1: cv2.typing.MatLike
:param distCoeffs1: Input/output vector of distortion coefficients, the same as in@ref calibrateCamera. 
:type distCoeffs1: cv2.typing.MatLike
:param cameraMatrix2: Input/output camera matrix for the first camera, the same as in@ref calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below. 
:type cameraMatrix2: cv2.typing.MatLike
:param distCoeffs2: Input/output vector of distortion coefficients, the same as in@ref calibrateCamera. 
:type distCoeffs2: cv2.typing.MatLike
:param E: The input essential matrix.
:type E: cv2.typing.MatLike | None
:param R: Output rotation matrix. Together with the translation vector, this matrix makes up a tuplethat performs a change of basis from the first camera's coordinate system to the second camera's coordinate system. Note that, in general, t can not be used for this tuple, see the parameter description below. 
:type R: cv2.typing.MatLike | None
:param t: Output translation vector. This vector is obtained by @ref decomposeEssentialMat andtherefore is only known up to scale, i.e. t is the direction of the translation vector and has unit length. 
:type t: cv2.typing.MatLike | None
:param method: Method for computing an essential matrix.-   @ref RANSAC for the RANSAC algorithm. -   @ref LMEDS for the LMedS algorithm. 
:type method: int
:param prob: Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level ofconfidence (probability) that the estimated matrix is correct. 
:type prob: float
:param threshold: Parameter used for RANSAC. It is the maximum distance from a point to an epipolarline in pixels, beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point localization, image resolution, and the image noise. 
:type threshold: float
:param mask: Input/output mask for inliers in points1 and points2. If it is not empty, then it marksinliers in points1 and points2 for the given essential matrix E. Only these inliers will be used to recover pose. In the output mask only inliers which pass the chirality check. 
:type mask: cv2.typing.MatLike | None
:param cameraMatrix: Camera intrinsic matrix $\cameramatrix{A}$ .Note that this function assumes that points1 and points2 are feature points from cameras with the same camera intrinsic matrix. 
:type cameraMatrix: 
:param focal: Focal length of the camera. Note that this function assumes that points1 and points2are feature points from cameras with same focal length and principal point. 
:type focal: 
:param pp: principal point of the camera.
:type pp: 
:param distanceThresh: threshold distance which is used to filter out far away points (i.e. infinitepoints). 
:type distanceThresh: 
:param triangulatedPoints: 3D points which were reconstructed by triangulation.
:type triangulatedPoints: 
:rtype: tuple[int, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img

Draws a simple, thick, or filled up-right rectangle.


The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners are pt1 and pt2. 
rectangle(img, rec, color[, thickness[, lineType[, shift]]]) -> img @overload 
use `rec` parameter as alternative specification of the drawn rectangle: `r.tl() and r.br()-Point(1,1)` are opposite corners 


:param img: Image.
:type img: cv2.typing.MatLike
:param pt1: Vertex of the rectangle.
:type pt1: cv2.typing.Point
:param pt2: Vertex of the rectangle opposite to pt1 .
:type pt2: cv2.typing.Point
:param color: Rectangle color or brightness (grayscale image).
:type color: cv2.typing.Scalar
:param thickness: Thickness of lines that make up the rectangle. Negative values, like #FILLED,mean that the function has to draw a filled rectangle. 
:type thickness: int
:param lineType: Type of the line. See #LineTypes
:type lineType: int
:param shift: Number of fractional bits in the point coordinates.
:type shift: int
:rtype: cv2.typing.MatLike
````


````{py:function} rectangleIntersectionArea(a, b) -> retval

Finds out if there is any intersection between two rectangles


mainly useful for language bindings 


:param a: First rectangle
:type a: cv2.typing.Rect2d
:param b: Second rectangle
:type b: cv2.typing.Rect2d
:return: the area of the intersection
:rtype: float
````


````{py:function} rectify3Collinear(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, cameraMatrix3, distCoeffs3, imgpt1, imgpt3, imageSize, R12, T12, R13, T13, alpha, newImgSize, flags[, R1[, R2[, R3[, P1[, P2[, P3[, Q]]]]]]]) -> retval, R1, R2, R3, P1, P2, P3, Q, roi1, roi2






:param cameraMatrix1: 
:type cameraMatrix1: cv2.typing.MatLike
:param distCoeffs1: 
:type distCoeffs1: cv2.typing.MatLike
:param cameraMatrix2: 
:type cameraMatrix2: cv2.typing.MatLike
:param distCoeffs2: 
:type distCoeffs2: cv2.typing.MatLike
:param cameraMatrix3: 
:type cameraMatrix3: cv2.typing.MatLike
:param distCoeffs3: 
:type distCoeffs3: cv2.typing.MatLike
:param imgpt1: 
:type imgpt1: _typing.Sequence[cv2.typing.MatLike]
:param imgpt3: 
:type imgpt3: _typing.Sequence[cv2.typing.MatLike]
:param imageSize: 
:type imageSize: cv2.typing.Size
:param R12: 
:type R12: cv2.typing.MatLike
:param T12: 
:type T12: cv2.typing.MatLike
:param R13: 
:type R13: cv2.typing.MatLike
:param T13: 
:type T13: cv2.typing.MatLike
:param alpha: 
:type alpha: float
:param newImgSize: 
:type newImgSize: cv2.typing.Size
:param flags: 
:type flags: int
:param R1: 
:type R1: cv2.typing.MatLike | None
:param R2: 
:type R2: cv2.typing.MatLike | None
:param R3: 
:type R3: cv2.typing.MatLike | None
:param P1: 
:type P1: cv2.typing.MatLike | None
:param P2: 
:type P2: cv2.typing.MatLike | None
:param P3: 
:type P3: cv2.typing.MatLike | None
:param Q: 
:type Q: cv2.typing.MatLike | None
:rtype: tuple[float, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.Rect, cv2.typing.Rect]
````


````{py:function} redirectError(onError) -> None






:param onError: 
:type onError: _typing.Callable[[int, str, str, str, int], None] | None
:rtype: None
````


````{py:function} reduce(src, dim, rtype[, dst[, dtype]]) -> dst

Reduces a matrix to a vector.


The function #reduce reduces the matrix to a vector by treating the matrix rows/columns as a set of 1D vectors and performing the specified operation on the vectors until a single row/column is obtained. For example, the function can be used to compute horizontal and vertical projections of a raster image. In case of #REDUCE_MAX and #REDUCE_MIN , the output image should have the same type as the source one. In case of #REDUCE_SUM, #REDUCE_SUM2 and #REDUCE_AVG , the output may have a larger element bit-depth to preserve accuracy. And multi-channel arrays are also supported in these two reduction modes. 
The following code demonstrates its usage for a single channel matrix. @snippet snippets/core_reduce.cpp example 
And the following code demonstrates its usage for a two-channel matrix. @snippet snippets/core_reduce.cpp example2 
**See also:** repeat, reduceArgMin, reduceArgMax


:param src: input 2D matrix.
:type src: cv2.typing.MatLike
:param dst: output vector. Its size and type is defined by dim and dtype parameters.
:type dst: cv2.typing.MatLike | None
:param dim: dimension index along which the matrix is reduced. 0 means that the matrix is reduced toa single row. 1 means that the matrix is reduced to a single column. 
:type dim: int
:param rtype: reduction operation that could be one of #ReduceTypes
:type rtype: int
:param dtype: when negative, the output vector will have the same type as the input matrix,otherwise, its type will be CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()). 
:type dtype: int
:rtype: cv2.typing.MatLike
````


````{py:function} reduceArgMax(src, axis[, dst[, lastIndex]]) -> dst

Finds indices of max elements along provided axis


@note - If input or output array is not continuous, this function will create an internal copy. - NaN handling is left unspecified, see patchNaNs(). - The returned index is always in bounds of input matrix. 
**See also:** reduceArgMin, minMaxLoc, min, max, compare, reduce


:param src: input single-channel array.
:type src: cv2.typing.MatLike
:param dst: output array of type CV_32SC1 with the same dimensionality as src,except for axis being reduced - it should be set to 1. 
:type dst: cv2.typing.MatLike | None
:param lastIndex: whether to get the index of first or last occurrence of max.
:type lastIndex: bool
:param axis: axis to reduce along.
:type axis: int
:rtype: cv2.typing.MatLike
````


````{py:function} reduceArgMin(src, axis[, dst[, lastIndex]]) -> dst

Finds indices of min elements along provided axis


@note - If input or output array is not continuous, this function will create an internal copy. - NaN handling is left unspecified, see patchNaNs(). - The returned index is always in bounds of input matrix. 
**See also:** reduceArgMax, minMaxLoc, min, max, compare, reduce


:param src: input single-channel array.
:type src: cv2.typing.MatLike
:param dst: output array of type CV_32SC1 with the same dimensionality as src,except for axis being reduced - it should be set to 1. 
:type dst: cv2.typing.MatLike | None
:param lastIndex: whether to get the index of first or last occurrence of min.
:type lastIndex: bool
:param axis: axis to reduce along.
:type axis: int
:rtype: cv2.typing.MatLike
````


````{py:function} remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) -> dst

Applies a generic geometrical transformation to an image.


The function remap transforms the source image using the specified map: 
$\texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))$ 
where values of pixels with non-integer coordinates are computed using one of available interpolation methods. $map_x$ and $map_y$ can be encoded as separate floating-point maps in $map_1$ and $map_2$ respectively, or interleaved floating-point maps of $(x,y)$ in $map_1$, or fixed-point maps created by using #convertMaps. The reason you might want to convert from floating to fixed-point representations of a map is that they can yield much faster (\~2x) remapping operations. In the converted case, $map_1$ contains pairs (cvFloor(x), cvFloor(y)) and $map_2$ contains indices in a table of interpolation coefficients. 
This function cannot operate in-place. 


:param src: Source image.
:type src: cv2.typing.MatLike
:param dst: Destination image. It has the same size as map1 and the same type as src .
:type dst: cv2.typing.MatLike | None
:param map1: The first map of either (x,y) points or just x values having the type CV_16SC2 ,CV_32FC1, or CV_32FC2. See #convertMaps for details on converting a floating point representation to fixed-point for speed. 
:type map1: cv2.typing.MatLike
:param map2: The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty mapif map1 is (x,y) points), respectively. 
:type map2: cv2.typing.MatLike
:param interpolation: Interpolation method (see #InterpolationFlags). The methods #INTER_AREAand #INTER_LINEAR_EXACT are not supported by this function. 
:type interpolation: int
:param borderMode: Pixel extrapolation method (see #BorderTypes). WhenborderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function. 
:type borderMode: int
:param borderValue: Value used in case of a constant border. By default, it is 0.@note Due to current implementation limitations the size of an input and output images should be less than 32767x32767. 
:type borderValue: cv2.typing.Scalar
:rtype: cv2.typing.MatLike
````


````{py:function} repeat(src, ny, nx[, dst]) -> dst

Fills the output array with repeated copies of the input array.


The function cv::repeat duplicates the input array one or more times along each of the two axes: $\texttt{dst} _{ij}= \texttt{src} _{i\mod src.rows, \; j\mod src.cols }$ The second variant of the function is more convenient to use with @ref MatrixExpressions. 
**See also:** cv::reduce


:param src: input array to replicate.
:type src: cv2.typing.MatLike
:param ny: Flag to specify how many times the `src` is repeated along thevertical axis. 
:type ny: int
:param nx: Flag to specify how many times the `src` is repeated along thehorizontal axis. 
:type nx: int
:param dst: output array of the same type as `src`.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} reprojectImageTo3D(disparity, Q[, _3dImage[, handleMissingValues[, ddepth]]]) -> _3dImage

Reprojects a disparity image to 3D space.


The function transforms a single-channel disparity map to a 3-channel image representing a 3D surface. That is, for each pixel (x,y) and the corresponding disparity d=disparity(x,y) , it computes: 
$\begin{bmatrix} X \\ Y \\ Z \\ W \end{bmatrix} = Q \begin{bmatrix} x \\ y \\ \texttt{disparity} (x,y) \\ z \end{bmatrix}.$ 
@sa To reproject a sparse set of points {(x,y,d),...} to 3D space, use perspectiveTransform. 


:param disparity: Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bitfloating-point disparity image. The values of 8-bit / 16-bit signed formats are assumed to have no fractional bits. If the disparity is 16-bit signed format, as computed by @ref StereoBM or @ref StereoSGBM and maybe other algorithms, it should be divided by 16 (and scaled to float) before being used here. 
:type disparity: cv2.typing.MatLike
:param _3dImage: Output 3-channel floating-point image of the same size as disparity. Each element of_3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map. If one uses Q obtained by @ref stereoRectify, then the returned points are represented in the first camera's rectified coordinate system. 
:type _3dImage: cv2.typing.MatLike | None
:param Q: $4 \times 4$ perspective transformation matrix that can be obtained with@ref stereoRectify. 
:type Q: cv2.typing.MatLike
:param handleMissingValues: Indicates, whether the function should handle missing values (i.e.points where the disparity was not computed). If handleMissingValues=true, then pixels with the minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed to 3D points with a very large Z value (currently set to 10000). 
:type handleMissingValues: bool
:param ddepth: The optional output array depth. If it is -1, the output image will have CV_32Fdepth. ddepth can also be set to CV_16S, CV_32S or CV_32F. 
:type ddepth: int
:rtype: cv2.typing.MatLike
````


````{py:function} resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst

Resizes an image.


The function resize resizes the image src down to or up to the specified size. Note that the initial dst type or size are not taken into account. Instead, the size and type are derived from the `src`,`dsize`,`fx`, and `fy`. If you want to resize src so that it fits the pre-created dst, you may call the function as follows: 
```c++
// explicitly specify dsize=dst.size(); fx and fy will be computed from that.
resize(src, dst, dst.size(), 0, 0, interpolation);
```
If you want to decimate the image by factor of 2 in each direction, you can call the function this way: 
```c++
// specify fx and fy and let the function compute the destination image size.
resize(src, dst, Size(), 0.5, 0.5, interpolation);
```
To shrink an image, it will generally look best with #INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with #INTER_CUBIC (slow) or #INTER_LINEAR (faster but still looks OK). 
**See also:**  warpAffine, warpPerspective, remap


:param src: input image.
:type src: cv2.typing.MatLike
:param dst: output image; it has the size dsize (when it is non-zero) or the size computed fromsrc.size(), fx, and fy; the type of dst is the same as of src. 
:type dst: cv2.typing.MatLike | None
:param dsize: output image size; if it equals zero (`None` in Python), it is computed as:$\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}$ Either dsize or both fx and fy must be non-zero. 
:type dsize: cv2.typing.Size | None
:param fx: scale factor along the horizontal axis; when it equals 0, it is computed as$\texttt{(double)dsize.width/src.cols}$ 
:type fx: float
:param fy: scale factor along the vertical axis; when it equals 0, it is computed as$\texttt{(double)dsize.height/src.rows}$ 
:type fy: float
:param interpolation: interpolation method, see #InterpolationFlags
:type interpolation: int
:rtype: cv2.typing.MatLike
````


````{py:function} resizeWindow(winname, width, height) -> None

Resizes the window to the specified size


resizeWindow(winname, size) -> None @overload 
```{note}
The specified window size is for the image area. Toolbars are not counted.Only windows created without cv::WINDOW_AUTOSIZE flag can be resized. 
```


:param winname: Window name.
:type winname: str
:param width: The new window width.
:type width: int
:param height: The new window height.
:type height: int
:param size: The new window size.
:type size: 
:rtype: None
````


````{py:function} rotate(src, rotateCode[, dst]) -> dst

Rotates a 2D array in multiples of 90 degrees.The function cv::rotate rotates the array in one of three different ways: Rotate by 90 degrees clockwise (rotateCode = ROTATE_90_CLOCKWISE). Rotate by 180 degrees clockwise (rotateCode = ROTATE_180). Rotate by 270 degrees clockwise (rotateCode = ROTATE_90_COUNTERCLOCKWISE). 


**See also:** transpose , repeat , completeSymm, flip, RotateFlags


:param src: input array.
:type src: cv2.typing.MatLike
:param dst: output array of the same type as src. The size is the same with ROTATE_180,and the rows and cols are switched for ROTATE_90_CLOCKWISE and ROTATE_90_COUNTERCLOCKWISE. 
:type dst: cv2.typing.MatLike | None
:param rotateCode: an enum to specify how to rotate the array; see the enum #RotateFlags
:type rotateCode: int
:rtype: cv2.typing.MatLike
````


````{py:function} rotatedRectangleIntersection(rect1, rect2[, intersectingRegion]) -> retval, intersectingRegion

Finds out if there is any intersection between two rotated rectangles.


If there is then the vertices of the intersecting region are returned as well. 
Below are some examples of intersection configurations. The hatched pattern indicates the intersecting region and the red vertices are returned by the function. 
![intersection examples](pics/intersection.png) 


:param rect1: First rectangle
:type rect1: cv2.typing.RotatedRect
:param rect2: Second rectangle
:type rect2: cv2.typing.RotatedRect
:param intersectingRegion: The output array of the vertices of the intersecting region. It returnsat most 8 vertices. Stored as std::vector\<cv::Point2f\> or cv::Mat as Mx1 of type CV_32FC2. 
:type intersectingRegion: cv2.typing.MatLike | None
:return: One of #RectanglesIntersectTypes
:rtype: tuple[int, cv2.typing.MatLike]
````


````{py:function} sampsonDistance(pt1, pt2, F) -> retval

Calculates the Sampson Distance between two points.


The function cv::sampsonDistance calculates and returns the first order approximation of the geometric error as: $ sd( \texttt{pt1} , \texttt{pt2} )= \frac{(\texttt{pt2}^t \cdot \texttt{F} \cdot \texttt{pt1})^2} {((\texttt{F} \cdot \texttt{pt1})(0))^2 + ((\texttt{F} \cdot \texttt{pt1})(1))^2 + ((\texttt{F}^t \cdot \texttt{pt2})(0))^2 + ((\texttt{F}^t \cdot \texttt{pt2})(1))^2} $ The fundamental matrix may be calculated using the #findFundamentalMat function. See @cite HartleyZ00 11.4.3 for details. 


:param pt1: first homogeneous 2d point
:type pt1: cv2.typing.MatLike
:param pt2: second homogeneous 2d point
:type pt2: cv2.typing.MatLike
:param F: fundamental matrix
:type F: cv2.typing.MatLike
:return: The computed Sampson distance.
:rtype: float
````


````{py:function} scaleAdd(src1, alpha, src2[, dst]) -> dst

Calculates the sum of a scaled array and another array.


The function scaleAdd is one of the classical primitive linear algebra operations, known as DAXPY or SAXPY in [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). It calculates the sum of a scaled array and another array: $\texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I) +  \texttt{src2} (I)$ The function can also be emulated with a matrix expression, for example: 
```cpp
Mat A(3, 3, CV_64F);

A.row(0) = A.row(1)*2 + A.row(2);
```

**See also:** add, addWeighted, subtract, Mat::dot, Mat::convertTo


:param src1: first input array.
:type src1: cv2.typing.MatLike
:param alpha: scale factor for the first array.
:type alpha: float
:param src2: second input array of the same size and type as src1.
:type src2: cv2.typing.MatLike
:param dst: output array of the same size and type as src1.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} seamlessClone(src, dst, mask, p, flags[, blend]) -> blend

Image editing tasks concern either global changes (color/intensity corrections, filters,deformations) or local changes concerned to a selection. Here we are interested in achieving local changes, ones that are restricted to a region manually selected (ROI), in a seamless and effortless manner. The extent of the changes ranges from slight distortions to complete replacement by novel content @cite PM03 . 




:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param dst: Input 8-bit 3-channel image.
:type dst: cv2.typing.MatLike
:param mask: Input 8-bit 1 or 3-channel image.
:type mask: cv2.typing.MatLike
:param p: Point in dst image where object is placed.
:type p: cv2.typing.Point
:param blend: Output image with the same size and type as dst.
:type blend: cv2.typing.MatLike | None
:param flags: Cloning method that could be cv::NORMAL_CLONE, cv::MIXED_CLONE or cv::MONOCHROME_TRANSFER
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} selectROI(windowName, img[, showCrosshair[, fromCenter[, printNotice]]]) -> retval

Allows users to select a ROI on the given image.


The function creates a window and allows users to select a ROI using the mouse. Controls: use `space` or `enter` to finish selection, use key `c` to cancel selection (function will return the zero cv::Rect). 
selectROI(img[, showCrosshair[, fromCenter[, printNotice]]]) -> retval @overload 
```{note}
The function sets it's own mouse callback for specified window using cv::setMouseCallback(windowName, ...).After finish of work an empty callback will be set for the used window. 
```


:param windowName: name of the window where selection process will be shown.
:type windowName: str
:param img: image to select a ROI.
:type img: cv2.typing.MatLike
:param showCrosshair: if true crosshair of selection rectangle will be shown.
:type showCrosshair: bool
:param fromCenter: if true center of selection will match initial mouse position. In opposite case a corner ofselection rectangle will correspont to the initial mouse position. 
:type fromCenter: bool
:param printNotice: if true a notice to select ROI or cancel selection will be printed in console.
:type printNotice: bool
:return: selected ROI or empty rect if selection canceled.
:rtype: cv2.typing.Rect
````


````{py:function} selectROIs(windowName, img[, showCrosshair[, fromCenter[, printNotice]]]) -> boundingBoxes

Allows users to select multiple ROIs on the given image.


The function creates a window and allows users to select multiple ROIs using the mouse. Controls: use `space` or `enter` to finish current selection and start a new one, use `esc` to terminate multiple ROI selection process. 
```{note}
The function sets it's own mouse callback for specified window using cv::setMouseCallback(windowName, ...).After finish of work an empty callback will be set for the used window. 
```


:param windowName: name of the window where selection process will be shown.
:type windowName: str
:param img: image to select a ROI.
:type img: cv2.typing.MatLike
:param boundingBoxes: selected ROIs.
:type boundingBoxes: 
:param showCrosshair: if true crosshair of selection rectangle will be shown.
:type showCrosshair: bool
:param fromCenter: if true center of selection will match initial mouse position. In opposite case a corner ofselection rectangle will correspont to the initial mouse position. 
:type fromCenter: bool
:param printNotice: if true a notice to select ROI or cancel selection will be printed in console.
:type printNotice: bool
:rtype: _typing.Sequence[cv2.typing.Rect]
````


````{py:function} sepFilter2D(src, ddepth, kernelX, kernelY[, dst[, anchor[, delta[, borderType]]]]) -> dst

Applies a separable linear filter to an image.


The function applies a separable linear filter to the image. That is, first, every row of src is filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D kernel kernelY. The final result shifted by delta is stored in dst . 
**See also:**  filter2D, Sobel, GaussianBlur, boxFilter, blur


:param src: Source image.
:type src: cv2.typing.MatLike
:param dst: Destination image of the same size and the same number of channels as src .
:type dst: cv2.typing.MatLike | None
:param ddepth: Destination image depth, see @ref filter_depths "combinations"
:type ddepth: int
:param kernelX: Coefficients for filtering each row.
:type kernelX: cv2.typing.MatLike
:param kernelY: Coefficients for filtering each column.
:type kernelY: cv2.typing.MatLike
:param anchor: Anchor position within the kernel. The default value $(-1,-1)$ means that the anchoris at the kernel center. 
:type anchor: cv2.typing.Point
:param delta: Value added to the filtered results before storing them.
:type delta: float
:param borderType: Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} setIdentity(mtx[, s]) -> mtx

Initializes a scaled identity matrix.


The function cv::setIdentity initializes a scaled identity matrix: $\texttt{mtx} (i,j)= \fork{\texttt{value}}{ if \(i=j\)}{0}{otherwise}$ 
The function can also be emulated using the matrix initializers and the matrix expressions: 
```c++
Mat A = Mat::eye(4, 3, CV_32F)*5;
// A will be set to [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0]]
```

**See also:** Mat::zeros, Mat::ones, Mat::setTo, Mat::operator=


:param mtx: matrix to initialize (not necessarily square).
:type mtx: cv2.typing.MatLike
:param s: value to assign to diagonal elements.
:type s: cv2.typing.Scalar
:rtype: cv2.typing.MatLike
````


````{py:function} setLogLevel(level) -> retval






:param level: 
:type level: int
:rtype: int
````


````{py:function} setMouseCallback(windowName, onMouse [, param]) -> None






:param windowName: 
:type windowName: str
:param onMouse: 
:type onMouse: _typing.Callable[[int, int, int, int, _typing.Any | None], None]
:param param: 
:type param: _typing.Any | None
:rtype: None
````


````{py:function} setNumThreads(nthreads) -> None

OpenCV will try to set the number of threads for subsequent parallel regions.


If threads == 1, OpenCV will disable threading optimizations and run all it's functions sequentially. Passing threads \< 0 will reset threads number to system default. The function is not thread-safe. It must not be called in parallel region or concurrent threads. 
OpenCV will try to run its functions with specified threads number, but some behaviour differs from framework: -   `TBB` - User-defined parallel constructions will run with the same threads number, if another is not specified. If later on user creates his own scheduler, OpenCV will use it. -   `OpenMP` - No special defined behaviour. -   `Concurrency` - If threads == 1, OpenCV will disable threading optimizations and run its functions sequentially. -   `GCD` - Supports only values \<= 0. -   `C=` - No special defined behaviour. 
**See also:** getNumThreads, getThreadNum


:param nthreads: Number of threads used by OpenCV.
:type nthreads: int
:rtype: None
````


````{py:function} setRNGSeed(seed) -> None

Sets state of default random number generator.


The function cv::setRNGSeed sets state of default random number generator to custom value. 
**See also:** RNG, randu, randn


:param seed: new state for default random number generator
:type seed: int
:rtype: None
````


````{py:function} setTrackbarMax(trackbarname, winname, maxval) -> None

Sets the trackbar maximum position.


The function sets the maximum position of the specified trackbar in the specified window. 
```{note}
[__Qt Backend Only__] winname can be empty if the trackbar is attached to the controlpanel. 
```


:param trackbarname: Name of the trackbar.
:type trackbarname: str
:param winname: Name of the window that is the parent of trackbar.
:type winname: str
:param maxval: New maximum position.
:type maxval: int
:rtype: None
````


````{py:function} setTrackbarMin(trackbarname, winname, minval) -> None

Sets the trackbar minimum position.


The function sets the minimum position of the specified trackbar in the specified window. 
```{note}
[__Qt Backend Only__] winname can be empty if the trackbar is attached to the controlpanel. 
```


:param trackbarname: Name of the trackbar.
:type trackbarname: str
:param winname: Name of the window that is the parent of trackbar.
:type winname: str
:param minval: New minimum position.
:type minval: int
:rtype: None
````


````{py:function} setTrackbarPos(trackbarname, winname, pos) -> None

Sets the trackbar position.


The function sets the position of the specified trackbar in the specified window. 
```{note}
[__Qt Backend Only__] winname can be empty if the trackbar is attached to the controlpanel. 
```


:param trackbarname: Name of the trackbar.
:type trackbarname: str
:param winname: Name of the window that is the parent of trackbar.
:type winname: str
:param pos: New position.
:type pos: int
:rtype: None
````


````{py:function} setUseOpenVX(flag) -> None






:param flag: 
:type flag: bool
:rtype: None
````


````{py:function} setUseOptimized(onoff) -> None

Enables or disables the optimized code.


The function can be used to dynamically turn on and off optimized dispatched code (code that uses SSE4.2, AVX/AVX2, and other instructions on the platforms that support it). It sets a global flag that is further checked by OpenCV functions. Since the flag is not checked in the inner OpenCV loops, it is only safe to call the function on the very top level in your application where you can be sure that no other OpenCV function is currently executed. 
By default, the optimized code is enabled unless you disable it in CMake. The current status can be retrieved using useOptimized. 


:param onoff: The boolean flag specifying whether the optimized code should be used (onoff=true)or not (onoff=false). 
:type onoff: bool
:rtype: None
````


````{py:function} setWindowProperty(winname, prop_id, prop_value) -> None

Changes parameters of a window dynamically.


The function setWindowProperty enables changing properties of a window. 


:param winname: Name of the window.
:type winname: str
:param prop_id: Window property to edit. The supported operation flags are: (cv::WindowPropertyFlags)
:type prop_id: int
:param prop_value: New value of the window property. The supported flags are: (cv::WindowFlags)
:type prop_value: float
:rtype: None
````


````{py:function} setWindowTitle(winname, title) -> None

Updates window title




:param winname: Name of the window.
:type winname: str
:param title: New title.
:type title: str
:rtype: None
````


````{py:function} solve(src1, src2[, dst[, flags]]) -> retval, dst

Solves one or more linear systems or least-squares problems.


The function cv::solve solves a linear system or least-squares problem (the latter is possible with SVD or QR methods, or by specifying the flag #DECOMP_NORMAL ): $\texttt{dst} =  \arg \min _X \| \texttt{src1} \cdot \texttt{X} -  \texttt{src2} \|$ 
If #DECOMP_LU or #DECOMP_CHOLESKY method is used, the function returns 1 if src1 (or $\texttt{src1}^T\texttt{src1}$ ) is non-singular. Otherwise, it returns 0. In the latter case, dst is not valid. Other methods find a pseudo-solution in case of a singular left-hand side part. 
```{note}
If you want to find a unity-norm solution of an under-definedsingular system $\texttt{src1}\cdot\texttt{dst}=0$ , the function solve will not do the work. Use SVD::solveZ instead. 
```
**See also:** invert, SVD, eigen


:param src1: input matrix on the left-hand side of the system.
:type src1: cv2.typing.MatLike
:param src2: input matrix on the right-hand side of the system.
:type src2: cv2.typing.MatLike
:param dst: output solution.
:type dst: cv2.typing.MatLike | None
:param flags: solution (matrix inversion) method (#DecompTypes)
:type flags: int
:rtype: tuple[bool, cv2.typing.MatLike]
````


````{py:function} solveCubic(coeffs[, roots]) -> retval, roots

Finds the real roots of a cubic equation.


The function solveCubic finds the real roots of a cubic equation: -   if coeffs is a 4-element vector: $\texttt{coeffs} [0] x^3 +  \texttt{coeffs} [1] x^2 +  \texttt{coeffs} [2] x +  \texttt{coeffs} [3] = 0$ -   if coeffs is a 3-element vector: $x^3 +  \texttt{coeffs} [0] x^2 +  \texttt{coeffs} [1] x +  \texttt{coeffs} [2] = 0$ 
The roots are stored in the roots array. 


:param coeffs: equation coefficients, an array of 3 or 4 elements.
:type coeffs: cv2.typing.MatLike
:param roots: output array of real roots that has 1 or 3 elements.
:type roots: cv2.typing.MatLike | None
:return: number of real roots. It can be 0, 1 or 2.
:rtype: tuple[int, cv2.typing.MatLike]
````


````{py:function} solveLP(Func, Constr, constr_eps[, z]) -> retval, z

Solve given (non-integer) linear programming problem using the Simplex Algorithm (Simplex Method).


What we mean here by "linear programming problem" (or LP problem, for short) can be formulated as: 
$\mbox{Maximize } c\cdot x\\ \mbox{Subject to:}\\ Ax\leq b\\ x\geq 0$ 
Where $c$ is fixed `1`-by-`n` row-vector, $A$ is fixed `m`-by-`n` matrix, $b$ is fixed `m`-by-`1` column vector and $x$ is an arbitrary `n`-by-`1` column vector, which satisfies the constraints. 
Simplex algorithm is one of many algorithms that are designed to handle this sort of problems efficiently. Although it is not optimal in theoretical sense (there exist algorithms that can solve any problem written as above in polynomial time, while simplex method degenerates to exponential time for some special cases), it is well-studied, easy to implement and is shown to work well for real-life purposes. 
The particular implementation is taken almost verbatim from **Introduction to Algorithms, third edition** by T. H. Cormen, C. E. Leiserson, R. L. Rivest and Clifford Stein. In particular, the Bland's rule <http://en.wikipedia.org/wiki/Bland%27s_rule> is used to prevent cycling. 
solveLP(Func, Constr[, z]) -> retval, z @overload 


:param Func: This row-vector corresponds to $c$ in the LP problem formulation (see above). It shouldcontain 32- or 64-bit floating point numbers. As a convenience, column-vector may be also submitted, in the latter case it is understood to correspond to $c^T$. 
:type Func: cv2.typing.MatLike
:param Constr: `m`-by-`n+1` matrix, whose rightmost column corresponds to $b$ in formulation aboveand the remaining to $A$. It should contain 32- or 64-bit floating point numbers. 
:type Constr: cv2.typing.MatLike
:param z: The solution will be returned here as a column-vector - it corresponds to $c$ in theformulation above. It will contain 64-bit floating point numbers. 
:type z: cv2.typing.MatLike | None
:param constr_eps: allowed numeric disparity for constraints
:type constr_eps: float
:return: One of cv::SolveLPResult
:rtype: tuple[int, cv2.typing.MatLike]
````


````{py:function} solveP3P(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags[, rvecs[, tvecs]]) -> retval, rvecs, tvecs

Finds an object pose from 3 3D-2D point correspondences.


The function estimates the object pose given 3 object points, their corresponding image projections, as well as the camera intrinsic matrix and the distortion coefficients. 
@note The solutions are sorted by reprojection errors (lowest to highest). 
**See also:** @ref calib3d_solvePnP


:param objectPoints: Array of object points in the object coordinate space, 3x3 1-channel or1x3/3x1 3-channel. vector\<Point3f\> can be also passed here. 
:type objectPoints: cv2.typing.MatLike
:param imagePoints: Array of corresponding image points, 3x2 1-channel or 1x3/3x1 2-channel.vector\<Point2f\> can be also passed here. 
:type imagePoints: cv2.typing.MatLike
:param cameraMatrix: Input camera intrinsic matrix $\cameramatrix{A}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$\distcoeffs$. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param rvecs: Output rotation vectors (see @ref Rodrigues ) that, together with tvecs, brings points fromthe model coordinate system to the camera coordinate system. A P3P problem has up to 4 solutions. 
:type rvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param tvecs: Output translation vectors.
:type tvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param flags: Method for solving a P3P problem:-   @ref SOLVEPNP_P3P Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang "Complete Solution Classification for the Perspective-Three-Point Problem" (@cite gao2003complete). -   @ref SOLVEPNP_AP3P Method is based on the paper of T. Ke and S. Roumeliotis. "An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17). 
:type flags: int
:rtype: tuple[int, _typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike]]
````


````{py:function} solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) -> retval, rvec, tvec

Finds an object pose from 3D-2D point correspondences.


This function returns the rotation and the translation vectors that transform a 3D point expressed in the object coordinate frame to the camera coordinate frame, using different methods: - P3P methods (@ref SOLVEPNP_P3P, @ref SOLVEPNP_AP3P): need 4 input points to return a unique solution. - @ref SOLVEPNP_IPPE Input points must be >= 4 and object points must be coplanar. - @ref SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation. Number of input points must be 4. Object points must be defined in the following order: - point 0: [-squareLength / 2,  squareLength / 2, 0] - point 1: [ squareLength / 2,  squareLength / 2, 0] - point 2: [ squareLength / 2, -squareLength / 2, 0] - point 3: [-squareLength / 2, -squareLength / 2, 0] - for all the other flags, number of input points must be >= 4 and object points can be in any configuration. 
More information about Perspective-n-Points is described in @ref calib3d_solvePnP 
@note -   An example of how to use solvePnP for planar augmented reality can be found at opencv_source_code/samples/python/plane_ar.py -   If you are using Python: - Numpy array slices won't work as input because solvePnP requires contiguous arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of modules/calib3d/src/solvepnp.cpp version 2.4.9) - The P3P algorithm requires image points to be in an array of shape (N,1,2) due to its calling of #undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9) which requires 2-channel information. - Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints = np.ascontiguousarray(D[:,:2]).reshape((N,1,2)) -   The methods @ref SOLVEPNP_DLS and @ref SOLVEPNP_UPNP cannot be used as the current implementations are unstable and sometimes give completely wrong results. If you pass one of these two flags, @ref SOLVEPNP_EPNP method will be used instead. -   The minimum number of points is 4 in the general case. In the case of @ref SOLVEPNP_P3P and @ref SOLVEPNP_AP3P methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error). -   With @ref SOLVEPNP_ITERATIVE method and `useExtrinsicGuess=true`, the minimum number of points is 3 (3 points are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the global solution to converge. -   With @ref SOLVEPNP_IPPE input points must be >= 4 and object points must be coplanar. -   With @ref SOLVEPNP_IPPE_SQUARE this is a special case suitable for marker pose estimation. Number of input points must be 4. Object points must be defined in the following order: - point 0: [-squareLength / 2,  squareLength / 2, 0] - point 1: [ squareLength / 2,  squareLength / 2, 0] - point 2: [ squareLength / 2, -squareLength / 2, 0] - point 3: [-squareLength / 2, -squareLength / 2, 0] -  With @ref SOLVEPNP_SQPNP input points must be >= 3 
**See also:** @ref calib3d_solvePnP


:param objectPoints: Array of object points in the object coordinate space, Nx3 1-channel or1xN/Nx1 3-channel, where N is the number of points. vector\<Point3d\> can be also passed here. 
:type objectPoints: cv2.typing.MatLike
:param imagePoints: Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,where N is the number of points. vector\<Point2d\> can be also passed here. 
:type imagePoints: cv2.typing.MatLike
:param cameraMatrix: Input camera intrinsic matrix $\cameramatrix{A}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$\distcoeffs$. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param rvec: Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points fromthe model coordinate system to the camera coordinate system. 
:type rvec: cv2.typing.MatLike | None
:param tvec: Output translation vector.
:type tvec: cv2.typing.MatLike | None
:param useExtrinsicGuess: Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function usesthe provided rvec and tvec values as initial approximations of the rotation and translation vectors, respectively, and further optimizes them. 
:type useExtrinsicGuess: bool
:param flags: Method for solving a PnP problem: see @ref calib3d_solvePnP_flags
:type flags: int
:rtype: tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} solvePnPGeneric(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvecs[, tvecs[, useExtrinsicGuess[, flags[, rvec[, tvec[, reprojectionError]]]]]]]) -> retval, rvecs, tvecs, reprojectionError

Finds an object pose from 3D-2D point correspondences.


This function returns a list of all the possible solutions (a solution is a <rotation vector, translation vector> couple), depending on the number of input points and the chosen method: - P3P methods (@ref SOLVEPNP_P3P, @ref SOLVEPNP_AP3P): 3 or 4 input points. Number of returned solutions can be between 0 and 4 with 3 input points. - @ref SOLVEPNP_IPPE Input points must be >= 4 and object points must be coplanar. Returns 2 solutions. - @ref SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation. Number of input points must be 4 and 2 solutions are returned. Object points must be defined in the following order: - point 0: [-squareLength / 2,  squareLength / 2, 0] - point 1: [ squareLength / 2,  squareLength / 2, 0] - point 2: [ squareLength / 2, -squareLength / 2, 0] - point 3: [-squareLength / 2, -squareLength / 2, 0] - for all the other flags, number of input points must be >= 4 and object points can be in any configuration. Only 1 solution is returned. 
More information is described in @ref calib3d_solvePnP 
@note -   An example of how to use solvePnP for planar augmented reality can be found at opencv_source_code/samples/python/plane_ar.py -   If you are using Python: - Numpy array slices won't work as input because solvePnP requires contiguous arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of modules/calib3d/src/solvepnp.cpp version 2.4.9) - The P3P algorithm requires image points to be in an array of shape (N,1,2) due to its calling of #undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9) which requires 2-channel information. - Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints = np.ascontiguousarray(D[:,:2]).reshape((N,1,2)) -   The methods @ref SOLVEPNP_DLS and @ref SOLVEPNP_UPNP cannot be used as the current implementations are unstable and sometimes give completely wrong results. If you pass one of these two flags, @ref SOLVEPNP_EPNP method will be used instead. -   The minimum number of points is 4 in the general case. In the case of @ref SOLVEPNP_P3P and @ref SOLVEPNP_AP3P methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error). -   With @ref SOLVEPNP_ITERATIVE method and `useExtrinsicGuess=true`, the minimum number of points is 3 (3 points are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the global solution to converge. -   With @ref SOLVEPNP_IPPE input points must be >= 4 and object points must be coplanar. -   With @ref SOLVEPNP_IPPE_SQUARE this is a special case suitable for marker pose estimation. Number of input points must be 4. Object points must be defined in the following order: - point 0: [-squareLength / 2,  squareLength / 2, 0] - point 1: [ squareLength / 2,  squareLength / 2, 0] - point 2: [ squareLength / 2, -squareLength / 2, 0] - point 3: [-squareLength / 2, -squareLength / 2, 0] 
**See also:** @ref calib3d_solvePnP


:param objectPoints: Array of object points in the object coordinate space, Nx3 1-channel or1xN/Nx1 3-channel, where N is the number of points. vector\<Point3d\> can be also passed here. 
:type objectPoints: cv2.typing.MatLike
:param imagePoints: Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,where N is the number of points. vector\<Point2d\> can be also passed here. 
:type imagePoints: cv2.typing.MatLike
:param cameraMatrix: Input camera intrinsic matrix $\cameramatrix{A}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$\distcoeffs$. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param rvecs: Vector of output rotation vectors (see @ref Rodrigues ) that, together with tvecs, brings points fromthe model coordinate system to the camera coordinate system. 
:type rvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param tvecs: Vector of output translation vectors.
:type tvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param useExtrinsicGuess: Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function usesthe provided rvec and tvec values as initial approximations of the rotation and translation vectors, respectively, and further optimizes them. 
:type useExtrinsicGuess: bool
:param flags: Method for solving a PnP problem: see @ref calib3d_solvePnP_flags
:type flags: SolvePnPMethod
:param rvec: Rotation vector used to initialize an iterative PnP refinement algorithm, when flag is @ref SOLVEPNP_ITERATIVEand useExtrinsicGuess is set to true. 
:type rvec: cv2.typing.MatLike | None
:param tvec: Translation vector used to initialize an iterative PnP refinement algorithm, when flag is @ref SOLVEPNP_ITERATIVEand useExtrinsicGuess is set to true. 
:type tvec: cv2.typing.MatLike | None
:param reprojectionError: Optional vector of reprojection error, that is the RMS error($ \text{RMSE} = \sqrt{\frac{\sum_{i}^{N} \left ( \hat{y_i} - y_i \right )^2}{N}} $) between the input image points and the 3D object points projected with the estimated pose. 
:type reprojectionError: cv2.typing.MatLike | None
:rtype: tuple[int, _typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike]
````


````{py:function} solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, iterationsCount[, reprojectionError[, confidence[, inliers[, flags]]]]]]]]) -> retval, rvec, tvec, inliers

Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.


The function estimates an object pose given a set of object points, their corresponding image projections, as well as the camera intrinsic matrix and the distortion coefficients. This function finds such a pose that minimizes reprojection error, that is, the sum of squared distances between the observed projections imagePoints and the projected (using @ref projectPoints ) objectPoints. The use of RANSAC makes the function resistant to outliers. 
@note -   An example of how to use solvePNPRansac for object detection can be found at opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/ -   The default method used to estimate the camera pose for the Minimal Sample Sets step is #SOLVEPNP_EPNP. Exceptions are: - if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used. - if the number of input points is equal to 4, #SOLVEPNP_P3P is used. -   The method used to estimate the camera pose using all the inliers is defined by the flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case, the method #SOLVEPNP_EPNP will be used instead. 
solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, inliers[, params]]]]) -> retval, cameraMatrix, rvec, tvec, inliers 
**See also:** @ref calib3d_solvePnP


:param objectPoints: Array of object points in the object coordinate space, Nx3 1-channel or1xN/Nx1 3-channel, where N is the number of points. vector\<Point3d\> can be also passed here. 
:type objectPoints: cv2.typing.MatLike
:param imagePoints: Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,where N is the number of points. vector\<Point2d\> can be also passed here. 
:type imagePoints: cv2.typing.MatLike
:param cameraMatrix: Input camera intrinsic matrix $\cameramatrix{A}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$\distcoeffs$. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param rvec: Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points fromthe model coordinate system to the camera coordinate system. 
:type rvec: cv2.typing.MatLike | None
:param tvec: Output translation vector.
:type tvec: cv2.typing.MatLike | None
:param useExtrinsicGuess: Parameter used for @ref SOLVEPNP_ITERATIVE. If true (1), the function usesthe provided rvec and tvec values as initial approximations of the rotation and translation vectors, respectively, and further optimizes them. 
:type useExtrinsicGuess: bool
:param iterationsCount: Number of iterations.
:type iterationsCount: int
:param reprojectionError: Inlier threshold value used by the RANSAC procedure. The parameter valueis the maximum allowed distance between the observed and computed point projections to consider it an inlier. 
:type reprojectionError: float
:param confidence: The probability that the algorithm produces a useful result.
:type confidence: float
:param inliers: Output vector that contains indices of inliers in objectPoints and imagePoints .
:type inliers: cv2.typing.MatLike | None
:param flags: Method for solving a PnP problem (see @ref solvePnP ).
:type flags: int
:rtype: tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec[, criteria]) -> rvec, tvec

Refine a pose (the translation and the rotation that transform a 3D point expressed in the object coordinate frameto the camera coordinate frame) from a 3D-2D point correspondences and starting from an initial solution. 


The function refines the object pose given at least 3 object points, their corresponding image projections, an initial solution for the rotation and translation vector, as well as the camera intrinsic matrix and the distortion coefficients. The function minimizes the projection error with respect to the rotation and the translation vectors, according to a Levenberg-Marquardt iterative minimization @cite Madsen04 @cite Eade13 process. 
**See also:** @ref calib3d_solvePnP


:param objectPoints: Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,where N is the number of points. vector\<Point3d\> can also be passed here. 
:type objectPoints: cv2.typing.MatLike
:param imagePoints: Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,where N is the number of points. vector\<Point2d\> can also be passed here. 
:type imagePoints: cv2.typing.MatLike
:param cameraMatrix: Input camera intrinsic matrix $\cameramatrix{A}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$\distcoeffs$. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param rvec: Input/Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points fromthe model coordinate system to the camera coordinate system. Input values are used as an initial solution. 
:type rvec: cv2.typing.MatLike
:param tvec: Input/Output translation vector. Input values are used as an initial solution.
:type tvec: cv2.typing.MatLike
:param criteria: Criteria when to stop the Levenberg-Marquard iterative algorithm.
:type criteria: cv2.typing.TermCriteria
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} solvePnPRefineVVS(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec[, criteria[, VVSlambda]]) -> rvec, tvec

Refine a pose (the translation and the rotation that transform a 3D point expressed in the object coordinate frameto the camera coordinate frame) from a 3D-2D point correspondences and starting from an initial solution. 


The function refines the object pose given at least 3 object points, their corresponding image projections, an initial solution for the rotation and translation vector, as well as the camera intrinsic matrix and the distortion coefficients. The function minimizes the projection error with respect to the rotation and the translation vectors, using a virtual visual servoing (VVS) @cite Chaumette06 @cite Marchand16 scheme. 
**See also:** @ref calib3d_solvePnP


:param objectPoints: Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,where N is the number of points. vector\<Point3d\> can also be passed here. 
:type objectPoints: cv2.typing.MatLike
:param imagePoints: Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,where N is the number of points. vector\<Point2d\> can also be passed here. 
:type imagePoints: cv2.typing.MatLike
:param cameraMatrix: Input camera intrinsic matrix $\cameramatrix{A}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$\distcoeffs$. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param rvec: Input/Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points fromthe model coordinate system to the camera coordinate system. Input values are used as an initial solution. 
:type rvec: cv2.typing.MatLike
:param tvec: Input/Output translation vector. Input values are used as an initial solution.
:type tvec: cv2.typing.MatLike
:param criteria: Criteria when to stop the Levenberg-Marquard iterative algorithm.
:type criteria: cv2.typing.TermCriteria
:param VVSlambda: Gain for the virtual visual servoing control law, equivalent to the $\alpha$gain in the Damped Gauss-Newton formulation. 
:type VVSlambda: float
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} solvePoly(coeffs[, roots[, maxIters]]) -> retval, roots

Finds the real or complex roots of a polynomial equation.


The function cv::solvePoly finds real and complex roots of a polynomial equation: $\texttt{coeffs} [n] x^{n} +  \texttt{coeffs} [n-1] x^{n-1} + ... +  \texttt{coeffs} [1] x +  \texttt{coeffs} [0] = 0$ 


:param coeffs: array of polynomial coefficients.
:type coeffs: cv2.typing.MatLike
:param roots: output (complex) array of roots.
:type roots: cv2.typing.MatLike | None
:param maxIters: maximum number of iterations the algorithm does.
:type maxIters: int
:rtype: tuple[float, cv2.typing.MatLike]
````


````{py:function} sort(src, flags[, dst]) -> dst

Sorts each row or each column of a matrix.


The function cv::sort sorts each matrix row or each matrix column in ascending or descending order. So you should pass two operation flags to get desired behaviour. If you want to sort matrix rows or columns lexicographically, you can use STL std::sort generic function with the proper comparison predicate. 
**See also:** sortIdx, randShuffle


:param src: input single-channel array.
:type src: cv2.typing.MatLike
:param dst: output array of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param flags: operation flags, a combination of #SortFlags
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} sortIdx(src, flags[, dst]) -> dst

Sorts each row or each column of a matrix.


The function cv::sortIdx sorts each matrix row or each matrix column in the ascending or descending order. So you should pass two operation flags to get desired behaviour. Instead of reordering the elements themselves, it stores the indices of sorted elements in the output array. For example: 
```c++
Mat A = Mat::eye(3,3,CV_32F), B;
sortIdx(A, B, SORT_EVERY_ROW + SORT_ASCENDING);
// B will probably contain
// (because of equal elements in A some permutations are possible):
// [[1, 2, 0], [0, 2, 1], [0, 1, 2]]
```

**See also:** sort, randShuffle


:param src: input single-channel array.
:type src: cv2.typing.MatLike
:param dst: output integer array of the same size as src.
:type dst: cv2.typing.MatLike | None
:param flags: operation flags that could be a combination of cv::SortFlags
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} spatialGradient(src[, dx[, dy[, ksize[, borderType]]]]) -> dx, dy

Calculates the first order image derivative in both x and y using a Sobel operator


Equivalent to calling: 

```c++
Sobel( src, dx, CV_16SC1, 1, 0, 3 );
Sobel( src, dy, CV_16SC1, 0, 1, 3 );
```

**See also:** Sobel


:param src: input image.
:type src: cv2.typing.MatLike
:param dx: output image with first-order derivative in x.
:type dx: cv2.typing.MatLike | None
:param dy: output image with first-order derivative in y.
:type dy: cv2.typing.MatLike | None
:param ksize: size of Sobel kernel. It must be 3.
:type ksize: int
:param borderType: pixel extrapolation method, see #BorderTypes.Only #BORDER_DEFAULT=#BORDER_REFLECT_101 and #BORDER_REPLICATE are supported. 
:type borderType: int
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} split(m[, mv]) -> mv




@overload 


:param m: input multi-channel array.
:type m: cv2.typing.MatLike
:param mv: output vector of arrays; the arrays themselves are reallocated, if needed.
:type mv: _typing.Sequence[cv2.typing.MatLike] | None
:rtype: _typing.Sequence[cv2.typing.MatLike]
````


````{py:function} sqrBoxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]) -> dst

Calculates the normalized sum of squares of the pixel values overlapping the filter.


For every pixel $ (x, y) $ in the source image, the function calculates the sum of squares of those neighboring pixel values which overlap the filter placed over the pixel $ (x, y) $. 
The unnormalized square box filter can be useful in computing local image statistics such as the local variance and standard deviation around the neighborhood of a pixel. 
**See also:** boxFilter


:param src: input image
:type src: cv2.typing.MatLike
:param dst: output image of the same size and type as src
:type dst: cv2.typing.MatLike | None
:param ddepth: the output image depth (-1 to use src.depth())
:type ddepth: int
:param ksize: kernel size
:type ksize: cv2.typing.Size
:param anchor: kernel anchor point. The default value of Point(-1, -1) denotes that the anchor is at the kernelcenter. 
:type anchor: cv2.typing.Point
:param normalize: flag, specifying whether the kernel is to be normalized by it's area or not.
:type normalize: bool
:param borderType: border mode used to extrapolate pixels outside of the image, see #BorderTypes. #BORDER_WRAP is not supported.
:type borderType: int
:rtype: cv2.typing.MatLike
````


````{py:function} sqrt(src[, dst]) -> dst

Calculates a square root of array elements.


The function cv::sqrt calculates a square root of each input array element. In case of multi-channel arrays, each channel is processed independently. The accuracy is approximately the same as of the built-in std::sqrt . 


:param src: input floating-point array.
:type src: cv2.typing.MatLike
:param dst: output array of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} stackBlur(src, ksize[, dst]) -> dst

Blurs an image using the stackBlur.


The function applies and stackBlur to an image. stackBlur can generate similar results as Gaussian blur, and the time consumption does not increase with the increase of kernel size. It creates a kind of moving stack of colors whilst scanning through the image. Thereby it just has to add one new block of color to the right side of the stack and remove the leftmost color. The remaining colors on the topmost layer of the stack are either added on or reduced by one, depending on if they are on the right or on the left side of the stack. The only supported borderType is BORDER_REPLICATE. Original paper was proposed by Mario Klingemann, which can be found http://underdestruction.com/2004/02/25/stackblur-2004. 


:param src: input image. The number of channels can be arbitrary, but the depth should be one ofCV_8U, CV_16U, CV_16S or CV_32F. 
:type src: cv2.typing.MatLike
:param dst: output image of the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param ksize: stack-blurring kernel size. The ksize.width and ksize.height can differ but they both must bepositive and odd. 
:type ksize: cv2.typing.Size
:rtype: cv2.typing.MatLike
````


````{py:function} startWindowThread() -> retval






:rtype: int
````


````{py:function} stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize[, R[, T[, E[, F[, flags[, criteria]]]]]]) -> retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F




stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, E[, F[, perViewErrors[, flags[, criteria]]]]]) -> retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, perViewErrors 


:param objectPoints: 
:type objectPoints: _typing.Sequence[cv2.typing.MatLike]
:param imagePoints1: 
:type imagePoints1: _typing.Sequence[cv2.typing.MatLike]
:param imagePoints2: 
:type imagePoints2: _typing.Sequence[cv2.typing.MatLike]
:param cameraMatrix1: 
:type cameraMatrix1: cv2.typing.MatLike
:param distCoeffs1: 
:type distCoeffs1: cv2.typing.MatLike
:param cameraMatrix2: 
:type cameraMatrix2: cv2.typing.MatLike
:param distCoeffs2: 
:type distCoeffs2: cv2.typing.MatLike
:param imageSize: 
:type imageSize: cv2.typing.Size
:param R: 
:type R: cv2.typing.MatLike | None
:param T: 
:type T: cv2.typing.MatLike | None
:param E: 
:type E: cv2.typing.MatLike | None
:param F: 
:type F: cv2.typing.MatLike | None
:param flags: 
:type flags: int
:param criteria: 
:type criteria: cv2.typing.TermCriteria
:rtype: tuple[float, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} stereoCalibrateExtended(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, E[, F[, rvecs[, tvecs[, perViewErrors[, flags[, criteria]]]]]]]) -> retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, rvecs, tvecs, perViewErrors

Calibrates a stereo camera set up. This function finds the intrinsic parametersfor each of the two cameras and the extrinsic parameters between the two cameras. 


-   @ref CALIB_SAME_FOCAL_LENGTH Enforce $f^{(0)}_x=f^{(1)}_x$ and $f^{(0)}_y=f^{(1)}_y$ . -   @ref CALIB_ZERO_TANGENT_DIST Set tangential distortion coefficients for each camera to zeros and fix there. -   @ref CALIB_FIX_K1,..., @ref CALIB_FIX_K6 Do not change the corresponding radial distortion coefficient during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0. -   @ref CALIB_RATIONAL_MODEL Enable coefficients k4, k5, and k6. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the rational model and return 8 coefficients. If the flag is not set, the function computes and returns only 5 distortion coefficients. -   @ref CALIB_THIN_PRISM_MODEL Coefficients s1, s2, s3 and s4 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the thin prism model and return 12 coefficients. If the flag is not set, the function computes and returns only 5 distortion coefficients. -   @ref CALIB_FIX_S1_S2_S3_S4 The thin prism distortion coefficients are not changed during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0. -   @ref CALIB_TILTED_MODEL Coefficients tauX and tauY are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the tilted sensor model and return 14 coefficients. If the flag is not set, the function computes and returns only 5 distortion coefficients. -   @ref CALIB_FIX_TAUX_TAUY The coefficients of the tilted sensor model are not changed during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0. 
The function estimates the transformation between two cameras making a stereo pair. If one computes the poses of an object relative to the first camera and to the second camera, ( $R_1$,$T_1$ ) and ($R_2$,$T_2$), respectively, for a stereo camera where the relative position and orientation between the two cameras are fixed, then those poses definitely relate to each other. This means, if the relative position and orientation ($R$,$T$) of the two cameras is known, it is possible to compute ($R_2$,$T_2$) when ($R_1$,$T_1$) is given. This is what the described function does. It computes ($R$,$T$) such that: 
$R_2=R R_1$ $T_2=R T_1 + T.$ 
Therefore, one can compute the coordinate representation of a 3D point for the second camera's coordinate system when given the point's coordinate representation in the first camera's coordinate system: 
$\begin{bmatrix} X_2 \\ Y_2 \\ Z_2 \\ 1 \end{bmatrix} = \begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix} \begin{bmatrix} X_1 \\ Y_1 \\ Z_1 \\ 1 \end{bmatrix}.$ 
Optionally, it computes the essential matrix E: 
$E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} R$ 
where $T_i$ are components of the translation vector $T$ : $T=[T_0, T_1, T_2]^T$ . And the function can also compute the fundamental matrix F: 
$F = cameraMatrix2^{-T}\cdot E \cdot cameraMatrix1^{-1}$ 
Besides the stereo-related information, the function can also perform a full calibration of each of the two cameras. However, due to the high dimensionality of the parameter space and noise in the input data, the function can diverge from the correct solution. If the intrinsic parameters can be estimated with high accuracy for each of the cameras individually (for example, using #calibrateCamera ), you are recommended to do so and then pass @ref CALIB_FIX_INTRINSIC flag to the function along with the computed intrinsic parameters. Otherwise, if all the parameters are estimated at once, it makes sense to restrict some parameters, for example, pass @ref CALIB_SAME_FOCAL_LENGTH and @ref CALIB_ZERO_TANGENT_DIST flags, which is usually a reasonable assumption. 
Similarly to #calibrateCamera, the function minimizes the total re-projection error for all the points in all the available views from both cameras. The function returns the final value of the re-projection error. 


:param objectPoints: Vector of vectors of the calibration pattern points. The same structure asin @ref calibrateCamera. For each pattern view, both cameras need to see the same object points. Therefore, objectPoints.size(), imagePoints1.size(), and imagePoints2.size() need to be equal as well as objectPoints[i].size(), imagePoints1[i].size(), and imagePoints2[i].size() need to be equal for each i. 
:type objectPoints: _typing.Sequence[cv2.typing.MatLike]
:param imagePoints1: Vector of vectors of the projections of the calibration pattern points,observed by the first camera. The same structure as in @ref calibrateCamera. 
:type imagePoints1: _typing.Sequence[cv2.typing.MatLike]
:param imagePoints2: Vector of vectors of the projections of the calibration pattern points,observed by the second camera. The same structure as in @ref calibrateCamera. 
:type imagePoints2: _typing.Sequence[cv2.typing.MatLike]
:param cameraMatrix1: Input/output camera intrinsic matrix for the first camera, the same as in@ref calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below. 
:type cameraMatrix1: cv2.typing.MatLike
:param distCoeffs1: Input/output vector of distortion coefficients, the same as in@ref calibrateCamera. 
:type distCoeffs1: cv2.typing.MatLike
:param cameraMatrix2: Input/output second camera intrinsic matrix for the second camera. See description forcameraMatrix1. 
:type cameraMatrix2: cv2.typing.MatLike
:param distCoeffs2: Input/output lens distortion coefficients for the second camera. Seedescription for distCoeffs1. 
:type distCoeffs2: cv2.typing.MatLike
:param imageSize: Size of the image used only to initialize the camera intrinsic matrices.
:type imageSize: cv2.typing.Size
:param R: Output rotation matrix. Together with the translation vector T, this matrix bringspoints given in the first camera's coordinate system to points in the second camera's coordinate system. In more technical terms, the tuple of R and T performs a change of basis from the first camera's coordinate system to the second camera's coordinate system. Due to its duality, this tuple is equivalent to the position of the first camera with respect to the second camera coordinate system. 
:type R: cv2.typing.MatLike
:param T: Output translation vector, see description above.
:type T: cv2.typing.MatLike
:param E: Output essential matrix.
:type E: cv2.typing.MatLike | None
:param F: Output fundamental matrix.
:type F: cv2.typing.MatLike | None
:param rvecs: Output vector of rotation vectors ( @ref Rodrigues ) estimated for each pattern view in thecoordinate system of the first camera of the stereo pair (e.g. std::vector<cv::Mat>). More in detail, each i-th rotation vector together with the corresponding i-th translation vector (see the next output parameter description) brings the calibration pattern from the object coordinate space (in which object points are specified) to the camera coordinate space of the first camera of the stereo pair. In more technical terms, the tuple of the i-th rotation and translation vector performs a change of basis from object coordinate space to camera coordinate space of the first camera of the stereo pair. 
:type rvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param tvecs: Output vector of translation vectors estimated for each pattern view, see parameter descriptionof previous output parameter ( rvecs ). 
:type tvecs: _typing.Sequence[cv2.typing.MatLike] | None
:param perViewErrors: Output vector of the RMS re-projection error estimated for each pattern view.
:type perViewErrors: cv2.typing.MatLike | None
:param flags: Different flags that may be zero or a combination of the following values:-   @ref CALIB_FIX_INTRINSIC Fix cameraMatrix? and distCoeffs? so that only R, T, E, and F matrices are estimated. -   @ref CALIB_USE_INTRINSIC_GUESS Optimize some or all of the intrinsic parameters according to the specified flags. Initial values are provided by the user. -   @ref CALIB_USE_EXTRINSIC_GUESS R and T contain valid initial values that are optimized further. Otherwise R and T are initialized to the median value of the pattern views (each dimension separately). -   @ref CALIB_FIX_PRINCIPAL_POINT Fix the principal points during the optimization. -   @ref CALIB_FIX_FOCAL_LENGTH Fix $f^{(j)}_x$ and $f^{(j)}_y$ . -   @ref CALIB_FIX_ASPECT_RATIO Optimize $f^{(j)}_y$ . Fix the ratio $f^{(j)}_x/f^{(j)}_y$ 
:type flags: int
:param criteria: Termination criteria for the iterative optimization algorithm.
:type criteria: cv2.typing.TermCriteria
:rtype: tuple[float, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, _typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike], cv2.typing.MatLike]
````


````{py:function} stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, R1[, R2[, P1[, P2[, Q[, flags[, alpha[, newImageSize]]]]]]]]) -> R1, R2, P1, P2, Q, validPixROI1, validPixROI2

Computes rectification transforms for each head of a calibrated stereo camera.


The function computes the rotation matrices for each camera that (virtually) make both camera image planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies the dense stereo correspondence problem. The function takes the matrices computed by #stereoCalibrate as input. As output, it provides two rotation matrices and also two projection matrices in the new coordinates. The function distinguishes the following two cases: 
-   **Horizontal stereo**: the first and the second camera views are shifted relative to each other mainly along the x-axis (with possible small vertical shift). In the rectified images, the corresponding epipolar lines in the left and right cameras are horizontal and have the same y-coordinate. P1 and P2 look like: 
$\texttt{P1} = \begin{bmatrix} f & 0 & cx_1 & 0 \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$ 
$\texttt{P2} = \begin{bmatrix} f & 0 & cx_2 & T_x \cdot f \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} ,$ 
$\texttt{Q} = \begin{bmatrix} 1 & 0 & 0 & -cx_1 \\ 0 & 1 & 0 & -cy \\ 0 & 0 & 0 & f \\ 0 & 0 & -\frac{1}{T_x} & \frac{cx_1 - cx_2}{T_x} \end{bmatrix} $ 
where $T_x$ is a horizontal shift between the cameras and $cx_1=cx_2$ if @ref CALIB_ZERO_DISPARITY is set. 
-   **Vertical stereo**: the first and the second camera views are shifted relative to each other mainly in the vertical direction (and probably a bit in the horizontal direction too). The epipolar lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like: 
$\texttt{P1} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_1 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$ 
$\texttt{P2} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_2 & T_y \cdot f \\ 0 & 0 & 1 & 0 \end{bmatrix},$ 
$\texttt{Q} = \begin{bmatrix} 1 & 0 & 0 & -cx \\ 0 & 1 & 0 & -cy_1 \\ 0 & 0 & 0 & f \\ 0 & 0 & -\frac{1}{T_y} & \frac{cy_1 - cy_2}{T_y} \end{bmatrix} $ 
where $T_y$ is a vertical shift between the cameras and $cy_1=cy_2$ if @ref CALIB_ZERO_DISPARITY is set. 
As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera matrices. The matrices, together with R1 and R2 , can then be passed to #initUndistortRectifyMap to initialize the rectification map for each camera. 
See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through the corresponding image regions. This means that the images are well rectified, which is what most stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that their interiors are all valid pixels. 
![image](pics/stereo_undistort.jpg) 


:param cameraMatrix1: First camera intrinsic matrix.
:type cameraMatrix1: cv2.typing.MatLike
:param distCoeffs1: First camera distortion parameters.
:type distCoeffs1: cv2.typing.MatLike
:param cameraMatrix2: Second camera intrinsic matrix.
:type cameraMatrix2: cv2.typing.MatLike
:param distCoeffs2: Second camera distortion parameters.
:type distCoeffs2: cv2.typing.MatLike
:param imageSize: Size of the image used for stereo calibration.
:type imageSize: cv2.typing.Size
:param R: Rotation matrix from the coordinate system of the first camera to the second camera,see @ref stereoCalibrate. 
:type R: cv2.typing.MatLike
:param T: Translation vector from the coordinate system of the first camera to the second camera,see @ref stereoCalibrate. 
:type T: cv2.typing.MatLike
:param R1: Output 3x3 rectification transform (rotation matrix) for the first camera. This matrixbrings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system. 
:type R1: cv2.typing.MatLike | None
:param R2: Output 3x3 rectification transform (rotation matrix) for the second camera. This matrixbrings points given in the unrectified second camera's coordinate system to points in the rectified second camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified second camera's coordinate system to the rectified second camera's coordinate system. 
:type R2: cv2.typing.MatLike | None
:param P1: Output 3x4 projection matrix in the new (rectified) coordinate systems for the firstcamera, i.e. it projects points given in the rectified first camera coordinate system into the rectified first camera's image. 
:type P1: cv2.typing.MatLike | None
:param P2: Output 3x4 projection matrix in the new (rectified) coordinate systems for the secondcamera, i.e. it projects points given in the rectified first camera coordinate system into the rectified second camera's image. 
:type P2: cv2.typing.MatLike | None
:param Q: Output $4 \times 4$ disparity-to-depth mapping matrix (see @ref reprojectImageTo3D).
:type Q: cv2.typing.MatLike | None
:param flags: Operation flags that may be zero or @ref CALIB_ZERO_DISPARITY . If the flag is set,the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area. 
:type flags: int
:param alpha: Free scaling parameter. If it is -1 or absent, the function performs the defaultscaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Any intermediate value yields an intermediate result between those two extreme cases. 
:type alpha: float
:param newImageSize: New image resolution after rectification. The same size should be passed to#initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0) is passed (default), it is set to the original imageSize . Setting it to a larger value can help you preserve details in the original image, especially when there is a big radial distortion. 
:type newImageSize: cv2.typing.Size
:param validPixROI1: Optional output rectangles inside the rectified images where all the pixelsare valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below). 
:type validPixROI1: 
:param validPixROI2: Optional output rectangles inside the rectified images where all the pixelsare valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below). 
:type validPixROI2: 
:rtype: tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.Rect, cv2.typing.Rect]
````


````{py:function} stereoRectifyUncalibrated(points1, points2, F, imgSize[, H1[, H2[, threshold]]]) -> retval, H1, H2

Computes a rectification transform for an uncalibrated stereo camera.


The function computes the rectification transformations without knowing intrinsic parameters of the cameras and their relative position in the space, which explains the suffix "uncalibrated". Another related difference from #stereoRectify is that the function outputs not the rectification transformations in the object (3D) space, but the planar perspective transformations encoded by the homography matrices H1 and H2 . The function implements the algorithm @cite Hartley99 . 
@note While the algorithm does not need to know the intrinsic parameters of the cameras, it heavily depends on the epipolar geometry. Therefore, if the camera lenses have a significant distortion, it would be better to correct it before computing the fundamental matrix and calling this function. For example, distortion coefficients can be estimated for each head of stereo camera separately by using #calibrateCamera . Then, the images can be corrected using #undistort , or just the point coordinates can be corrected with #undistortPoints . 


:param points1: Array of feature points in the first image.
:type points1: cv2.typing.MatLike
:param points2: The corresponding points in the second image. The same formats as in#findFundamentalMat are supported. 
:type points2: cv2.typing.MatLike
:param F: Input fundamental matrix. It can be computed from the same set of point pairs using#findFundamentalMat . 
:type F: cv2.typing.MatLike
:param imgSize: Size of the image.
:type imgSize: cv2.typing.Size
:param H1: Output rectification homography matrix for the first image.
:type H1: cv2.typing.MatLike | None
:param H2: Output rectification homography matrix for the second image.
:type H2: cv2.typing.MatLike | None
:param threshold: Optional threshold used to filter out the outliers. If the parameter is greaterthan zero, all the point pairs that do not comply with the epipolar geometry (that is, the points for which $|\texttt{points2[i]}^T \cdot \texttt{F} \cdot \texttt{points1[i]}|>\texttt{threshold}$ ) are rejected prior to computing the homographies. Otherwise, all the points are considered inliers. 
:type threshold: float
:rtype: tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]
````


````{py:function} stylization(src[, dst[, sigma_s[, sigma_r]]]) -> dst

Stylization aims to produce digital imagery with a wide variety of effects not focused onphotorealism. Edge-aware filters are ideal for stylization, as they can abstract regions of low contrast while preserving, or enhancing, high-contrast features. 




:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param dst: Output image with the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param sigma_s: %Range between 0 to 200.
:type sigma_s: float
:param sigma_r: %Range between 0 to 1.
:type sigma_r: float
:rtype: cv2.typing.MatLike
````


````{py:function} subtract(src1, src2[, dst[, mask[, dtype]]]) -> dst

Calculates the per-element difference between two arrays or array and a scalar.


The function subtract calculates: - Difference between two arrays, when both input arrays have the same size and the same number of channels: $\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0$ - Difference between an array and a scalar, when src2 is constructed from Scalar or has the same number of elements as `src1.channels()`: $\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0$ - Difference between a scalar and an array, when src1 is constructed from Scalar or has the same number of elements as `src2.channels()`: $\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0$ - The reverse difference between a scalar and an array in the case of `SubRS`: $\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0$ where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each channel is processed independently. 
The first function in the list above can be replaced with matrix expressions: 
```cpp
dst = src1 - src2;
dst -= src1; // equivalent to subtract(dst, src1, dst);
```
The input arrays and the output array can all have the same or different depths. For example, you can subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of the output array is determined by dtype parameter. In the second and third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this case the output array will have the same depth as the input array, be it src1, src2 or both. 
```{note}
Saturation is not applied when the output array has the depth CV_32S. You may even getresult of an incorrect sign in the case of overflow. 
```
```{note}
(Python) Be careful to difference behaviour between src1/src2 are single number and they are tuple/array.`subtract(src,X)` means `subtract(src,(X,X,X,X))`. `subtract(src,(X,))` means `subtract(src,(X,0,0,0))`. 
```
**See also:**  add, addWeighted, scaleAdd, Mat::convertTo


:param src1: first input array or a scalar.
:type src1: cv2.typing.MatLike
:param src2: second input array or a scalar.
:type src2: cv2.typing.MatLike
:param dst: output array of the same size and the same number of channels as the input array.
:type dst: cv2.typing.MatLike | None
:param mask: optional operation mask; this is an 8-bit single channel array that specifies elementsof the output array to be changed. 
:type mask: cv2.typing.MatLike | None
:param dtype: optional depth of the output array
:type dtype: int
:rtype: cv2.typing.MatLike
````


````{py:function} sumElems(src) -> retval

Calculates the sum of array elements.


The function cv::sum calculates and returns the sum of array elements, independently for each channel. 
**See also:**  countNonZero, mean, meanStdDev, norm, minMaxLoc, reduce


:param src: input array that must have from 1 to 4 channels.
:type src: cv2.typing.MatLike
:rtype: cv2.typing.Scalar
````


````{py:function} textureFlattening(src, mask[, dst[, low_threshold[, high_threshold[, kernel_size]]]]) -> dst

By retaining only the gradients at edge locations, before integrating with the Poisson solver, onewashes out the texture of the selected region, giving its contents a flat aspect. Here Canny Edge %Detector is used. 


@note The algorithm assumes that the color of the source image is close to that of the destination. This assumption means that when the colors don't match, the source image color gets tinted toward the color of the destination image. 


:param src: Input 8-bit 3-channel image.
:type src: cv2.typing.MatLike
:param mask: Input 8-bit 1 or 3-channel image.
:type mask: cv2.typing.MatLike
:param dst: Output image with the same size and type as src.
:type dst: cv2.typing.MatLike | None
:param low_threshold: %Range from 0 to 100.
:type low_threshold: float
:param high_threshold: Value \> 100.
:type high_threshold: float
:param kernel_size: The size of the Sobel kernel to be used.
:type kernel_size: int
:rtype: cv2.typing.MatLike
````


````{py:function} threshold(src, thresh, maxval, type[, dst]) -> retval, dst

Applies a fixed-level threshold to each array element.


The function applies fixed-level thresholding to a multiple-channel array. The function is typically used to get a bi-level (binary) image out of a grayscale image ( #compare could be also used for this purpose) or for removing a noise, that is, filtering out pixels with too small or too large values. There are several types of thresholding supported by the function. They are determined by type parameter. 
Also, the special values #THRESH_OTSU or #THRESH_TRIANGLE may be combined with one of the above values. In these cases, the function determines the optimal threshold value using the Otsu's or Triangle algorithm and uses it instead of the specified thresh. 
```{note}
Currently, the Otsu's and Triangle methods are implemented only for 8-bit single-channel images.
```
**See also:**  adaptiveThreshold, findContours, compare, min, max


:param src: input array (multiple-channel, 8-bit or 32-bit floating point).
:type src: cv2.typing.MatLike
:param dst: output array of the same size and type and the same number of channels as src.
:type dst: cv2.typing.MatLike | None
:param thresh: threshold value.
:type thresh: float
:param maxval: maximum value to use with the #THRESH_BINARY and #THRESH_BINARY_INV thresholdingtypes. 
:type maxval: float
:param type: thresholding type (see #ThresholdTypes).
:type type: int
:return: the computed threshold value if Otsu's or Triangle methods used.
:rtype: tuple[float, cv2.typing.MatLike]
````


````{py:function} trace(mtx) -> retval

Returns the trace of a matrix.


The function cv::trace returns the sum of the diagonal elements of the matrix mtx . $\mathrm{tr} ( \texttt{mtx} ) =  \sum _i  \texttt{mtx} (i,i)$ 


:param mtx: input matrix.
:type mtx: cv2.typing.MatLike
:rtype: cv2.typing.Scalar
````


````{py:function} transform(src, m[, dst]) -> dst

Performs the matrix transformation of every array element.


The function cv::transform performs the matrix transformation of every element of the array src and stores the results in dst : $\texttt{dst} (I) =  \texttt{m} \cdot \texttt{src} (I)$ (when m.cols=src.channels() ), or $\texttt{dst} (I) =  \texttt{m} \cdot [ \texttt{src} (I); 1]$ (when m.cols=src.channels()+1 ) 
Every element of the N -channel array src is interpreted as N -element vector that is transformed using the M x N or M x (N+1) matrix m to M-element vector - the corresponding element of the output array dst . 
The function may be used for geometrical transformation of N -dimensional points, arbitrary linear color space transformation (such as various kinds of RGB to YUV transforms), shuffling the image channels, and so forth. 
**See also:** perspectiveTransform, getAffineTransform, estimateAffine2D, warpAffine, warpPerspective


:param src: input array that must have as many channels (1 to 4) asm.cols or m.cols-1. 
:type src: cv2.typing.MatLike
:param dst: output array of the same size and depth as src; it has asmany channels as m.rows. 
:type dst: cv2.typing.MatLike | None
:param m: transformation 2x2 or 2x3 floating-point matrix.
:type m: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````


````{py:function} transpose(src[, dst]) -> dst

Transposes a matrix.


The function cv::transpose transposes the matrix src : $\texttt{dst} (i,j) =  \texttt{src} (j,i)$ 
```{note}
No complex conjugation is done in case of a complex matrix. Itshould be done separately if needed. 
```


:param src: input array.
:type src: cv2.typing.MatLike
:param dst: output array of the same type as src.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} transposeND(src, order[, dst]) -> dst

Transpose for n-dimensional matrices.


```{note}
Input should be continuous single-channel matrix.
```


:param src: input array.
:type src: cv2.typing.MatLike
:param order: a permutation of [0,1,..,N-1] where N is the number of axes of src.The i&#8217;th axis of dst will correspond to the axis numbered order[i] of the input. 
:type order: _typing.Sequence[int]
:param dst: output array of the same type as src.
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2[, points4D]) -> points4D

This function reconstructs 3-dimensional points (in homogeneous coordinates) by usingtheir observations with a stereo camera. 


@note Keep in mind that all input data should be of float type in order for this function to work. 
@note If the projection matrices from @ref stereoRectify are used, then the returned points are represented in the first camera's rectified coordinate system. 
@sa reprojectImageTo3D 


:param projMatr1: 3x4 projection matrix of the first camera, i.e. this matrix projects 3D pointsgiven in the world's coordinate system into the first image. 
:type projMatr1: cv2.typing.MatLike
:param projMatr2: 3x4 projection matrix of the second camera, i.e. this matrix projects 3D pointsgiven in the world's coordinate system into the second image. 
:type projMatr2: cv2.typing.MatLike
:param projPoints1: 2xN array of feature points in the first image. In the case of the c++ version,it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1. 
:type projPoints1: cv2.typing.MatLike
:param projPoints2: 2xN array of corresponding points in the second image. In the case of the c++version, it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1. 
:type projPoints2: cv2.typing.MatLike
:param points4D: 4xN array of reconstructed points in homogeneous coordinates. These points arereturned in the world's coordinate system. 
:type points4D: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]) -> dst

Transforms an image to compensate for lens distortion.


The function transforms an image to compensate radial and tangential lens distortion. 
The function is simply a combination of #initUndistortRectifyMap (with unity R ) and #remap (with bilinear interpolation). See the former function for details of the transformation being performed. 
Those pixels in the destination image, for which there is no correspondent pixels in the source image, are filled with zeros (black color). 
A particular subset of the source image that will be visible in the corrected image can be regulated by newCameraMatrix. You can use #getOptimalNewCameraMatrix to compute the appropriate newCameraMatrix depending on your requirements. 
The camera matrix and the distortion parameters can be determined using #calibrateCamera. If the resolution of images is different from the resolution used at the calibration stage, $f_x, f_y, c_x$ and $c_y$ need to be scaled accordingly, while the distortion coefficients remain the same. 


:param src: Input (distorted) image.
:type src: cv2.typing.MatLike
:param dst: Output (corrected) image that has the same size and type as src .
:type dst: cv2.typing.MatLike | None
:param cameraMatrix: Input camera matrix $A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param newCameraMatrix: Camera matrix of the distorted image. By default, it is the same ascameraMatrix but you may additionally scale and shift the result by using a different matrix. 
:type newCameraMatrix: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} undistortImagePoints(src, cameraMatrix, distCoeffs[, dst[, arg1]]) -> dst

Compute undistorted image points position




:param src: Observed points position, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 orCV_64FC2) (or vector\<Point2f\> ). 
:type src: cv2.typing.MatLike
:param dst: Output undistorted points position (1xN/Nx1 2-channel or vector\<Point2f\> ).
:type dst: cv2.typing.MatLike | None
:param cameraMatrix: Camera matrix $\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Distortion coefficients
:type distCoeffs: cv2.typing.MatLike
:param arg1: 
:type arg1: cv2.typing.TermCriteria
:rtype: cv2.typing.MatLike
````


````{py:function} undistortPoints(src, cameraMatrix, distCoeffs[, dst[, R[, P]]]) -> dst

Computes the ideal point coordinates from the observed point coordinates.


The function is similar to #undistort and #initUndistortRectifyMap but it operates on a sparse set of points instead of a raster image. Also the function performs a reverse transformation to  #projectPoints. In case of a 3D object, it does not reconstruct its 3D coordinates, but for a planar object, it does, up to a translation vector, if the proper R is specified. 
For each observed point coordinate $(u, v)$ the function computes: $ \begin{array}{l} x^{"}  \leftarrow (u - c_x)/f_x  \\ y^{"}  \leftarrow (v - c_y)/f_y  \\ (x',y') = undistort(x^{"},y^{"}, \texttt{distCoeffs}) \\ {[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\ x  \leftarrow X/W  \\ y  \leftarrow Y/W  \\ \text{only performed if P is specified:} \\ u'  \leftarrow x {f'}_x + {c'}_x  \\ v'  \leftarrow y {f'}_y + {c'}_y \end{array} $ 
where *undistort* is an approximate iterative algorithm that estimates the normalized original point coordinates out of the normalized distorted point coordinates ("normalized" means that the coordinates do not depend on the camera matrix). 
The function can be used for both a stereo camera head or a monocular camera (when R is empty). 


:param src: Observed point coordinates, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or CV_64FC2) (orvector\<Point2f\> ). 
:type src: cv2.typing.MatLike
:param dst: Output ideal point coordinates (1xN/Nx1 2-channel or vector\<Point2f\> ) after undistortion and reverse perspectivetransformation. If matrix P is identity or omitted, dst will contain normalized point coordinates. 
:type dst: cv2.typing.MatLike | None
:param cameraMatrix: Camera matrix $\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}$ .
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: Input vector of distortion coefficients$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])$ of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
:type distCoeffs: cv2.typing.MatLike
:param R: Rectification transformation in the object space (3x3 matrix). R1 or R2 computed by#stereoRectify can be passed here. If the matrix is empty, the identity transformation is used. 
:type R: cv2.typing.MatLike | None
:param P: New camera matrix (3x3) or new projection matrix (3x4) $\begin{bmatrix} {f'}_x & 0 & {c'}_x & t_x \\ 0 & {f'}_y & {c'}_y & t_y \\ 0 & 0 & 1 & t_z \end{bmatrix}$. P1 or P2 computed by#stereoRectify can be passed here. If the matrix is empty, the identity new camera matrix is used. 
:type P: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} undistortPointsIter(src, cameraMatrix, distCoeffs, R, P, criteria[, dst]) -> dst




@overload 
```{note}
Default version of #undistortPoints does 5 iterations to compute undistorted points.
```


:param src: 
:type src: cv2.typing.MatLike
:param cameraMatrix: 
:type cameraMatrix: cv2.typing.MatLike
:param distCoeffs: 
:type distCoeffs: cv2.typing.MatLike
:param R: 
:type R: cv2.typing.MatLike
:param P: 
:type P: cv2.typing.MatLike
:param criteria: 
:type criteria: cv2.typing.TermCriteria
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} useOpenVX() -> retval






:rtype: bool
````


````{py:function} useOptimized() -> retval

Returns the status of optimized code usage.


The function returns true if the optimized code is enabled. Otherwise, it returns false. 


:rtype: bool
````


````{py:function} validateDisparity(disparity, cost, minDisparity, numberOfDisparities[, disp12MaxDisp]) -> disparity






:param disparity: 
:type disparity: cv2.typing.MatLike
:param cost: 
:type cost: cv2.typing.MatLike
:param minDisparity: 
:type minDisparity: int
:param numberOfDisparities: 
:type numberOfDisparities: int
:param disp12MaxDisp: 
:type disp12MaxDisp: int
:rtype: cv2.typing.MatLike
````


````{py:function} vconcat(src[, dst]) -> dst




@overload 
```cpp
std::vector<cv::Mat> matrices = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),
cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),
cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};

cv::Mat out;
cv::vconcat( matrices, out );
//out:
//[1,   1,   1,   1;
// 2,   2,   2,   2;
// 3,   3,   3,   3]
```



:param src: input array or vector of matrices. all of the matrices must have the same number of cols and the same depth
:type src: _typing.Sequence[cv2.typing.MatLike]
:param dst: output array. It has the same number of cols and depth as the src, and the sum of rows of the src.same depth. 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} waitKey([, delay]) -> retval

Waits for a pressed key.


The function waitKey waits for a key event infinitely (when $\texttt{delay}\leq 0$ ) or for delay milliseconds, when it is positive. Since the OS has a minimum time between switching threads, the function will not wait exactly delay ms, it will wait at least delay ms, depending on what else is running on your computer at that time. It returns the code of the pressed key or -1 if no key was pressed before the specified time had elapsed. To check for a key press but not wait for it, use #pollKey. 
```{note}
The functions #waitKey and #pollKey are the only methods in HighGUI that can fetch and handleGUI events, so one of them needs to be called periodically for normal event processing unless HighGUI is used within an environment that takes care of event processing. 
```
```{note}
The function only works if there is at least one HighGUI window created and the window isactive. If there are several HighGUI windows, any of them can be active. 
```


:param delay: Delay in milliseconds. 0 is the special value that means "forever".
:type delay: int
:rtype: int
````


````{py:function} waitKeyEx([, delay]) -> retval

Similar to #waitKey, but returns full key code.


```{note}
Key code is implementation specific and depends on used backend: QT/GTK/Win32/etc
```


:param delay: 
:type delay: int
:rtype: int
````


````{py:function} warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst

Applies an affine transformation to an image.


The function warpAffine transforms the source image using the specified matrix: 
$\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})$ 
when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with #invertAffineTransform and then put in the formula above instead of M. The function cannot operate in-place. 
**See also:**  warpPerspective, resize, remap, getRectSubPix, transform


:param src: input image.
:type src: cv2.typing.MatLike
:param dst: output image that has the size dsize and the same type as src .
:type dst: cv2.typing.MatLike | None
:param M: $2\times 3$ transformation matrix.
:type M: cv2.typing.MatLike
:param dsize: size of the output image.
:type dsize: cv2.typing.Size
:param flags: combination of interpolation methods (see #InterpolationFlags) and the optionalflag #WARP_INVERSE_MAP that means that M is the inverse transformation ( $\texttt{dst}\rightarrow\texttt{src}$ ). 
:type flags: int
:param borderMode: pixel extrapolation method (see #BorderTypes); whenborderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function. 
:type borderMode: int
:param borderValue: value used in case of a constant border; by default, it is 0.
:type borderValue: cv2.typing.Scalar
:rtype: cv2.typing.MatLike
````


````{py:function} warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst

Applies a perspective transformation to an image.


The function warpPerspective transforms the source image using the specified matrix: 
$\texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} , \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )$ 
when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert and then put in the formula above instead of M. The function cannot operate in-place. 
**See also:**  warpAffine, resize, remap, getRectSubPix, perspectiveTransform


:param src: input image.
:type src: cv2.typing.MatLike
:param dst: output image that has the size dsize and the same type as src .
:type dst: cv2.typing.MatLike | None
:param M: $3\times 3$ transformation matrix.
:type M: cv2.typing.MatLike
:param dsize: size of the output image.
:type dsize: cv2.typing.Size
:param flags: combination of interpolation methods (#INTER_LINEAR or #INTER_NEAREST) and theoptional flag #WARP_INVERSE_MAP, that sets M as the inverse transformation ( $\texttt{dst}\rightarrow\texttt{src}$ ). 
:type flags: int
:param borderMode: pixel extrapolation method (#BORDER_CONSTANT or #BORDER_REPLICATE).
:type borderMode: int
:param borderValue: value used in case of a constant border; by default, it equals 0.
:type borderValue: cv2.typing.Scalar
:rtype: cv2.typing.MatLike
````


````{py:function} warpPolar(src, dsize, center, maxRadius, flags[, dst]) -> dst




\brief Remaps an image to polar or semilog-polar coordinates space 
@anchor polar_remaps_reference_image ![Polar remaps reference](pics/polar_remap_doc.png) 
Transform the source image using the following transformation: $ dst(\rho , \phi ) = src(x,y) $ 
where $ \begin{array}{l} \vec{I} = (x - center.x, \;y - center.y) \\ \phi = Kangle \cdot \texttt{angle} (\vec{I}) \\ \rho = \left\{\begin{matrix} Klin \cdot \texttt{magnitude} (\vec{I}) & default \\ Klog \cdot log_e(\texttt{magnitude} (\vec{I})) & if \; semilog \\ \end{matrix}\right. \end{array} $ 
and $ \begin{array}{l} Kangle = dsize.height / 2\Pi \\ Klin = dsize.width / maxRadius \\ Klog = dsize.width / log_e(maxRadius) \\ \end{array} $ 
\par Linear vs semilog mapping 
Polar mapping can be linear or semi-log. Add one of #WarpPolarMode to `flags` to specify the polar mapping mode. 
Linear is the default mode. 
The semilog mapping emulates the human "foveal" vision that permit very high acuity on the line of sight (central vision) in contrast to peripheral vision where acuity is minor. 
\par Option on `dsize`: 
- if both values in `dsize <=0 ` (default), the destination image will have (almost) same area of source bounding circle: $\begin{array}{l} dsize.area  \leftarrow (maxRadius^2 \cdot \Pi) \\ dsize.width = \texttt{cvRound}(maxRadius) \\ dsize.height = \texttt{cvRound}(maxRadius \cdot \Pi) \\ \end{array}$ 
- if only `dsize.height <= 0`, the destination image area will be proportional to the bounding circle area but scaled by `Kx * Kx`: $\begin{array}{l} dsize.height = \texttt{cvRound}(dsize.width \cdot \Pi) \\ \end{array} $ 
- if both values in `dsize > 0 `, the destination image will have the given size therefore the area of the bounding circle will be scaled to `dsize`. 
\par Reverse mapping 
You can get reverse mapping adding #WARP_INVERSE_MAP to `flags` \snippet polar_transforms.cpp InverseMap 
In addiction, to calculate the original coordinate from a polar mapped coordinate $(rho, phi)->(x, y)$: \snippet polar_transforms.cpp InverseCoordinate 
**See also:** cv::remap


:param src: Source image.
:type src: cv2.typing.MatLike
:param dst: Destination image. It will have same type as src.
:type dst: cv2.typing.MatLike | None
:param dsize: The destination image size (see description for valid options).
:type dsize: cv2.typing.Size
:param center: The transformation center.
:type center: cv2.typing.Point2f
:param maxRadius: The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.
:type maxRadius: float
:param flags: A combination of interpolation methods, #InterpolationFlags + #WarpPolarMode.- Add #WARP_POLAR_LINEAR to select linear polar mapping (default) - Add #WARP_POLAR_LOG to select semilog polar mapping - Add #WARP_INVERSE_MAP for reverse mapping. @note -  The function can not operate in-place. -  To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees. -  This function uses #remap. Due to current implementation limitations the size of an input and output images should be less than 32767x32767. 
:type flags: int
:rtype: cv2.typing.MatLike
````


````{py:function} watershed(image, markers) -> markers

Performs a marker-based image segmentation using the watershed algorithm.


The function implements one of the variants of watershed, non-parametric marker-based segmentation algorithm, described in @cite Meyer92 . 
Before passing the image to the function, you have to roughly outline the desired regions in the image markers with positive (\>0) indices. So, every region is represented as one or more connected components with the pixel values 1, 2, 3, and so on. Such markers can be retrieved from a binary mask using #findContours and #drawContours (see the watershed.cpp demo). The markers are "seeds" of the future image regions. All the other pixels in markers , whose relation to the outlined regions is not known and should be defined by the algorithm, should be set to 0's. In the function output, each pixel in markers is set to a value of the "seed" components or to -1 at boundaries between the regions. 
```{note}
Any two neighbor connected components are not necessarily separated by a watershed boundary(-1's pixels); for example, they can touch each other in the initial marker image passed to the function. 
```
**See also:** findContours


:param image: Input 8-bit 3-channel image.
:type image: cv2.typing.MatLike
:param markers: Input/output 32-bit single-channel image (map) of markers. It should have the samesize as image . 
:type markers: cv2.typing.MatLike
:rtype: cv2.typing.MatLike
````


````{py:function} writeOpticalFlow(path, flow) -> retval

Write a .flo to disk


The function stores a flow field in a file, returns true on success, false otherwise. The flow field must be a 2-channel, floating-point matrix (CV_32FC2). First channel corresponds to the flow in the horizontal direction (u), second - vertical (v). 


:param path: Path to the file to be written
:type path: str
:param flow: Flow field to be stored
:type flow: cv2.typing.MatLike
:rtype: bool
````



