# `cv2.dnn`
```{py:module} cv2.dnn
None
```
## Attributes
```{py:attribute} DNN_BACKEND_CANN
:type: int
```


```{py:attribute} DNN_BACKEND_CUDA
:type: int
```


```{py:attribute} DNN_BACKEND_DEFAULT
:type: int
```


```{py:attribute} DNN_BACKEND_HALIDE
:type: int
```


```{py:attribute} DNN_BACKEND_INFERENCE_ENGINE
:type: int
```


```{py:attribute} DNN_BACKEND_OPENCV
:type: int
```


```{py:attribute} DNN_BACKEND_TIMVX
:type: int
```


```{py:attribute} DNN_BACKEND_VKCOM
:type: int
```


```{py:attribute} DNN_BACKEND_WEBNN
:type: int
```


```{py:attribute} DNN_LAYOUT_NCDHW
:type: int
```


```{py:attribute} DNN_LAYOUT_NCHW
:type: int
```


```{py:attribute} DNN_LAYOUT_ND
:type: int
```


```{py:attribute} DNN_LAYOUT_NDHWC
:type: int
```


```{py:attribute} DNN_LAYOUT_NHWC
:type: int
```


```{py:attribute} DNN_LAYOUT_PLANAR
:type: int
```


```{py:attribute} DNN_LAYOUT_UNKNOWN
:type: int
```


```{py:attribute} DNN_PMODE_CROP_CENTER
:type: int
```


```{py:attribute} DNN_PMODE_LETTERBOX
:type: int
```


```{py:attribute} DNN_PMODE_NULL
:type: int
```


```{py:attribute} DNN_TARGET_CPU
:type: int
```


```{py:attribute} DNN_TARGET_CPU_FP16
:type: int
```


```{py:attribute} DNN_TARGET_CUDA
:type: int
```


```{py:attribute} DNN_TARGET_CUDA_FP16
:type: int
```


```{py:attribute} DNN_TARGET_FPGA
:type: int
```


```{py:attribute} DNN_TARGET_HDDL
:type: int
```


```{py:attribute} DNN_TARGET_MYRIAD
:type: int
```


```{py:attribute} DNN_TARGET_NPU
:type: int
```


```{py:attribute} DNN_TARGET_OPENCL
:type: int
```


```{py:attribute} DNN_TARGET_OPENCL_FP16
:type: int
```


```{py:attribute} DNN_TARGET_VULKAN
:type: int
```


```{py:attribute} SOFT_NMSMETHOD_SOFTNMS_GAUSSIAN
:type: int
```


```{py:attribute} SOFT_NMSMETHOD_SOFTNMS_LINEAR
:type: int
```


```{py:attribute} SoftNMSMethod_SOFTNMS_GAUSSIAN
:type: int
```


```{py:attribute} SoftNMSMethod_SOFTNMS_LINEAR
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
`````{py:class} ClassificationModel




````{py:method} __init__(self, model: str, config: str=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param model: 
:type model: str
:param config: 
:type config: str
:rtype: None
````

````{py:method} __init__(self, network: Net)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param network: 
:type network: Net
:rtype: None
````

````{py:method} classify(frame) -> classId, conf




@overload 


:param self: 
:type self: 
:param frame: 
:type frame: cv2.typing.MatLike
:rtype: tuple[int, float]
````

````{py:method} classify(frame) -> classId, conf




@overload 


:param self: 
:type self: 
:param frame: 
:type frame: cv2.UMat
:rtype: tuple[int, float]
````

````{py:method} setEnableSoftmaxPostProcessing(enable) -> retval
Set enable/disable softmax post processing option.


If this option is true, softmax is applied after forward inference within the classify() function to convert the confidences range to [0.0-1.0]. This function allows you to toggle this behavior. Please turn true when not contain softmax layer in model. 


:param self: 
:type self: 
:param enable: [in] Set enable softmax post processing within the classify() function.
:type enable: bool
:rtype: ClassificationModel
````

````{py:method} getEnableSoftmaxPostProcessing() -> retval
Get enable/disable softmax post processing option.


This option defaults to false, softmax post processing is not applied within the classify() function. 


:param self: 
:type self: 
:rtype: bool
````


`````


`````{py:class} DetectionModel




````{py:method} __init__(self, model: str, config: str=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param model: 
:type model: str
:param config: 
:type config: str
:rtype: None
````

````{py:method} __init__(self, network: Net)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param network: 
:type network: Net
:rtype: None
````

````{py:method} detect(frame[, confThreshold[, nmsThreshold]]) -> classIds, confidences, boxes

Given the @p input frame, create input blob, run net and return result detections.




:param self: 
:type self: 
:param frame: [in] The input image.
:type frame: cv2.typing.MatLike
:param confThreshold: [in] A threshold used to filter boxes by confidences.
:type confThreshold: float
:param nmsThreshold: [in] A threshold used in non maximum suppression.
:type nmsThreshold: float
:param classIds: [out] Class indexes in result detection.
:type classIds: 
:param confidences: [out] A set of corresponding confidences.
:type confidences: 
:param boxes: [out] A set of bounding boxes.
:type boxes: 
:rtype: tuple[_typing.Sequence[int], _typing.Sequence[float], _typing.Sequence[cv2.typing.Rect]]
````

````{py:method} detect(frame[, confThreshold[, nmsThreshold]]) -> classIds, confidences, boxes

Given the @p input frame, create input blob, run net and return result detections.




:param self: 
:type self: 
:param frame: [in] The input image.
:type frame: cv2.UMat
:param confThreshold: [in] A threshold used to filter boxes by confidences.
:type confThreshold: float
:param nmsThreshold: [in] A threshold used in non maximum suppression.
:type nmsThreshold: float
:param classIds: [out] Class indexes in result detection.
:type classIds: 
:param confidences: [out] A set of corresponding confidences.
:type confidences: 
:param boxes: [out] A set of bounding boxes.
:type boxes: 
:rtype: tuple[_typing.Sequence[int], _typing.Sequence[float], _typing.Sequence[cv2.typing.Rect]]
````

````{py:method} setNmsAcrossClasses(value) -> retval
nmsAcrossClasses defaults to false,such that when non max suppression is used during the detect() function, it will do so per-class. This function allows you to toggle this behaviour. 




:param self: 
:type self: 
:param value: [in] The new value for nmsAcrossClasses
:type value: bool
:rtype: DetectionModel
````

````{py:method} getNmsAcrossClasses() -> retval
Getter for nmsAcrossClasses. This variable defaults to false,such that when non max suppression is used during the detect() function, it will do so only per-class 




:param self: 
:type self: 
:rtype: bool
````


`````


`````{py:class} DictValue




````{py:method} __init__(self, i: int)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param i: 
:type i: int
:rtype: None
````

````{py:method} __init__(self, p: float)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param p: 
:type p: float
:rtype: None
````

````{py:method} __init__(self, s: str)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param s: 
:type s: str
:rtype: None
````

````{py:method} isInt() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isString() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} isReal() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} getIntValue([, idx]) -> retval





:param self: 
:type self: 
:param idx: 
:type idx: int
:rtype: int
````

````{py:method} getRealValue([, idx]) -> retval





:param self: 
:type self: 
:param idx: 
:type idx: int
:rtype: float
````

````{py:method} getStringValue([, idx]) -> retval





:param self: 
:type self: 
:param idx: 
:type idx: int
:rtype: str
````


`````


`````{py:class} Image2BlobParams




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, scalefactor: cv2.typing.Scalar, size: cv2.typing.Size=..., mean: cv2.typing.Scalar=..., swapRB: bool=..., ddepth: int=..., datalayout: DataLayout=..., mode: ImagePaddingMode=..., borderValue: cv2.typing.Scalar=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param scalefactor: 
:type scalefactor: cv2.typing.Scalar
:param size: 
:type size: cv2.typing.Size
:param mean: 
:type mean: cv2.typing.Scalar
:param swapRB: 
:type swapRB: bool
:param ddepth: 
:type ddepth: int
:param datalayout: 
:type datalayout: DataLayout
:param mode: 
:type mode: ImagePaddingMode
:param borderValue: 
:type borderValue: cv2.typing.Scalar
:rtype: None
````

````{py:method} blobRectToImageRect(rBlob, size) -> retval
Get rectangle coordinates in original image system from rectangle in blob coordinates.




:param self: 
:type self: 
:param rBlob: rect in blob coordinates.
:type rBlob: cv2.typing.Rect
:param size: original input image size.
:type size: cv2.typing.Size
:return: rectangle in original image coordinates.
:rtype: cv2.typing.Rect
````

````{py:method} blobRectsToImageRects(rBlob, size) -> rImg
Get rectangle coordinates in original image system from rectangle in blob coordinates.




:param self: 
:type self: 
:param rBlob: rect in blob coordinates.
:type rBlob: _typing.Sequence[cv2.typing.Rect]
:param size: original input image size.
:type size: cv2.typing.Size
:param rImg: result rect in image coordinates.
:type rImg: 
:rtype: _typing.Sequence[cv2.typing.Rect]
````

```{py:attribute} scalefactor
:type: cv2.typing.Scalar
```

```{py:attribute} size
:type: cv2.typing.Size
```

```{py:attribute} mean
:type: cv2.typing.Scalar
```

```{py:attribute} swapRB
:type: bool
```

```{py:attribute} ddepth
:type: int
```

```{py:attribute} datalayout
:type: DataLayout
```

```{py:attribute} paddingmode
:type: ImagePaddingMode
```

```{py:attribute} borderValue
:type: cv2.typing.Scalar
```


`````


`````{py:class} KeypointsModel




````{py:method} __init__(self, model: str, config: str=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param model: 
:type model: str
:param config: 
:type config: str
:rtype: None
````

````{py:method} __init__(self, network: Net)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param network: 
:type network: Net
:rtype: None
````

````{py:method} estimate(frame[, thresh]) -> retval

Given the @p input frame, create input blob, run net




:param self: 
:type self: 
:param frame: [in] The input image.
:type frame: cv2.typing.MatLike
:param thresh: minimum confidence threshold to select a keypoint
:type thresh: float
:return: a vector holding the x and y coordinates of each detected keypoint
:rtype: _typing.Sequence[cv2.typing.Point2f]
````

````{py:method} estimate(frame[, thresh]) -> retval

Given the @p input frame, create input blob, run net




:param self: 
:type self: 
:param frame: [in] The input image.
:type frame: cv2.UMat
:param thresh: minimum confidence threshold to select a keypoint
:type thresh: float
:return: a vector holding the x and y coordinates of each detected keypoint
:rtype: _typing.Sequence[cv2.typing.Point2f]
````


`````


`````{py:class} Layer




````{py:method} name






:param self: 
:type self: 
:rtype: str
````

````{py:method} type






:param self: 
:type self: 
:rtype: str
````

````{py:method} preferableTarget






:param self: 
:type self: 
:rtype: int
````

````{py:method} finalize(inputs[, outputs]) -> outputs

Computes and sets internal parameters according to inputs, outputs and blobs.


If this method is called after network has allocated all memory for input and output blobs and before inferencing. 


:param self: 
:type self: 
:param inputs: [in] vector of already allocated input blobs
:type inputs: _typing.Sequence[cv2.typing.MatLike]
:param outputs: [out] vector of already allocated output blobs
:type outputs: _typing.Sequence[cv2.typing.MatLike] | None
:rtype: _typing.Sequence[cv2.typing.MatLike]
````

````{py:method} finalize(inputs[, outputs]) -> outputs

Computes and sets internal parameters according to inputs, outputs and blobs.


If this method is called after network has allocated all memory for input and output blobs and before inferencing. 


:param self: 
:type self: 
:param inputs: [in] vector of already allocated input blobs
:type inputs: _typing.Sequence[cv2.UMat]
:param outputs: [out] vector of already allocated output blobs
:type outputs: _typing.Sequence[cv2.UMat] | None
:rtype: _typing.Sequence[cv2.UMat]
````

````{py:method} run(inputs, internals[, outputs]) -> outputs, internals
Allocates layer and computes output.


```{deprecated} unknown
This method will be removed in the future release.
```


:param self: 
:type self: 
:param inputs: 
:type inputs: _typing.Sequence[cv2.typing.MatLike]
:param internals: 
:type internals: _typing.Sequence[cv2.typing.MatLike]
:param outputs: 
:type outputs: _typing.Sequence[cv2.typing.MatLike] | None
:rtype: tuple[_typing.Sequence[cv2.typing.MatLike], _typing.Sequence[cv2.typing.MatLike]]
````

````{py:method} outputNameToIndex(outputName) -> retval
Returns index of output blob in output array.


**See also:** inputNameToIndex()


:param self: 
:type self: 
:param outputName: 
:type outputName: str
:rtype: int
````

```{py:attribute} blobs
:type: _typing.Sequence[cv2.typing.MatLike]
```


`````


`````{py:class} Model




````{py:method} __init__(self, model: str, config: str=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param model: 
:type model: str
:param config: 
:type config: str
:rtype: None
````

````{py:method} __init__(self, network: Net)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param network: 
:type network: Net
:rtype: None
````

````{py:method} setInputSize(size) -> retval

Set input size for frame.


setInputSize(width, height) -> retval @overload 
```{note}
If shape of the new blob less than 0, then frame size not change.
```


:param self: 
:type self: 
:param size: [in] New input size.
:type size: cv2.typing.Size
:param width: [in] New input width.
:type width: 
:param height: [in] New input height.
:type height: 
:rtype: Model
````

````{py:method} setInputSize(size) -> retval

Set input size for frame.


setInputSize(width, height) -> retval @overload 
```{note}
If shape of the new blob less than 0, then frame size not change.
```


:param self: 
:type self: 
:param width: [in] New input width.
:type width: int
:param height: [in] New input height.
:type height: int
:param size: [in] New input size.
:type size: 
:rtype: Model
````

````{py:method} predict(frame[, outs]) -> outs

Given the @p input frame, create input blob, run net and return the output @p blobs.




:param self: 
:type self: 
:param frame: [in] The input image.
:type frame: cv2.typing.MatLike
:param outs: [out] Allocated output blobs, which will store results of the computation.
:type outs: _typing.Sequence[cv2.typing.MatLike] | None
:rtype: _typing.Sequence[cv2.typing.MatLike]
````

````{py:method} predict(frame[, outs]) -> outs

Given the @p input frame, create input blob, run net and return the output @p blobs.




:param self: 
:type self: 
:param frame: [in] The input image.
:type frame: cv2.UMat
:param outs: [out] Allocated output blobs, which will store results of the computation.
:type outs: _typing.Sequence[cv2.UMat] | None
:rtype: _typing.Sequence[cv2.UMat]
````

````{py:method} setInputMean(mean) -> retval
Set mean value for frame.




:param self: 
:type self: 
:param mean: [in] Scalar with mean values which are subtracted from channels.
:type mean: cv2.typing.Scalar
:rtype: Model
````

````{py:method} setInputScale(scale) -> retval
Set scalefactor value for frame.




:param self: 
:type self: 
:param scale: [in] Multiplier for frame values.
:type scale: cv2.typing.Scalar
:rtype: Model
````

````{py:method} setInputCrop(crop) -> retval
Set flag crop for frame.




:param self: 
:type self: 
:param crop: [in] Flag which indicates whether image will be cropped after resize or not.
:type crop: bool
:rtype: Model
````

````{py:method} setInputSwapRB(swapRB) -> retval
Set flag swapRB for frame.




:param self: 
:type self: 
:param swapRB: [in] Flag which indicates that swap first and last channels.
:type swapRB: bool
:rtype: Model
````

````{py:method} setInputParams([, scale[, size[, mean[, swapRB[, crop]]]]]) -> None
Set preprocessing parameters for frame.




:param self: 
:type self: 
:param scale: [in] Multiplier for frame values.
:type scale: float
:param size: [in] New input size.
:type size: cv2.typing.Size
:param mean: [in] Scalar with mean values which are subtracted from channels.
:type mean: cv2.typing.Scalar
:param swapRB: [in] Flag which indicates that swap first and last channels.
:type swapRB: bool
:param crop: [in] Flag which indicates whether image will be cropped after resize or not.blob(n, c, y, x) = scale * resize( frame(y, x, c) ) - mean(c) ) 
:type crop: bool
:rtype: None
````

````{py:method} setPreferableBackend(backendId) -> retval





:param self: 
:type self: 
:param backendId: 
:type backendId: Backend
:rtype: Model
````

````{py:method} setPreferableTarget(targetId) -> retval





:param self: 
:type self: 
:param targetId: 
:type targetId: Target
:rtype: Model
````

````{py:method} enableWinograd(useWinograd) -> retval





:param self: 
:type self: 
:param useWinograd: 
:type useWinograd: bool
:rtype: Model
````


`````


`````{py:class} Net




````{py:method} readFromModelOptimizer(xml, bin) -> retval
:classmethod:
Create a network from Intel's Model Optimizer in-memory buffers with intermediate representation (IR).


readFromModelOptimizer(bufferModelConfig, bufferWeights) -> retval 


:param cls: 
:type cls: 
:param xml: [in] XML configuration file with network's topology.
:type xml: str
:param bin: [in] Binary file with trained weights.Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine backend. 
:type bin: str
:param bufferModelConfig: [in] buffer with model's configuration.
:type bufferModelConfig: 
:param bufferWeights: [in] buffer with model's trained weights.
:type bufferWeights: 
:return: Net object.
:rtype: Net
````

````{py:method} readFromModelOptimizer(xml, bin) -> retval
:classmethod:
Create a network from Intel's Model Optimizer in-memory buffers with intermediate representation (IR).


readFromModelOptimizer(bufferModelConfig, bufferWeights) -> retval 


:param cls: 
:type cls: 
:param bufferModelConfig: [in] buffer with model's configuration.
:type bufferModelConfig: numpy.ndarray[_typing.Any, numpy.dtype[numpy.uint8]]
:param bufferWeights: [in] buffer with model's trained weights.
:type bufferWeights: numpy.ndarray[_typing.Any, numpy.dtype[numpy.uint8]]
:param xml: [in] XML configuration file with network's topology.
:type xml: 
:param bin: [in] Binary file with trained weights.Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine backend. 
:type bin: 
:return: Net object.
:rtype: Net
````

````{py:method} getLayer(layerId) -> retval

Returns pointer to layer with specified id or name which the network use.


getLayer(layerName) -> retval @overload 
```{deprecated} unknown
Use int getLayerId(const String &layer)
```


:param self: 
:type self: 
:param layerId: 
:type layerId: int
:rtype: Layer
````

````{py:method} getLayer(layerId) -> retval

Returns pointer to layer with specified id or name which the network use.


getLayer(layerName) -> retval @overload 
```{deprecated} unknown
Use int getLayerId(const String &layer)
```


:param self: 
:type self: 
:param layerName: 
:type layerName: str
:rtype: Layer
````

````{py:method} getLayer(layerId) -> retval

Returns pointer to layer with specified id or name which the network use.


getLayer(layerName) -> retval @overload 
```{deprecated} unknown
Use int getLayerId(const String &layer)
```


:param self: 
:type self: 
:param layerId: 
:type layerId: cv2.typing.LayerId
:rtype: Layer
````

````{py:method} forward([, outputName]) -> retval

Runs forward pass to compute outputs of layers listed in @p outBlobNames.


forward([, outputBlobs[, outputName]]) -> outputBlobs 
forward(outBlobNames[, outputBlobs]) -> outputBlobs 


:param self: 
:type self: 
:param outputName: name for layer which output is needed to get@details If @p outputName is empty, runs forward pass for the whole network. 
:type outputName: str
:param outputBlobs: contains blobs for first outputs of specified layers.
:type outputBlobs: 
:param outBlobNames: names for layers which outputs are needed to get
:type outBlobNames: 
:return: blob for first output of specified layer.@details By default runs forward pass for the whole network. 
:rtype: cv2.typing.MatLike
````

````{py:method} forward([, outputName]) -> retval

Runs forward pass to compute outputs of layers listed in @p outBlobNames.


forward([, outputBlobs[, outputName]]) -> outputBlobs 
forward(outBlobNames[, outputBlobs]) -> outputBlobs 


:param self: 
:type self: 
:param outputBlobs: contains blobs for first outputs of specified layers.
:type outputBlobs: _typing.Sequence[cv2.typing.MatLike] | None
:param outputName: name for layer which output is needed to get@details If @p outputName is empty, runs forward pass for the whole network. 
:type outputName: str
:param outBlobNames: names for layers which outputs are needed to get
:type outBlobNames: 
:return: blob for first output of specified layer.@details By default runs forward pass for the whole network. 
:rtype: _typing.Sequence[cv2.typing.MatLike]
````

````{py:method} forward([, outputName]) -> retval

Runs forward pass to compute outputs of layers listed in @p outBlobNames.


forward([, outputBlobs[, outputName]]) -> outputBlobs 
forward(outBlobNames[, outputBlobs]) -> outputBlobs 


:param self: 
:type self: 
:param outputBlobs: contains blobs for first outputs of specified layers.
:type outputBlobs: _typing.Sequence[cv2.UMat] | None
:param outputName: name for layer which output is needed to get@details If @p outputName is empty, runs forward pass for the whole network. 
:type outputName: str
:param outBlobNames: names for layers which outputs are needed to get
:type outBlobNames: 
:return: blob for first output of specified layer.@details By default runs forward pass for the whole network. 
:rtype: _typing.Sequence[cv2.UMat]
````

````{py:method} forward([, outputName]) -> retval

Runs forward pass to compute outputs of layers listed in @p outBlobNames.


forward([, outputBlobs[, outputName]]) -> outputBlobs 
forward(outBlobNames[, outputBlobs]) -> outputBlobs 


:param self: 
:type self: 
:param outBlobNames: names for layers which outputs are needed to get
:type outBlobNames: _typing.Sequence[str]
:param outputBlobs: contains blobs for first outputs of specified layers.
:type outputBlobs: _typing.Sequence[cv2.typing.MatLike] | None
:param outputName: name for layer which output is needed to get@details If @p outputName is empty, runs forward pass for the whole network. 
:type outputName: 
:return: blob for first output of specified layer.@details By default runs forward pass for the whole network. 
:rtype: _typing.Sequence[cv2.typing.MatLike]
````

````{py:method} forward([, outputName]) -> retval

Runs forward pass to compute outputs of layers listed in @p outBlobNames.


forward([, outputBlobs[, outputName]]) -> outputBlobs 
forward(outBlobNames[, outputBlobs]) -> outputBlobs 


:param self: 
:type self: 
:param outBlobNames: names for layers which outputs are needed to get
:type outBlobNames: _typing.Sequence[str]
:param outputBlobs: contains blobs for first outputs of specified layers.
:type outputBlobs: _typing.Sequence[cv2.UMat] | None
:param outputName: name for layer which output is needed to get@details If @p outputName is empty, runs forward pass for the whole network. 
:type outputName: 
:return: blob for first output of specified layer.@details By default runs forward pass for the whole network. 
:rtype: _typing.Sequence[cv2.UMat]
````

````{py:method} quantize(calibData, inputsDtype, outputsDtype[, perChannel]) -> retval

Returns a quantized Net from a floating-point Net.




:param self: 
:type self: 
:param calibData: Calibration data to compute the quantization parameters.
:type calibData: _typing.Sequence[cv2.typing.MatLike]
:param inputsDtype: Datatype of quantized net's inputs. Can be CV_32F or CV_8S.
:type inputsDtype: int
:param outputsDtype: Datatype of quantized net's outputs. Can be CV_32F or CV_8S.
:type outputsDtype: int
:param perChannel: Quantization granularity of quantized Net. The default is true, that means quantize modelin per-channel way (channel-wise). Set it false to quantize model in per-tensor way (or tensor-wise). 
:type perChannel: bool
:rtype: Net
````

````{py:method} quantize(calibData, inputsDtype, outputsDtype[, perChannel]) -> retval

Returns a quantized Net from a floating-point Net.




:param self: 
:type self: 
:param calibData: Calibration data to compute the quantization parameters.
:type calibData: _typing.Sequence[cv2.UMat]
:param inputsDtype: Datatype of quantized net's inputs. Can be CV_32F or CV_8S.
:type inputsDtype: int
:param outputsDtype: Datatype of quantized net's outputs. Can be CV_32F or CV_8S.
:type outputsDtype: int
:param perChannel: Quantization granularity of quantized Net. The default is true, that means quantize modelin per-channel way (channel-wise). Set it false to quantize model in per-tensor way (or tensor-wise). 
:type perChannel: bool
:rtype: Net
````

````{py:method} setInput(blob[, name[, scalefactor[, mean]]]) -> None

Sets the new input value for the network


If scale or mean values are specified, a final input blob is computed as: $input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)$ 
**See also:** connect(String, String) to know format of the descriptor.


:param self: 
:type self: 
:param blob: A new blob. Should have CV_32F or CV_8U depth.
:type blob: cv2.typing.MatLike
:param name: A name of input layer.
:type name: str
:param scalefactor: An optional normalization scale.
:type scalefactor: float
:param mean: An optional mean subtraction values.
:type mean: cv2.typing.Scalar
:rtype: None
````

````{py:method} setInput(blob[, name[, scalefactor[, mean]]]) -> None

Sets the new input value for the network


If scale or mean values are specified, a final input blob is computed as: $input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)$ 
**See also:** connect(String, String) to know format of the descriptor.


:param self: 
:type self: 
:param blob: A new blob. Should have CV_32F or CV_8U depth.
:type blob: cv2.UMat
:param name: A name of input layer.
:type name: str
:param scalefactor: An optional normalization scale.
:type scalefactor: float
:param mean: An optional mean subtraction values.
:type mean: cv2.typing.Scalar
:rtype: None
````

````{py:method} setParam(layer, numParam, blob) -> None

Sets the new value for the learned param of the layer.


setParam(layerName, numParam, blob) -> None 
**See also:** Layer::blobs
```{note}
If shape of the new blob differs from the previous shape,then the following forward pass may fail. 
```


:param self: 
:type self: 
:param layer: name or id of the layer.
:type layer: int
:param numParam: index of the layer parameter in the Layer::blobs array.
:type numParam: int
:param blob: the new value.
:type blob: cv2.typing.MatLike
:rtype: None
````

````{py:method} setParam(layer, numParam, blob) -> None

Sets the new value for the learned param of the layer.


setParam(layerName, numParam, blob) -> None 
**See also:** Layer::blobs
```{note}
If shape of the new blob differs from the previous shape,then the following forward pass may fail. 
```


:param self: 
:type self: 
:param layerName: 
:type layerName: str
:param numParam: index of the layer parameter in the Layer::blobs array.
:type numParam: int
:param blob: the new value.
:type blob: cv2.typing.MatLike
:param layer: name or id of the layer.
:type layer: 
:rtype: None
````

````{py:method} getParam(layer[, numParam]) -> retval

Returns parameter blob of the layer.


getParam(layerName[, numParam]) -> retval 
**See also:** Layer::blobs


:param self: 
:type self: 
:param layer: name or id of the layer.
:type layer: int
:param numParam: index of the layer parameter in the Layer::blobs array.
:type numParam: int
:rtype: cv2.typing.MatLike
````

````{py:method} getParam(layer[, numParam]) -> retval

Returns parameter blob of the layer.


getParam(layerName[, numParam]) -> retval 
**See also:** Layer::blobs


:param self: 
:type self: 
:param layerName: 
:type layerName: str
:param numParam: index of the layer parameter in the Layer::blobs array.
:type numParam: int
:param layer: name or id of the layer.
:type layer: 
:rtype: cv2.typing.MatLike
````

````{py:method} getLayersShapes(netInputShapes) -> layersIds, inLayersShapes, outLayersShapes

Returns input and output shapes for all layers in loaded model;preliminary inferencing isn't necessary. 


getLayersShapes(netInputShape) -> layersIds, inLayersShapes, outLayersShapes @overload 


:param self: 
:type self: 
:param netInputShapes: shapes for all input blobs in net input layer.
:type netInputShapes: _typing.Sequence[cv2.typing.MatShape]
:param layersIds: output parameter for layer IDs.
:type layersIds: 
:param inLayersShapes: output parameter for input layers shapes;order is the same as in layersIds 
:type inLayersShapes: 
:param outLayersShapes: output parameter for output layers shapes;order is the same as in layersIds 
:type outLayersShapes: 
:rtype: tuple[_typing.Sequence[int], _typing.Sequence[_typing.Sequence[cv2.typing.MatShape]], _typing.Sequence[_typing.Sequence[cv2.typing.MatShape]]]
````

````{py:method} getLayersShapes(netInputShapes) -> layersIds, inLayersShapes, outLayersShapes

Returns input and output shapes for all layers in loaded model;preliminary inferencing isn't necessary. 


getLayersShapes(netInputShape) -> layersIds, inLayersShapes, outLayersShapes @overload 


:param self: 
:type self: 
:param netInputShape: 
:type netInputShape: cv2.typing.MatShape
:param netInputShapes: shapes for all input blobs in net input layer.
:type netInputShapes: 
:param layersIds: output parameter for layer IDs.
:type layersIds: 
:param inLayersShapes: output parameter for input layers shapes;order is the same as in layersIds 
:type inLayersShapes: 
:param outLayersShapes: output parameter for output layers shapes;order is the same as in layersIds 
:type outLayersShapes: 
:rtype: tuple[_typing.Sequence[int], _typing.Sequence[_typing.Sequence[cv2.typing.MatShape]], _typing.Sequence[_typing.Sequence[cv2.typing.MatShape]]]
````

````{py:method} getFLOPS(netInputShapes) -> retval

Computes FLOP for whole loaded model with specified input shapes.


getFLOPS(netInputShape) -> retval @overload 
getFLOPS(layerId, netInputShapes) -> retval @overload 
getFLOPS(layerId, netInputShape) -> retval @overload 


:param self: 
:type self: 
:param netInputShapes: vector of shapes for all net inputs.
:type netInputShapes: _typing.Sequence[cv2.typing.MatShape]
:return: computed FLOP.
:rtype: int
````

````{py:method} getFLOPS(netInputShapes) -> retval

Computes FLOP for whole loaded model with specified input shapes.


getFLOPS(netInputShape) -> retval @overload 
getFLOPS(layerId, netInputShapes) -> retval @overload 
getFLOPS(layerId, netInputShape) -> retval @overload 


:param self: 
:type self: 
:param netInputShape: 
:type netInputShape: cv2.typing.MatShape
:param netInputShapes: vector of shapes for all net inputs.
:type netInputShapes: 
:return: computed FLOP.
:rtype: int
````

````{py:method} getFLOPS(netInputShapes) -> retval

Computes FLOP for whole loaded model with specified input shapes.


getFLOPS(netInputShape) -> retval @overload 
getFLOPS(layerId, netInputShapes) -> retval @overload 
getFLOPS(layerId, netInputShape) -> retval @overload 


:param self: 
:type self: 
:param layerId: 
:type layerId: int
:param netInputShapes: vector of shapes for all net inputs.
:type netInputShapes: _typing.Sequence[cv2.typing.MatShape]
:return: computed FLOP.
:rtype: int
````

````{py:method} getFLOPS(netInputShapes) -> retval

Computes FLOP for whole loaded model with specified input shapes.


getFLOPS(netInputShape) -> retval @overload 
getFLOPS(layerId, netInputShapes) -> retval @overload 
getFLOPS(layerId, netInputShape) -> retval @overload 


:param self: 
:type self: 
:param layerId: 
:type layerId: int
:param netInputShape: 
:type netInputShape: cv2.typing.MatShape
:param netInputShapes: vector of shapes for all net inputs.
:type netInputShapes: 
:return: computed FLOP.
:rtype: int
````

````{py:method} getMemoryConsumption(netInputShape) -> weights, blobs




@overload 
getMemoryConsumption(layerId, netInputShapes) -> weights, blobs @overload 
getMemoryConsumption(layerId, netInputShape) -> weights, blobs @overload 


:param self: 
:type self: 
:param netInputShape: 
:type netInputShape: cv2.typing.MatShape
:rtype: tuple[int, int]
````

````{py:method} getMemoryConsumption(netInputShape) -> weights, blobs




@overload 
getMemoryConsumption(layerId, netInputShapes) -> weights, blobs @overload 
getMemoryConsumption(layerId, netInputShape) -> weights, blobs @overload 


:param self: 
:type self: 
:param layerId: 
:type layerId: int
:param netInputShapes: 
:type netInputShapes: _typing.Sequence[cv2.typing.MatShape]
:rtype: tuple[int, int]
````

````{py:method} getMemoryConsumption(netInputShape) -> weights, blobs




@overload 
getMemoryConsumption(layerId, netInputShapes) -> weights, blobs @overload 
getMemoryConsumption(layerId, netInputShape) -> weights, blobs @overload 


:param self: 
:type self: 
:param layerId: 
:type layerId: int
:param netInputShape: 
:type netInputShape: cv2.typing.MatShape
:rtype: tuple[int, int]
````

````{py:method} __init__(self)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} empty() -> retval



Returns true if there are no layers in the network. 


:param self: 
:type self: 
:rtype: bool
````

````{py:method} dump() -> retval
Dump net to String




:param self: 
:type self: 
:return: String with structure, hyperparameters, backend, target and fusionCall method after setInput(). To see correct backend, target and fusion run after forward(). 
:rtype: str
````

````{py:method} dumpToFile(path) -> None
Dump net structure, hyperparameters, backend, target and fusion to dot file


**See also:** dump()


:param self: 
:type self: 
:param path: path to output file with .dot extension
:type path: str
:rtype: None
````

````{py:method} getLayerId(layer) -> retval
Converts string name of the layer to the integer identifier.




:param self: 
:type self: 
:param layer: 
:type layer: str
:return: id of the layer, or -1 if the layer wasn't found.
:rtype: int
````

````{py:method} getLayerNames() -> retval





:param self: 
:type self: 
:rtype: _typing.Sequence[str]
````

````{py:method} connect(outPin, inpPin) -> None
Connects output of the first layer to input of the second layer.


Descriptors have the following template <DFN>&lt;layer_name&gt;[.input_number]</DFN>: - the first part of the template <DFN>layer_name</DFN> is string name of the added layer. If this part is empty then the network input pseudo layer will be used; - the second optional part of the template <DFN>input_number</DFN> is either number of the layer input, either label one. If this part is omitted then the first layer input will be used. 
**See also:** setNetInputs(), Layer::inputNameToIndex(), Layer::outputNameToIndex()


:param self: 
:type self: 
:param outPin: descriptor of the first layer output.
:type outPin: str
:param inpPin: descriptor of the second layer input.
:type inpPin: str
:rtype: None
````

````{py:method} setInputsNames(inputBlobNames) -> None
Sets outputs names of the network input pseudo layer.


Each net always has special own the network input pseudo layer with id=0. This layer stores the user blobs only and don't make any computations. In fact, this layer provides the only way to pass user data into the network. As any other layer, this layer can label its outputs and this function provides an easy way to do this. 


:param self: 
:type self: 
:param inputBlobNames: 
:type inputBlobNames: _typing.Sequence[str]
:rtype: None
````

````{py:method} setInputShape(inputName, shape) -> None
Specify shape of network input.




:param self: 
:type self: 
:param inputName: 
:type inputName: str
:param shape: 
:type shape: cv2.typing.MatShape
:rtype: None
````

````{py:method} forwardAsync([, outputName]) -> retval
Runs forward pass to compute output of layer with name @p outputName.


This is an asynchronous version of forward(const String&). dnn::DNN_BACKEND_INFERENCE_ENGINE backend is required. 


:param self: 
:type self: 
:param outputName: name for layer which output is needed to get@details By default runs forward pass for the whole network. 
:type outputName: str
:rtype: cv2.AsyncArray
````

````{py:method} forwardAndRetrieve(outBlobNames) -> outputBlobs
Runs forward pass to compute outputs of layers listed in @p outBlobNames.




:param self: 
:type self: 
:param outBlobNames: names for layers which outputs are needed to get
:type outBlobNames: _typing.Sequence[str]
:param outputBlobs: contains all output blobs for each layer specified in @p outBlobNames.
:type outputBlobs: 
:rtype: _typing.Sequence[_typing.Sequence[cv2.typing.MatLike]]
````

````{py:method} getInputDetails() -> scales, zeropoints
Returns input scale and zeropoint for a quantized Net.




:param self: 
:type self: 
:param scales: output parameter for returning input scales.
:type scales: 
:param zeropoints: output parameter for returning input zeropoints.
:type zeropoints: 
:rtype: tuple[_typing.Sequence[float], _typing.Sequence[int]]
````

````{py:method} getOutputDetails() -> scales, zeropoints
Returns output scale and zeropoint for a quantized Net.




:param self: 
:type self: 
:param scales: output parameter for returning output scales.
:type scales: 
:param zeropoints: output parameter for returning output zeropoints.
:type zeropoints: 
:rtype: tuple[_typing.Sequence[float], _typing.Sequence[int]]
````

````{py:method} setHalideScheduler(scheduler) -> None
Compile Halide layers.


Schedule layers that support Halide backend. Then compile them for specific target. For layers that not represented in scheduling file or if no manual scheduling used at all, automatic scheduling will be applied. 
**See also:** setPreferableBackend


:param self: 
:type self: 
:param scheduler: [in] Path to YAML file with scheduling directives.
:type scheduler: str
:rtype: None
````

````{py:method} setPreferableBackend(backendId) -> None
Ask network to use specific computation backend where it supported.


**See also:** Backend


:param self: 
:type self: 
:param backendId: [in] backend identifier.
:type backendId: int
:rtype: None
````

````{py:method} setPreferableTarget(targetId) -> None
Ask network to make computations on specific target device.


List of supported combinations backend / target: |                        | DNN_BACKEND_OPENCV | DNN_BACKEND_INFERENCE_ENGINE | DNN_BACKEND_HALIDE |  DNN_BACKEND_CUDA | |------------------------|--------------------|------------------------------|--------------------|-------------------| | DNN_TARGET_CPU         |                  + |                            + |                  + |                   | | DNN_TARGET_OPENCL      |                  + |                            + |                  + |                   | | DNN_TARGET_OPENCL_FP16 |                  + |                            + |                    |                   | | DNN_TARGET_MYRIAD      |                    |                            + |                    |                   | | DNN_TARGET_FPGA        |                    |                            + |                    |                   | | DNN_TARGET_CUDA        |                    |                              |                    |                 + | | DNN_TARGET_CUDA_FP16   |                    |                              |                    |                 + | | DNN_TARGET_HDDL        |                    |                            + |                    |                   | 
**See also:** Target


:param self: 
:type self: 
:param targetId: [in] target identifier.
:type targetId: int
:rtype: None
````

````{py:method} getUnconnectedOutLayers() -> retval
Returns indexes of layers with unconnected outputs.


FIXIT: Rework API to registerOutput() approach, deprecate this call 


:param self: 
:type self: 
:rtype: _typing.Sequence[int]
````

````{py:method} getUnconnectedOutLayersNames() -> retval
Returns names of layers with unconnected outputs.


FIXIT: Rework API to registerOutput() approach, deprecate this call 


:param self: 
:type self: 
:rtype: _typing.Sequence[str]
````

````{py:method} getLayerTypes() -> layersTypes
Returns list of types for layer used in model.




:param self: 
:type self: 
:param layersTypes: output parameter for returning types.
:type layersTypes: 
:rtype: _typing.Sequence[str]
````

````{py:method} getLayersCount(layerType) -> retval
Returns count of layers of specified type.




:param self: 
:type self: 
:param layerType: type.
:type layerType: str
:return: count of layers
:rtype: int
````

````{py:method} enableFusion(fusion) -> None
Enables or disables layer fusion in the network.




:param self: 
:type self: 
:param fusion: true to enable the fusion, false to disable. The fusion is enabled by default.
:type fusion: bool
:rtype: None
````

````{py:method} enableWinograd(useWinograd) -> None
Enables or disables the Winograd compute branch. The Winograd compute branch can speed up3x3 Convolution at a small loss of accuracy. 




:param self: 
:type self: 
:param useWinograd: true to enable the Winograd compute branch. The default is true.
:type useWinograd: bool
:rtype: None
````

````{py:method} getPerfProfile() -> retval, timings
Returns overall time for inference and timings (in ticks) for layers.


Indexes in returned vector correspond to layers ids. Some layers can be fused with others, in this case zero ticks count will be return for that skipped layers. Supported by DNN_BACKEND_OPENCV on DNN_TARGET_CPU only. 


:param self: 
:type self: 
:param timings: [out] vector for tick timings for all layers.
:type timings: 
:return: overall ticks for model inference.
:rtype: tuple[int, _typing.Sequence[float]]
````


`````


`````{py:class} SegmentationModel




````{py:method} __init__(self, model: str, config: str=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param model: 
:type model: str
:param config: 
:type config: str
:rtype: None
````

````{py:method} __init__(self, network: Net)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param network: 
:type network: Net
:rtype: None
````

````{py:method} segment(frame[, mask]) -> mask

Given the @p input frame, create input blob, run net




:param self: 
:type self: 
:param frame: [in] The input image.
:type frame: cv2.typing.MatLike
:param mask: [out] Allocated class prediction for each pixel
:type mask: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} segment(frame[, mask]) -> mask

Given the @p input frame, create input blob, run net




:param self: 
:type self: 
:param frame: [in] The input image.
:type frame: cv2.UMat
:param mask: [out] Allocated class prediction for each pixel
:type mask: cv2.UMat | None
:rtype: cv2.UMat
````


`````


`````{py:class} TextDetectionModel




````{py:method} detect(frame) -> detections, confidences

Performs detection


Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections. 
Each result is quadrangle's 4 points in this order: - bottom-left - top-left - top-right - bottom-right 
Use cv::getPerspectiveTransform function to retrieve image region without perspective transformations. 
detect(frame) -> detections @overload 
```{note}
If DL model doesn't support that kind of output then result may be derived from detectTextRectangles() output.
```


:param self: 
:type self: 
:param frame: [in] The input image
:type frame: cv2.typing.MatLike
:param detections: [out] array with detections' quadrangles (4 points per result)
:type detections: 
:param confidences: [out] array with detection confidences
:type confidences: 
:rtype: tuple[_typing.Sequence[_typing.Sequence[cv2.typing.Point]], _typing.Sequence[float]]
````

````{py:method} detect(frame) -> detections, confidences

Performs detection


Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections. 
Each result is quadrangle's 4 points in this order: - bottom-left - top-left - top-right - bottom-right 
Use cv::getPerspectiveTransform function to retrieve image region without perspective transformations. 
detect(frame) -> detections @overload 
```{note}
If DL model doesn't support that kind of output then result may be derived from detectTextRectangles() output.
```


:param self: 
:type self: 
:param frame: [in] The input image
:type frame: cv2.UMat
:param detections: [out] array with detections' quadrangles (4 points per result)
:type detections: 
:param confidences: [out] array with detection confidences
:type confidences: 
:rtype: tuple[_typing.Sequence[_typing.Sequence[cv2.typing.Point]], _typing.Sequence[float]]
````

````{py:method} detect(frame) -> detections, confidences

Performs detection


Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections. 
Each result is quadrangle's 4 points in this order: - bottom-left - top-left - top-right - bottom-right 
Use cv::getPerspectiveTransform function to retrieve image region without perspective transformations. 
detect(frame) -> detections @overload 
```{note}
If DL model doesn't support that kind of output then result may be derived from detectTextRectangles() output.
```


:param self: 
:type self: 
:param frame: [in] The input image
:type frame: cv2.typing.MatLike
:param detections: [out] array with detections' quadrangles (4 points per result)
:type detections: 
:param confidences: [out] array with detection confidences
:type confidences: 
:rtype: _typing.Sequence[_typing.Sequence[cv2.typing.Point]]
````

````{py:method} detect(frame) -> detections, confidences

Performs detection


Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections. 
Each result is quadrangle's 4 points in this order: - bottom-left - top-left - top-right - bottom-right 
Use cv::getPerspectiveTransform function to retrieve image region without perspective transformations. 
detect(frame) -> detections @overload 
```{note}
If DL model doesn't support that kind of output then result may be derived from detectTextRectangles() output.
```


:param self: 
:type self: 
:param frame: [in] The input image
:type frame: cv2.UMat
:param detections: [out] array with detections' quadrangles (4 points per result)
:type detections: 
:param confidences: [out] array with detection confidences
:type confidences: 
:rtype: _typing.Sequence[_typing.Sequence[cv2.typing.Point]]
````

````{py:method} detectTextRectangles(frame) -> detections, confidences

Performs detection


Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections. 
Each result is rotated rectangle. 
detectTextRectangles(frame) -> detections @overload 
```{note}
Result may be inaccurate in case of strong perspective transformations.
```


:param self: 
:type self: 
:param frame: [in] the input image
:type frame: cv2.typing.MatLike
:param detections: [out] array with detections' RotationRect results
:type detections: 
:param confidences: [out] array with detection confidences
:type confidences: 
:rtype: tuple[_typing.Sequence[cv2.typing.RotatedRect], _typing.Sequence[float]]
````

````{py:method} detectTextRectangles(frame) -> detections, confidences

Performs detection


Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections. 
Each result is rotated rectangle. 
detectTextRectangles(frame) -> detections @overload 
```{note}
Result may be inaccurate in case of strong perspective transformations.
```


:param self: 
:type self: 
:param frame: [in] the input image
:type frame: cv2.UMat
:param detections: [out] array with detections' RotationRect results
:type detections: 
:param confidences: [out] array with detection confidences
:type confidences: 
:rtype: tuple[_typing.Sequence[cv2.typing.RotatedRect], _typing.Sequence[float]]
````

````{py:method} detectTextRectangles(frame) -> detections, confidences

Performs detection


Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections. 
Each result is rotated rectangle. 
detectTextRectangles(frame) -> detections @overload 
```{note}
Result may be inaccurate in case of strong perspective transformations.
```


:param self: 
:type self: 
:param frame: [in] the input image
:type frame: cv2.typing.MatLike
:param detections: [out] array with detections' RotationRect results
:type detections: 
:param confidences: [out] array with detection confidences
:type confidences: 
:rtype: _typing.Sequence[cv2.typing.RotatedRect]
````

````{py:method} detectTextRectangles(frame) -> detections, confidences

Performs detection


Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections. 
Each result is rotated rectangle. 
detectTextRectangles(frame) -> detections @overload 
```{note}
Result may be inaccurate in case of strong perspective transformations.
```


:param self: 
:type self: 
:param frame: [in] the input image
:type frame: cv2.UMat
:param detections: [out] array with detections' RotationRect results
:type detections: 
:param confidences: [out] array with detection confidences
:type confidences: 
:rtype: _typing.Sequence[cv2.typing.RotatedRect]
````


`````


`````{py:class} TextDetectionModel_DB




````{py:method} __init__(self, network: Net)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param network: 
:type network: Net
:rtype: None
````

````{py:method} __init__(self, model: str, config: str=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param model: 
:type model: str
:param config: 
:type config: str
:rtype: None
````

````{py:method} setBinaryThreshold(binaryThreshold) -> retval





:param self: 
:type self: 
:param binaryThreshold: 
:type binaryThreshold: float
:rtype: TextDetectionModel_DB
````

````{py:method} getBinaryThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setPolygonThreshold(polygonThreshold) -> retval





:param self: 
:type self: 
:param polygonThreshold: 
:type polygonThreshold: float
:rtype: TextDetectionModel_DB
````

````{py:method} getPolygonThreshold() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setUnclipRatio(unclipRatio) -> retval





:param self: 
:type self: 
:param unclipRatio: 
:type unclipRatio: float
:rtype: TextDetectionModel_DB
````

````{py:method} getUnclipRatio() -> retval





:param self: 
:type self: 
:rtype: float
````

````{py:method} setMaxCandidates(maxCandidates) -> retval





:param self: 
:type self: 
:param maxCandidates: 
:type maxCandidates: int
:rtype: TextDetectionModel_DB
````

````{py:method} getMaxCandidates() -> retval





:param self: 
:type self: 
:rtype: int
````


`````


`````{py:class} TextDetectionModel_EAST




````{py:method} __init__(self, network: Net)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param network: 
:type network: Net
:rtype: None
````

````{py:method} __init__(self, model: str, config: str=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param model: 
:type model: str
:param config: 
:type config: str
:rtype: None
````

````{py:method} setConfidenceThreshold(confThreshold) -> retval
Set the detection confidence threshold




:param self: 
:type self: 
:param confThreshold: [in] A threshold used to filter boxes by confidences
:type confThreshold: float
:rtype: TextDetectionModel_EAST
````

````{py:method} getConfidenceThreshold() -> retval
Get the detection confidence threshold




:param self: 
:type self: 
:rtype: float
````

````{py:method} setNMSThreshold(nmsThreshold) -> retval
Set the detection NMS filter threshold




:param self: 
:type self: 
:param nmsThreshold: [in] A threshold used in non maximum suppression
:type nmsThreshold: float
:rtype: TextDetectionModel_EAST
````

````{py:method} getNMSThreshold() -> retval
Get the detection confidence threshold




:param self: 
:type self: 
:rtype: float
````


`````


`````{py:class} TextRecognitionModel




````{py:method} __init__(self, network: Net)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param network: 
:type network: Net
:rtype: None
````

````{py:method} __init__(self, model: str, config: str=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param model: 
:type model: str
:param config: 
:type config: str
:rtype: None
````

````{py:method} recognize(frame) -> retval

Given the @p input frame, create input blob, run net and return recognition result


recognize(frame, roiRects) -> results 


:param self: 
:type self: 
:param frame: [in] The input image
:type frame: cv2.typing.MatLike
:param roiRects: [in] List of text detection regions of interest (cv::Rect, CV_32SC4). ROIs is be cropped as the network inputs
:type roiRects: 
:param results: [out] A set of text recognition results.
:type results: 
:return: The text recognition result
:rtype: str
````

````{py:method} recognize(frame) -> retval

Given the @p input frame, create input blob, run net and return recognition result


recognize(frame, roiRects) -> results 


:param self: 
:type self: 
:param frame: [in] The input image
:type frame: cv2.UMat
:param roiRects: [in] List of text detection regions of interest (cv::Rect, CV_32SC4). ROIs is be cropped as the network inputs
:type roiRects: 
:param results: [out] A set of text recognition results.
:type results: 
:return: The text recognition result
:rtype: str
````

````{py:method} recognize(frame) -> retval

Given the @p input frame, create input blob, run net and return recognition result


recognize(frame, roiRects) -> results 


:param self: 
:type self: 
:param frame: [in] The input image
:type frame: cv2.typing.MatLike
:param roiRects: [in] List of text detection regions of interest (cv::Rect, CV_32SC4). ROIs is be cropped as the network inputs
:type roiRects: _typing.Sequence[cv2.typing.MatLike]
:param results: [out] A set of text recognition results.
:type results: 
:return: The text recognition result
:rtype: _typing.Sequence[str]
````

````{py:method} recognize(frame) -> retval

Given the @p input frame, create input blob, run net and return recognition result


recognize(frame, roiRects) -> results 


:param self: 
:type self: 
:param frame: [in] The input image
:type frame: cv2.UMat
:param roiRects: [in] List of text detection regions of interest (cv::Rect, CV_32SC4). ROIs is be cropped as the network inputs
:type roiRects: _typing.Sequence[cv2.UMat]
:param results: [out] A set of text recognition results.
:type results: 
:return: The text recognition result
:rtype: _typing.Sequence[str]
````

````{py:method} setDecodeType(decodeType) -> retval
Set the decoding method of translating the network output into string




:param self: 
:type self: 
:param decodeType: [in] The decoding method of translating the network output into string, currently supported type:- `"CTC-greedy"` greedy decoding for the output of CTC-based methods - `"CTC-prefix-beam-search"` Prefix beam search decoding for the output of CTC-based methods 
:type decodeType: str
:rtype: TextRecognitionModel
````

````{py:method} getDecodeType() -> retval
Get the decoding method




:param self: 
:type self: 
:return: the decoding method
:rtype: str
````

````{py:method} setDecodeOptsCTCPrefixBeamSearch(beamSize[, vocPruneSize]) -> retval
Set the decoding method options for `"CTC-prefix-beam-search"` decode usage




:param self: 
:type self: 
:param beamSize: [in] Beam size for search
:type beamSize: int
:param vocPruneSize: [in] Parameter to optimize big vocabulary search,only take top @p vocPruneSize tokens in each search step, @p vocPruneSize <= 0 stands for disable this prune. 
:type vocPruneSize: int
:rtype: TextRecognitionModel
````

````{py:method} setVocabulary(vocabulary) -> retval
Set the vocabulary for recognition.




:param self: 
:type self: 
:param vocabulary: [in] the associated vocabulary of the network.
:type vocabulary: _typing.Sequence[str]
:rtype: TextRecognitionModel
````

````{py:method} getVocabulary() -> retval
Get the vocabulary for recognition.




:param self: 
:type self: 
:return: vocabulary the associated vocabulary
:rtype: _typing.Sequence[str]
````


`````



## Functions
````{py:function} NMSBoxes(bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]]) -> indices

Performs non maximum suppression given boxes and corresponding scores.




:param bboxes: a set of bounding boxes to apply NMS.
:type bboxes: _typing.Sequence[cv2.typing.Rect2d]
:param scores: a set of corresponding confidences.
:type scores: _typing.Sequence[float]
:param score_threshold: a threshold used to filter boxes by score.
:type score_threshold: float
:param nms_threshold: a threshold used in non maximum suppression.
:type nms_threshold: float
:param indices: the kept indices of bboxes after NMS.
:type indices: 
:param eta: a coefficient in adaptive threshold formula: $nms\_threshold_{i+1}=eta\cdot nms\_threshold_i$.
:type eta: float
:param top_k: if `>0`, keep at most @p top_k picked indices.
:type top_k: int
:rtype: _typing.Sequence[int]
````


````{py:function} NMSBoxesBatched(bboxes, scores, class_ids, score_threshold, nms_threshold[, eta[, top_k]]) -> indices

Performs batched non maximum suppression on given boxes and corresponding scores across different classes.




:param bboxes: a set of bounding boxes to apply NMS.
:type bboxes: _typing.Sequence[cv2.typing.Rect2d]
:param scores: a set of corresponding confidences.
:type scores: _typing.Sequence[float]
:param class_ids: a set of corresponding class ids. Ids are integer and usually start from 0.
:type class_ids: _typing.Sequence[int]
:param score_threshold: a threshold used to filter boxes by score.
:type score_threshold: float
:param nms_threshold: a threshold used in non maximum suppression.
:type nms_threshold: float
:param indices: the kept indices of bboxes after NMS.
:type indices: 
:param eta: a coefficient in adaptive threshold formula: $nms\_threshold_{i+1}=eta\cdot nms\_threshold_i$.
:type eta: float
:param top_k: if `>0`, keep at most @p top_k picked indices.
:type top_k: int
:rtype: _typing.Sequence[int]
````


````{py:function} NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]]) -> indices






:param bboxes: 
:type bboxes: _typing.Sequence[cv2.typing.RotatedRect]
:param scores: 
:type scores: _typing.Sequence[float]
:param score_threshold: 
:type score_threshold: float
:param nms_threshold: 
:type nms_threshold: float
:param eta: 
:type eta: float
:param top_k: 
:type top_k: int
:rtype: _typing.Sequence[int]
````


````{py:function} Net_readFromModelOptimizer(xml, bin) -> retval

Create a network from Intel's Model Optimizer in-memory buffers with intermediate representation (IR).


Net_readFromModelOptimizer(bufferModelConfig, bufferWeights) -> retval 


:param xml: [in] XML configuration file with network's topology.
:type xml: 
:param bin: [in] Binary file with trained weights.Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine backend. 
:type bin: 
:param bufferModelConfig: [in] buffer with model's configuration.
:type bufferModelConfig: 
:param bufferWeights: [in] buffer with model's trained weights.
:type bufferWeights: 
:return: Net object.
:rtype: object
````


````{py:function} blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]]) -> retval

Creates 4-dimensional blob from image. Optionally resizes and crops @p image from center,subtract @p mean values, scales values by @p scalefactor, swap Blue and Red channels. 


@note The order and usage of `scalefactor` and `mean` are (input - mean) * scalefactor. 


:param image: input image (with 1-, 3- or 4-channels).
:type image: cv2.typing.MatLike
:param scalefactor: multiplier for @p images values.
:type scalefactor: float
:param size: spatial size for output image
:type size: cv2.typing.Size
:param mean: scalar with mean values which are subtracted from channels. Values are intendedto be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true. 
:type mean: cv2.typing.Scalar
:param swapRB: flag which indicates that swap first and last channelsin 3-channel image is necessary. 
:type swapRB: bool
:param crop: flag which indicates whether image will be cropped after resize or not
:type crop: bool
:param ddepth: Depth of output blob. Choose CV_32F or CV_8U.@details if @p crop is true, input image is resized so one side after resize is equal to corresponding dimension in @p size and another one is equal or larger. Then, crop from the center is performed. If @p crop is false, direct resize without cropping and preserving aspect ratio is performed. 
:type ddepth: int
:return: 4-dimensional Mat with NCHW dimensions order.
:rtype: cv2.typing.MatLike
````


````{py:function} blobFromImageWithParams(image[, param]) -> retval

Creates 4-dimensional blob from image with given params.


@details This function is an extension of @ref blobFromImage to meet more image preprocess needs. Given input image and preprocessing parameters, and function outputs the blob. 
blobFromImageWithParams(image[, blob[, param]]) -> blob @overload 


:param image: input image (all with 1-, 3- or 4-channels).
:type image: cv2.typing.MatLike
:param param: struct of Image2BlobParams, contains all parameters needed by processing of image to blob.
:type param: Image2BlobParams
:return: 4-dimensional Mat.
:rtype: cv2.typing.MatLike
````


````{py:function} blobFromImages(images[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]]) -> retval

Creates 4-dimensional blob from series of images. Optionally resizes andcrops @p images from center, subtract @p mean values, scales values by @p scalefactor, swap Blue and Red channels. 


@note The order and usage of `scalefactor` and `mean` are (input - mean) * scalefactor. 


:param images: input images (all with 1-, 3- or 4-channels).
:type images: _typing.Sequence[cv2.typing.MatLike]
:param size: spatial size for output image
:type size: cv2.typing.Size
:param mean: scalar with mean values which are subtracted from channels. Values are intendedto be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true. 
:type mean: cv2.typing.Scalar
:param scalefactor: multiplier for @p images values.
:type scalefactor: float
:param swapRB: flag which indicates that swap first and last channelsin 3-channel image is necessary. 
:type swapRB: bool
:param crop: flag which indicates whether image will be cropped after resize or not
:type crop: bool
:param ddepth: Depth of output blob. Choose CV_32F or CV_8U.@details if @p crop is true, input image is resized so one side after resize is equal to corresponding dimension in @p size and another one is equal or larger. Then, crop from the center is performed. If @p crop is false, direct resize without cropping and preserving aspect ratio is performed. 
:type ddepth: int
:return: 4-dimensional Mat with NCHW dimensions order.
:rtype: cv2.typing.MatLike
````


````{py:function} blobFromImagesWithParams(images[, param]) -> retval

Creates 4-dimensional blob from series of images with given params.


@details This function is an extension of @ref blobFromImages to meet more image preprocess needs. Given input image and preprocessing parameters, and function outputs the blob. 
blobFromImagesWithParams(images[, blob[, param]]) -> blob @overload 


:param images: input image (all with 1-, 3- or 4-channels).
:type images: _typing.Sequence[cv2.typing.MatLike]
:param param: struct of Image2BlobParams, contains all parameters needed by processing of image to blob.
:type param: Image2BlobParams
:return: 4-dimensional Mat.
:rtype: cv2.typing.MatLike
````


````{py:function} getAvailableTargets(be) -> retval






:param be: 
:type be: Backend
:rtype: _typing.Sequence[Target]
````


````{py:function} imagesFromBlob(blob_[, images_]) -> images_

Parse a 4D blob and output the images it contains as 2D arrays through a simpler data structure(std::vector<cv::Mat>). 




:param blob_: [in] 4 dimensional array (images, channels, height, width) in floating point precision (CV_32F) fromwhich you would like to extract the images. 
:type blob_: cv2.typing.MatLike
:param images_: [out] array of 2D Mat containing the images extracted from the blob in floating point precision(CV_32F). They are non normalized neither mean added. The number of returned images equals the first dimension of the blob (batch size). Every image has a number of channels equals to the second dimension of the blob (depth). 
:type images_: _typing.Sequence[cv2.typing.MatLike] | None
:rtype: _typing.Sequence[cv2.typing.MatLike]
````


````{py:function} readNet(model[, config[, framework]]) -> retval

Read deep learning network represented in one of the supported formats.@details This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts. 


This function automatically detects an origin framework of trained model and calls an appropriate function such @ref readNetFromCaffe, @ref readNetFromTensorflow, @ref readNetFromTorch or @ref readNetFromDarknet. An order of @p model and @p config arguments does not matter. 
readNet(framework, bufferModel[, bufferConfig]) -> retval 


:param model: [in] Binary file contains trained weights. The following fileextensions are expected for models from different frameworks: `*.caffemodel` (Caffe, http://caffe.berkeleyvision.org/) `*.pb` (TensorFlow, https://www.tensorflow.org/) `*.t7` | `*.net` (Torch, http://torch.ch/) `*.weights` (Darknet, https://pjreddie.com/darknet/) `*.bin` | `*.onnx` (OpenVINO, https://software.intel.com/openvino-toolkit) `*.onnx` (ONNX, https://onnx.ai/) 
:type model: str
:param config: [in] Text file contains network configuration. It could be afile with the following extensions: `*.prototxt` (Caffe, http://caffe.berkeleyvision.org/) `*.pbtxt` (TensorFlow, https://www.tensorflow.org/) `*.cfg` (Darknet, https://pjreddie.com/darknet/) `*.xml` (OpenVINO, https://software.intel.com/openvino-toolkit) 
:type config: str
:param framework: [in] Name of origin framework.
:type framework: str
:param bufferModel: [in] A buffer with a content of binary file with weights
:type bufferModel: 
:param bufferConfig: [in] A buffer with a content of text file contains network configuration.
:type bufferConfig: 
:return: Net object.
:rtype: Net
````


````{py:function} readNetFromCaffe(prototxt[, caffeModel]) -> retval

Reads a network model stored in Caffe model in memory.


readNetFromCaffe(bufferProto[, bufferModel]) -> retval 


:param prototxt: path to the .prototxt file with text description of the network architecture.
:type prototxt: str
:param caffeModel: path to the .caffemodel file with learned network.
:type caffeModel: str
:param bufferProto: buffer containing the content of the .prototxt file
:type bufferProto: 
:param bufferModel: buffer containing the content of the .caffemodel file
:type bufferModel: 
:return: Net object.
:rtype: Net
````


````{py:function} readNetFromDarknet(cfgFile[, darknetModel]) -> retval

Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.


readNetFromDarknet(bufferCfg[, bufferModel]) -> retval 


:param cfgFile: path to the .cfg file with text description of the network architecture.
:type cfgFile: str
:param darknetModel: path to the .weights file with learned network.
:type darknetModel: str
:param bufferCfg: A buffer contains a content of .cfg file with text description of the network architecture.
:type bufferCfg: 
:param bufferModel: A buffer contains a content of .weights file with learned network.
:type bufferModel: 
:return: Net object.
:rtype: Net
````


````{py:function} readNetFromModelOptimizer(xml[, bin]) -> retval

Load a network from Intel's Model Optimizer intermediate representation.


readNetFromModelOptimizer(bufferModelConfig, bufferWeights) -> retval 


:param xml: [in] XML configuration file with network's topology.
:type xml: str
:param bin: [in] Binary file with trained weights.
:type bin: str
:param bufferModelConfig: [in] Buffer contains XML configuration with network's topology.
:type bufferModelConfig: 
:param bufferWeights: [in] Buffer contains binary data with trained weights.
:type bufferWeights: 
:return: Net object.Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine backend. 
:rtype: Net
````


````{py:function} readNetFromONNX(onnxFile) -> retval

Reads a network model from <a href="https://onnx.ai/">ONNX</a>in-memory buffer. 


readNetFromONNX(buffer) -> retval 


:param onnxFile: path to the .onnx file with text description of the network architecture.
:type onnxFile: str
:param buffer: in-memory buffer that stores the ONNX model bytes.
:type buffer: 
:return: Network object that ready to do forward, throw an exceptionin failure cases. 
:rtype: Net
````


````{py:function} readNetFromTFLite(model) -> retval

Reads a network model stored in <a href="https://www.tensorflow.org/lite">TFLite</a> framework's format.


readNetFromTFLite(bufferModel) -> retval 


:param model: path to the .tflite file with binary flatbuffers description of the network architecture
:type model: str
:param bufferModel: buffer containing the content of the tflite file
:type bufferModel: 
:return: Net object.
:rtype: Net
````


````{py:function} readNetFromTensorflow(model[, config]) -> retval

Reads a network model stored in <a href="https://www.tensorflow.org/">TensorFlow</a> framework's format.


readNetFromTensorflow(bufferModel[, bufferConfig]) -> retval 


:param model: path to the .pb file with binary protobuf description of the network architecture
:type model: str
:param config: path to the .pbtxt file that contains text graph definition in protobuf format.Resulting Net object is built by text graph using weights from a binary one that let us make it more flexible. 
:type config: str
:param bufferModel: buffer containing the content of the pb file
:type bufferModel: 
:param bufferConfig: buffer containing the content of the pbtxt file
:type bufferConfig: 
:return: Net object.
:rtype: Net
````


````{py:function} readNetFromTorch(model[, isBinary[, evaluate]]) -> retval

Reads a network model stored in <a href="http://torch.ch">Torch7</a> framework's format.


The loading file must contain serialized <a href="https://github.com/torch/nn/blob/master/doc/module.md">nn.Module</a> object with importing network. Try to eliminate a custom objects from serialazing data to avoid importing errors. 
List of supported layers (i.e. object instances derived from Torch nn.Module class): - nn.Sequential - nn.Parallel - nn.Concat - nn.Linear - nn.SpatialConvolution - nn.SpatialMaxPooling, nn.SpatialAveragePooling - nn.ReLU, nn.TanH, nn.Sigmoid - nn.Reshape - nn.SoftMax, nn.LogSoftMax 
Also some equivalents of these classes from cunn, cudnn, and fbcunn may be successfully imported. 
```{note}
Ascii mode of Torch serializer is more preferable, because binary mode extensively use `long` type of C language,which has various bit-length on different systems. 
```


:param model: path to the file, dumped from Torch by using torch.save() function.
:type model: str
:param isBinary: specifies whether the network was serialized in ascii mode or binary.
:type isBinary: bool
:param evaluate: specifies testing phase of network. If true, it's similar to evaluate() method in Torch.
:type evaluate: bool
:return: Net object.
:rtype: Net
````


````{py:function} readTensorFromONNX(path) -> retval

Creates blob from .pb file.




:param path: to the .pb file with input tensor.
:type path: str
:return: Mat.
:rtype: cv2.typing.MatLike
````


````{py:function} readTorchBlob(filename[, isBinary]) -> retval

Loads blob which was serialized as torch.Tensor object of Torch7 framework.@warning This function has the same limitations as readNetFromTorch(). 




:param filename: 
:type filename: str
:param isBinary: 
:type isBinary: bool
:rtype: cv2.typing.MatLike
````


````{py:function} shrinkCaffeModel(src, dst[, layersTypes]) -> None

Convert all weights of Caffe network to half precision floating point.


```{note}
Shrinked model has no origin float32 weights so it can't be usedin origin Caffe framework anymore. However the structure of data is taken from NVidia's Caffe fork: https://github.com/NVIDIA/caffe. So the resulting model may be used there. 
```


:param src: Path to origin model from Caffe framework contains singleprecision floating point weights (usually has `.caffemodel` extension). 
:type src: str
:param dst: Path to destination model with updated weights.
:type dst: str
:param layersTypes: Set of layers types which parameters will be converted.By default, converts only Convolutional and Fully-Connected layers' weights. 
:type layersTypes: _typing.Sequence[str]
:rtype: None
````


````{py:function} softNMSBoxes(bboxes, scores, score_threshold, nms_threshold[, top_k[, sigma[, method]]]) -> updated_scores, indices

Performs soft non maximum suppression given boxes and corresponding scores.Reference: https://arxiv.org/abs/1704.04503 


**See also:** SoftNMSMethod


:param bboxes: a set of bounding boxes to apply Soft NMS.
:type bboxes: _typing.Sequence[cv2.typing.Rect]
:param scores: a set of corresponding confidences.
:type scores: _typing.Sequence[float]
:param updated_scores: a set of corresponding updated confidences.
:type updated_scores: 
:param score_threshold: a threshold used to filter boxes by score.
:type score_threshold: float
:param nms_threshold: a threshold used in non maximum suppression.
:type nms_threshold: float
:param indices: the kept indices of bboxes after NMS.
:type indices: 
:param top_k: keep at most @p top_k picked indices.
:type top_k: int
:param sigma: parameter of Gaussian weighting.
:type sigma: float
:param method: Gaussian or linear.
:type method: SoftNMSMethod
:rtype: tuple[_typing.Sequence[float], _typing.Sequence[int]]
````


````{py:function} writeTextGraph(model, output) -> None

Create a text representation for a binary network stored in protocol buffer format.


```{note}
To reduce output file size, trained weights are not included.
```


:param model: [in] A path to binary network.
:type model: str
:param output: [in] A path to output text file to be created.
:type output: str
:rtype: None
````



