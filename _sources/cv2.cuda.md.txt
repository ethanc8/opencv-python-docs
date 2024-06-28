# `cv2.cuda`
```{py:module} cv2.cuda
None
```
## Attributes
```{py:attribute} DEVICE_INFO_COMPUTE_MODE_DEFAULT
:type: int
```


```{py:attribute} DEVICE_INFO_COMPUTE_MODE_EXCLUSIVE
:type: int
```


```{py:attribute} DEVICE_INFO_COMPUTE_MODE_EXCLUSIVE_PROCESS
:type: int
```


```{py:attribute} DEVICE_INFO_COMPUTE_MODE_PROHIBITED
:type: int
```


```{py:attribute} DYNAMIC_PARALLELISM
:type: int
```


```{py:attribute} DeviceInfo_ComputeModeDefault
:type: int
```


```{py:attribute} DeviceInfo_ComputeModeExclusive
:type: int
```


```{py:attribute} DeviceInfo_ComputeModeExclusiveProcess
:type: int
```


```{py:attribute} DeviceInfo_ComputeModeProhibited
:type: int
```


```{py:attribute} EVENT_BLOCKING_SYNC
:type: int
```


```{py:attribute} EVENT_DEFAULT
:type: int
```


```{py:attribute} EVENT_DISABLE_TIMING
:type: int
```


```{py:attribute} EVENT_INTERPROCESS
:type: int
```


```{py:attribute} Event_BLOCKING_SYNC
:type: int
```


```{py:attribute} Event_DEFAULT
:type: int
```


```{py:attribute} Event_DISABLE_TIMING
:type: int
```


```{py:attribute} Event_INTERPROCESS
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_10
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_11
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_12
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_13
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_20
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_21
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_30
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_32
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_35
:type: int
```


```{py:attribute} FEATURE_SET_COMPUTE_50
:type: int
```


```{py:attribute} GLOBAL_ATOMICS
:type: int
```


```{py:attribute} HOST_MEM_PAGE_LOCKED
:type: int
```


```{py:attribute} HOST_MEM_SHARED
:type: int
```


```{py:attribute} HOST_MEM_WRITE_COMBINED
:type: int
```


```{py:attribute} HostMem_PAGE_LOCKED
:type: int
```


```{py:attribute} HostMem_SHARED
:type: int
```


```{py:attribute} HostMem_WRITE_COMBINED
:type: int
```


```{py:attribute} NATIVE_DOUBLE
:type: int
```


```{py:attribute} SHARED_ATOMICS
:type: int
```


```{py:attribute} WARP_SHUFFLE_FUNCTIONS
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
`````{py:class} BufferPool




````{py:method} getBuffer(rows, cols, type) -> retval




getBuffer(size, type) -> retval 


:param self: 
:type self: 
:param rows: 
:type rows: int
:param cols: 
:type cols: int
:param type: 
:type type: int
:rtype: GpuMat
````

````{py:method} getBuffer(rows, cols, type) -> retval




getBuffer(size, type) -> retval 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param type: 
:type type: int
:rtype: GpuMat
````

````{py:method} __init__(self, stream: Stream)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param stream: 
:type stream: Stream
:rtype: None
````

````{py:method} getAllocator() -> retval





:param self: 
:type self: 
:rtype: GpuMat.Allocator
````


`````


`````{py:class} DeviceInfo




````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, device_id: int)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param device_id: 
:type device_id: int
:rtype: None
````

````{py:method} deviceID() -> retval
Returns system index of the CUDA device starting with 0.




:param self: 
:type self: 
:rtype: int
````

````{py:method} totalGlobalMem() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} sharedMemPerBlock() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} regsPerBlock() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} warpSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} memPitch() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} maxThreadsPerBlock() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} maxThreadsDim() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec3i
````

````{py:method} maxGridSize() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec3i
````

````{py:method} clockRate() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} totalConstMem() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} majorVersion() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} minorVersion() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} textureAlignment() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} texturePitchAlignment() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} multiProcessorCount() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} kernelExecTimeoutEnabled() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} integrated() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} canMapHostMemory() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} computeMode() -> retval





:param self: 
:type self: 
:rtype: DeviceInfo_ComputeMode
````

````{py:method} maxTexture1D() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} maxTexture1DMipmap() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} maxTexture1DLinear() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} maxTexture2D() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec2i
````

````{py:method} maxTexture2DMipmap() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec2i
````

````{py:method} maxTexture2DLinear() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec3i
````

````{py:method} maxTexture2DGather() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec2i
````

````{py:method} maxTexture3D() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec3i
````

````{py:method} maxTextureCubemap() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} maxTexture1DLayered() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec2i
````

````{py:method} maxTexture2DLayered() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec3i
````

````{py:method} maxTextureCubemapLayered() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec2i
````

````{py:method} maxSurface1D() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} maxSurface2D() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec2i
````

````{py:method} maxSurface3D() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec3i
````

````{py:method} maxSurface1DLayered() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec2i
````

````{py:method} maxSurface2DLayered() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec3i
````

````{py:method} maxSurfaceCubemap() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} maxSurfaceCubemapLayered() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Vec2i
````

````{py:method} surfaceAlignment() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} concurrentKernels() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} ECCEnabled() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} pciBusID() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} pciDeviceID() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} pciDomainID() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} tccDriver() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} asyncEngineCount() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} unifiedAddressing() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} memoryClockRate() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} memoryBusWidth() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} l2CacheSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} maxThreadsPerMultiProcessor() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} queryMemory(totalMemory, freeMemory) -> None





:param self: 
:type self: 
:param totalMemory: 
:type totalMemory: int
:param freeMemory: 
:type freeMemory: int
:rtype: None
````

````{py:method} freeMemory() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} totalMemory() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} isCompatible() -> retval
Checks the CUDA module and device compatibility.


This function returns true if the CUDA module can be run on the specified device. Otherwise, it returns false . 


:param self: 
:type self: 
:rtype: bool
````


`````


`````{py:class} Event




````{py:method} __init__(self, flags: Event_CreateFlags=...)



Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param flags: 
:type flags: Event_CreateFlags
:rtype: None
````

````{py:method} record([, stream]) -> None





:param self: 
:type self: 
:param stream: 
:type stream: Stream
:rtype: None
````

````{py:method} queryIfComplete() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} waitForCompletion() -> None





:param self: 
:type self: 
:rtype: None
````

````{py:method} elapsedTime(start, end) -> retval
:staticmethod:





:param start: 
:type start: Event
:param end: 
:type end: Event
:rtype: float
````


`````


`````{py:class} GpuData





`````


`````{py:class} GpuMat




````{py:method} step






:param self: 
:type self: 
:rtype: int
````

````{py:method} __init__(self, allocator: GpuMat.Allocator=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} __init__(self, rows: int, cols: int, type: int, allocator: GpuMat.Allocator=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param rows: 
:type rows: int
:param cols: 
:type cols: int
:param type: 
:type type: int
:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} __init__(self, size: cv2.typing.Size, type: int, allocator: GpuMat.Allocator=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param type: 
:type type: int
:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} __init__(self, rows: int, cols: int, type: int, s: cv2.typing.Scalar, allocator: GpuMat.Allocator=...)




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
:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} __init__(self, size: cv2.typing.Size, type: int, s: cv2.typing.Scalar, allocator: GpuMat.Allocator=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param type: 
:type type: int
:param s: 
:type s: cv2.typing.Scalar
:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} __init__(self, m: GpuMat)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param m: 
:type m: GpuMat
:rtype: None
````

````{py:method} __init__(self, m: GpuMat, rowRange: cv2.typing.Range, colRange: cv2.typing.Range)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param m: 
:type m: GpuMat
:param rowRange: 
:type rowRange: cv2.typing.Range
:param colRange: 
:type colRange: cv2.typing.Range
:rtype: None
````

````{py:method} __init__(self, m: GpuMat, roi: cv2.typing.Rect)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param m: 
:type m: GpuMat
:param roi: 
:type roi: cv2.typing.Rect
:rtype: None
````

````{py:method} __init__(self, arr: cv2.typing.MatLike, allocator: GpuMat.Allocator=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arr: 
:type arr: cv2.typing.MatLike
:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} __init__(self, arr: GpuMat, allocator: GpuMat.Allocator=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arr: 
:type arr: GpuMat
:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} __init__(self, arr: cv2.UMat, allocator: GpuMat.Allocator=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arr: 
:type arr: cv2.UMat
:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} create(rows, cols, type) -> None




create(size, type) -> None 


:param self: 
:type self: 
:param rows: 
:type rows: int
:param cols: 
:type cols: int
:param type: 
:type type: int
:rtype: None
````

````{py:method} create(rows, cols, type) -> None




create(size, type) -> None 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param type: 
:type type: int
:rtype: None
````

````{py:method} upload(arr) -> None

Performs data upload to GpuMat (Non-Blocking call)


This function copies data from host memory to device memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
upload(arr, stream) -> None 
This function copies data from host memory to device memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param arr: 
:type arr: cv2.typing.MatLike
:rtype: None
````

````{py:method} upload(arr) -> None

Performs data upload to GpuMat (Non-Blocking call)


This function copies data from host memory to device memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
upload(arr, stream) -> None 
This function copies data from host memory to device memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param arr: 
:type arr: GpuMat
:rtype: None
````

````{py:method} upload(arr) -> None

Performs data upload to GpuMat (Non-Blocking call)


This function copies data from host memory to device memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
upload(arr, stream) -> None 
This function copies data from host memory to device memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param arr: 
:type arr: cv2.UMat
:rtype: None
````

````{py:method} upload(arr) -> None

Performs data upload to GpuMat (Non-Blocking call)


This function copies data from host memory to device memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
upload(arr, stream) -> None 
This function copies data from host memory to device memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param arr: 
:type arr: cv2.typing.MatLike
:param stream: 
:type stream: Stream
:rtype: None
````

````{py:method} upload(arr) -> None

Performs data upload to GpuMat (Non-Blocking call)


This function copies data from host memory to device memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
upload(arr, stream) -> None 
This function copies data from host memory to device memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param arr: 
:type arr: GpuMat
:param stream: 
:type stream: Stream
:rtype: None
````

````{py:method} upload(arr) -> None

Performs data upload to GpuMat (Non-Blocking call)


This function copies data from host memory to device memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
upload(arr, stream) -> None 
This function copies data from host memory to device memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param arr: 
:type arr: cv2.UMat
:param stream: 
:type stream: Stream
:rtype: None
````

````{py:method} download([, dst]) -> dst

Performs data download from GpuMat (Non-Blocking call)


This function copies data from device memory to host memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
download(stream[, dst]) -> dst 
This function copies data from device memory to host memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} download([, dst]) -> dst

Performs data download from GpuMat (Non-Blocking call)


This function copies data from device memory to host memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
download(stream[, dst]) -> dst 
This function copies data from device memory to host memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param dst: 
:type dst: GpuMat | None
:rtype: GpuMat
````

````{py:method} download([, dst]) -> dst

Performs data download from GpuMat (Non-Blocking call)


This function copies data from device memory to host memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
download(stream[, dst]) -> dst 
This function copies data from device memory to host memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param dst: 
:type dst: cv2.UMat | None
:rtype: cv2.UMat
````

````{py:method} download([, dst]) -> dst

Performs data download from GpuMat (Non-Blocking call)


This function copies data from device memory to host memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
download(stream[, dst]) -> dst 
This function copies data from device memory to host memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param stream: 
:type stream: Stream
:param dst: 
:type dst: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````

````{py:method} download([, dst]) -> dst

Performs data download from GpuMat (Non-Blocking call)


This function copies data from device memory to host memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
download(stream[, dst]) -> dst 
This function copies data from device memory to host memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param stream: 
:type stream: Stream
:param dst: 
:type dst: GpuMat | None
:rtype: GpuMat
````

````{py:method} download([, dst]) -> dst

Performs data download from GpuMat (Non-Blocking call)


This function copies data from device memory to host memory. As being a blocking call, it is guaranteed that the copy operation is finished when this function returns. 
download(stream[, dst]) -> dst 
This function copies data from device memory to host memory. As being a non-blocking call, this function may return even if the copy operation is not finished. 
The copy operation may be overlapped with operations in other non-default streams if \p stream is not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option. 


:param self: 
:type self: 
:param stream: 
:type stream: Stream
:param dst: 
:type dst: cv2.UMat | None
:rtype: cv2.UMat
````

````{py:method} copyTo([, dst]) -> dst




copyTo(stream[, dst]) -> dst 
copyTo(mask[, dst]) -> dst 
copyTo(mask, stream[, dst]) -> dst 


:param self: 
:type self: 
:param dst: 
:type dst: GpuMat | None
:rtype: GpuMat
````

````{py:method} copyTo([, dst]) -> dst




copyTo(stream[, dst]) -> dst 
copyTo(mask[, dst]) -> dst 
copyTo(mask, stream[, dst]) -> dst 


:param self: 
:type self: 
:param stream: 
:type stream: Stream
:param dst: 
:type dst: GpuMat | None
:rtype: GpuMat
````

````{py:method} copyTo([, dst]) -> dst




copyTo(stream[, dst]) -> dst 
copyTo(mask[, dst]) -> dst 
copyTo(mask, stream[, dst]) -> dst 


:param self: 
:type self: 
:param mask: 
:type mask: GpuMat
:param dst: 
:type dst: GpuMat | None
:rtype: GpuMat
````

````{py:method} copyTo([, dst]) -> dst




copyTo(stream[, dst]) -> dst 
copyTo(mask[, dst]) -> dst 
copyTo(mask, stream[, dst]) -> dst 


:param self: 
:type self: 
:param mask: 
:type mask: GpuMat
:param stream: 
:type stream: Stream
:param dst: 
:type dst: GpuMat | None
:rtype: GpuMat
````

````{py:method} setTo(s) -> retval




setTo(s, stream) -> retval 
setTo(s, mask) -> retval 
setTo(s, mask, stream) -> retval 


:param self: 
:type self: 
:param s: 
:type s: cv2.typing.Scalar
:rtype: GpuMat
````

````{py:method} setTo(s) -> retval




setTo(s, stream) -> retval 
setTo(s, mask) -> retval 
setTo(s, mask, stream) -> retval 


:param self: 
:type self: 
:param s: 
:type s: cv2.typing.Scalar
:param stream: 
:type stream: Stream
:rtype: GpuMat
````

````{py:method} setTo(s) -> retval




setTo(s, stream) -> retval 
setTo(s, mask) -> retval 
setTo(s, mask, stream) -> retval 


:param self: 
:type self: 
:param s: 
:type s: cv2.typing.Scalar
:param mask: 
:type mask: cv2.typing.MatLike
:rtype: GpuMat
````

````{py:method} setTo(s) -> retval




setTo(s, stream) -> retval 
setTo(s, mask) -> retval 
setTo(s, mask, stream) -> retval 


:param self: 
:type self: 
:param s: 
:type s: cv2.typing.Scalar
:param mask: 
:type mask: GpuMat
:rtype: GpuMat
````

````{py:method} setTo(s) -> retval




setTo(s, stream) -> retval 
setTo(s, mask) -> retval 
setTo(s, mask, stream) -> retval 


:param self: 
:type self: 
:param s: 
:type s: cv2.typing.Scalar
:param mask: 
:type mask: cv2.UMat
:rtype: GpuMat
````

````{py:method} setTo(s) -> retval




setTo(s, stream) -> retval 
setTo(s, mask) -> retval 
setTo(s, mask, stream) -> retval 


:param self: 
:type self: 
:param s: 
:type s: cv2.typing.Scalar
:param mask: 
:type mask: cv2.typing.MatLike
:param stream: 
:type stream: Stream
:rtype: GpuMat
````

````{py:method} setTo(s) -> retval




setTo(s, stream) -> retval 
setTo(s, mask) -> retval 
setTo(s, mask, stream) -> retval 


:param self: 
:type self: 
:param s: 
:type s: cv2.typing.Scalar
:param mask: 
:type mask: GpuMat
:param stream: 
:type stream: Stream
:rtype: GpuMat
````

````{py:method} setTo(s) -> retval




setTo(s, stream) -> retval 
setTo(s, mask) -> retval 
setTo(s, mask, stream) -> retval 


:param self: 
:type self: 
:param s: 
:type s: cv2.typing.Scalar
:param mask: 
:type mask: cv2.UMat
:param stream: 
:type stream: Stream
:rtype: GpuMat
````

````{py:method} convertTo(rtype, stream[, dst]) -> dst




convertTo(rtype[, dst[, alpha[, beta]]]) -> dst 
convertTo(rtype, alpha, beta, stream[, dst]) -> dst 


:param self: 
:type self: 
:param rtype: 
:type rtype: int
:param stream: 
:type stream: Stream
:param dst: 
:type dst: GpuMat | None
:rtype: GpuMat
````

````{py:method} convertTo(rtype, stream[, dst]) -> dst




convertTo(rtype[, dst[, alpha[, beta]]]) -> dst 
convertTo(rtype, alpha, beta, stream[, dst]) -> dst 


:param self: 
:type self: 
:param rtype: 
:type rtype: int
:param dst: 
:type dst: GpuMat | None
:param alpha: 
:type alpha: float
:param beta: 
:type beta: float
:rtype: GpuMat
````

````{py:method} convertTo(rtype, stream[, dst]) -> dst




convertTo(rtype[, dst[, alpha[, beta]]]) -> dst 
convertTo(rtype, alpha, beta, stream[, dst]) -> dst 


:param self: 
:type self: 
:param rtype: 
:type rtype: int
:param alpha: 
:type alpha: float
:param beta: 
:type beta: float
:param stream: 
:type stream: Stream
:param dst: 
:type dst: GpuMat | None
:rtype: GpuMat
````

````{py:method} rowRange(startrow, endrow) -> retval




rowRange(r) -> retval 


:param self: 
:type self: 
:param startrow: 
:type startrow: int
:param endrow: 
:type endrow: int
:rtype: GpuMat
````

````{py:method} rowRange(startrow, endrow) -> retval




rowRange(r) -> retval 


:param self: 
:type self: 
:param r: 
:type r: cv2.typing.Range
:rtype: GpuMat
````

````{py:method} colRange(startcol, endcol) -> retval




colRange(r) -> retval 


:param self: 
:type self: 
:param startcol: 
:type startcol: int
:param endcol: 
:type endcol: int
:rtype: GpuMat
````

````{py:method} colRange(startcol, endcol) -> retval




colRange(r) -> retval 


:param self: 
:type self: 
:param r: 
:type r: cv2.typing.Range
:rtype: GpuMat
````

````{py:method} defaultAllocator() -> retval
:staticmethod:





:rtype: GpuMat.Allocator
````

````{py:method} setDefaultAllocator(allocator) -> None
:staticmethod:





:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} release() -> None





:param self: 
:type self: 
:rtype: None
````

````{py:method} swap(mat) -> None





:param self: 
:type self: 
:param mat: 
:type mat: GpuMat
:rtype: None
````

````{py:method} clone() -> retval





:param self: 
:type self: 
:rtype: GpuMat
````

````{py:method} assignTo(m[, type]) -> None





:param self: 
:type self: 
:param m: 
:type m: GpuMat
:param type: 
:type type: int
:rtype: None
````

````{py:method} row(y) -> retval





:param self: 
:type self: 
:param y: 
:type y: int
:rtype: GpuMat
````

````{py:method} col(x) -> retval





:param self: 
:type self: 
:param x: 
:type x: int
:rtype: GpuMat
````

````{py:method} reshape(cn[, rows]) -> retval





:param self: 
:type self: 
:param cn: 
:type cn: int
:param rows: 
:type rows: int
:rtype: GpuMat
````

````{py:method} locateROI(wholeSize, ofs) -> None





:param self: 
:type self: 
:param wholeSize: 
:type wholeSize: cv2.typing.Size
:param ofs: 
:type ofs: cv2.typing.Point
:rtype: None
````

````{py:method} adjustROI(dtop, dbottom, dleft, dright) -> retval





:param self: 
:type self: 
:param dtop: 
:type dtop: int
:param dbottom: 
:type dbottom: int
:param dleft: 
:type dleft: int
:param dright: 
:type dright: int
:rtype: GpuMat
````

````{py:method} isContinuous() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} elemSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} elemSize1() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} type() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} depth() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} channels() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} step1() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} size() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} empty() -> retval





:param self: 
:type self: 
:rtype: bool
````

````{py:method} cudaPtr() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.IntPointer
````

````{py:method} updateContinuityFlag() -> None





:param self: 
:type self: 
:rtype: None
````


`````


`````{py:class} GpuMatND





`````


`````{py:class} HostMem




````{py:method} step






:param self: 
:type self: 
:rtype: int
````

````{py:method} __init__(self, alloc_type: HostMem_AllocType=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param alloc_type: 
:type alloc_type: HostMem_AllocType
:rtype: None
````

````{py:method} __init__(self, rows: int, cols: int, type: int, alloc_type: HostMem_AllocType=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param rows: 
:type rows: int
:param cols: 
:type cols: int
:param type: 
:type type: int
:param alloc_type: 
:type alloc_type: HostMem_AllocType
:rtype: None
````

````{py:method} __init__(self, size: cv2.typing.Size, type: int, alloc_type: HostMem_AllocType=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param size: 
:type size: cv2.typing.Size
:param type: 
:type type: int
:param alloc_type: 
:type alloc_type: HostMem_AllocType
:rtype: None
````

````{py:method} __init__(self, arr: cv2.typing.MatLike, alloc_type: HostMem_AllocType=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arr: 
:type arr: cv2.typing.MatLike
:param alloc_type: 
:type alloc_type: HostMem_AllocType
:rtype: None
````

````{py:method} __init__(self, arr: GpuMat, alloc_type: HostMem_AllocType=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arr: 
:type arr: GpuMat
:param alloc_type: 
:type alloc_type: HostMem_AllocType
:rtype: None
````

````{py:method} __init__(self, arr: cv2.UMat, alloc_type: HostMem_AllocType=...)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param arr: 
:type arr: cv2.UMat
:param alloc_type: 
:type alloc_type: HostMem_AllocType
:rtype: None
````

````{py:method} swap(b) -> None





:param self: 
:type self: 
:param b: 
:type b: HostMem
:rtype: None
````

````{py:method} clone() -> retval





:param self: 
:type self: 
:rtype: HostMem
````

````{py:method} create(rows, cols, type) -> None





:param self: 
:type self: 
:param rows: 
:type rows: int
:param cols: 
:type cols: int
:param type: 
:type type: int
:rtype: None
````

````{py:method} reshape(cn[, rows]) -> retval





:param self: 
:type self: 
:param cn: 
:type cn: int
:param rows: 
:type rows: int
:rtype: HostMem
````

````{py:method} createMatHeader() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.MatLike
````

````{py:method} isContinuous() -> retval
Maps CPU memory to GPU address space and creates the cuda::GpuMat header without reference countingfor it. 


This can be done only if memory was allocated with the SHARED flag and if it is supported by the hardware. Laptops often share video and CPU memory, so address spaces can be mapped, which eliminates an extra copy. 


:param self: 
:type self: 
:rtype: bool
````

````{py:method} elemSize() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} elemSize1() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} type() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} depth() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} channels() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} step1() -> retval





:param self: 
:type self: 
:rtype: int
````

````{py:method} size() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.Size
````

````{py:method} empty() -> retval





:param self: 
:type self: 
:rtype: bool
````


`````


`````{py:class} Stream




````{py:method} Null() -> retval
:classmethod:
Adds a callback to be called on the host after all currently enqueued items in the stream havecompleted. 


```{note}
Callbacks must not make any CUDA API calls. Callbacks must not perform any synchronizationthat may depend on outstanding device work or other callbacks that are not mandated to run earlier. Callbacks without a mandated order (in independent streams) execute in undefined order and may be serialized. 
```


:param cls: 
:type cls: 
:rtype: Stream
````

````{py:method} __init__(self)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, allocator: GpuMat.Allocator)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param allocator: 
:type allocator: GpuMat.Allocator
:rtype: None
````

````{py:method} __init__(self, cudaFlags: int)




Initialize self.  See help(type(self)) for accurate signature. 


:param self: 
:type self: 
:param cudaFlags: 
:type cudaFlags: int
:rtype: None
````

````{py:method} queryIfComplete() -> retval
Returns true if the current stream queue is finished. Otherwise, it returns false.




:param self: 
:type self: 
:rtype: bool
````

````{py:method} waitForCompletion() -> None
Blocks the current CPU thread until all operations in the stream are complete.




:param self: 
:type self: 
:rtype: None
````

````{py:method} waitEvent(event) -> None
Makes a compute stream wait on an event.




:param self: 
:type self: 
:param event: 
:type event: Event
:rtype: None
````

````{py:method} cudaPtr() -> retval





:param self: 
:type self: 
:rtype: cv2.typing.IntPointer
````


`````


`````{py:class} TargetArchs




````{py:method} has(major, minor) -> retval
:staticmethod:
There is a set of methods to check whether the module contains intermediate (PTX) or binary CUDAcode for the given architecture(s): 




:param major: Major compute capability version.
:type major: int
:param minor: Minor compute capability version.
:type minor: int
:rtype: bool
````

````{py:method} hasPtx(major, minor) -> retval
:staticmethod:





:param major: 
:type major: int
:param minor: 
:type minor: int
:rtype: bool
````

````{py:method} hasBin(major, minor) -> retval
:staticmethod:





:param major: 
:type major: int
:param minor: 
:type minor: int
:rtype: bool
````

````{py:method} hasEqualOrLessPtx(major, minor) -> retval
:staticmethod:





:param major: 
:type major: int
:param minor: 
:type minor: int
:rtype: bool
````

````{py:method} hasEqualOrGreater(major, minor) -> retval
:staticmethod:





:param major: 
:type major: int
:param minor: 
:type minor: int
:rtype: bool
````

````{py:method} hasEqualOrGreaterPtx(major, minor) -> retval
:staticmethod:





:param major: 
:type major: int
:param minor: 
:type minor: int
:rtype: bool
````

````{py:method} hasEqualOrGreaterBin(major, minor) -> retval
:staticmethod:





:param major: 
:type major: int
:param minor: 
:type minor: int
:rtype: bool
````


`````



## Functions
````{py:function} Event_elapsedTime(start, end) -> retval






:rtype: object
````


````{py:function} GpuMat_defaultAllocator() -> retval






:rtype: object
````


````{py:function} GpuMat_setDefaultAllocator(allocator) -> None






:rtype: object
````


````{py:function} Stream_Null() -> retval

Adds a callback to be called on the host after all currently enqueued items in the stream havecompleted. 


```{note}
Callbacks must not make any CUDA API calls. Callbacks must not perform any synchronizationthat may depend on outstanding device work or other callbacks that are not mandated to run earlier. Callbacks without a mandated order (in independent streams) execute in undefined order and may be serialized. 
```


:rtype: object
````


````{py:function} TargetArchs_has(major, minor) -> retval

There is a set of methods to check whether the module contains intermediate (PTX) or binary CUDAcode for the given architecture(s): 




:param major: Major compute capability version.
:type major: 
:param minor: Minor compute capability version.
:type minor: 
:rtype: object
````


````{py:function} TargetArchs_hasBin(major, minor) -> retval






:rtype: object
````


````{py:function} TargetArchs_hasEqualOrGreater(major, minor) -> retval






:rtype: object
````


````{py:function} TargetArchs_hasEqualOrGreaterBin(major, minor) -> retval






:rtype: object
````


````{py:function} TargetArchs_hasEqualOrGreaterPtx(major, minor) -> retval






:rtype: object
````


````{py:function} TargetArchs_hasEqualOrLessPtx(major, minor) -> retval






:rtype: object
````


````{py:function} TargetArchs_hasPtx(major, minor) -> retval






:rtype: object
````


````{py:function} createContinuous(rows, cols, type[, arr]) -> arr

Creates a continuous matrix.


Matrix is called continuous if its elements are stored continuously, that is, without gaps at the end of each row. 


:param rows: Row count.
:type rows: int
:param cols: Column count.
:type cols: int
:param type: Type of the matrix.
:type type: int
:param arr: Destination matrix. This parameter changes only if it has a proper type and area ($\texttt{rows} \times \texttt{cols}$ ). 
:type arr: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} createGpuMatFromCudaMemory(rows, cols, type, cudaMemoryAddress[, step]) -> retval

Bindings overload to create a GpuMat from existing GPU memory.


createGpuMatFromCudaMemory(size, type, cudaMemoryAddress[, step]) -> retval @overload 
```{note}
Overload for generation of bindings only, not exported or intended for use internally from C++.
```
```{note}
Overload for generation of bindings only, not exported or intended for use internally from C++.
```


:param rows: Row count.
:type rows: int
:param cols: Column count.
:type cols: int
:param type: Type of the matrix.
:type type: int
:param cudaMemoryAddress: Address of the allocated GPU memory on the device. This does not allocate matrix data. Instead, it just initializes the matrix header that points to the specified \a cudaMemoryAddress, which means that no data is copied. This operation is very efficient and can be used to process external data using OpenCV functions. The external data is not automatically deallocated, so you should take care of it.
:type cudaMemoryAddress: int
:param step: Number of bytes each matrix row occupies. The value should include the padding bytes at the end of each row, if any. If the parameter is missing (set to Mat::AUTO_STEP ), no padding is assumed and the actual step is calculated as cols*elemSize(). See GpuMat::elemSize.
:type step: int
:param size: 2D array size: Size(cols, rows). In the Size() constructor, the number of rows and the number of columns go in the reverse order.
:type size: 
:rtype: GpuMat
````


````{py:function} ensureSizeIsEnough(rows, cols, type[, arr]) -> arr

Ensures that the size of a matrix is big enough and the matrix has a proper type.


The function does not reallocate memory if the matrix has proper attributes already. 


:param rows: Minimum desired number of rows.
:type rows: int
:param cols: Minimum desired number of columns.
:type cols: int
:param type: Desired matrix type.
:type type: int
:param arr: Destination matrix.
:type arr: cv2.typing.MatLike | None
:rtype: cv2.typing.MatLike
````


````{py:function} fastNlMeansDenoising(src, h[, dst[, search_window[, block_size[, stream]]]]) -> dst

Perform image denoising using Non-local Means Denoising algorithm<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising> with several computational optimizations. Noise expected to be a gaussian white noise 


This function expected to be applied to grayscale images. For colored images look at FastNonLocalMeansDenoising::labMethod. 
@sa fastNlMeansDenoising 


:param src: Input 8-bit 1-channel, 2-channel or 3-channel image.
:type src: GpuMat
:param dst: Output image with the same size and type as src .
:type dst: GpuMat | None
:param h: Parameter regulating filter strength. Big h value perfectly removes noise but alsoremoves image details, smaller h value preserves details but also preserves some noise 
:type h: float
:param search_window: Size in pixels of the window that is used to compute weighted average forgiven pixel. Should be odd. Affect performance linearly: greater search_window - greater denoising time. Recommended value 21 pixels 
:type search_window: int
:param block_size: Size in pixels of the template patch that is used to compute weights. Should beodd. Recommended value 7 pixels 
:type block_size: int
:param stream: Stream for the asynchronous invocations.
:type stream: Stream
:rtype: GpuMat
````


````{py:function} fastNlMeansDenoisingColored(src, h_luminance, photo_render[, dst[, search_window[, block_size[, stream]]]]) -> dst

Modification of fastNlMeansDenoising function for colored images


The function converts image to CIELAB colorspace and then separately denoise L and AB components with given h parameters using FastNonLocalMeansDenoising::simpleMethod function. 
@sa fastNlMeansDenoisingColored 


:param src: Input 8-bit 3-channel image.
:type src: GpuMat
:param dst: Output image with the same size and type as src .
:type dst: GpuMat | None
:param h_luminance: Parameter regulating filter strength. Big h value perfectly removes noise butalso removes image details, smaller h value preserves details but also preserves some noise 
:type h_luminance: float
:param photo_render: float The same as h but for color components. For most images value equals 10 will beenough to remove colored noise and do not distort colors 
:type photo_render: float
:param search_window: Size in pixels of the window that is used to compute weighted average forgiven pixel. Should be odd. Affect performance linearly: greater search_window - greater denoising time. Recommended value 21 pixels 
:type search_window: int
:param block_size: Size in pixels of the template patch that is used to compute weights. Should beodd. Recommended value 7 pixels 
:type block_size: int
:param stream: Stream for the asynchronous invocations.
:type stream: Stream
:rtype: GpuMat
````


````{py:function} getCudaEnabledDeviceCount() -> retval

Returns the number of installed CUDA-enabled devices.


Use this function before any other CUDA functions calls. If OpenCV is compiled without CUDA support, this function returns 0. If the CUDA driver is not installed, or is incompatible, this function returns -1. 


:rtype: int
````


````{py:function} getDevice() -> retval

Returns the current device index set by cuda::setDevice or initialized by default.




:rtype: int
````


````{py:function} nonLocalMeans(src, h[, dst[, search_window[, block_size[, borderMode[, stream]]]]]) -> dst

Performs pure non local means denoising without any simplification, and thus it is not fast.


@sa fastNlMeansDenoising 


:param src: Source image. Supports only CV_8UC1, CV_8UC2 and CV_8UC3.
:type src: GpuMat
:param dst: Destination image.
:type dst: GpuMat | None
:param h: Filter sigma regulating filter strength for color.
:type h: float
:param search_window: Size of search window.
:type search_window: int
:param block_size: Size of block used for computing weights.
:type block_size: int
:param borderMode: Border type. See borderInterpolate for details. BORDER_REFLECT101 ,BORDER_REPLICATE , BORDER_CONSTANT , BORDER_REFLECT and BORDER_WRAP are supported for now. 
:type borderMode: int
:param stream: Stream for the asynchronous version.
:type stream: Stream
:rtype: GpuMat
````


````{py:function} printCudaDeviceInfo(device) -> None






:param device: 
:type device: int
:rtype: None
````


````{py:function} printShortCudaDeviceInfo(device) -> None






:param device: 
:type device: int
:rtype: None
````


````{py:function} registerPageLocked(m) -> None

Page-locks the memory of matrix and maps it for the device(s).




:param m: Input matrix.
:type m: cv2.typing.MatLike
:rtype: None
````


````{py:function} resetDevice() -> None

Explicitly destroys and cleans up all resources associated with the current device in the currentprocess. 


Any subsequent API call to this device will reinitialize the device. 


:rtype: None
````


````{py:function} setBufferPoolConfig(deviceId, stackSize, stackCount) -> None






:param deviceId: 
:type deviceId: int
:param stackSize: 
:type stackSize: int
:param stackCount: 
:type stackCount: int
:rtype: None
````


````{py:function} setBufferPoolUsage(on) -> None






:param on: 
:type on: bool
:rtype: None
````


````{py:function} setDevice(device) -> None

Sets a device and initializes it for the current thread.


If the call of this function is omitted, a default device is initialized at the fist CUDA usage. 


:param device: System index of a CUDA device starting with 0.
:type device: int
:rtype: None
````


````{py:function} unregisterPageLocked(m) -> None

Unmaps the memory of matrix and makes it pageable again.




:param m: Input matrix.
:type m: cv2.typing.MatLike
:rtype: None
````


````{py:function} wrapStream(cudaStreamMemoryAddress) -> retval

Bindings overload to create a Stream object from the address stored in an existing CUDA Runtime API stream pointer (cudaStream_t).


```{note}
Overload for generation of bindings only, not exported or intended for use internally from C++.
```


:param cudaStreamMemoryAddress: Memory address stored in a CUDA Runtime API stream pointer (cudaStream_t). The created Stream object does not perform any allocation or deallocation and simply wraps existing raw CUDA Runtime API stream pointer.
:type cudaStreamMemoryAddress: int
:rtype: Stream
````



