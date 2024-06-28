# `cv2.barcode`
```{py:module} cv2.barcode
None
```
## Attributes
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
`````{py:class} BarcodeDetector




````{py:method} __init__(self)






:param self: 
:type self: 
:rtype: None
````

````{py:method} __init__(self, prototxt_path: str, model_path: str)






:param self: 
:type self: 
:param prototxt_path: 
:type prototxt_path: str
:param model_path: 
:type model_path: str
:rtype: None
````

````{py:method} decodeWithType(img, points) -> retval, decoded_info, decoded_type

Decodes barcode in image once it's found by the detect() method.     *
     * @param img grayscale or color (BGR) image containing bar code.
     * @param points vector of rotated rectangle vertices found by detect() method (or some other algorithm).
     * For N detected barcodes, the dimensions of this array should be [N][4].
     * Order of four points in vector<Point2f> is bottomLeft, topLeft, topRight, bottomRight.
     * @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector strings, specifies the type of these barcodes
     * @return true if at least one valid barcode have been found





:param self: 
:type self: 
:param img: 
:type img: cv2.typing.MatLike
:param points: 
:type points: cv2.typing.MatLike
:rtype: tuple[bool, _typing.Sequence[str], _typing.Sequence[str]]
````

````{py:method} decodeWithType(img, points) -> retval, decoded_info, decoded_type

Decodes barcode in image once it's found by the detect() method.     *
     * @param img grayscale or color (BGR) image containing bar code.
     * @param points vector of rotated rectangle vertices found by detect() method (or some other algorithm).
     * For N detected barcodes, the dimensions of this array should be [N][4].
     * Order of four points in vector<Point2f> is bottomLeft, topLeft, topRight, bottomRight.
     * @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector strings, specifies the type of these barcodes
     * @return true if at least one valid barcode have been found





:param self: 
:type self: 
:param img: 
:type img: cv2.UMat
:param points: 
:type points: cv2.UMat
:rtype: tuple[bool, _typing.Sequence[str], _typing.Sequence[str]]
````

````{py:method} detectAndDecodeWithType(img[, points]) -> retval, decoded_info, decoded_type, points

Both detects and decodes barcode


     * @param img grayscale or color (BGR) image containing barcode.
     * @param decoded_info UTF8-encoded output vector of string(s) or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector of strings, specifies the type of these barcodes
     * @param points optional output vector of vertices of the found  barcode rectangle. Will be empty if not found.
     * @return true if at least one valid barcode have been found



:param self: 
:type self: 
:param img: 
:type img: cv2.typing.MatLike
:param points: 
:type points: cv2.typing.MatLike | None
:rtype: tuple[bool, _typing.Sequence[str], _typing.Sequence[str], cv2.typing.MatLike]
````

````{py:method} detectAndDecodeWithType(img[, points]) -> retval, decoded_info, decoded_type, points

Both detects and decodes barcode


     * @param img grayscale or color (BGR) image containing barcode.
     * @param decoded_info UTF8-encoded output vector of string(s) or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector of strings, specifies the type of these barcodes
     * @param points optional output vector of vertices of the found  barcode rectangle. Will be empty if not found.
     * @return true if at least one valid barcode have been found



:param self: 
:type self: 
:param img: 
:type img: cv2.UMat
:param points: 
:type points: cv2.UMat | None
:rtype: tuple[bool, _typing.Sequence[str], _typing.Sequence[str], cv2.UMat]
````


`````



## Functions

