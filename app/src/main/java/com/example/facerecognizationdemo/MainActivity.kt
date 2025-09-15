package com.example.facerecognizationdemo

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.graphics.*
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.android.gms.tflite.gpu.GpuDelegate
import com.google.common.util.concurrent.ListenableFuture
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    private lateinit var detector: FaceDetector
    private lateinit var previewView: PreviewView
    private lateinit var tfLite: Interpreter
    private lateinit var recoName: TextView
    private lateinit var camera_switch: Button
    var cam_face: Int = CameraSelector.LENS_FACING_BACK // Default Back Camera

    private var registered: HashMap<String, MutableList<FloatArray>> = HashMap()
    private var camFace = CameraSelector.LENS_FACING_BACK
    private var flipX = false
    private var start = true

    // 🔧 Lowered threshold
    private val COSINE_THRESHOLD = 0.5f
    private val inputSize = 112
    private val outputSize = 192

    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private var cameraProvider: ProcessCameraProvider? = null

    private val modelFile = "mobile_face_net.tflite"
    private val context: Context = this

    // static faces (multiple refs per person recommended)
    private val staticFacesToRegister = listOf(
        Pair(R.drawable.elonmusk, "Elon Musk"),
        Pair(R.drawable.elon, "Elon Musk"),
        Pair(R.drawable.narendra, "Narendra Modi")
    )

    // frame skipping counter
    private var frameCounter = 0
    // deprecated
    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        recoName = findViewById(R.id.textView)
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
//        camera_switch = findViewById(R.id.camera_switch)
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
        }

        // Load model with GPU delegate if available
        tfLite = loadModelWithGPU()
//        camera_switch.setOnClickListener {
//            Log.i("camera_switch", "onCreate: clicked")
//            if (cam_face == CameraSelector.LENS_FACING_BACK) {
//                cam_face = CameraSelector.LENS_FACING_FRONT
//                flipX = true
//            } else {
//                cam_face = CameraSelector.LENS_FACING_BACK
//                flipX = false
//            }
//            cameraProvider?.unbindAll()
//            cameraBind()
//        }

        // 🔧 Fast for live detection
        val opts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()
        detector = FaceDetection.getClient(opts)

        // Register static faces with more accuracy
        registerStaticFacesThenStartCamera()
    }

    /** Try GPU delegate, fallback to CPU */
    private fun loadModelWithGPU(): Interpreter {
        val options = Interpreter.Options()
        try {
            val gpuDelegate = GpuDelegate()
            options.addDelegate(gpuDelegate)
            Log.i("TFLITE", "Using GPU Delegate")
        } catch (e: Exception) {
            Log.w("TFLITE", "GPU not supported, fallback to CPU")
            options.setUseNNAPI(true)
        }
        return Interpreter(loadModelFile(this, modelFile), options)
    }

    private fun registerStaticFacesThenStartCamera() {
        val accurateOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()
        val accurateDetector = FaceDetection.getClient(accurateOpts)

        val remaining = AtomicInteger(staticFacesToRegister.size)
        for ((resId, label) in staticFacesToRegister) {
            registerStaticFaceAsync(resId, label, accurateDetector) {
                val left = remaining.decrementAndGet()
                if (left <= 0) {
                    Log.i("REGISTER", "All static faces registered: ${registered.size}")
                    cameraBind()
                }
            }
        }
    }

    private fun registerStaticFaceAsync(resourceId: Int, name: String, regDetector: FaceDetector, done: () -> Unit) {
        val bitmap = BitmapFactory.decodeResource(resources, resourceId)
        val image = InputImage.fromBitmap(bitmap, 0)
        regDetector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    val face = faces[0]
                    val cropped = cropFaceOnly(bitmap, face)
                    val aligned = alignFace(cropped, face) // 🔧 alignment
                    val scaled = getResizedBitmap(aligned, inputSize, inputSize)
                    val emb = runRecognitionForBitmap(scaled)
                    val list = registered.getOrPut(name) { mutableListOf() }
                    list.add(emb)
                    Log.i("REGISTER", "Registered $name")
                } else {
                    Log.e("REGISTER", "No face detected for $name")
                }
                done()
            }
            .addOnFailureListener {
                Log.e("REGISTER", "Error registering $name", it)
                done()
            }
    }

    private fun loadModelFile(activity: Activity, modelFile: String): MappedByteBuffer {
        val fd: AssetFileDescriptor = activity.assets.openFd(modelFile)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    private fun cameraBind() {
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider!!)
        }, ContextCompat.getMainExecutor(this))
    }

    @OptIn(ExperimentalGetImage::class)
    private fun bindPreview(cameraProvider: ProcessCameraProvider) {
        val preview = Preview.Builder().build()
        val cameraSelector = CameraSelector.Builder().requireLensFacing(camFace).build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        val executor: Executor = Executors.newSingleThreadExecutor()
        imageAnalysis.setAnalyzer(executor) { imageProxy ->
            frameCounter++
            if (frameCounter % 2 != 0) { // 🔧 process every 2nd frame (faster)
                imageProxy.close()
                return@setAnalyzer
            }

            val mediaImage = imageProxy.image
            if (mediaImage != null) {
                val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
                detector.process(inputImage)
                    .addOnSuccessListener { faces ->
                        if (faces.isNotEmpty()) {
                            val face = faces[0]
                            val frameBmp = toBitmap(mediaImage)
                            val rot = imageProxy.imageInfo.rotationDegrees
                            val frameBmp1 = rotateBitmap(frameBmp, rot, false, false)
                            val cropped = cropFaceOnly(frameBmp1, face)
                            val aligned = alignFace(cropped, face) // 🔧 align live face
                            val finalFace = if (flipX) rotateBitmap(aligned, 0, true, false) else aligned
                            val scaled = getResizedBitmap(finalFace, inputSize, inputSize)
                            if (start) recognizeAndCompare(scaled)
                        } else {
                            runOnUiThread { recoName.text = "No Face Detected!" }
                        }
                    }
                    .addOnCompleteListener { imageProxy.close() }
            } else imageProxy.close()
        }
        cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, imageAnalysis, preview)
    }

    /** Compare embeddings with registered faces */
    private fun recognizeAndCompare(bitmap: Bitmap) {
        val emb = runRecognitionForBitmap(bitmap) ?: return
        var bestName = "Unknown"
        var bestSim = -1f
        for ((name, list) in registered) {
            for (knownEmb in list) {
                val sim = cosineSimilarity(emb, knownEmb)
                if (sim > bestSim) {
                    bestSim = sim
                    bestName = name
                }
            }
        }
        runOnUiThread {
            recoName.text = if (bestSim >= COSINE_THRESHOLD) {
                "Matched: $bestName (${String.format("%.2f", bestSim)})"
            } else {
                "Not Matched (best: $bestName ${String.format("%.2f", bestSim)})"
            }
        }
    }

    /** Face-only crop (ignores clothes/neck) */
    private fun cropFaceOnly(bitmap: Bitmap, face: Face): Bitmap {
        val rect = face.boundingBox
        val top = rect.top
        val bottom = rect.top + (rect.height() * 0.85f).toInt()
        val newRect = Rect(rect.left, top, rect.right, bottom)
        val safeRect = expandRect(newRect, 1.1f, bitmap.width, bitmap.height)
        return Bitmap.createBitmap(
            bitmap,
            safeRect.left,
            safeRect.top,
            safeRect.width(),
            safeRect.height()
        )
    }

    /** 🔧 Align face using eyes */
    private fun alignFace(bitmap: Bitmap, face: Face): Bitmap {
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
        if (leftEye != null && rightEye != null) {
            val dx = (rightEye.x - leftEye.x).toDouble()
            val dy = (rightEye.y - leftEye.y).toDouble()
            val angle = Math.toDegrees(Math.atan2(dy, dx)).toFloat()
            return rotateBitmap(bitmap, -angle.toInt(), false, false)
        }
        return bitmap
    }

    private fun runRecognitionForBitmap(bitmap: Bitmap): FloatArray {
        val bm = if (bitmap.width != inputSize || bitmap.height != inputSize)
            getResizedBitmap(bitmap, inputSize, inputSize) else bitmap

        val imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        imgData.order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputSize * inputSize)
        bm.getPixels(intValues, 0, bm.width, 0, 0, bm.width, bm.height)

        imgData.rewind()
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]
                val r = (pixelValue shr 16 and 0xFF) / 255.0f
                val g = (pixelValue shr 8 and 0xFF) / 255.0f
                val b = (pixelValue and 0xFF) / 255.0f
                imgData.putFloat((r - 0.5f) * 2)
                imgData.putFloat((g - 0.5f) * 2)
                imgData.putFloat((b - 0.5f) * 2)
            }
        }

        val inputArray = arrayOf<Any>(imgData)
        val output = Array(1) { FloatArray(outputSize) }
        val outputMap = hashMapOf(0 to output)
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap as Map<Int, Any>)

        return l2Normalize(output[0])
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        val denom = sqrt(na) * sqrt(nb)
        return if (denom == 0f) 0f else dot / denom
    }

    private fun l2Normalize(src: FloatArray): FloatArray {
        val norm = sqrt(src.sumOf { (it * it).toDouble() }).toFloat()
        return if (norm == 0f) src else src.map { it / norm }.toFloatArray()
    }

    private fun expandRect(rect: Rect, scale: Float, imgW: Int, imgH: Int): Rect {
        val w = rect.width()
        val h = rect.height()
        val newW = (w * scale).toInt()
        val newH = (h * scale).toInt()
        val cx = rect.centerX()
        val cy = rect.centerY()
        val left = max(0, cx - newW / 2)
        val top = max(0, cy - newH / 2)
        val right = min(imgW - 1, cx + newW / 2)
        val bottom = min(imgH - 1, cy + newH / 2)
        return Rect(left, top, right, bottom)
    }

    private fun getResizedBitmap(bm: Bitmap, newW: Int, newH: Int): Bitmap {
        val matrix = Matrix()
        matrix.postScale(newW.toFloat() / bm.width, newH.toFloat() / bm.height)
        return Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, matrix, false)
    }

    private fun rotateBitmap(bm: Bitmap, rot: Int, flipX: Boolean, flipY: Boolean): Bitmap {
        val matrix = Matrix()
        if (rot != 0) matrix.postRotate(rot.toFloat())
        matrix.postScale(if (flipX) -1f else 1f, if (flipY) -1f else 1f)
        return Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, matrix, true)
    }

    private fun toBitmap(image: Image): Bitmap {
        val nv21 = YUV_420_888toNV21(image)
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, yuv.width, yuv.height), 90, out)
        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }

    private fun YUV_420_888toNV21(image: Image): ByteArray {
        val w = image.width; val h = image.height
        val ySize = w * h; val uvSize = w * h / 4
        val nv21 = ByteArray(ySize + uvSize * 2)
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        yBuffer.get(nv21, 0, ySize)
        val rowStride = image.planes[2].rowStride
        val pixelStride = image.planes[2].pixelStride
        var pos = ySize
        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                val vuPos = col * pixelStride + row * rowStride
                nv21[pos++] = vBuffer.get(vuPos)
                nv21[pos++] = uBuffer.get(vuPos)
            }
        }
        return nv21
    }
}
