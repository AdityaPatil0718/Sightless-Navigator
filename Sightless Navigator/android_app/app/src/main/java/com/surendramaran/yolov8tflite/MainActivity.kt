package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.Detector.DetectorListener
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), DetectorListener, TextToSpeech.OnInitListener {
    private var binding: ActivityMainBinding? = null
    private val isFrontCamera = false
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null
    private var cameraExecutor: ExecutorService? = null
    private var textToSpeech: TextToSpeech? = null
    private var frameCounter = 0

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding!!.root)

        textToSpeech = TextToSpeech(this, this)

        detector = Detector(
            baseContext, MODEL_PATH, LABELS_PATH, this,
            textToSpeech = textToSpeech!!
        )
        detector!!.setup()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Camera initialization failed.", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        if (cameraProvider == null) return

        val rotation = binding?.viewFinder?.display?.rotation ?: return
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetResolution(Size(640, 640))  // Updated for compatibility
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 640))  // Updated for compatibility
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer!!.setAnalyzer(cameraExecutor!!) { imageProxy: ImageProxy ->
            frameCounter++
            if (frameCounter % 3 != 0) { // Process every 3rd frame to reduce CPU load
                imageProxy.close()
                return@setAnalyzer
            }

            val bitmapBuffer = Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )

            imageProxy.use {
                bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
            }

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                if (isFrontCamera) {
                    postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
            )

            detector?.detect(rotatedBitmap)
            imageProxy.close() // Ensure proper closing
        }

        cameraProvider!!.unbindAll()
        try {
            camera = cameraProvider!!.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            preview?.setSurfaceProvider(binding?.viewFinder?.surfaceProvider)
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    private fun allPermissionsGranted(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = textToSpeech?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(TAG, "Language not supported for TextToSpeech")
            }
        } else {
            Log.e(TAG, "TextToSpeech initialization failed")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.clear()
        cameraExecutor?.shutdown()
        textToSpeech?.apply {
            stop()
            shutdown()
        }
    }

    override fun onEmptyDetect() {
        binding?.overlay?.invalidate()
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding?.inferenceTime?.text = String.format(Locale.getDefault(), "%d ms", inferenceTime)
            binding?.overlay?.setResults(boundingBoxes)
            binding?.overlay?.invalidate()

            boundingBoxes.takeIf { it.isNotEmpty() }?.joinToString(", ") { it.clsName }?.let { detectedText ->
                textToSpeech?.let {
                    if (!it.isSpeaking) {
                        it.speak(detectedText, TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                }
            }
        }
    }

    companion object {
        private const val TAG = "Camera"
    }
}
