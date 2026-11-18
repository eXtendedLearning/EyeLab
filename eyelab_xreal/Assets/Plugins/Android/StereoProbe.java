package it.univpm.eyelab.probes;

import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Size;

import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;

/**
 * T6.2 (gate BS-2): can an app open the Beam Pro's hidden logical rear camera and get both physical streams?
 * The HAL has 4 cameras, the app list 2; id "3" is a logical camera of physical "0" and "2" (sync APPROXIMATE),
 * and it answered getCameraCharacteristics by id (inventory 2026-11-17). This opens it with one small YUV stream
 * per physical camera and logs, per capture, the logical and both physical SENSOR_TIMESTAMPs (stereo.csv), plus
 * every image arrival (images.csv). A few Y planes per side are saved as PGM. Every refusal is recorded, not thrown.
 */
public final class StereoProbe {
    private static final Object LOCK = new Object();
    private static HandlerThread thread;
    private static Handler handler;
    private static CameraDevice device;
    private static CameraCaptureSession session;
    private static final List<ImageReader> readers = new ArrayList<>();
    private static BufferedWriter captures, images;
    private static final JSONObject state = new JSONObject();
    private static long nCaptures, nImages, nSaved;
    private static File dir;

    private StereoProbe() { }

    /** Starts the probe; returns a JSON status at once (the camera opens asynchronously). */
    public static String start(Context ctx, String runDir, String logicalId) {
        synchronized (LOCK) {
            try {
                dir = new File(runDir);
                new File(dir, "stereo").mkdirs();
                captures = new BufferedWriter(new FileWriter(new File(dir, "stereo.csv")));
                captures.write("frame,logical_ts,phys_a,ts_a,phys_b,ts_b,boot_ns_at_result\n");
                images = new BufferedWriter(new FileWriter(new File(dir, "stereo_images.csv")));
                images.write("physical_id,image_ts,boot_ns_at_arrival,w,h\n");
                thread = new HandlerThread("eyelab-stereo");
                thread.start();
                handler = new Handler(thread.getLooper());
                put("logical_id", logicalId);
                if (Build.VERSION.SDK_INT < 28) { put("result", "needs API 28"); return state.toString(); }

                CameraManager cm = (CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
                CameraCharacteristics lc = cm.getCameraCharacteristics(logicalId);
                List<String> phys = new ArrayList<>(lc.getPhysicalCameraIds());
                put("physical_ids", phys.toString());
                if (phys.size() < 2) { put("result", "not a logical camera"); return state.toString(); }
                List<OutputConfiguration> outs = new ArrayList<>();
                for (int i = 0; i < 2; i++) {
                    String pid = phys.get(i);
                    Size size = smallYuv(cm.getCameraCharacteristics(pid), lc);
                    put("size_" + pid, size == null ? "none" : size.toString());
                    if (size == null) { put("result", "no YUV size for physical " + pid); return state.toString(); }
                    ImageReader r = ImageReader.newInstance(size.getWidth(), size.getHeight(), ImageFormat.YUV_420_888, 4);
                    r.setOnImageAvailableListener(rd -> onImage(rd, pid), handler);
                    readers.add(r);
                    OutputConfiguration oc = new OutputConfiguration(r.getSurface());
                    oc.setPhysicalCameraId(pid);
                    outs.add(oc);
                }
                final String pa = phys.get(0), pb = phys.get(1);
                cm.openCamera(logicalId, new CameraDevice.StateCallback() {
                    @Override public void onOpened(CameraDevice cd) {
                        synchronized (LOCK) { device = cd; put("opened_boot_ns", SystemClock.elapsedRealtimeNanos()); }
                        try {
                            Executor ex = handler::post;
                            cd.createCaptureSession(new SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outs, ex,
                                new CameraCaptureSession.StateCallback() {
                                    @Override public void onConfigured(CameraCaptureSession s) {
                                        synchronized (LOCK) { session = s; }
                                        try {
                                            CaptureRequest.Builder b = cd.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                                            for (ImageReader r : readers) b.addTarget(r.getSurface());
                                            s.setRepeatingRequest(b.build(), new CameraCaptureSession.CaptureCallback() {
                                                @Override public void onCaptureCompleted(CameraCaptureSession ss, CaptureRequest rq, TotalCaptureResult res) {
                                                    onResult(res, pa, pb);
                                                }
                                            }, handler);
                                            put("result", "streaming");
                                        } catch (Exception e) { put("result", "request failed: " + e); }
                                    }
                                    @Override public void onConfigureFailed(CameraCaptureSession s) { put("result", "session configure failed"); }
                                }));
                        } catch (Exception e) { put("result", "session failed: " + e); }
                    }
                    @Override public void onDisconnected(CameraDevice cd) { put("disconnected", true); cd.close(); }
                    @Override public void onError(CameraDevice cd, int error) { put("result", "open error " + error); cd.close(); }
                }, handler);
                put("result", "opening");
            } catch (Exception e) {
                put("result", "refused: " + e);
            }
            return state.toString();
        }
    }

    /** The smallest YUV size at least 640 wide that both the physical and the logical camera list. */
    private static Size smallYuv(CameraCharacteristics pc, CameraCharacteristics lc) {
        StreamConfigurationMap m = pc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (m == null) m = lc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (m == null) return null;
        Size best = null;
        for (Size s : m.getOutputSizes(ImageFormat.YUV_420_888)) {
            if (s.getWidth() < 640) continue;
            if (best == null || (long) s.getWidth() * s.getHeight() < (long) best.getWidth() * best.getHeight()) best = s;
        }
        return best;
    }

    private static void onResult(TotalCaptureResult res, String pa, String pb) {
        long boot = SystemClock.elapsedRealtimeNanos();
        Long lts = res.get(CaptureResult.SENSOR_TIMESTAMP);
        Map<String, CaptureResult> ph = res.getPhysicalCameraResults();
        Long ta = ph.containsKey(pa) ? ph.get(pa).get(CaptureResult.SENSOR_TIMESTAMP) : null;
        Long tb = ph.containsKey(pb) ? ph.get(pb).get(CaptureResult.SENSOR_TIMESTAMP) : null;
        synchronized (LOCK) {
            try {
                if (captures != null) {
                    captures.write(res.getFrameNumber() + "," + lts + "," + pa + "," + ta + "," + pb + "," + tb + "," + boot + "\n");
                    nCaptures++;
                }
            } catch (Exception ignored) { }
        }
    }

    private static void onImage(ImageReader rd, String pid) {
        long boot = SystemClock.elapsedRealtimeNanos();
        Image img = rd.acquireLatestImage();
        if (img == null) return;
        try {
            synchronized (LOCK) {
                if (images == null) return;
                images.write(pid + "," + img.getTimestamp() + "," + boot + "," + img.getWidth() + "," + img.getHeight() + "\n");
                nImages++;
                if (nImages % 150 == 1 && nSaved < 12) {   // about every 5 s per side
                    savePgm(img, new File(dir, "stereo/y_" + pid + "_" + img.getTimestamp() + ".pgm"));
                    nSaved++;
                }
            }
        } catch (Exception ignored) {
        } finally {
            img.close();
        }
    }

    private static void savePgm(Image img, File f) throws Exception {
        Image.Plane y = img.getPlanes()[0];
        ByteBuffer buf = y.getBuffer();
        int w = img.getWidth(), h = img.getHeight(), stride = y.getRowStride();
        byte[] row = new byte[w];
        try (FileOutputStream out = new FileOutputStream(f)) {
            out.write(("P5\n" + w + " " + h + "\n255\n").getBytes("US-ASCII"));
            for (int r = 0; r < h; r++) {
                buf.position(r * stride);
                buf.get(row, 0, w);
                out.write(row);
            }
        }
    }

    /** Stops everything; returns the JSON status with counts. Safe to call twice. */
    public static String stop() {
        synchronized (LOCK) {
            try { if (session != null) session.close(); } catch (Exception ignored) { }
            try { if (device != null) device.close(); } catch (Exception ignored) { }
            for (ImageReader r : readers) { try { r.close(); } catch (Exception ignored) { } }
            readers.clear();
            session = null;
            device = null;
            try { if (captures != null) captures.close(); } catch (Exception ignored) { }
            try { if (images != null) images.close(); } catch (Exception ignored) { }
            captures = null;
            images = null;
            if (thread != null) thread.quitSafely();
            thread = null;
            put("captures", nCaptures);
            put("images", nImages);
            put("saved", nSaved);
            String s = state.toString();
            nCaptures = nImages = nSaved = 0;
            return s;
        }
    }

    private static void put(String k, Object v) {
        try { synchronized (state) { state.put(k, v); } } catch (Exception ignored) { }
    }
}
