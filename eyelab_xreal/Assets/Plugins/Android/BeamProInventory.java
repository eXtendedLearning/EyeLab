package it.univpm.eyelab.probes;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.hardware.display.DisplayManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Range;
import android.util.Size;
import android.util.SizeF;
import android.view.Display;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * EyeLab T6.0 (gate BS): what a third-party app on the Beam Pro can see of the host's own sensors.
 * Read-only: camera characteristics need no permission and nothing is opened. Called from
 * BeamProInventoryProbe.cs through AndroidJavaClass. Every method returns JSON text or a CSV path,
 * so the C# side never handles Java arrays.
 */
public final class BeamProInventory {
    private BeamProInventory() {}

    // ---------------------------------------------------------------- device and packages
    public static String device(Context ctx) {
        JSONObject o = new JSONObject();
        try {
            o.put("manufacturer", Build.MANUFACTURER);
            o.put("model", Build.MODEL);
            o.put("device", Build.DEVICE);
            o.put("product", Build.PRODUCT);
            o.put("hardware", Build.HARDWARE);
            o.put("soc_model", Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : "");
            o.put("android_release", Build.VERSION.RELEASE);
            o.put("sdk_int", Build.VERSION.SDK_INT);
            o.put("security_patch", Build.VERSION.SECURITY_PATCH);
            o.put("build_id", Build.ID);
            o.put("build_display", Build.DISPLAY);
            o.put("build_fingerprint", Build.FINGERPRINT);
            o.put("build_incremental", Build.VERSION.INCREMENTAL);
            JSONArray pk = new JSONArray();
            PackageManager pm = ctx.getPackageManager();
            for (PackageInfo p : pm.getInstalledPackages(0)) {
                String n = p.packageName.toLowerCase();
                if (n.contains("xreal") || n.contains("nreal") || n.contains("nebula") || n.contains("myglasses")
                        || n.contains("controlglasses") || n.equals(ctx.getPackageName())) {
                    JSONObject e = new JSONObject();
                    e.put("package", p.packageName);
                    e.put("version_name", p.versionName);
                    e.put("version_code", Build.VERSION.SDK_INT >= 28 ? p.getLongVersionCode() : p.versionCode);
                    pk.put(e);
                }
            }
            o.put("packages", pk);
        } catch (Exception e) {
            put(o, "error", e.toString());
        }
        return o.toString();
    }

    // ---------------------------------------------------------------- cameras (Camera2)
    public static String cameras(Context ctx) {
        JSONObject o = new JSONObject();
        try {
            CameraManager cm = (CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
            String[] ids = cm.getCameraIdList();
            o.put("camera_id_list", new JSONArray(ids));
            JSONArray cams = new JSONArray();
            java.util.HashSet<String> seen = new java.util.HashSet<>();
            for (String id : ids) {
                cams.put(camera(cm, id, true));
                seen.add(id);
            }
            // Physical members of logical cameras are not always in the id list: describe them too.
            if (Build.VERSION.SDK_INT >= 28) {
                for (String id : ids) {
                    for (String ph : cm.getCameraCharacteristics(id).getPhysicalCameraIds()) {
                        if (seen.add(ph)) cams.put(camera(cm, ph, false));
                    }
                }
            }
            o.put("cameras", cams);
            if (Build.VERSION.SDK_INT >= 30) {
                JSONArray conc = new JSONArray();
                for (Set<String> s : cm.getConcurrentCameraIds()) conc.put(new JSONArray(s));
                o.put("concurrent_camera_ids", conc);
            }
        } catch (Exception e) {
            put(o, "error", e.toString());
        }
        return o.toString();
    }

    private static JSONObject camera(CameraManager cm, String id, boolean listed) throws Exception {
        CameraCharacteristics c = cm.getCameraCharacteristics(id);
        JSONObject o = new JSONObject();
        o.put("id", id);
        o.put("in_id_list", listed);
        Integer facing = c.get(CameraCharacteristics.LENS_FACING);
        o.put("facing", facing == null ? "?" : facing == CameraCharacteristics.LENS_FACING_BACK ? "back"
                : facing == CameraCharacteristics.LENS_FACING_FRONT ? "front" : "external");
        Integer level = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        o.put("hardware_level", level == null ? JSONObject.NULL : level);
        int[] caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
        o.put("capabilities", ints(caps));
        if (Build.VERSION.SDK_INT >= 28) {
            o.put("physical_ids", new JSONArray(c.getPhysicalCameraIds()));
            Integer sync = c.get(CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE);
            o.put("sensor_sync_type", sync == null ? JSONObject.NULL
                    : sync == CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_CALIBRATED ? "calibrated" : "approximate");
            o.put("lens_distortion", floats(c.get(CameraCharacteristics.LENS_DISTORTION)));
            Integer ref = c.get(CameraCharacteristics.LENS_POSE_REFERENCE);
            o.put("lens_pose_reference", ref == null ? JSONObject.NULL : ref);
        }
        Integer ts = c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
        o.put("timestamp_source", ts == null ? "?" : ts == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME ? "realtime" : "unknown");
        o.put("lens_intrinsic_calibration", floats(c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)));
        o.put("lens_pose_translation", floats(c.get(CameraCharacteristics.LENS_POSE_TRANSLATION)));
        o.put("lens_pose_rotation", floats(c.get(CameraCharacteristics.LENS_POSE_ROTATION)));
        o.put("focal_lengths_mm", floats(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)));
        o.put("apertures", floats(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)));
        Float minFocus = c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE);
        o.put("min_focus_diopter", minFocus == null ? JSONObject.NULL : minFocus);
        Integer focusCal = c.get(CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION);
        o.put("focus_distance_calibration", focusCal == null ? JSONObject.NULL : focusCal);
        Integer orient = c.get(CameraCharacteristics.SENSOR_ORIENTATION);
        o.put("sensor_orientation_deg", orient == null ? JSONObject.NULL : orient);
        SizeF phys = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
        o.put("sensor_physical_size_mm", phys == null ? JSONObject.NULL : phys.getWidth() + "x" + phys.getHeight());
        Size px = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE);
        o.put("pixel_array", px == null ? JSONObject.NULL : px.toString());
        Rect act = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        o.put("active_array", act == null ? JSONObject.NULL : act.flattenToString());
        if (Build.VERSION.SDK_INT >= 28) {
            Rect pre = c.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE);
            o.put("pre_correction_active_array", pre == null ? JSONObject.NULL : pre.flattenToString());
        }
        Range<Integer>[] fps = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        JSONArray fr = new JSONArray();
        if (fps != null) for (Range<Integer> r : fps) fr.put(r.toString());
        o.put("ae_fps_ranges", fr);
        StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map != null) {
            JSONArray yuv = new JSONArray();
            Size[] sizes = map.getOutputSizes(ImageFormat.YUV_420_888);
            if (sizes != null) for (Size s : sizes) {
                long d = map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, s);
                yuv.put(s.toString() + "@" + (d > 0 ? String.format(java.util.Locale.ROOT, "%.1f", 1e9 / d) : "?"));
            }
            o.put("yuv_sizes_at_max_fps", yuv);
            JSONArray hs = new JSONArray();
            Size[] hsz = map.getHighSpeedVideoSizes();
            if (hsz != null) for (Size s : hsz) hs.put(s.toString());
            o.put("high_speed_video_sizes", hs);
        }
        return o;
    }

    // ---------------------------------------------------------------- sensors and displays
    public static String sensors(Context ctx) {
        JSONObject o = new JSONObject();
        try {
            SensorManager sm = (SensorManager) ctx.getSystemService(Context.SENSOR_SERVICE);
            JSONArray arr = new JSONArray();
            for (Sensor s : sm.getSensorList(Sensor.TYPE_ALL)) {
                JSONObject e = new JSONObject();
                e.put("name", s.getName());
                e.put("vendor", s.getVendor());
                e.put("type", s.getType());
                e.put("string_type", s.getStringType());
                e.put("version", s.getVersion());
                e.put("resolution", (double) s.getResolution());
                e.put("max_range", (double) s.getMaximumRange());
                e.put("min_delay_us", s.getMinDelay());
                e.put("max_delay_us", s.getMaxDelay());
                e.put("fifo_reserved", s.getFifoReservedEventCount());
                e.put("fifo_max", s.getFifoMaxEventCount());
                e.put("wake_up", s.isWakeUpSensor());
                e.put("dynamic", s.isDynamicSensor());
                e.put("direct_report_max_level", s.getHighestDirectReportRateLevel());
                e.put("power_ma", (double) s.getPower());
                arr.put(e);
            }
            o.put("sensors", arr);
        } catch (Exception e) {
            put(o, "error", e.toString());
        }
        return o.toString();
    }

    public static String displays(Context ctx) {
        JSONObject o = new JSONObject();
        try {
            DisplayManager dm = (DisplayManager) ctx.getSystemService(Context.DISPLAY_SERVICE);
            JSONArray arr = new JSONArray();
            for (Display d : dm.getDisplays()) {
                JSONObject e = new JSONObject();
                e.put("id", d.getDisplayId());
                e.put("name", d.getName());
                e.put("flags", d.getFlags());
                e.put("refresh_hz", (double) d.getRefreshRate());
                Display.Mode m = d.getMode();
                e.put("mode", m.getPhysicalWidth() + "x" + m.getPhysicalHeight() + "@" + m.getRefreshRate());
                JSONArray modes = new JSONArray();
                for (Display.Mode sm : d.getSupportedModes())
                    modes.put(sm.getPhysicalWidth() + "x" + sm.getPhysicalHeight() + "@" + sm.getRefreshRate());
                e.put("supported_modes", modes);
                e.put("app_vsync_offset_ns", d.getAppVsyncOffsetNanos());
                e.put("presentation_deadline_ns", d.getPresentationDeadlineNanos());
                arr.put(e);
            }
            o.put("displays", arr);
        } catch (Exception e) {
            put(o, "error", e.toString());
        }
        return o.toString();
    }

    // ---------------------------------------------------------------- IMU recorder
    private static HandlerThread sThread;
    private static SensorManager sSm;
    private static BufferedWriter sOut;
    private static final Map<Integer, long[]> sCounts = new HashMap<>();
    private static long sStartBoot;

    private static final SensorEventListener LISTENER = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent ev) {
            long boot = SystemClock.elapsedRealtimeNanos();
            long mono = System.nanoTime();
            BufferedWriter w = sOut;
            if (w == null) return;
            try {
                float[] v = ev.values;
                StringBuilder sb = new StringBuilder(96);
                sb.append(ev.sensor.getType()).append(',').append(ev.timestamp).append(',').append(boot).append(',').append(mono);
                for (int i = 0; i < 3; i++) sb.append(',').append(i < v.length ? v[i] : Float.NaN);
                sb.append(',').append(ev.accuracy).append('\n');
                w.write(sb.toString());
                synchronized (sCounts) {
                    long[] c = sCounts.get(ev.sensor.getType());
                    if (c == null) { c = new long[3]; c[1] = ev.timestamp; sCounts.put(ev.sensor.getType(), c); }
                    c[0]++;
                    c[2] = ev.timestamp;
                }
            } catch (Exception ignored) { }
        }

        @Override
        public void onAccuracyChanged(Sensor s, int a) { }
    };

    /** Records accelerometer, gyroscope and their uncalibrated variants at rateUs (0 = fastest) into csvPath. */
    public static String imuStart(Context ctx, String csvPath, int rateUs) {
        JSONObject o = new JSONObject();
        try {
            imuStop();
            sOut = new BufferedWriter(new FileWriter(csvPath), 1 << 16);
            sOut.write("type,event_ns,boot_ns,mono_ns,v0,v1,v2,accuracy\n");
            sCounts.clear();
            sThread = new HandlerThread("eyelab-imu");
            sThread.start();
            Handler h = new Handler(sThread.getLooper());
            sSm = (SensorManager) ctx.getSystemService(Context.SENSOR_SERVICE);
            int[] types = { Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE,
                    Sensor.TYPE_ACCELEROMETER_UNCALIBRATED, Sensor.TYPE_GYROSCOPE_UNCALIBRATED };
            JSONArray reg = new JSONArray();
            for (int t : types) {
                Sensor s = sSm.getDefaultSensor(t);
                if (s == null) continue;
                boolean ok = sSm.registerListener(LISTENER, s, rateUs <= 0 ? SensorManager.SENSOR_DELAY_FASTEST : rateUs, h);
                reg.put(t + ":" + s.getName() + ":" + ok);
            }
            sStartBoot = SystemClock.elapsedRealtimeNanos();
            o.put("registered", reg);
            o.put("start_boot_ns", sStartBoot);
        } catch (Exception e) {
            put(o, "error", e.toString());
        }
        return o.toString();
    }

    public static String imuStop() {
        JSONObject o = new JSONObject();
        try {
            if (sSm != null) sSm.unregisterListener(LISTENER);
            if (sThread != null) sThread.quitSafely();
            BufferedWriter w = sOut;
            sOut = null;
            if (w != null) { w.flush(); w.close(); }
            synchronized (sCounts) {
                for (Map.Entry<Integer, long[]> e : sCounts.entrySet()) {
                    long[] c = e.getValue();
                    double hz = c[0] > 1 && c[2] > c[1] ? (c[0] - 1) * 1e9 / (c[2] - c[1]) : 0;
                    o.put(String.valueOf(e.getKey()), c[0] + " events, " + String.format(java.util.Locale.ROOT, "%.1f", hz) + " Hz");
                }
            }
        } catch (Exception e) {
            put(o, "error", e.toString());
        }
        sSm = null;
        sThread = null;
        return o.toString();
    }

    // ---------------------------------------------------------------- helpers
    public static String clocks() {
        return SystemClock.elapsedRealtimeNanos() + "," + System.nanoTime() + "," + System.currentTimeMillis();
    }

    private static JSONArray ints(int[] a) {
        JSONArray r = new JSONArray();
        if (a != null) for (int v : a) r.put(v);
        return r;
    }

    private static Object floats(float[] a) {
        if (a == null) return JSONObject.NULL;
        JSONArray r = new JSONArray();
        for (float v : a) { try { r.put((double) v); } catch (Exception ignored) { } }
        return r;
    }

    private static void put(JSONObject o, String k, Object v) {
        try { o.put(k, v); } catch (Exception ignored) { }
    }
}
