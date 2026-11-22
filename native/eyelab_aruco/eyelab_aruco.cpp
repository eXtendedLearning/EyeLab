// EyeLab ArUco plugin (B3a, ADR-0004 option A2): marker detection and board pose on the device.
//
// It mirrors the Python pipeline (python/pose_estimator.py) step by step, so the parity test can compare the two
// on the same frames:
//   1. CLAHE, clip limit from the tuning, 8x8 tiles            (preprocess_frame)
//   2. ArucoDetector.detectMarkers with the same parameters    (make_detector_parameters)
//   3. Board.matchImagePoints, then for planar point sets       (_solve_board_pose)
//      solvePnPGeneric IPPE (best solution, ambiguity ratio and separation), otherwise SOLVEPNP_ITERATIVE,
//      then solvePnPRefineLM; reprojection RMS                  (_reprojection_rms_px)
// Not mirrored yet: board-guided refineDetectedMarkers, the pose lock and the Kalman filter (they come after B3a).
//
// T6.7 (ABI 3) adds, in ela_process_roi:
//   - a region of interest: CLAHE and detection on a crop of the frame, corners returned in full-frame pixels;
//   - the box of the whole board projected with the pose (where to look next frame);
//   - a prior rotation: of the two IPPE solutions, the one nearest the prior is kept (pose_estimator's `prior`);
// and ela_set_nice for the calling thread's scheduling priority.
//
// Plain C ABI for P/Invoke from Unity (EyeLab.Aruco.ArucoNative). Units: metres for object points, pixels for
// image points. One handle per thread: nothing here is shared between handles except OpenCV's thread pool.

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define ELA_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

// 2 (0.4.008): ambiguity_ratio fixed (abi 1 read IPPE's float errors as double, so its ratio was garbage).
// 3 (0.4.009): ela_process_roi, ela_set_nice; ElaResult gains the ROI, the projected board box and the solution.
// 4 (0.4.010): the region is grown to multiples of 8 (CLAHE's tiles) and the perimeter rates are scaled by
//   frame / region, so a region keeps the frame's pixel limits.
constexpr int kAbi = 4;

// The order of ArucoDetectorTuning's fields in python/pose_estimator.py.
enum Tuning {
    kClipLimit, kWinSizeMax, kWinSizeStep, kThreshConstant, kMinPerimeterRate, kPolyApproxRate,
    kErrorCorrectionRate, kCornerRefineWin, kMinCornerDistRate, kMinMarkerDistRate, kMinDistToBorder,
    kPerspPixelPerCell, kPerspIgnoredMargin, kMinOtsuStdDev, kTuningCount
};

int OddAtLeast(int v, int minimum) {
    v = std::max(v, minimum);
    return v % 2 == 1 ? v : v + 1;
}

cv::aruco::DetectorParameters MakeParams(const float* t) {
    cv::aruco::DetectorParameters p;
    p.adaptiveThreshWinSizeMin = 3;
    p.adaptiveThreshWinSizeMax = OddAtLeast(static_cast<int>(t[kWinSizeMax]), 3);
    p.adaptiveThreshWinSizeStep = std::max(1, static_cast<int>(t[kWinSizeStep]));
    p.adaptiveThreshConstant = t[kThreshConstant];
    p.minMarkerPerimeterRate = std::max(0.001, static_cast<double>(t[kMinPerimeterRate]));
    p.maxMarkerPerimeterRate = 4.0;
    p.polygonalApproxAccuracyRate = std::max(0.001, static_cast<double>(t[kPolyApproxRate]));
    p.minCornerDistanceRate = std::max(0.001, static_cast<double>(t[kMinCornerDistRate]));
    p.minMarkerDistanceRate = std::max(0.001, static_cast<double>(t[kMinMarkerDistRate]));
    p.minDistanceToBorder = std::max(0, static_cast<int>(t[kMinDistToBorder]));
    p.errorCorrectionRate = std::min(1.0, std::max(0.0, static_cast<double>(t[kErrorCorrectionRate])));
    p.markerBorderBits = 1;
    p.perspectiveRemovePixelPerCell = std::max(1, static_cast<int>(t[kPerspPixelPerCell]));
    p.perspectiveRemoveIgnoredMarginPerCell = std::min(0.49, std::max(0.0, static_cast<double>(t[kPerspIgnoredMargin])));
    p.minOtsuStdDev = std::max(0.0, static_cast<double>(t[kMinOtsuStdDev]));
    p.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    p.cornerRefinementWinSize = std::max(1, static_cast<int>(t[kCornerRefineWin]));
    p.cornerRefinementMaxIterations = 50;
    p.cornerRefinementMinAccuracy = 0.02;
    return p;
}

double MsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double RotationAngleDeg(const cv::Mat& r1, const cv::Mat& r2) {
    cv::Mat a, b, R1, R2;
    r1.convertTo(a, CV_64F);   // IPPE's rvecs and a caller's prior may differ in depth
    r2.convertTo(b, CV_64F);
    cv::Rodrigues(a, R1);
    cv::Rodrigues(b, R2);
    cv::Mat R = R1.t() * R2;
    double c = std::max(-1.0, std::min(1.0, (cv::trace(R)[0] - 1.0) / 2.0));
    return std::acos(c) * 180.0 / CV_PI;
}

bool Planar(const std::vector<cv::Point3f>& pts, double tol_m = 1e-4) {
    if (pts.size() < 4) return true;
    cv::Mat m(static_cast<int>(pts.size()), 3, CV_64F);
    cv::Point3d mean(0, 0, 0);
    for (const auto& p : pts) mean += cv::Point3d(p.x, p.y, p.z);
    mean *= 1.0 / pts.size();
    for (int i = 0; i < m.rows; ++i) {
        m.at<double>(i, 0) = pts[i].x - mean.x;
        m.at<double>(i, 1) = pts[i].y - mean.y;
        m.at<double>(i, 2) = pts[i].z - mean.z;
    }
    cv::Mat w;
    cv::SVD::compute(m, w, cv::SVD::NO_UV);
    return w.at<double>(w.rows - 1) < tol_m;
}

struct Handle {
    cv::aruco::ArucoDetector detector;
    cv::aruco::DetectorParameters params;   // the frame's parameters; a region scales its perimeter rates (ABI 4)
    double perimeter_scale = 1.0;           // the scale the detector holds now
    cv::Ptr<cv::CLAHE> clahe;
    float clip = 0.f;
    std::unique_ptr<cv::aruco::Board> board;
    cv::Mat enhanced;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    std::vector<int> ids;
    std::string error;
};

}  // namespace

// Outputs of ela_process, one struct so the C# side reads them with one marshal. Keep in step with
// EyeLab.Aruco.ArucoNative.Result.
struct ElaResult {
    int32_t markers;       // markers detected (all ids)
    int32_t rejected;      // rejected candidates
    int32_t board_markers; // detected markers that belong to the board
    int32_t pose_ok;       // 1 = rvec/tvec valid
    int32_t planar;        // 1 = the matched points were planar (IPPE path)
    int32_t points;        // matched object/image points used by the solve
    double clahe_ms, detect_ms, pose_ms, total_ms;
    double rvec[3], tvec[3];
    double rms_px;         // reprojection RMS over the matched points, NaN without a pose
    double ambiguity_ratio, ambiguity_deg;  // IPPE second/best error and the angle between them; NaN otherwise
    // ABI 3
    int32_t roi_x, roi_y, roi_w, roi_h;     // the region processed (the whole frame without a ROI)
    int32_t solution;      // IPPE solution kept: 0 = lowest error, 1 = the other; -1 = not the IPPE path
    int32_t prior_used;    // 1 = a prior was given and decided between two IPPE solutions
    double proj_x0, proj_y0, proj_x1, proj_y1;  // box of every board corner projected with the pose; NaN without
};

ELA_EXPORT int ela_abi() { return kAbi; }

ELA_EXPORT int ela_info(char* buf, int n) {
    if (!buf || n <= 0) return 0;
    std::snprintf(buf, static_cast<size_t>(n), "eyelab_aruco abi %d; opencv %s; threads %d; cpus %d; neon %d", kAbi,
                  CV_VERSION, cv::getNumThreads(), cv::getNumberOfCPUs(), cv::checkHardwareSupport(CV_CPU_NEON) ? 1 : 0);
    return static_cast<int>(std::strlen(buf));
}

ELA_EXPORT int ela_set_threads(int n) {
    cv::setNumThreads(n);
    return cv::getNumThreads();
}

// Scheduling priority (nice, -20..19; higher yields more) of the calling thread only. OpenCV's pool threads keep
// theirs, so this matters most with 1 OpenCV thread. Returns the nice value now in effect, or -100 on error.
ELA_EXPORT int ela_set_nice(int nice) {
    const id_t tid = static_cast<id_t>(gettid());
    if (setpriority(PRIO_PROCESS, tid, nice) != 0) return -100;
    errno = 0;
    int now = getpriority(PRIO_PROCESS, tid);
    return errno != 0 ? -100 : now;
}

ELA_EXPORT void* ela_create(int dictionary_id, const float* tuning, int n_tuning) {
    if (!tuning || n_tuning < kTuningCount) return nullptr;
    try {
        auto h = new Handle{cv::aruco::ArucoDetector(cv::aruco::getPredefinedDictionary(dictionary_id), MakeParams(tuning))};
        h->params = MakeParams(tuning);
        h->clip = tuning[kClipLimit];
        if (h->clip > 0.f) h->clahe = cv::createCLAHE(h->clip, cv::Size(8, 8));
        return h;
    } catch (...) {
        return nullptr;
    }
}

ELA_EXPORT void ela_destroy(void* handle) { delete static_cast<Handle*>(handle); }

// Board: n markers, ids[n], object corners obj[n*12] (4 corners x xyz, metres, in the order
// marker_object_corners gives them). Returns n, or -1.
ELA_EXPORT int ela_set_board(void* handle, const int* ids, const float* obj, int n) {
    auto h = static_cast<Handle*>(handle);
    if (!h || n <= 0 || !ids || !obj) return -1;
    try {
        std::vector<std::vector<cv::Point3f>> pts(n);
        std::vector<int> v(ids, ids + n);
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 4; ++c) pts[i].emplace_back(obj[i * 12 + c * 3], obj[i * 12 + c * 3 + 1], obj[i * 12 + c * 3 + 2]);
        h->board = std::make_unique<cv::aruco::Board>(pts, h->detector.getDictionary(), v);
        return n;
    } catch (const cv::Exception& e) {
        h->error = e.what();
        return -1;
    }
}

ELA_EXPORT int ela_last_error(void* handle, char* buf, int n) {
    auto h = static_cast<Handle*>(handle);
    if (!h || !buf || n <= 0) return 0;
    std::snprintf(buf, static_cast<size_t>(n), "%s", h->error.c_str());
    return static_cast<int>(std::strlen(buf));
}

// One frame: CLAHE, detection and (with a board and K) the board pose, on the whole frame (B3a; the parity path).
ELA_EXPORT int ela_process_roi(void* handle, const uint8_t* y, int width, int height, int stride, const int* roi,
                               const double* K, const double* dist, int n_dist, const double* prior_rvec, int max_out,
                               int* ids_out, float* corners_out, ElaResult* res);

// y: 8-bit luma, rows of `stride` bytes. K: 3x3 row-major, may be null (no pose). dist: n_dist coefficients.
// ids_out[max_out], corners_out[max_out*8] receive the detected markers (all ids). Returns markers, or -1.
ELA_EXPORT int ela_process(void* handle, const uint8_t* y, int width, int height, int stride, const double* K,
                           const double* dist, int n_dist, int max_out, int* ids_out, float* corners_out, ElaResult* res) {
    return ela_process_roi(handle, y, width, height, stride, nullptr, K, dist, n_dist, nullptr, max_out, ids_out,
                           corners_out, res);
}

// T6.7. roi: {x, y, w, h} in frame pixels, or null for the whole frame; it is clipped to the frame, and a region
// under 32 px either way falls back to the whole frame. CLAHE and detection run on the crop only (so CLAHE's 8x8
// tiles cover the crop, not the frame), and the corners come back in frame pixels. prior_rvec: a rotation (Rodrigues,
// camera frame) or null; for a planar point set with two IPPE solutions, the one nearer the prior is kept, as
// pose_estimator's _solve_board_pose does with a prior.
ELA_EXPORT int ela_process_roi(void* handle, const uint8_t* y, int width, int height, int stride, const int* roi,
                               const double* K, const double* dist, int n_dist, const double* prior_rvec, int max_out,
                               int* ids_out, float* corners_out, ElaResult* res) {
    auto h = static_cast<Handle*>(handle);
    if (!h || !y || !res || width <= 0 || height <= 0 || stride < width) return -1;
    const double nan = std::nan("");
    *res = ElaResult{};
    res->rms_px = res->ambiguity_ratio = res->ambiguity_deg = nan;
    res->proj_x0 = res->proj_y0 = res->proj_x1 = res->proj_y1 = nan;
    res->solution = -1;
    try {
        auto t0 = std::chrono::steady_clock::now();
        cv::Mat gray(height, width, CV_8UC1, const_cast<uint8_t*>(y), static_cast<size_t>(stride));
        cv::Rect rect(0, 0, width, height);
        if (roi) {
            cv::Rect r = cv::Rect(roi[0], roi[1], roi[2], roi[3]) & rect;
            if (r.width >= 32 && r.height >= 32) {
                // 0.4.010: grown to multiples of 8 (CLAHE's tile grid), moved back inside the frame if needed. An odd
                // size makes CLAHE pad the crop and interpolate over fractional tiles, where the device and Python
                // differed by up to 0.74 px on 3/23 bench frames of T6.7 run 1; whole-sized tiles matched exactly.
                int w8 = std::min(width - width % 8, (r.width + 7) / 8 * 8);
                int h8 = std::min(height - height % 8, (r.height + 7) / 8 * 8);
                r.x = std::max(0, std::min(r.x, width - w8));
                r.y = std::max(0, std::min(r.y, height - h8));
                r.width = w8;
                r.height = h8;
                rect = r;
            }
        }
        res->roi_x = rect.x;
        res->roi_y = rect.y;
        res->roi_w = rect.width;
        res->roi_h = rect.height;
        cv::Mat src = gray(rect);
        const cv::Mat* in = &src;
        if (h->clahe) {
            h->clahe->apply(src, h->enhanced);
            in = &h->enhanced;
        }
        res->clahe_ms = MsSince(t0);

        auto t1 = std::chrono::steady_clock::now();
        h->corners.clear();
        h->ids.clear();
        h->rejected.clear();
        // 0.4.010 (ABI 4): ArUco's perimeter rates are fractions of the image's larger side, so on a crop they fall
        // in pixels and let smaller contours through. Scaled by frame / crop they keep the frame's pixel limits: the
        // region finds what the full frame would, and detection on run 1's regions took 30 % less time.
        const double scale = static_cast<double>(std::max(width, height)) / std::max(rect.width, rect.height);
        if (scale != h->perimeter_scale) {
            cv::aruco::DetectorParameters p = h->params;
            p.minMarkerPerimeterRate *= scale;
            p.maxMarkerPerimeterRate *= scale;
            h->detector.setDetectorParameters(p);
            h->perimeter_scale = scale;
        }
        h->detector.detectMarkers(*in, h->corners, h->ids, h->rejected);
        if (rect.x != 0 || rect.y != 0) {
            const cv::Point2f off(static_cast<float>(rect.x), static_cast<float>(rect.y));
            for (auto& c : h->corners)
                for (auto& p : c) p += off;
        }
        res->detect_ms = MsSince(t1);
        res->markers = static_cast<int32_t>(h->ids.size());
        res->rejected = static_cast<int32_t>(h->rejected.size());
        int n = std::min(res->markers, max_out);
        for (int i = 0; i < n && ids_out && corners_out; ++i) {
            ids_out[i] = h->ids[i];
            for (int c = 0; c < 4; ++c) {
                corners_out[i * 8 + c * 2] = h->corners[i][c].x;
                corners_out[i * 8 + c * 2 + 1] = h->corners[i][c].y;
            }
        }

        auto t2 = std::chrono::steady_clock::now();
        if (h->board && K && !h->ids.empty()) {
            const auto& board_ids = h->board->getIds();
            for (int id : h->ids)
                if (std::find(board_ids.begin(), board_ids.end(), id) != board_ids.end()) res->board_markers++;
            std::vector<cv::Point3f> obj;
            std::vector<cv::Point2f> img;
            if (res->board_markers > 0) h->board->matchImagePoints(h->corners, h->ids, obj, img);
            res->points = static_cast<int32_t>(obj.size());
            if (obj.size() >= 4) {
                cv::Mat Km(3, 3, CV_64F, const_cast<double*>(K));
                cv::Mat D = n_dist > 0 && dist ? cv::Mat(1, n_dist, CV_64F, const_cast<double*>(dist)) : cv::Mat();
                cv::Mat rvec, tvec;
                bool ok = false;
                if (Planar(obj)) {
                    res->planar = 1;
                    std::vector<cv::Mat> rvecs, tvecs;
                    cv::Mat errs;
                    int sol = 0;
                    try {
                        sol = cv::solvePnPGeneric(obj, img, Km, D, rvecs, tvecs, false, cv::SOLVEPNP_IPPE, cv::noArray(),
                                                  cv::noArray(), errs);
                    } catch (const cv::Exception&) {
                        sol = 0;
                    }
                    if (sol > 0) {
                        int k = 0;
                        if (sol > 1) {
                            // solvePnPGeneric returns the errors in the points' depth (float for Point2f/Point3f),
                            // so convert before reading them as double, as Python does with np.float64.
                            cv::Mat e;
                            errs.convertTo(e, CV_64F);
                            res->ambiguity_ratio = e.at<double>(1) / std::max(e.at<double>(0), 1e-9);
                            res->ambiguity_deg = RotationAngleDeg(rvecs[0], rvecs[1]);
                            if (prior_rvec) {
                                cv::Mat prior(3, 1, CV_64F, const_cast<double*>(prior_rvec));
                                k = RotationAngleDeg(rvecs[1], prior) < RotationAngleDeg(rvecs[0], prior) ? 1 : 0;
                                res->prior_used = 1;
                            }
                        }
                        res->solution = k;
                        rvec = rvecs[k];
                        tvec = tvecs[k];
                        ok = true;
                    }
                }
                if (!ok) ok = cv::solvePnP(obj, img, Km, D, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
                if (ok) {
                    try {
                        cv::solvePnPRefineLM(obj, img, Km, D, rvec, tvec);
                    } catch (const cv::Exception&) {
                    }
                    rvec.convertTo(rvec, CV_64F);
                    tvec.convertTo(tvec, CV_64F);
                    for (int k = 0; k < 3; ++k) {
                        res->rvec[k] = rvec.at<double>(k);
                        res->tvec[k] = tvec.at<double>(k);
                    }
                    std::vector<cv::Point2f> proj;
                    cv::projectPoints(obj, rvec, tvec, Km, D, proj);
                    double s = 0;
                    for (size_t i = 0; i < proj.size(); ++i) {
                        cv::Point2f d = proj[i] - img[i];
                        s += d.x * d.x + d.y * d.y;
                    }
                    res->rms_px = std::sqrt(s / proj.size());
                    res->pose_ok = 1;
                    // where the whole board is: every marker corner projected, also those not detected
                    std::vector<cv::Point3f> all;
                    for (const auto& m : h->board->getObjPoints()) all.insert(all.end(), m.begin(), m.end());
                    std::vector<cv::Point2f> pa;
                    cv::projectPoints(all, rvec, tvec, Km, D, pa);
                    if (!pa.empty()) {
                        float x0 = pa[0].x, y0 = pa[0].y, x1 = pa[0].x, y1 = pa[0].y;
                        for (const auto& p : pa) {
                            x0 = std::min(x0, p.x);
                            y0 = std::min(y0, p.y);
                            x1 = std::max(x1, p.x);
                            y1 = std::max(y1, p.y);
                        }
                        res->proj_x0 = x0;
                        res->proj_y0 = y0;
                        res->proj_x1 = x1;
                        res->proj_y1 = y1;
                    }
                }
            }
        }
        res->pose_ms = MsSince(t2);
        res->total_ms = MsSince(t0);
        return res->markers;
    } catch (const cv::Exception& e) {
        h->error = e.what();
        return -1;
    }
}
