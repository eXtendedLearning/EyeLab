#!/usr/bin/env bash
# Build the EyeLab ArUco plugin (native/eyelab_aruco) for Android arm64 and stage it in the Unity project.
#
#   bash tools/build_aruco_plugin.sh
#
# Needs: Unity 6000.4.5f1 with Android Build Support (its NDK r27c and SDK CMake 3.22.1), and the OpenCV Android
# SDK 4.13.0 unpacked at vendor/opencv-android/sdk-4.13.0/OpenCV-android-sdk (gitignored; official release
# asset opencv-4.13.0-android-sdk.zip, SHA-256 edfda20fdf65d0bd45391d168ec5261dd30b600b00279c4d910d7f1c3e020f0f).
# Output: eyelab_xreal/Assets/Plugins/Android/libs/arm64-v8a/libeyelab_aruco.so (gitignored build artefact).
# Override the locations with UNITY_ANDROID, OPENCV_ANDROID_SDK.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
android="${UNITY_ANDROID:-/c/Program Files/Unity/Hub/Editor/6000.4.5f1/Editor/Data/PlaybackEngines/AndroidPlayer}"
sdk="${OPENCV_ANDROID_SDK:-$root/vendor/opencv-android/sdk-4.13.0/OpenCV-android-sdk}"
ndk="$android/NDK"
cmake_bin="$android/SDK/cmake/3.22.1/bin"
build="$root/native/eyelab_aruco/build-arm64"
out="$root/eyelab_xreal/Assets/Plugins/Android/libs/arm64-v8a"

[ -f "$sdk/sdk/native/jni/OpenCVConfig.cmake" ] || { echo "OpenCV Android SDK not found at $sdk" >&2; exit 1; }
[ -f "$ndk/build/cmake/android.toolchain.cmake" ] || { echo "NDK not found at $ndk" >&2; exit 1; }

"$cmake_bin/cmake.exe" -S "$root/native/eyelab_aruco" -B "$build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$cmake_bin/ninja.exe" \
  -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$sdk/sdk/native/jni"
"$cmake_bin/cmake.exe" --build "$build"
mkdir -p "$out"
cp "$build/libeyelab_aruco.so" "$out/"
ls -l "$out/libeyelab_aruco.so"
