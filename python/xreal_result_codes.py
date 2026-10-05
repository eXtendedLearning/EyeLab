#!/usr/bin/env python3
"""NRSDK result codes, recovered from the vendor host's own metadata.

Every EyeLab probe until 2026-09-23 reported raw integers, on the rule that
naming them would be a guess (see ``xreal_native.NR_SUCCESS``). The names
below are not a guess. Nebula for Windows is an IL2CPP build, so its managed
enums survive in ``Nebula for Windows_Data/il2cpp_data/Metadata/global-metadata.dat``:
``XREALErrorCode`` is there in declaration order, with its constants stored as
zigzag varints in the field-default blob. The block below decodes cleanly and
sequentially (0-24), which is its own corroboration. Method and evidence:
``docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md`` section 11.

Codes above :data:`VERIFIED_MAX` exist in the enum - channel errors, permission
errors - but their constants did not decode unambiguously, so they are
deliberately absent here. :func:`describe` calls them unknown rather than
inventing a name.

Pure data and pure functions: no ctypes, no hardware, testable anywhere.
"""

from __future__ import annotations

from typing import Optional

#: ``XREALErrorCode`` 0-24, declaration order, constants verified.
RESULT_NAMES: dict[int, str] = {
    0: "Success",
    1: "Failure",
    2: "InvalidArgument",
    3: "NotEnoughMemory",
    4: "UnSupported",
    5: "GlassesDisconnect",
    6: "SdkVersionMismatch",
    7: "SdcardPermissionDeny",
    8: "RGBCameraDeviceNotFind",
    9: "DPDeviceNotFind",
    10: "TrackingNotRunning",
    11: "GetDisplayFailure",
    12: "GetDisplayModeMismatch",
    13: "InTheCoolDown",
    14: "UnSupportedHandtrackingCalculation",
    15: "Busy",
    16: "Processing",
    17: "NumberLimited",
    18: "DisplayNoInStereoMode",
    19: "InvalidData",
    20: "NotFindRuntime",
    21: "TimeOut",
    22: "LicenseFeatureUnsupported",
    23: "LicenseDeviceUnsupported",
    24: "LicenseExpiration",
}

#: Above this, the enum continues but the constants were not recovered.
VERIFIED_MAX = 24

SUCCESS = 0

#: The licence block. A call that returns one of these was refused before it
#: did any work: no device traffic, no state change. 22 is what S3's first
#: ``NRImuCreate`` returned.
LICENCE_CODES = (22, 23, 24)

LICENSE_FEATURE_UNSUPPORTED = 22


def name(code: Optional[int]) -> Optional[str]:
    """``22`` -> ``"LicenseFeatureUnsupported"``; unrecovered codes -> ``None``."""
    if code is None:
        return None
    return RESULT_NAMES.get(code)


def describe(code: Optional[int]) -> str:
    """Render a code for humans, never inventing a name.

    ``22`` -> ``"22 (LicenseFeatureUnsupported)"``, ``31`` -> ``"31 (unnamed)"``,
    ``None`` -> ``"not called"``.
    """
    if code is None:
        return "not called"
    known = RESULT_NAMES.get(code)
    return f"{code} ({known})" if known else f"{code} (unnamed)"


def is_licence_failure(code: Optional[int]) -> bool:
    """True for the licence block, which is a refusal rather than a fault."""
    return code in LICENCE_CODES

#: ``NotTrackingReason`` - why the head tracker is not tracking. Same source and
#: method as :data:`RESULT_NAMES` (Nebula's ``Unity.XR.XREAL`` metadata), and the
#: whole block decodes sequentially. ``None`` (0) means tracking is fine.
#: ``NRHeadPoseGetTrackingReason`` returns one of these.
NOT_TRACKING_REASONS: dict[int, str] = {
    0: "None",
    1: "Initializing",
    2: "Relocalizing",
    3: "InsufficientLight",
    4: "InsufficientFeatures",
    5: "ExcessiveMotion",
    6: "Unsupported",
    7: "CameraUnavailable",
    8: "Scanning",
}

#: Reasons that mean the tracker is working on it rather than refusing.
TRANSIENT_REASONS = (1, 2, 8)


def describe_reason(reason: Optional[int]) -> str:
    """``1`` -> ``"1 (Initializing)"``; unknown -> ``"9 (unnamed)"``."""
    if reason is None:
        return "not read"
    known = NOT_TRACKING_REASONS.get(reason)
    return f"{reason} ({known})" if known else f"{reason} (unnamed)"
