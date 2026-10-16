using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Windows display layout: for every display attached to the desktop, its monitor name, mode,
    /// position and rotation (0/90/180/270), as one text line. Logged at start and after Enter XR,
    /// and again whenever it changes (polled every 2 s). Added for the 14 Oct G2 run in which the whole
    /// XR view appeared upside down in the glasses while every recorded pose and camera frame was
    /// upright: a 180 degree rotation here would mean Windows (e.g. Surface auto-rotate) flipped it.
    /// </summary>
    public sealed class DisplayProbe : MonoBehaviour
    {
        const double PollSeconds = 2.0;

        GateRunLog m_Log;
        string m_Last;
        double m_LastPoll = double.NegativeInfinity;

        public void Attach(GateRunLog log) => m_Log = log;

        /// <summary>Logs the layout now, as a <c>display</c> event with <paramref name="stage"/> as result.</summary>
        public void Log(string stage)
        {
            if (m_Log == null) return;
            m_Last = Describe();
            m_Log.Event("display", stage, m_Last);
        }

        void Update()
        {
            if (m_Log == null) return;
            double now = Time.realtimeSinceStartupAsDouble;
            if (now - m_LastPoll < PollSeconds) return;
            m_LastPoll = now;
            string d = Describe();
            if (m_Last != null && d != m_Last)
                m_Log.Event("display", "CHANGED", d);
            m_Last = d;
        }

        public static string Describe()
        {
            var sb = new StringBuilder();
            sb.Append($"unity screen {Screen.width}x{Screen.height} mode {Screen.fullScreenMode} displays {Display.displays.Length}");
            if (Application.platform != RuntimePlatform.WindowsPlayer && Application.platform != RuntimePlatform.WindowsEditor)
                return sb.ToString();
            try
            {
                foreach (string line in Windows())
                    sb.Append(" | ").Append(line);
            }
            catch (Exception e)
            {
                sb.Append(" | windows query failed: ").Append(e.GetType().Name);
            }
            return sb.ToString();
        }

        static IEnumerable<string> Windows()
        {
            var adapter = new DISPLAY_DEVICE { cb = Marshal.SizeOf<DISPLAY_DEVICE>() };
            for (uint i = 0; EnumDisplayDevices(null, i, ref adapter, 0); i++)
            {
                if ((adapter.StateFlags & AttachedToDesktop) != 0)
                {
                    var monitor = new DISPLAY_DEVICE { cb = Marshal.SizeOf<DISPLAY_DEVICE>() };
                    string name = EnumDisplayDevices(adapter.DeviceName, 0, ref monitor, 0) ? monitor.DeviceString : "?";
                    var mode = new DEVMODE { dmSize = (short)Marshal.SizeOf<DEVMODE>() };
                    string text = EnumDisplaySettings(adapter.DeviceName, EnumCurrentSettings, ref mode)
                        ? $"{mode.dmPelsWidth}x{mode.dmPelsHeight}@{mode.dmDisplayFrequency} at {mode.dmPositionX},{mode.dmPositionY} rot {mode.dmDisplayOrientation * 90}"
                        : "mode ?";
                    bool primary = (adapter.StateFlags & PrimaryDevice) != 0;
                    yield return $"{adapter.DeviceName.TrimStart('\\', '.')} '{name}' {text}{(primary ? " primary" : "")}";
                }
                adapter = new DISPLAY_DEVICE { cb = Marshal.SizeOf<DISPLAY_DEVICE>() };
            }
        }

        const int AttachedToDesktop = 0x1, PrimaryDevice = 0x4, EnumCurrentSettings = -1;

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct DISPLAY_DEVICE
        {
            public int cb;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string DeviceName;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceString;
            public int StateFlags;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceID;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceKey;
        }

        /// <summary>DEVMODEW, display variant of the union (220 bytes).</summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct DEVMODE
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
            public short dmSpecVersion, dmDriverVersion, dmSize, dmDriverExtra;
            public int dmFields;
            public int dmPositionX, dmPositionY, dmDisplayOrientation, dmDisplayFixedOutput;
            public short dmColor, dmDuplex, dmYResolution, dmTTOption, dmCollate;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
            public short dmLogPixels;
            public int dmBitsPerPel, dmPelsWidth, dmPelsHeight, dmDisplayFlags, dmDisplayFrequency;
            public int dmICMMethod, dmICMIntent, dmMediaType, dmDitherType, dmReserved1, dmReserved2, dmPanningWidth, dmPanningHeight;
        }

        [DllImport("user32.dll", CharSet = CharSet.Unicode, EntryPoint = "EnumDisplayDevicesW")]
        static extern bool EnumDisplayDevices(string device, uint devNum, ref DISPLAY_DEVICE displayDevice, uint flags);

        [DllImport("user32.dll", CharSet = CharSet.Unicode, EntryPoint = "EnumDisplaySettingsW")]
        static extern bool EnumDisplaySettings(string deviceName, int modeNum, ref DEVMODE devMode);
    }
}
