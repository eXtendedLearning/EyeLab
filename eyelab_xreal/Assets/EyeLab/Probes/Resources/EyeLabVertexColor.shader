// T6.10: the solid overlay style (PoseOverlay). Unlit, vertex colour times _Color, with depth test and depth write so a
// box's near faces hide its far ones. Stereo-instancing macros so it draws in both eyes. In Resources so the build keeps
// it (loaded by name, no material asset). T6.11: _EyeMask draws in one eye only (0 both, 1 left only, 2 right only;
// Unity's eye index 0 is the left eye), for the per-eye SPAAM crosshair.
Shader "EyeLab/VertexColorUnlit"
{
    Properties
    {
        _Color ("Color", Color) = (1, 1, 1, 1)
        _EyeMask ("Eye mask (0 both, 1 left, 2 right)", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }
        Pass
        {
            ZWrite On
            ZTest LEqual
            Cull Off
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            fixed4 _Color;
            float _EyeMask;

            struct appdata
            {
                float4 vertex : POSITION;
                fixed4 color : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                fixed4 color : COLOR;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = UnityObjectToClipPos(v.vertex);
#if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_STEREO_MULTIVIEW_ENABLED) || defined(UNITY_SINGLE_PASS_STEREO)
                // the other eye's vertices go outside the clip volume
                if ((_EyeMask > 0.5 && _EyeMask < 1.5 && unity_StereoEyeIndex != 0) || (_EyeMask > 1.5 && unity_StereoEyeIndex != 1))
                    o.pos = float4(2, 2, 2, 1);
#endif
                o.color = v.color * _Color;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return i.color;
            }
            ENDCG
        }
    }
}
