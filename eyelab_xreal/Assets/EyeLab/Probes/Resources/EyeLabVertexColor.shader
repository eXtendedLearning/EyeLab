// T6.10: the solid overlay style (PoseOverlay). Unlit, vertex colour times _Color, with depth test and depth write so a
// box's near faces hide its far ones. Stereo-instancing macros so it draws in both eyes. In Resources so the build keeps
// it (loaded by name, no material asset). T6.11: _EyeMask draws in one eye only (0 both, 1 left only, 2 right only;
// Unity's eye index 0 is the left eye), for the per-eye SPAAM crosshair. T6.12: _EyeWarp draws each vertex where the
// wearer's eye sees it (EyeModel.cs): in the SDK eye's frame (_EyeW2E*, set every frame by EyeWarp.cs), with the eye
// centre C and w = 1/D in _EyeModel*, the point P is drawn at Q = (uv * P_z, P_z),
// uv = w C_xy + (1 - w C_z) (P_xy - C_xy) / (P_z - C_z).
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
            float _EyeWarp;
            float4x4 _EyeW2E0, _EyeW2E1, _EyeE2W0, _EyeE2W1;
            float4 _EyeModel0, _EyeModel1;

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
                float4 wpos = mul(unity_ObjectToWorld, float4(v.vertex.xyz, 1.0));
#if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_STEREO_MULTIVIEW_ENABLED) || defined(UNITY_SINGLE_PASS_STEREO)
                if (_EyeWarp > 0.5)
                {
                    float4x4 w2e = _EyeW2E1;
                    float4x4 e2w = _EyeE2W1;
                    float4 m = _EyeModel1;
                    if (unity_StereoEyeIndex == 0) { w2e = _EyeW2E0; e2w = _EyeE2W0; m = _EyeModel0; }
                    float3 P = mul(w2e, wpos).xyz;
                    float dz = P.z - m.z;
                    if (dz > 0.01)
                    {
                        float2 uv = m.w * m.xy + (1.0 - m.w * m.z) * (P.xy - m.xy) / dz;
                        wpos = mul(e2w, float4(uv * P.z, P.z, 1.0));
                    }
                }
#endif
                o.pos = mul(UNITY_MATRIX_VP, wpos);
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
