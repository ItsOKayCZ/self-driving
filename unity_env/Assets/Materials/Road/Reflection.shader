Shader "Custom/Reflection"
{
    Properties
    {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _ReflectionStrength ("Reflection strength", Range(0, 1)) = 0.5
        _NoiseScale ("Noise scale", Float) = 1.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
        #pragma surface surf Standard fullforwardshadows

        // Use shader model 3.0 target, to get nicer looking lighting
        #pragma target 3.0

        sampler2D _MainTex;
        float _ReflectionStrength;
        float _NoiseScale;

        struct Input
        {
            float2 uv_MainTex;
            float3 worldRefl;
            float3 viewDir;
        };

        float wglnoise_mod(float x, float y)
        {
            return x - y * floor(x / y);
        }

        float2 wglnoise_mod(float2 x, float2 y)
        {
            return x - y * floor(x / y);
        }

        float3 wglnoise_mod(float3 x, float3 y)
        {
            return x - y * floor(x / y);
        }

        float4 wglnoise_mod(float4 x, float4 y)
        {
            return x - y * floor(x / y);
        }

        float2 wglnoise_fade(float2 t)
        {
            return t * t * t * (t * (t * 6 - 15) + 10);
        }

        float3 wglnoise_fade(float3 t)
        {
            return t * t * t * (t * (t * 6 - 15) + 10);
        }

        float wglnoise_mod289(float x)
        {
            return x - floor(x / 289) * 289;
        }

        float2 wglnoise_mod289(float2 x)
        {
            return x - floor(x / 289) * 289;
        }

        float3 wglnoise_mod289(float3 x)
        {
            return x - floor(x / 289) * 289;
        }

        float4 wglnoise_mod289(float4 x)
        {
            return x - floor(x / 289) * 289;
        }

        float3 wglnoise_permute(float3 x)
        {
            return wglnoise_mod289((x * 34 + 1) * x);
        }

        float4 wglnoise_permute(float4 x)
        {
            return wglnoise_mod289((x * 34 + 1) * x);
        }

        float ClassicNoise_impl(float2 pi0, float2 pf0, float2 pi1, float2 pf1)
        {
            pi0 = wglnoise_mod289(pi0); // To avoid truncation effects in permutation
            pi1 = wglnoise_mod289(pi1);

            float4 ix = float2(pi0.x, pi1.x).xyxy;
            float4 iy = float2(pi0.y, pi1.y).xxyy;
            float4 fx = float2(pf0.x, pf1.x).xyxy;
            float4 fy = float2(pf0.y, pf1.y).xxyy;

            float4 i = wglnoise_permute(wglnoise_permute(ix) + iy);

            float4 phi = i / 41 * 3.14159265359 * 2;
            float2 g00 = float2(cos(phi.x), sin(phi.x));
            float2 g10 = float2(cos(phi.y), sin(phi.y));
            float2 g01 = float2(cos(phi.z), sin(phi.z));
            float2 g11 = float2(cos(phi.w), sin(phi.w));

            float n00 = dot(g00, float2(fx.x, fy.x));
            float n10 = dot(g10, float2(fx.y, fy.y));
            float n01 = dot(g01, float2(fx.z, fy.z));
            float n11 = dot(g11, float2(fx.w, fy.w));

            float2 fade_xy = wglnoise_fade(pf0);
            float2 n_x = lerp(float2(n00, n01), float2(n10, n11), fade_xy.x);
            float n_xy = lerp(n_x.x, n_x.y, fade_xy.y);
            return 1.44 * n_xy;
        }

        // Classic Perlin noise
        float ClassicNoise(float2 p)
        {
            float2 i = floor(p);
            float2 f = frac(p);
            return ClassicNoise_impl(i, f, i + 1, f - 1);
        }

        // Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
        // See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
        // #pragma instancing_options assumeuniformscaling
        UNITY_INSTANCING_BUFFER_START(Props)
            // put more per-instance properties here
        UNITY_INSTANCING_BUFFER_END(Props)

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            // fixed4 albedo = tex2D(_MainTex, IN.uv_MainTex);
            float2 noiseInput = IN.uv_MainTex * _NoiseScale;
            float noiseValue = ClassicNoise(noiseInput);

            fixed4 albedo = tex2D(_MainTex, IN.uv_MainTex);
            o.Albedo = albedo * noiseValue - noiseValue;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
