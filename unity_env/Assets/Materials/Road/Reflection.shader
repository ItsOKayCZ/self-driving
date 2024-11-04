Shader "Custom/Reflection"
{
    Properties
    {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        [Maincolor] _MainColor ("Main color", Color) = (1., 1., 0., 1.)
        _SetTexture ("Texture set", Float) = 0.
        _NoiseTexture ("Noise texture", 2D) = "white" {}
        _ReflectionStrength ("Reflection strength", Float) = 1.0
        _NoiseScaleX ("Noise scale X", Float) = 1.0
        _NoiseScaleY ("Noise scale Y", Float) = 1.0
        _Speed ("Noise speed", Range(0, 1)) = 0.5
        _ReflectionColor ("Reflection color", Color) = (1., 1., 1., 1.)
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
        sampler2D _NoiseTexture;
        float4 _MainColor;
        float _ReflectionStrength;
        float _NoiseScaleX;
        float _NoiseScaleY;
        float _Speed;
        float4 _ReflectionColor;
        float _SetTexture;

        struct Input
        {
            float2 uv_MainTex;
            float3 worldRefl;
            float3 worldPos;
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

        float ClassicNoise_impl(float3 pi0, float3 pf0, float3 pi1, float3 pf1)
        {
            pi0 = wglnoise_mod289(pi0);
            pi1 = wglnoise_mod289(pi1);

            float4 ix = float4(pi0.x, pi1.x, pi0.x, pi1.x);
            float4 iy = float4(pi0.y, pi0.y, pi1.y, pi1.y);
            float4 iz0 = pi0.z;
            float4 iz1 = pi1.z;

            float4 ixy = wglnoise_permute(wglnoise_permute(ix) + iy);
            float4 ixy0 = wglnoise_permute(ixy + iz0);
            float4 ixy1 = wglnoise_permute(ixy + iz1);

            float4 gx0 = lerp(-1, 1, frac(floor(ixy0 / 7) / 7));
            float4 gy0 = lerp(-1, 1, frac(floor(ixy0 % 7) / 7));
            float4 gz0 = 1 - abs(gx0) - abs(gy0);

            bool4 zn0 = gz0 < -0.01;
            gx0 += zn0 * (gx0 < -0.01 ? 1 : -1);
            gy0 += zn0 * (gy0 < -0.01 ? 1 : -1);

            float4 gx1 = lerp(-1, 1, frac(floor(ixy1 / 7) / 7));
            float4 gy1 = lerp(-1, 1, frac(floor(ixy1 % 7) / 7));
            float4 gz1 = 1 - abs(gx1) - abs(gy1);

            bool4 zn1 = gz1 < -0.01;
            gx1 += zn1 * (gx1 < -0.01 ? 1 : -1);
            gy1 += zn1 * (gy1 < -0.01 ? 1 : -1);

            float3 g000 = normalize(float3(gx0.x, gy0.x, gz0.x));
            float3 g100 = normalize(float3(gx0.y, gy0.y, gz0.y));
            float3 g010 = normalize(float3(gx0.z, gy0.z, gz0.z));
            float3 g110 = normalize(float3(gx0.w, gy0.w, gz0.w));
            float3 g001 = normalize(float3(gx1.x, gy1.x, gz1.x));
            float3 g101 = normalize(float3(gx1.y, gy1.y, gz1.y));
            float3 g011 = normalize(float3(gx1.z, gy1.z, gz1.z));
            float3 g111 = normalize(float3(gx1.w, gy1.w, gz1.w));

            float n000 = dot(g000, pf0);
            float n100 = dot(g100, float3(pf1.x, pf0.y, pf0.z));
            float n010 = dot(g010, float3(pf0.x, pf1.y, pf0.z));
            float n110 = dot(g110, float3(pf1.x, pf1.y, pf0.z));
            float n001 = dot(g001, float3(pf0.x, pf0.y, pf1.z));
            float n101 = dot(g101, float3(pf1.x, pf0.y, pf1.z));
            float n011 = dot(g011, float3(pf0.x, pf1.y, pf1.z));
            float n111 = dot(g111, pf1);

            float3 fade_xyz = wglnoise_fade(pf0);
            float4 n_z = lerp(float4(n000, n100, n010, n110),
                            float4(n001, n101, n011, n111), fade_xyz.z);
            float2 n_yz = lerp(n_z.xy, n_z.zw, fade_xyz.y);
            float n_xyz = lerp(n_yz.x, n_yz.y, fade_xyz.x);
            return 1.46 * n_xyz;
        }

        // Classic Perlin noise
        float ClassicNoise(float3 p)
        {
            float3 i = floor(p);
            float3 f = frac(p);
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
            // float2 noiseInput = IN.uv_MainTex * _NoiseScale;
            // float2 noiseInput = IN.worldPos.xz;
            // noiseInput.x *= _NoiseScaleX;
            // noiseInput.y *= _NoiseScaleY;
            // float noiseValue = clamp(ClassicNoise(float3(noiseInput.x, noiseInput.y, _Time.y * _Speed)), 0.0, 1.0) * _ReflectionStrength;

            // float2 noiseInput2 = IN.worldPos.xz * .1;
            // noiseInput2.x *= 1.5;
            // noiseInput2.y *= .01;
            // float noiseValue2 = clamp(ClassicNoise(float3(noiseInput2.x, noiseInput2.y, _Time.y * _Speed * 2.)), 0., 1.) * 1.;

            // fixed4 albedo = tex2D(_MainTex, IN.uv_MainTex) * _SetTexture + _MainColor * (1. - _SetTexture);

            // o.Albedo = albedo + clamp(noiseValue + noiseValue2, 0., 1.) * _ReflectionColor.rgb;

            float2 noiseInput = IN.worldPos.xz * .02;
            noiseInput.x += -_Time * _Speed;
            noiseInput.y += _Time * _Speed;
            noiseInput.x *= .3 * _NoiseScaleX;
            noiseInput.y *= .05 * _NoiseScaleY;
            float noiseValue = clamp((tex2D(_NoiseTexture, noiseInput) - 0.1) * 2., 0., 1.) * _ReflectionStrength;

            fixed4 albedo = tex2D(_MainTex, IN.uv_MainTex) * _SetTexture + _MainColor * (1. - _SetTexture);

            o.Albedo = albedo + noiseValue * _ReflectionColor.rgb;
        }
        ENDCG
    }
    FallBack "Diffuse"
}