import{j as r}from"./index-Dz89mzWx.js";import"./helperFunctions-D1d7R4lh.js";import"./index-D6-13Dgy.js";import"./svelte/svelte.js";const e="rgbdEncodePixelShader",t=`varying vUV: vec2f;var textureSamplerSampler: sampler;var textureSampler: texture_2d<f32>;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
@fragment
fn main(input: FragmentInputs)->FragmentOutputs {fragmentOutputs.color=toRGBD(textureSample(textureSampler,textureSamplerSampler,input.vUV).rgb);}`;r.ShadersStoreWGSL[e]||(r.ShadersStoreWGSL[e]=t);const p={name:e,shader:t};export{p as rgbdEncodePixelShaderWGSL};
//# sourceMappingURL=rgbdEncode.fragment-DX28CJfQ.js.map
