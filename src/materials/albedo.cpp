
// materials/albedo.cpp*
#include "interaction.h"
#include "materials/albedo.h"
#include "paramset.h"
#include "reflection.h"
#include "texture.h"

namespace pbrt {

// AlbedoMaterial Method Definitions
void AlbedoMaterial::ComputeScatteringFunctions(SurfaceInteraction *si,
                                               MemoryArena &arena,
                                               TransportMode mode,
                                               bool allowMultipleLobes) const {

    // Evaluate textures for AlbedoMaterial_ material and allocate BRDF
    if (bumpMap) Bump(bumpMap, si);
    Float eta = 1.5f;
    si->bsdf = ARENA_ALLOC(arena, BSDF)(*si, eta);
    Spectrum r = reflect->Evaluate(*si).Clamp();
    Spectrum t = Spectrum(1.0) - r;
	Spectrum ks = Ks->Evaluate(*si).Clamp();
 //   if (!ks.IsBlack() && (!r.IsBlack()||!t.IsBlack())) {
	//	Float rough = roughness->Evaluate(*si);
 //       // two lobe micro-facet specular lobe
 //       MicrofacetDistribution *distrib1 =
 //           ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(rough, rough);
 //       MicrofacetDistribution *distrib2 =
	//		ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(rough*0.5, rough*0.5);
 //       if (!r.IsBlack()) {
 //           Fresnel *fresnel = ARENA_ALLOC(arena, FresnelDielectric)(1.f, eta);
 //           si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetReflection)(
 //               r * ks, distrib1, fresnel));
 //           si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetReflection)(
 //               r * ks, distrib2, fresnel));
 //       }
 //       if (!t.IsBlack())
 //           si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetTransmission)(
 //               t * ks, distrib1, 1.f, eta, mode)); 
 //   
	//} else {
 //       si->bsdf = nullptr;
	//}
	//
        si->bsdf = nullptr;
}

AlbedoMaterial *CreateAlbedoMaterial(const TextureParams &mp) {
    std::shared_ptr<Texture<Spectrum>> Kd =
        mp.GetSpectrumTexture("Kd", Spectrum(0.0f));
    std::shared_ptr<Texture<Spectrum>> Ks =
        mp.GetSpectrumTexture("Ks", Spectrum(0.0f));
    std::shared_ptr<Texture<Spectrum>> reflect =
        mp.GetSpectrumTexture("reflect", Spectrum(0.0f));
    std::shared_ptr<Texture<Float>> bumpMap =
        mp.GetFloatTextureOrNull("bumpmap");
    std::shared_ptr<Texture<Float>> roughness =
        mp.GetFloatTexture("roughness", .1f);
    return new AlbedoMaterial(Kd,Ks,roughness,reflect,bumpMap);
}

Spectrum AlbedoMaterial::GetAlbedo(SurfaceInteraction *si) const
{
    return Kd->Evaluate(*si).Clamp();
}

}  // namespace pbrt
