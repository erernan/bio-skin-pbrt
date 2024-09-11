#include "materials/skin.h"
#include "spectrum.h"
#include "reflection.h"
#include "paramset.h"
#include "texture.h"
#include "interaction.h"

namespace pbrt {

// SkinMaterial Method Definitions
void SkinMaterial::ComputeScatteringFunctions(
    SurfaceInteraction *si, MemoryArena &arena, TransportMode mode,
    bool allowMultipleLobes) const {
	
	// Perform bump mapping with _bumpMap_, if present
    if (bumpMap) Bump(bumpMap, si);
    Float eta = 1.5f;
    si->bsdf = ARENA_ALLOC(arena, BSDF)(*si, eta);

    Spectrum r = reflect->Evaluate(*si).Clamp();
    //Spectrum t = Spectrum(1.0) - r;
    Spectrum t = Spectrum(0.0);

    Spectrum kd = Kd->Evaluate(*si).Clamp();
    if (!kd.IsBlack()) {
        if (!r.IsBlack())
            si->bsdf->Add(ARENA_ALLOC(arena, LambertianReflection)(r * kd));
    }
    Spectrum ks = Ks->Evaluate(*si).Clamp();
    if (!ks.IsBlack() && (!r.IsBlack() || !t.IsBlack())) {
        Float rough = roughness->Evaluate(*si);
        if (remapRoughness)
            rough = TrowbridgeReitzDistribution::RoughnessToAlpha(rough);
		// two lobe micro-facet specular lobe
        MicrofacetDistribution *distrib1 =
            ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(rough, rough);
        //MicrofacetDistribution *distrib2 =
        //    ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(rough*2, rough*2);
        if (!r.IsBlack()) {
            Fresnel *fresnel = ARENA_ALLOC(arena, FresnelDielectric)(1.f, eta);
            si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetReflection)(
                r * ks, distrib1, fresnel));
            //si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetReflection)(
            //    r * ks, distrib2, fresnel));
        }
		// might change to dielectric transmission
        if (!t.IsBlack())
            si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetTransmission)(
                t * ks, distrib1, 1.f, eta, mode));
    }
}


SkinMaterial *CreateSkinMaterial(const TextureParams &mp) { 

	std::shared_ptr<Texture<Spectrum>> Kd =
        mp.GetSpectrumTexture("Kd", Spectrum(0.25f));
    std::shared_ptr<Texture<Spectrum>> Ks =
        mp.GetSpectrumTexture("Ks", Spectrum(0.25f));
    std::shared_ptr<Texture<Spectrum>> reflect =
        mp.GetSpectrumTexture("reflect", Spectrum(0.5f));
    std::shared_ptr<Texture<Float>> roughness =
        mp.GetFloatTexture("roughness", .1f);
    std::shared_ptr<Texture<Float>> bumpMap =
        mp.GetFloatTextureOrNull("bumpmap");
    bool remapRoughness = mp.FindBool("remaproughness", true);
    return new SkinMaterial(Kd, Ks, roughness, reflect, bumpMap,
                            remapRoughness);
}

}  // namespace pbrt