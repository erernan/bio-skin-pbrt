#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_ALBEDO_H
#define PBRT_MATERIALS_ALBEDO_H

// materials/albedo.h*
#include "material.h"
#include "pbrt.h"

namespace pbrt {

// AlbedoMaterial Declarations
class AlbedoMaterial : public Material {
  public:
    // AlbedoMaterial Public Methods
    AlbedoMaterial(const std::shared_ptr<Texture<Spectrum>> &kd,
                   const std::shared_ptr<Texture<Spectrum>> &ks,
                   const std::shared_ptr<Texture<Float>> &rough,
                   const std::shared_ptr<Texture<Spectrum>> &refl,
                   const std::shared_ptr<Texture<Float>> &bump) {
        Kd = kd;
        Ks = ks;
        reflect = refl;
        bumpMap = bump;
        roughness = rough;
    }
    void ComputeScatteringFunctions(SurfaceInteraction *si, MemoryArena &arena,
                                    TransportMode mode,
                                    bool allowMultipleLobes) const;
    Spectrum GetAlbedo(SurfaceInteraction *si) const;
    std::shared_ptr<Texture<Spectrum>> GetTexture() const { return this->Kd; }
  private:
    // AlbedoMaterial Private Data
    std::shared_ptr<Texture<Spectrum>> Kd, Ks;
    std::shared_ptr<Texture<Spectrum>> reflect;
    std::shared_ptr<Texture<Float>> bumpMap;
    std::shared_ptr<Texture<Float>> roughness;
};

AlbedoMaterial *CreateAlbedoMaterial(const TextureParams &mp);

}  // namespace pbrt

#endif  // PBRT_MATERIALS_Albedo_H
