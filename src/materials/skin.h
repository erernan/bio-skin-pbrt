
#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_SKIN_H
#define PBRT_MATERIALS_SKIN_H

 // materials/skin.h*
#include "pbrt.h"
#include "material.h"

namespace pbrt {

	// SkinMaterial Declarations
	class SkinMaterial : public Material {
	public:
		// SkinMaterial Public Methods
		SkinMaterial(const std::shared_ptr<Texture<Spectrum>> &kd,
			const std::shared_ptr<Texture<Spectrum>> &ks,
			const std::shared_ptr<Texture<Float>> &rough,
			const std::shared_ptr<Texture<Spectrum>> &refl,
			const std::shared_ptr<Texture<Float>> &bump,
			bool remap) {
			Kd = kd;
			Ks = ks;
			roughness = rough;
			reflect = refl;
			bumpMap = bump;
			remapRoughness = remap;
		}
		void ComputeScatteringFunctions(SurfaceInteraction *si, MemoryArena &arena,
			TransportMode mode,
			bool allowMultipleLobes) const;

	private:
		// SkinMaterial Private Data
		std::shared_ptr<Texture<Spectrum>> Kd, Ks;
		std::shared_ptr<Texture<Float>> roughness;
		std::shared_ptr<Texture<Spectrum>> reflect;
		std::shared_ptr<Texture<Float>> bumpMap;
		bool remapRoughness;
	};

	SkinMaterial *CreateSkinMaterial(const TextureParams &mp);

}  // namespace pbrt

#endif  // PBRT_MATERIALS_SKIN_H
