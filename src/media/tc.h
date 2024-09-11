// USE THIS
#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MEDIA_TCMEDIUM_H
#define PBRT_MEDIA_TCMEDIUM_H

// media/tc.h*
#include "medium.h"

namespace pbrt {

// TCMedium Declarations
class TCMedium : public Medium {
  public:
    // TCMedium Public Methods
    TCMedium(const Spectrum &sigma_a, const Spectrum &sigma_s, Float g,
             const int32_t flag, const std::shared_ptr<Texture<Spectrum>> &sig_a,
             Spectrum maxT, Float scale)
        : sigma_a(sigma_a),
          sigma_s(sigma_s),
          sigma_t(sigma_s + sigma_a),
          g(g),
          sig_a(sig_a),
		      flag(flag),
	        maxT(maxT),
		      scale(scale){}
		
    Spectrum Tr(const Ray &ray, Sampler &sampler, int channel, const Scene &scene,
                SurfaceInteraction &isect,
                const std::vector<Vector3f> &colorvec,
                const std::vector<Vector3f> &sig_aouter,
                const std::vector<Vector3f> &sig_ainner) const;
    Spectrum Tr(const Ray &ray, Sampler &sampler, int channel = 0) const;
    Spectrum Sample(const Ray &ray, Sampler &sampler, MemoryArena &arena,
                    MediumInteraction *mi, int channel)const;
    Spectrum SampleKd(const Ray &ray, Sampler &sampler, MemoryArena &arena,
                      MediumInteraction *mi, int channel,
                      SurfaceInteraction &isect, const Scene &scene,
                      const std::vector<Vector3f> &colorvec,
                      const std::vector<Vector3f> &sig_aouter,
                      const std::vector<Vector3f> &sig_ainner) const;
    int32_t GetFlag() const { return this->flag; }
	
	std::shared_ptr<Texture<Spectrum>> get_sig_a() const { return this->sig_a; }


  private:
    // TCMedium Private Data
    Spectrum sigma_a, sigma_s, sigma_t;
    std::shared_ptr<Texture<Spectrum>> sig_a;
    Spectrum maxT;
    Float scale;
    const Float g;
	// add flag, flag indicates which layer is this medium, the outer or inner or none medium
    const int flag; // 0 for none, 1 for outer 2 for inner
};

}  // namespace pbrt

#endif  // PBRT_MEDIA_TCMedium_H
