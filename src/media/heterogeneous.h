// ABORT
#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MEDIA_HETEROGONEOUS_H
#define PBRT_MEDIA_HETEROGENEOUS_H

// media/heterogeneous.h*
#include "medium.h"

namespace pbrt {

// HeterogeneousMedium Declarations
class HeterogeneousMedium : public Medium {
  public:
    // HeterogeneousMedium Public Methods
    HeterogeneousMedium(const Spectrum sigma_s, const Spectrum sigma_a, const Float maxT,
                        Float g, const int32_t flag,
                        const std::shared_ptr<Texture<Spectrum>> &Kd, const float scale)
        : maxT(maxT),
          Kd(Kd),
          g(g),
          flag(flag),
          sigma_s(sigma_s),
          sigma_a(sigma_a),
		  sigma_t(sigma_a + sigma_s),
		  scale(scale){}
		
    Spectrum Tr(const Ray &ray, Sampler &sampler, int channel = 0) const; 
	
	Spectrum Tr(const Ray &ray, Sampler &sampler,
                const Scene &scene, SurfaceInteraction &isect,
                const std::vector<Vector3f> &colorvec,
                const std::vector<Vector3f> &sig_aouter,
                const std::vector<Vector3f> &sig_ainner) const;

    Spectrum Sample(const Ray &ray, Sampler &sampler, MemoryArena &arena,
                    MediumInteraction *mi, int channel = 0) const;

	Spectrum Sample(const Ray &ray, Sampler &sampler, MemoryArena &arena,
                    MediumInteraction *mi, Spectrum w_front, int channel = 0) const;

	Spectrum SampleKd(const Ray &ray, Sampler &sampler, MemoryArena &arena,
                          MediumInteraction *mi, Spectrum w_front,
						  std::shared_ptr<Texture<Spectrum>> &Kd,
                          SurfaceInteraction &isect, const Scene &scene,
                          int flag, const std::vector<Vector3f> &colorvec,
                          const std::vector<Vector3f> &sig_aouter,
                          const std::vector<Vector3f> &sig_ainner) const;

    int32_t GetFlag() const { return this->flag; }

	std::shared_ptr<Texture<Spectrum>> getKd() const { return this->Kd; }
	// compute the us at a point
    Spectrum MiuS(const Point3f &p) const;
	void FindCloest(Spectrum &color, Spectrum &sigmaa, int32_t flag,
                    const std::vector<Vector3f> &colorvec,
                    const std::vector<Vector3f> &sig_aouter,
                    const std::vector<Vector3f> &sig_ainner) const;


  private:
    // HeterogeneousMedium Private Data
    std::shared_ptr<Texture<Spectrum>> Kd;
    const Float g;
    const Spectrum sigma_s, sigma_t, sigma_a;
    const Float scale;
	// add flag, flag indicates which layer is this medium, the outer or inner or none medium
    const int flag; // 0 for none, 1 for outer 2 for inner

	// need to know the max ��t before compute Tr and sample points
    const Float maxT = 1.0;


};

}  // namespace pbrt

#endif  // PBRT_MEDIA_HETEROGONEOUS_H
