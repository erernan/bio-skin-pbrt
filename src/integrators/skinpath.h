// ABORT!!!!
#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_SKINPATH_H
#define PBRT_INTEGRATORS_SKINPATH_H

// integrators/skinpath.h*
#include "pbrt.h"
#include "integrator.h"
#include "lightdistrib.h"

namespace pbrt {

// SkinPathIntegrator Declarations
class SkinPathIntegrator : public SamplerIntegrator {
  public:
    // VolPathIntegrator Public Methods
    SkinPathIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
                      std::shared_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds, Float rrThreshold = 1,
                      const std::string &lightSampleStrategy = "spatial")
        : SamplerIntegrator(camera, sampler, pixelBounds),
          maxDepth(maxDepth),
          rrThreshold(rrThreshold),
          lightSampleStrategy(lightSampleStrategy) { }
    void Preprocess(const Scene &scene, Sampler &sampler);
    int id() { return 1; }
    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;

  private:
    // SkinPathIntegrator Private Data
    const int maxDepth;
    const Float rrThreshold;
    const std::string lightSampleStrategy;
    std::unique_ptr<LightDistribution> lightDistribution;

	// add method
    void FindCloest(Spectrum &color, Spectrum &sigmas, int32_t flag) const;
};

SkinPathIntegrator *CreateSkinPathIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_SKINPATH_H
