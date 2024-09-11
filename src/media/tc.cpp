
// USE THIS
// media/tc.cpp*
#include "media/tc.h"
#include "interaction.h"
#include "paramset.h"
#include "sampler.h"
#include "scene.h"
#include "stats.h"

namespace pbrt {

// TCMedium Method Definitions
Spectrum TCMedium::Tr(const Ray &ray, Sampler &sampler, int cha,
                      const Scene &scene, SurfaceInteraction &isect,
                      const std::vector<Vector3f> &colorvec,
                      const std::vector<Vector3f> &sig_aouter,
                      const std::vector<Vector3f> &sig_ainner) const {
    ProfilePhase _(Prof::MediumTr);
    CHECK(cha >= 0 && cha <= 2);
    int channel = cha;

    // Perform ratio tracking to estimate the transmittance value
    Float Tr = 1, t = 0.0;
	//
    CHECK(ray.tMax != Infinity);
	// Optimize. Only need one intersection per Tr and sample. Because ray.d doesn't 
	// change during compute
    Ray ray_reverse = Ray(ray(ray.tMax * 0.5), -ray.d);   // Ray origin is the middle of ray
    SurfaceInteraction isect_reverse;
    bool foundReverse = scene.Intersect(ray_reverse, &isect_reverse);
	// Get uv onece 
    Point2f uv_reverse, uv_front;
    uv_front = isect.uv;
    uv_reverse = isect_reverse.uv;

    while (true) {
        Float dist = -std::log(1 - sampler.Get1D()) * (1.0 / maxT[channel]);
        t += std::min(dist / ray.d.Length(), ray.tMax);
        if (t >= ray.tMax) break;
        // interpolate to get uv at ray(t)
        Point2f uv{0.0, 0.0};
        Float bilinear = (ray.tMax - t) / (ray.tMax * 0.5 + ray_reverse.tMax);
        //Float bilinear = 1.0;
        if (bilinear > 0.99) {
            uv = uv_reverse;
        } else if (bilinear <= 0.01) {
            uv = uv_front;
        } else {
            uv = bilinear * uv_reverse + (1.0 - bilinear) * uv_front;
        }
        // Get uv
        SurfaceInteraction isect_albedo;
        isect_albedo.uv = uv;
        // Albedo2sigma
        Spectrum sigma_a_albedo = sig_a -> Evaluate(isect_albedo);
        Float sigma_t_channel = (sigma_a_albedo * scale + sigma_s)[channel];
        Tr *= 1 - std::max((Float)0, sigma_t_channel * (1.0 / maxT[channel]));
        // Added after book publication: when transmittance gets low,
        // start applying Russian roulette to terminate sampling.
        const Float rrThreshold = .001;
        if (Tr < rrThreshold) {
            Float q = std::max((Float).05, 1 - Tr);
            if (sampler.Get1D() < q) return 0;
            Tr /= 1 - q;
        }
    }
    return Spectrum(Tr);
}

Spectrum TCMedium::Tr(const Ray &ray, Sampler &sampler, int channel) const {
    return Spectrum();
}

Spectrum TCMedium::Sample(const Ray &ray, Sampler &sampler, MemoryArena &arena,
                          MediumInteraction *mi, int cha) const {
    ProfilePhase _(Prof::MediumSample);
    // Sample a channel and distance along the ray
    CHECK(cha >= 0 && cha <= 2);
    int channel = cha;
    Spectrum sig_t_cha = Spectrum(sigma_t[cha]);
    // Run delta-tracking iterations to sample a medium interaction
    Float t = 0;
    while (true) {
        Float dist = -std::log(1 - sampler.Get1D()) / maxT[channel];
        t += std::min(dist / ray.d.Length(), ray.tMax);
        if (t >= ray.tMax) break;
        if ((sig_t_cha[0] / maxT[channel]) > sampler.Get1D()) {
            // Populate _mi_ with medium interaction information and return
            PhaseFunction *phase = ARENA_ALLOC(arena, HenyeyGreenstein)(g);
            *mi = MediumInteraction(ray(t), -ray.d, ray.time, this,
                                    ARENA_ALLOC(arena, HenyeyGreenstein)(g));
            return Spectrum(sigma_s[cha]) / sig_t_cha;
        }
    }
    return Spectrum(1.f);
}

Spectrum TCMedium::SampleKd(const Ray &ray, Sampler &sampler,
                            MemoryArena &arena, MediumInteraction *mi, int cha,
                            SurfaceInteraction &isect, const Scene &scene,
							const std::vector<Vector3f> &colorvec,
                            const std::vector<Vector3f> &sig_aouter,
                            const std::vector<Vector3f> &sig_ainner) const {
    ProfilePhase _(Prof::MediumSample);
    // Sample a channel and distance along the ray
    CHECK(cha >= 0 && cha <= 2);
    int channel = cha;
    // Run delta-tracking iterations to sample a medium interaction
    Float t = 0;
    // Optimize. Only need one intersection per Tr and sample. Because ray.d
    // doesn't
    // change during compute
    Ray ray_reverse = Ray(ray(ray.tMax * 0.5), -ray.d);  // Ray origin is the middle of ray
    SurfaceInteraction isect_reverse;
    bool foundReverse = scene.Intersect(ray_reverse, &isect_reverse);
    // Get uv onece
    Point2f uv_reverse, uv_front;
    uv_front = isect.uv;
    uv_reverse = isect_reverse.uv;

    while (true) {
        Float dist = -std::log(1 - sampler.Get1D()) / maxT[channel];
        t += std::min(dist / ray.d.Length(), ray.tMax);
        if (t >= ray.tMax) break;

        // interpolate to get uv at ray(t)
        Point2f uv{0.0, 0.0};
        Float bilinear = (ray.tMax - t) / (ray.tMax * 0.5 + ray_reverse.tMax);
        //Float bilinear = 1.0;
        if (bilinear > 0.99) {
            uv = uv_reverse;
        } else if (bilinear <= 0.01) {
            uv = uv_front;
        } else {
            uv = bilinear * uv_reverse + (1.0 - bilinear) * uv_front;
        }
        // Get uv
        SurfaceInteraction isect_albedo;
        isect_albedo.uv = uv;
        // Albedo2sigma
        Spectrum sigma_a_albedo = sig_a->Evaluate(isect_albedo);
        Float sigma_t_channel = (sigma_a_albedo * scale + sigma_s)[channel];
        if ((sigma_t_channel / maxT[channel]) > sampler.Get1D()) {
            // Populate _mi_ with medium interaction information and return
            PhaseFunction *phase = ARENA_ALLOC(arena, HenyeyGreenstein)(g);
            *mi = MediumInteraction(ray(t), -ray.d, ray.time, this,
                                    ARENA_ALLOC(arena, HenyeyGreenstein)(g));
            return Spectrum(sigma_s[channel]) / sigma_t_channel;
        }
    }
    return Spectrum(1.f);
}
}